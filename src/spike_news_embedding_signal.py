"""Embedding-based news signal tests for volatility spike days.

This script is deliberately standalone so it does not change the existing pilot
or detailed-analysis code paths. It focuses on the next question after metadata
tests: does the actual embedded news text help on spike days, and does it beat
simple placebo variants?

Example:

    python -m src.spike_news_embedding_signal \
      --num_tickers 25 \
      --forecast_horizons 1 3 5 \
      --targets log_gk total_volatility log_abs_return \
      --text_configurations target_only related_only target_sector news_all \
      --embedding_device cuda \
      --output_dir outputs/spike_embedding_signal_tests
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, mean_absolute_error, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, RANDOM_SEED, TEXT_CACHE_VERSION
from .detailed_news_signal import TargetSpec, split_by_time
from .features import add_detailed_volatility_targets, add_volatility_features, target_column_for, time_series_feature_columns
from .load_data import load_fintexts
from .models import make_ridge_model
from .text_features import detect_text_columns, joined_text_for_configuration, normalize_text_value, write_text_schema_validation


SPIKE_PERCENTILES = [85, 90, 95, 97.5]
DEFAULT_TEXT_CONFIGS = ["target_only", "related_only", "target_sector", "news_all"]
DEFAULT_ALIGNMENT_MODES = ["same_day_text", "shifted_1_day_text", "window_3d_text", "window_5d_text"]
PLACEBO_VARIANTS = ["correct_text", "shuffled_within_ticker", "cross_ticker_same_day", "stale_20d"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run embedding-based FinTexTS news signal tests on volatility spike days.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "total_volatility", "log_abs_return"])
    parser.add_argument("--text_configurations", nargs="+", default=DEFAULT_TEXT_CONFIGS)
    parser.add_argument("--alignment_modes", nargs="+", default=DEFAULT_ALIGNMENT_MODES)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto", help="auto, cpu, cuda, cuda:0, etc. Passed to SentenceTransformer when not auto.")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--output_dir", default="outputs/spike_embedding_signal_tests")
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--run_placebo", action="store_true", help="Run embedding placebo variants. Slower but more diagnostic.")
    return parser.parse_args()


def _row_identity(df: pd.DataFrame, ticker_col: str, date_col: str, text_configuration: str, alignment_mode: str, variant: str) -> pd.Series:
    return (
        df[ticker_col].astype(str)
        + "|"
        + df[date_col].astype(str)
        + "|"
        + text_configuration
        + "|"
        + alignment_mode
        + "|"
        + variant
    )


def _hash_series(values: pd.Series) -> str:
    normalized = values.map(normalize_text_value).tolist()
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def _embedding_cache_path(
    cache_dir: Path,
    texts: pd.Series,
    row_identity: pd.Series,
    model_name: str,
    text_configuration: str,
    alignment_mode: str,
    variant: str,
) -> Path:
    payload = {
        "cache_version": TEXT_CACHE_VERSION,
        "script": "spike_news_embedding_signal",
        "model": model_name,
        "text_configuration": text_configuration,
        "alignment_mode": alignment_mode,
        "variant": variant,
        "normalized_text_hash": _hash_series(texts),
        "row_identity_hash": hashlib.sha256("\n".join(row_identity.astype(str).tolist()).encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"spike_embeddings_{TEXT_CACHE_VERSION}_{digest}.npy"


def _load_sentence_transformer(model_name: str, device: str):
    from sentence_transformers import SentenceTransformer

    if device == "auto":
        return SentenceTransformer(model_name)
    return SentenceTransformer(model_name, device=device)


def encode_texts(
    texts: pd.Series,
    row_identity: pd.Series,
    model_name: str,
    device: str,
    cache_dir: Path,
    text_configuration: str,
    alignment_mode: str,
    variant: str,
) -> np.ndarray:
    path = _embedding_cache_path(cache_dir, texts, row_identity, model_name, text_configuration, alignment_mode, variant)
    if path.exists():
        return np.load(path)
    normalized = texts.map(normalize_text_value).tolist()
    model = _load_sentence_transformer(model_name, device)
    matrix = model.encode(normalized, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    np.save(path, matrix)
    return matrix


def train_only_pca(raw: np.ndarray, train_index: pd.Index, pca_dim: int, prefix: str) -> tuple[pd.DataFrame, list[str]]:
    if pca_dim and 0 < pca_dim < raw.shape[1]:
        n_components = min(pca_dim, len(train_index), raw.shape[1])
        pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        pca.fit(raw[train_index.to_numpy()])
        matrix = pca.transform(raw)
    else:
        matrix = raw
    columns = [f"{prefix}_emb_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=columns), columns


def apply_alignment(texts: pd.Series, df: pd.DataFrame, ticker_col: str, mode: str) -> pd.Series:
    work = texts.copy()
    if mode == "same_day_text":
        return work
    if mode == "shifted_1_day_text":
        return work.groupby(df[ticker_col], sort=False).shift(1).fillna("")
    if mode == "shifted_2_day_text":
        return work.groupby(df[ticker_col], sort=False).shift(2).fillna("")
    if mode == "window_3d_text":
        return _rolling_join(work, df[ticker_col], window=3)
    if mode == "window_5d_text":
        return _rolling_join(work, df[ticker_col], window=5)
    raise ValueError(f"Unknown alignment mode: {mode}")


def _rolling_join(texts: pd.Series, tickers: pd.Series, window: int) -> pd.Series:
    out = pd.Series([""] * len(texts), index=texts.index)
    for _, idx in texts.groupby(tickers, sort=False).groups.items():
        idx_list = list(idx)
        values = texts.loc[idx_list].map(normalize_text_value).tolist()
        combined = []
        for pos in range(len(values)):
            start = max(0, pos - window + 1)
            seen: set[str] = set()
            parts: list[str] = []
            for text in values[start : pos + 1]:
                if text and text not in seen:
                    seen.add(text)
                    parts.append(text)
            combined.append("\n".join(parts))
        out.loc[idx_list] = combined
    return out


def apply_placebo(texts: pd.Series, df: pd.DataFrame, ticker_col: str, date_col: str, variant: str) -> pd.Series:
    rng = np.random.default_rng(RANDOM_SEED)
    if variant == "correct_text":
        return texts.copy()
    if variant == "shuffled_within_ticker":
        out = texts.copy()
        for _, idx in df.groupby(ticker_col, sort=False).groups.items():
            idx_list = list(idx)
            out.loc[idx_list] = texts.loc[idx_list].iloc[rng.permutation(len(idx_list))].to_numpy()
        return out
    if variant == "cross_ticker_same_day":
        out = texts.copy()
        for _, idx in df.groupby(date_col, sort=False).groups.items():
            idx_list = list(idx)
            if len(idx_list) <= 1:
                continue
            perm = rng.permutation(len(idx_list))
            if np.array_equal(perm, np.arange(len(idx_list))):
                perm = np.roll(perm, 1)
            out.loc[idx_list] = texts.loc[idx_list].iloc[perm].to_numpy()
        return out
    if variant == "stale_20d":
        return texts.groupby(df[ticker_col], sort=False).shift(20).fillna("")
    raise ValueError(f"Unknown placebo variant: {variant}")


def prepare_data(args: argparse.Namespace):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, column_map = load_fintexts(num_tickers=args.num_tickers)
    df = add_volatility_features(raw, column_map, forecast_horizon=1, max_lag=args.max_lag)
    df = add_detailed_volatility_targets(df, column_map, sorted(set(args.forecast_horizons)))
    write_text_schema_validation(df, output_dir)
    text_columns = detect_text_columns(df)
    ts_features = time_series_feature_columns(args.max_lag)
    df = df.dropna(subset=ts_features + ["logGKVol"]).copy().reset_index(drop=False).rename(columns={"index": "row_id"})
    df = df.reset_index(drop=True)
    train, val, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train, val], ignore_index=False)
    specs = [TargetSpec(target, horizon, target_column_for(target, horizon)) for target in args.targets for horizon in args.forecast_horizons]
    return df, train_full, test, specs, column_map, ts_features, text_columns


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ticker_col: str, target_col: str) -> np.ndarray:
    model = make_ridge_model(features, [ticker_col])
    model.fit(train[features + [ticker_col]], train[target_col])
    return model.predict(test[features + [ticker_col]])


def classification_metrics(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
    if y_true.nunique() < 2:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    pred = score >= 0.5
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    except Exception:
        pass
    return out


def run_regression_and_classifier(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    column_map,
    ts_features: list[str],
    embedding_cols: list[str],
    text_configuration: str,
    alignment_mode: str,
    variant: str,
) -> tuple[list[dict], list[dict]]:
    regression_rows: list[dict] = []
    classifier_rows: list[dict] = []
    feature_cols = ts_features + embedding_cols
    ticker_col = column_map.ticker

    for spec in specs:
        tr = train.dropna(subset=feature_cols + [spec.column])
        te = test.dropna(subset=feature_cols + [spec.column])
        if tr.empty or te.empty:
            continue

        ts_pred = fit_predict_ridge(tr, te, ts_features, ticker_col, spec.column)
        emb_pred = fit_predict_ridge(tr, te, feature_cols, ticker_col, spec.column)

        for pct in SPIKE_PERCENTILES:
            thresholds = tr.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            scored = te[[ticker_col, spec.column]].copy()
            scored["ts_pred"] = ts_pred
            scored["embedding_pred"] = emb_pred
            scored = scored.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
            scored["is_spike"] = scored[spec.column] > scored["threshold"]

            for segment, mask in {"all": pd.Series(True, index=scored.index), "spike": scored["is_spike"], "non_spike": ~scored["is_spike"]}.items():
                subset = scored[mask]
                if subset.empty:
                    continue
                ts_abs = (subset["ts_pred"] - subset[spec.column]).abs()
                emb_abs = (subset["embedding_pred"] - subset[spec.column]).abs()
                regression_rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "text_configuration": text_configuration,
                        "alignment_mode": alignment_mode,
                        "placebo_variant": variant,
                        "segment": segment,
                        "ts_only_mae": float(ts_abs.mean()),
                        "embedding_mae": float(emb_abs.mean()),
                        "mae_improvement_pct": (float(ts_abs.mean()) - float(emb_abs.mean())) / float(ts_abs.mean()) * 100 if float(ts_abs.mean()) else np.nan,
                        "ts_underprediction": float((subset[spec.column] - subset["ts_pred"]).clip(lower=0).mean()),
                        "embedding_underprediction": float((subset[spec.column] - subset["embedding_pred"]).clip(lower=0).mean()),
                        "num_samples": int(len(subset)),
                    }
                )

            # Lightweight spike classifier using ridge predictions as one score and
            # embedding features as a separate diagnostic classifier.
            tr_cls = tr.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
            te_cls = te.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
            tr_cls["is_spike"] = (tr_cls[spec.column] > tr_cls["threshold"]).astype(int)
            te_cls["is_spike"] = (te_cls[spec.column] > te_cls["threshold"]).astype(int)
            if tr_cls["is_spike"].nunique() >= 2 and te_cls["is_spike"].nunique() >= 2:
                clf = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "logit",
                            LogisticRegression(
                                max_iter=5000,
                                class_weight="balanced",
                                random_state=RANDOM_SEED,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                clf.fit(tr_cls[feature_cols], tr_cls["is_spike"])
                score = pd.Series(clf.predict_proba(te_cls[feature_cols])[:, 1], index=te_cls.index)
                classifier_rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "text_configuration": text_configuration,
                        "alignment_mode": alignment_mode,
                        "placebo_variant": variant,
                        "num_train_spikes": int(tr_cls["is_spike"].sum()),
                        "num_test_spikes": int(te_cls["is_spike"].sum()),
                        "num_test_samples": int(len(te_cls)),
                        **classification_metrics(te_cls["is_spike"], score),
                    }
                )
    return regression_rows, classifier_rows


def write_summary(output_dir: Path, regression: pd.DataFrame, classifier: pd.DataFrame) -> None:
    lines = ["# Spike News Embedding Signal Summary", ""]
    if not regression.empty:
        correct = regression[(regression["segment"] == "spike") & (regression["placebo_variant"] == "correct_text")]
        lines.extend(["## Best Correct-Text Embedding Spike Improvements", ""])
        for _, row in correct.sort_values("mae_improvement_pct", ascending=False).head(12).iterrows():
            lines.append(
                f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']} "
                f"{row['text_configuration']} {row['alignment_mode']}: "
                f"{row['mae_improvement_pct']:.2f}% on {int(row['num_samples'])} samples"
            )
        ranks = []
        for keys, group in regression[regression["segment"] == "spike"].groupby(["target_name", "horizon", "spike_percentile", "text_configuration", "alignment_mode"]):
            sorted_group = group.sort_values("embedding_mae").reset_index(drop=True)
            if "correct_text" in set(sorted_group["placebo_variant"]):
                rank = int(sorted_group.index[sorted_group["placebo_variant"].eq("correct_text")][0] + 1)
                ranks.append(rank)
        if ranks:
            lines.extend(["", "## Placebo Check", ""])
            lines.append(f"- Correct text ranked best in {sum(rank == 1 for rank in ranks)} of {len(ranks)} comparable spike cases.")
    if not classifier.empty:
        correct = classifier[classifier["placebo_variant"] == "correct_text"].sort_values("pr_auc", ascending=False).head(8)
        lines.extend(["", "## Best Correct-Text Classifier PR-AUC", ""])
        for _, row in correct.iterrows():
            lines.append(
                f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']} "
                f"{row['text_configuration']} {row['alignment_mode']}: PR-AUC={row['pr_auc']:.4f}, recall={row['recall']:.4f}"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- PCA is fit on train rows only, then applied to validation/test rows.",
            "- Alignment modes are same-day, shifted, or backward-looking rolling windows; no future text is used.",
            "- Treat positive results as predictive association until correct text beats placebo consistently.",
        ]
    )
    (output_dir / "spike_news_embedding_signal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    df, train, test, specs, column_map, ts_features, text_columns = prepare_data(args)
    cache_dir = Path(args.cache_dir)

    regression_rows: list[dict] = []
    classifier_rows: list[dict] = []
    variants = PLACEBO_VARIANTS if args.run_placebo else ["correct_text"]

    for text_configuration in args.text_configurations:
        base_text = joined_text_for_configuration(df, text_columns, text_configuration)
        for alignment_mode in args.alignment_modes:
            aligned = apply_alignment(base_text, df, column_map.ticker, alignment_mode)
            for variant in variants:
                texts = apply_placebo(aligned, df, column_map.ticker, column_map.date, variant)
                identity = _row_identity(df, column_map.ticker, column_map.date, text_configuration, alignment_mode, variant)
                raw = encode_texts(
                    texts,
                    identity,
                    args.embedding_model,
                    args.embedding_device,
                    cache_dir,
                    text_configuration,
                    alignment_mode,
                    variant,
                )
                emb_df, emb_cols = train_only_pca(raw, train.index, args.pca_dim, f"{text_configuration}_{alignment_mode}_{variant}")
                work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                train_work = work.loc[train.index]
                test_work = work.loc[test.index]
                reg_rows, cls_rows = run_regression_and_classifier(
                    work,
                    train_work,
                    test_work,
                    specs,
                    column_map,
                    ts_features,
                    emb_cols,
                    text_configuration,
                    alignment_mode,
                    variant,
                )
                regression_rows.extend(reg_rows)
                classifier_rows.extend(cls_rows)

    regression = pd.DataFrame(regression_rows)
    classifier = pd.DataFrame(classifier_rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    regression.to_csv(output_dir / "spike_embedding_regression.csv", index=False)
    classifier.to_csv(output_dir / "spike_embedding_classifier_metrics.csv", index=False)
    write_summary(output_dir, regression, classifier)
    print(f"Spike embedding signal outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
