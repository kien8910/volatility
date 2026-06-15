"""Controlled embedding experiments for FinTexTS volatility spike days.

This script is separate from the existing pilot and diagnostics modules. It
tests whether more careful text embedding construction improves spike-day
forecasting signal:

- chunk-level target/sector embeddings instead of one large joined text;
- event-keyword filtered chunks;
- target/sector weighted embeddings;
- delta embeddings versus a trailing context window;
- strict placebo variants using the same split and spike thresholds.

Example:

    python -m src.spike_controlled_embedding \
      --num_tickers 25 \
      --tickers BA AMD AXP ABBV AMZN BLK \
      --targets log_abs_return log_gk \
      --forecast_horizons 1 3 \
      --spike_percentiles 90 95 \
      --alignment_modes shifted_1_day_text same_day_text \
      --output_dir outputs/spike_controlled_embedding
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, EVENT_KEYWORDS, RANDOM_SEED
from .detailed_news_signal import split_by_time
from .spike_news_embedding_signal import (
    PLACEBO_VARIANTS,
    _row_identity,
    apply_alignment,
    apply_placebo,
    encode_texts,
    fit_predict_ridge,
    prepare_data,
    train_only_pca,
)
from .text_features import add_event_keyword_features, add_text_presence_flags, normalize_text_value


DEFAULT_CONTROLLED_VARIANTS = [
    "target_chunk_mean",
    "target_event_filtered",
    "target_sector_weighted",
    "target_sector_event_weighted",
    "target_delta_20d",
    "target_sector_delta_20d",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled embedding experiments on volatility spike days.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker subset after loading num_tickers.")
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_abs_return", "log_gk"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[90.0, 95.0])
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text", "same_day_text"])
    parser.add_argument("--embedding_variants", nargs="+", default=DEFAULT_CONTROLLED_VARIANTS)
    parser.add_argument("--placebo_variants", nargs="+", default=PLACEBO_VARIANTS)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--delta_window", type=int, default=20)
    parser.add_argument("--target_weight", type=float, default=2.0)
    parser.add_argument("--sector_weight", type=float, default=1.0)
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output_dir", default="outputs/spike_controlled_embedding")
    parser.add_argument(
        "--require_target_event_keyword",
        action="store_true",
        help="Evaluate only rows where target-company text contains at least one configured event keyword.",
    )
    return parser.parse_args()


def _event_pattern() -> str:
    keywords = [kw for values in EVENT_KEYWORDS.values() for kw in values]
    escaped = [kw.lower().replace(" ", r"\s+") for kw in keywords]
    return "|".join(escaped)


def _column_text(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[column].map(normalize_text_value)


def _event_filter(texts: pd.Series) -> pd.Series:
    pattern = _event_pattern()
    normalized = texts.map(normalize_text_value)
    mask = normalized.str.lower().str.contains(pattern, regex=True, na=False)
    return normalized.where(mask, "")


def _encode_series(
    texts: pd.Series,
    df: pd.DataFrame,
    ticker_col: str,
    date_col: str,
    args: argparse.Namespace,
    cache_label: str,
    alignment_mode: str,
    placebo_variant: str,
) -> np.ndarray:
    identity = _row_identity(df, ticker_col, date_col, cache_label, alignment_mode, placebo_variant)
    return encode_texts(
        texts,
        identity,
        args.embedding_model,
        args.embedding_device,
        Path(args.cache_dir),
        cache_label,
        alignment_mode,
        placebo_variant,
    )


def _weighted_average(matrices: list[np.ndarray], weights: list[pd.Series]) -> np.ndarray:
    if not matrices:
        raise ValueError("No embedding matrices to average.")
    out = np.zeros_like(matrices[0], dtype=float)
    denom = np.zeros((matrices[0].shape[0], 1), dtype=float)
    for matrix, weight in zip(matrices, weights, strict=True):
        w = weight.to_numpy(dtype=float).reshape(-1, 1)
        out += matrix * w
        denom += w
    return np.divide(out, np.maximum(denom, 1.0), out=np.zeros_like(out), where=denom > 0)


def _rolling_context_delta(matrix: np.ndarray, df: pd.DataFrame, ticker_col: str, window: int) -> np.ndarray:
    result = np.zeros_like(matrix, dtype=float)
    for _, idx in df.groupby(ticker_col, sort=False).groups.items():
        idx_list = list(idx)
        values = matrix[idx_list]
        context = pd.DataFrame(values).rolling(window=window, min_periods=1).mean().shift(1).fillna(0.0).to_numpy()
        result[idx_list] = values - context
    return result


def _base_group_embedding(
    df: pd.DataFrame,
    text_columns: dict[str, list[str]],
    group: str,
    args: argparse.Namespace,
    alignment_mode: str,
    placebo_variant: str,
    event_filtered: bool,
) -> tuple[np.ndarray, pd.Series]:
    matrices: list[np.ndarray] = []
    weights: list[pd.Series] = []
    for column in text_columns.get(group, []):
        raw = _column_text(df, column)
        raw = _event_filter(raw) if event_filtered else raw
        aligned = apply_alignment(raw, df, args._ticker_col, alignment_mode)
        text = apply_placebo(aligned, df, args._ticker_col, args._date_col, placebo_variant)
        nonempty = text.map(normalize_text_value).ne("").astype(float)
        matrix = _encode_series(
            text,
            df,
            args._ticker_col,
            args._date_col,
            args,
            f"{group}_{column}_{'event' if event_filtered else 'raw'}",
            alignment_mode,
            placebo_variant,
        )
        matrices.append(matrix)
        weights.append(nonempty)
    if not matrices:
        raise ValueError(f"No configured text columns for group: {group}")
    return _weighted_average(matrices, weights), sum(weights, pd.Series(0.0, index=df.index))


def build_controlled_embedding(
    df: pd.DataFrame,
    text_columns: dict[str, list[str]],
    variant: str,
    alignment_mode: str,
    placebo_variant: str,
    args: argparse.Namespace,
) -> np.ndarray:
    if variant == "target_chunk_mean":
        matrix, _ = _base_group_embedding(df, text_columns, "target", args, alignment_mode, placebo_variant, event_filtered=False)
        return matrix
    if variant == "target_event_filtered":
        matrix, _ = _base_group_embedding(df, text_columns, "target", args, alignment_mode, placebo_variant, event_filtered=True)
        return matrix
    if variant == "target_sector_weighted":
        target, target_count = _base_group_embedding(df, text_columns, "target", args, alignment_mode, placebo_variant, event_filtered=False)
        sector, sector_count = _base_group_embedding(df, text_columns, "sector", args, alignment_mode, placebo_variant, event_filtered=False)
        return _weighted_average([target, sector], [target_count.gt(0).astype(float) * args.target_weight, sector_count.gt(0).astype(float) * args.sector_weight])
    if variant == "target_sector_event_weighted":
        target, target_count = _base_group_embedding(df, text_columns, "target", args, alignment_mode, placebo_variant, event_filtered=True)
        sector, sector_count = _base_group_embedding(df, text_columns, "sector", args, alignment_mode, placebo_variant, event_filtered=True)
        return _weighted_average([target, sector], [target_count.gt(0).astype(float) * args.target_weight, sector_count.gt(0).astype(float) * args.sector_weight])
    if variant == "target_delta_20d":
        matrix, _ = _base_group_embedding(df, text_columns, "target", args, alignment_mode, placebo_variant, event_filtered=False)
        return _rolling_context_delta(matrix, df, args._ticker_col, args.delta_window)
    if variant == "target_sector_delta_20d":
        matrix = build_controlled_embedding(df, text_columns, "target_sector_weighted", alignment_mode, placebo_variant, args)
        return _rolling_context_delta(matrix, df, args._ticker_col, args.delta_window)
    raise ValueError(f"Unknown controlled embedding variant: {variant}")


def _classification_metrics(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
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


def _classifier_score(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target_col: str) -> pd.Series | None:
    if train["is_spike"].nunique() < 2 or test["is_spike"].nunique() < 2:
        return None
    clf = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "logit",
                LogisticRegression(max_iter=5000, class_weight="balanced", random_state=RANDOM_SEED, solver="liblinear"),
            ),
        ]
    )
    clf.fit(train[features], train["is_spike"].astype(int))
    return pd.Series(clf.predict_proba(test[features])[:, 1], index=test.index)


def _event_preview(df: pd.DataFrame, text_columns: dict[str, list[str]], idx: int) -> str:
    parts: list[str] = []
    for group in ["target", "sector"]:
        for column in text_columns.get(group, []):
            text = normalize_text_value(df.loc[idx, column]) if column in df.columns else ""
            if text:
                parts.append(text.replace("\n", " ")[:180])
    return " | ".join(parts)[:360]


def evaluate_variant(
    work: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs,
    ts_features: list[str],
    embedding_cols: list[str],
    args: argparse.Namespace,
    embedding_variant: str,
    alignment_mode: str,
    placebo_variant: str,
    text_columns: dict[str, list[str]],
) -> tuple[list[dict], list[dict], list[dict]]:
    metric_rows: list[dict] = []
    classifier_rows: list[dict] = []
    event_rows: list[dict] = []
    feature_cols = ts_features + embedding_cols

    for spec in specs:
        tr = train.dropna(subset=feature_cols + [spec.column])
        te = test.dropna(subset=feature_cols + [spec.column])
        if args.require_target_event_keyword:
            te = te[te["target_event_keyword_count"].fillna(0).gt(0)]
        if tr.empty or te.empty:
            continue
        ts_pred = fit_predict_ridge(tr, te, ts_features, args._ticker_col, spec.column)
        emb_pred = fit_predict_ridge(tr, te, feature_cols, args._ticker_col, spec.column)

        for pct in args.spike_percentiles:
            thresholds = tr.groupby(args._ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            scored = te[[args._ticker_col, args._date_col, spec.column]].copy()
            scored["ts_only_prediction"] = ts_pred
            scored["embedding_prediction"] = emb_pred
            scored = scored.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left")
            scored["is_spike"] = scored[spec.column] > scored["threshold"]
            scored["ts_abs_error"] = (scored["ts_only_prediction"] - scored[spec.column]).abs()
            scored["embedding_abs_error"] = (scored["embedding_prediction"] - scored[spec.column]).abs()
            scored["abs_error_improvement"] = scored["ts_abs_error"] - scored["embedding_abs_error"]
            scored["underprediction_reduction"] = (
                (scored[spec.column] - scored["ts_only_prediction"]).clip(lower=0)
                - (scored[spec.column] - scored["embedding_prediction"]).clip(lower=0)
            )

            tr_cls = tr.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left").copy()
            te_cls = te.merge(thresholds, left_on=args._ticker_col, right_index=True, how="left").copy()
            tr_cls["is_spike"] = tr_cls[spec.column] > tr_cls["threshold"]
            te_cls["is_spike"] = te_cls[spec.column] > te_cls["threshold"]
            score = _classifier_score(tr_cls, te_cls, feature_cols, spec.column)
            if score is not None:
                for ticker, idx in te_cls.groupby(args._ticker_col, sort=False).groups.items():
                    idx_list = list(idx)
                    metrics = _classification_metrics(te_cls.loc[idx_list, "is_spike"].astype(int), score.loc[idx_list])
                    classifier_rows.append(
                        {
                            "ticker": ticker,
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "embedding_variant": embedding_variant,
                            "alignment_mode": alignment_mode,
                            "placebo_variant": placebo_variant,
                            "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                            "num_test_spikes": int(te_cls.loc[idx_list, "is_spike"].sum()),
                            "num_test_samples": int(len(idx_list)),
                            **metrics,
                        }
                    )

            for ticker, group in scored.groupby(args._ticker_col, sort=False):
                spike = group[group["is_spike"]]
                if spike.empty:
                    continue
                ts_mae = float(spike["ts_abs_error"].mean())
                emb_mae = float(spike["embedding_abs_error"].mean())
                metric_rows.append(
                    {
                        "ticker": ticker,
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "embedding_variant": embedding_variant,
                        "alignment_mode": alignment_mode,
                        "placebo_variant": placebo_variant,
                        "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                        "num_spike_samples": int(len(spike)),
                        "ts_only_mae": ts_mae,
                        "embedding_mae": emb_mae,
                        "mae_improvement_pct": (ts_mae - emb_mae) / ts_mae * 100 if ts_mae else np.nan,
                        "underprediction_reduction": float(spike["underprediction_reduction"].mean()),
                    }
                )

            if placebo_variant == "correct_text":
                helpful = scored[scored["is_spike"]].copy()
                for idx, row in helpful.iterrows():
                    event_rows.append(
                        {
                            "ticker": row[args._ticker_col],
                            "date": row[args._date_col],
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "embedding_variant": embedding_variant,
                            "alignment_mode": alignment_mode,
                            "evaluation_filter": "target_event_keyword" if args.require_target_event_keyword else "all_test_rows",
                            "actual_target": row[spec.column],
                            "threshold": row["threshold"],
                            "ts_only_prediction": row["ts_only_prediction"],
                            "embedding_prediction": row["embedding_prediction"],
                            "abs_error_improvement": row["abs_error_improvement"],
                            "underprediction_reduction": row["underprediction_reduction"],
                            "has_target_text": int(work.loc[idx].get("has_target_text", 0)),
                            "has_sector_text": int(work.loc[idx].get("has_sector_text", 0)),
                            "event_keyword_count": int(work.loc[idx].get("event_keyword_count", 0)),
                            "target_event_keyword_count": int(work.loc[idx].get("target_event_keyword_count", 0)),
                            "text_preview": _event_preview(work, text_columns, idx),
                        }
                    )
    return metric_rows, classifier_rows, event_rows


def summarize_placebo(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame()
    keys = ["ticker", "target_name", "horizon", "spike_percentile", "embedding_variant", "alignment_mode"]
    rows = []
    for key, group in metrics.groupby(keys, dropna=False):
        ordered = group.sort_values("embedding_mae", ascending=True).reset_index(drop=True)
        if "correct_text" not in set(ordered["placebo_variant"]):
            continue
        correct = ordered[ordered["placebo_variant"].eq("correct_text")].iloc[0]
        rows.append(
            {
                **dict(zip(keys, key)),
                "best_variant": ordered.iloc[0]["placebo_variant"],
                "correct_text_rank": int(ordered.index[ordered["placebo_variant"].eq("correct_text")][0]) + 1,
                "num_variants": int(len(ordered)),
                "correct_text_mae_improvement_pct": correct["mae_improvement_pct"],
                "correct_text_num_spike_samples": correct["num_spike_samples"],
                "correct_text_underprediction_reduction": correct["underprediction_reduction"],
            }
        )
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, metrics: pd.DataFrame, ranks: pd.DataFrame, classifier: pd.DataFrame) -> None:
    lines = ["# Controlled Spike Embedding Summary", ""]
    if not metrics.empty:
        correct = metrics[metrics["placebo_variant"].eq("correct_text")]
        lines.extend(["## Correct Text Regression", ""])
        lines.append(f"- Rows: {len(correct)}")
        lines.append(f"- Positive MAE improvement rate: {(correct['mae_improvement_pct'] > 0).mean() * 100:.2f}%")
        lines.append(f"- Mean MAE improvement: {correct['mae_improvement_pct'].mean():.2f}%")
        lines.append(f"- Median MAE improvement: {correct['mae_improvement_pct'].median():.2f}%")
        lines.extend(["", "## By Embedding Variant", ""])
        by_variant = correct.groupby("embedding_variant")["mae_improvement_pct"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        for name, row in by_variant.iterrows():
            lines.append(f"- {name}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
    if not ranks.empty:
        lines.extend(["", "## Placebo Rank", ""])
        lines.append(f"- Correct text ranked best in {(ranks['correct_text_rank'] == 1).sum()} of {len(ranks)} cases.")
        by_variant = ranks.assign(best=ranks["correct_text_rank"].eq(1)).groupby("embedding_variant")["best"].mean().sort_values(ascending=False)
        for name, value in by_variant.items():
            lines.append(f"- {name}: correct-best rate {value * 100:.2f}%")
    if not classifier.empty:
        correct_cls = classifier[classifier["placebo_variant"].eq("correct_text")]
        lines.extend(["", "## Classifier", ""])
        lines.append(f"- Mean PR-AUC: {correct_cls['pr_auc'].mean():.4f}")
        lines.append(f"- Median PR-AUC: {correct_cls['pr_auc'].median():.4f}")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- Treat a variant as promising only if correct text improves spike MAE, reduces underprediction, and beats placebo in at least half of comparable ticker/configuration cases.",
            "- Prefer shifted alignment when it remains competitive with same-day alignment.",
            "- Do not claim causal effect; this is predictive signal analysis.",
        ]
    )
    (output_dir / "controlled_embedding_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, train, test, specs, column_map, ts_features, text_columns = prepare_data(args)
    if args.tickers:
        keep = set(args.tickers)
        df = df[df[column_map.ticker].isin(keep)].copy()
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    df = df.reset_index(drop=True)
    train_part, val_part, test = split_by_time(df, column_map.ticker, column_map.date)
    train = pd.concat([train_part, val_part], ignore_index=False)
    args._ticker_col = column_map.ticker
    args._date_col = column_map.date

    metric_rows: list[dict] = []
    classifier_rows: list[dict] = []
    event_rows: list[dict] = []

    for embedding_variant in args.embedding_variants:
        for alignment_mode in args.alignment_modes:
            for placebo_variant in args.placebo_variants:
                raw = build_controlled_embedding(df, text_columns, embedding_variant, alignment_mode, placebo_variant, args)
                emb_df, emb_cols = train_only_pca(raw, train.index, args.pca_dim, f"{embedding_variant}_{alignment_mode}_{placebo_variant}")
                work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                train_work = work.loc[train.index]
                test_work = work.loc[test.index]
                rows, cls_rows, ev_rows = evaluate_variant(
                    work,
                    train_work,
                    test_work,
                    specs,
                    ts_features,
                    emb_cols,
                    args,
                    embedding_variant,
                    alignment_mode,
                    placebo_variant,
                    text_columns,
                )
                metric_rows.extend(rows)
                classifier_rows.extend(cls_rows)
                event_rows.extend(ev_rows)

    metrics = pd.DataFrame(metric_rows)
    ranks = summarize_placebo(metrics)
    classifier = pd.DataFrame(classifier_rows)
    events = pd.DataFrame(event_rows)

    metrics.to_csv(output_dir / "controlled_embedding_metrics.csv", index=False)
    ranks.to_csv(output_dir / "controlled_embedding_placebo_ranks.csv", index=False)
    classifier.to_csv(output_dir / "controlled_embedding_classifier_metrics.csv", index=False)
    if not events.empty:
        events.to_csv(output_dir / "controlled_embedding_event_examples.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=False).head(300).to_csv(output_dir / "controlled_embedding_top_improvements.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=True).head(300).to_csv(output_dir / "controlled_embedding_top_failures.csv", index=False)
    write_summary(output_dir, metrics, ranks, classifier)
    print(f"Controlled spike embedding outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
