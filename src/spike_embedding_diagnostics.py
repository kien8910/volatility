"""Diagnostics for embedding-based news signal on volatility spike days.

This module is intentionally separate from the pilot and from
``spike_news_embedding_signal``. It reuses the existing data preparation,
alignment, placebo, embedding cache, and train-only PCA helpers, then writes
more granular diagnostics:

- per-ticker correct-text versus placebo ranks;
- event-level spike rows where text helps or hurts;
- underprediction reduction by ticker;
- keyword/text-level overlays for spike rows.

Example:

    python -m src.spike_embedding_diagnostics \
      --num_tickers 25 \
      --forecast_horizons 1 3 5 \
      --targets log_gk total_volatility log_abs_return \
      --text_configurations target_only target_sector news_all \
      --alignment_modes same_day_text shifted_1_day_text window_5d_text \
      --output_dir outputs/spike_embedding_diagnostics
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

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, RANDOM_SEED
from .spike_news_embedding_signal import (
    DEFAULT_ALIGNMENT_MODES,
    DEFAULT_TEXT_CONFIGS,
    PLACEBO_VARIANTS,
    SPIKE_PERCENTILES,
    _row_identity,
    apply_alignment,
    apply_placebo,
    encode_texts,
    fit_predict_ridge,
    prepare_data,
    train_only_pca,
)
from .text_features import add_event_keyword_features, add_text_presence_flags, joined_text_for_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run granular diagnostics for embedding news signal on spike days.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "total_volatility", "log_abs_return"])
    parser.add_argument("--text_configurations", nargs="+", default=["target_only", "target_sector", "news_all"])
    parser.add_argument("--alignment_modes", nargs="+", default=["same_day_text", "shifted_1_day_text", "window_5d_text"])
    parser.add_argument("--placebo_variants", nargs="+", default=PLACEBO_VARIANTS)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto", help="auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output_dir", default="outputs/spike_embedding_diagnostics")
    parser.add_argument("--top_events", type=int, default=500)
    return parser.parse_args()


def _classification_metrics(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
    if len(y_true) == 0 or y_true.nunique() < 2:
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


def _fit_classifier(train: pd.DataFrame, test: pd.DataFrame, features: list[str], target_col: str, threshold_col: str) -> pd.Series | None:
    tr = train.dropna(subset=features + [target_col, threshold_col]).copy()
    te = test.dropna(subset=features + [target_col, threshold_col]).copy()
    if tr.empty or te.empty:
        return None
    tr["is_spike"] = (tr[target_col] > tr[threshold_col]).astype(int)
    if tr["is_spike"].nunique() < 2:
        return None
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
    clf.fit(tr[features], tr["is_spike"])
    return pd.Series(clf.predict_proba(test[features])[:, 1], index=test.index)


def _add_text_diagnostics(df: pd.DataFrame, text_columns: dict[str, list[str]], ticker_col: str) -> pd.DataFrame:
    out = add_text_presence_flags(df, text_columns, ticker_col=ticker_col)
    out = add_event_keyword_features(out, text_columns)
    return out


def _keyword_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col.startswith("event_keyword_")]


def _event_preview(texts: pd.Series, index: int, limit: int = 260) -> str:
    if index not in texts.index:
        return ""
    value = str(texts.loc[index]).replace("\n", " ").strip()
    return value[:limit]


def _spike_scored_frame(
    test: pd.DataFrame,
    ticker_col: str,
    date_col: str,
    target_col: str,
    threshold_by_ticker: pd.DataFrame,
    ts_pred: np.ndarray,
    emb_pred: np.ndarray,
) -> pd.DataFrame:
    scored = test[[ticker_col, date_col, target_col]].copy()
    scored["ts_only_prediction"] = ts_pred
    scored["embedding_prediction"] = emb_pred
    scored = scored.merge(threshold_by_ticker, left_on=ticker_col, right_index=True, how="left")
    scored["is_spike"] = scored[target_col] > scored["threshold"]
    scored["ts_abs_error"] = (scored["ts_only_prediction"] - scored[target_col]).abs()
    scored["embedding_abs_error"] = (scored["embedding_prediction"] - scored[target_col]).abs()
    scored["abs_error_improvement"] = scored["ts_abs_error"] - scored["embedding_abs_error"]
    scored["underprediction_reduction"] = (
        (scored[target_col] - scored["ts_only_prediction"]).clip(lower=0)
        - (scored[target_col] - scored["embedding_prediction"]).clip(lower=0)
    )
    return scored


def _append_event_rows(
    rows: list[dict],
    scored: pd.DataFrame,
    test: pd.DataFrame,
    aligned_text: pd.Series,
    text_configuration: str,
    alignment_mode: str,
    placebo_variant: str,
    target_name: str,
    horizon: int,
    spike_percentile: float,
    ticker_col: str,
    date_col: str,
    target_col: str,
) -> None:
    keyword_cols = _keyword_columns(test)
    text_flag_cols = [
        "has_macro_text",
        "has_sector_text",
        "has_related_text",
        "has_target_text",
        "has_any_news_text",
        "news_text_count_total",
        "news_text_length_total",
        "event_keyword_count",
        "target_event_keyword_count",
    ]
    for idx, row in scored[scored["is_spike"]].iterrows():
        active_keywords = [col.replace("event_keyword_", "") for col in keyword_cols if bool(test.loc[idx].get(col, 0))]
        out = {
            "ticker": row[ticker_col],
            "date": row[date_col],
            "target_name": target_name,
            "horizon": horizon,
            "spike_percentile": spike_percentile,
            "text_configuration": text_configuration,
            "alignment_mode": alignment_mode,
            "placebo_variant": placebo_variant,
            "actual_target": row[target_col],
            "threshold": row["threshold"],
            "ts_only_prediction": row["ts_only_prediction"],
            "embedding_prediction": row["embedding_prediction"],
            "ts_abs_error": row["ts_abs_error"],
            "embedding_abs_error": row["embedding_abs_error"],
            "abs_error_improvement": row["abs_error_improvement"],
            "underprediction_reduction": row["underprediction_reduction"],
            "active_keyword_groups": ",".join(active_keywords),
            "aligned_text_preview": _event_preview(aligned_text, idx),
        }
        for col in text_flag_cols:
            out[col] = test.loc[idx].get(col, np.nan)
        rows.append(out)


def _summarize_placebo_ranks(per_variant: pd.DataFrame) -> pd.DataFrame:
    if per_variant.empty:
        return pd.DataFrame()
    key_cols = ["ticker", "target_name", "horizon", "spike_percentile", "text_configuration", "alignment_mode"]
    rows = []
    for key, group in per_variant.groupby(key_cols, dropna=False):
        ordered = group.sort_values("embedding_mae", ascending=True).reset_index(drop=True)
        if "correct_text" not in set(ordered["placebo_variant"]):
            continue
        correct = ordered[ordered["placebo_variant"].eq("correct_text")].iloc[0]
        rows.append(
            {
                **dict(zip(key_cols, key)),
                "best_placebo_variant": ordered.iloc[0]["placebo_variant"],
                "correct_text_rank": int(ordered.index[ordered["placebo_variant"].eq("correct_text")][0]) + 1,
                "num_variants": int(len(ordered)),
                "correct_text_mae_improvement_pct": correct["mae_improvement_pct"],
                "correct_text_num_spike_samples": correct["num_spike_samples"],
                "correct_text_underprediction_reduction": correct["underprediction_reduction"],
            }
        )
    return pd.DataFrame(rows)


def _write_report(
    output_dir: Path,
    per_variant: pd.DataFrame,
    ranks: pd.DataFrame,
    events: pd.DataFrame,
    classifier: pd.DataFrame,
) -> None:
    lines = ["# Spike Embedding Diagnostics Report", ""]
    lines.append("This report focuses on whether correct news embeddings beat placebo variants on volatility spike days.")
    lines.append("")

    if not per_variant.empty:
        correct = per_variant[per_variant["placebo_variant"].eq("correct_text")]
        lines.extend(["## Correct Text Spike Regression", ""])
        lines.append(f"- Rows: {len(correct)}")
        lines.append(f"- Positive MAE improvement rate: {(correct['mae_improvement_pct'] > 0).mean() * 100:.2f}%")
        lines.append(f"- Mean MAE improvement: {correct['mae_improvement_pct'].mean():.2f}%")
        lines.append(f"- Median MAE improvement: {correct['mae_improvement_pct'].median():.2f}%")
        best = correct.sort_values("mae_improvement_pct", ascending=False).head(8)
        lines.extend(["", "## Best Correct-Text Cases", ""])
        for _, row in best.iterrows():
            lines.append(
                f"- {row['ticker']} {row['target_name']} h={row['horizon']} p{row['spike_percentile']} "
                f"{row['text_configuration']} {row['alignment_mode']}: "
                f"{row['mae_improvement_pct']:.2f}% on {int(row['num_spike_samples'])} spike samples"
            )

    if not ranks.empty:
        lines.extend(["", "## Placebo Rank Check", ""])
        lines.append(
            f"- Correct text ranked best in {(ranks['correct_text_rank'] == 1).sum()} "
            f"of {len(ranks)} ticker/configuration cases."
        )
        by_config = ranks.assign(correct_best=ranks["correct_text_rank"].eq(1)).groupby("text_configuration")["correct_best"].mean()
        for name, value in by_config.sort_values(ascending=False).items():
            lines.append(f"- {name}: correct-best rate {value * 100:.2f}%")

    if not classifier.empty:
        correct_cls = classifier[classifier["placebo_variant"].eq("correct_text")]
        lines.extend(["", "## Classifier Diagnostics", ""])
        lines.append(f"- Mean PR-AUC: {correct_cls['pr_auc'].mean():.4f}")
        lines.append(f"- Median PR-AUC: {correct_cls['pr_auc'].median():.4f}")
        lines.append(f"- Max PR-AUC: {correct_cls['pr_auc'].max():.4f}")

    if not events.empty:
        helpful = events[events["placebo_variant"].eq("correct_text")].sort_values("abs_error_improvement", ascending=False).head(5)
        harmful = events[events["placebo_variant"].eq("correct_text")].sort_values("abs_error_improvement", ascending=True).head(5)
        lines.extend(["", "## Event-Level Inspection", ""])
        lines.append(f"- Event rows written: {len(events)}")
        lines.append("- Top helpful spike rows:")
        for _, row in helpful.iterrows():
            lines.append(
                f"  - {row['ticker']} {row['date']} {row['target_name']} h={row['horizon']} "
                f"improvement={row['abs_error_improvement']:.4f} keywords={row['active_keyword_groups'] or 'none'}"
            )
        lines.append("- Top harmful spike rows:")
        for _, row in harmful.iterrows():
            lines.append(
                f"  - {row['ticker']} {row['date']} {row['target_name']} h={row['horizon']} "
                f"improvement={row['abs_error_improvement']:.4f} keywords={row['active_keyword_groups'] or 'none'}"
            )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Spike thresholds are fit on train rows by ticker.",
            "- PCA is fit on train rows only.",
            "- Alignment modes use same-day, shifted, or backward-looking text only.",
            "- Treat positive results as predictive association unless correct text beats placebo consistently.",
        ]
    )
    (output_dir / "spike_embedding_diagnostics_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, train, test, specs, column_map, ts_features, text_columns = prepare_data(args)
    df = _add_text_diagnostics(df, text_columns, column_map.ticker)
    train = df.loc[train.index]
    test = df.loc[test.index]

    cache_dir = Path(args.cache_dir)
    per_variant_rows: list[dict] = []
    event_rows: list[dict] = []
    classifier_rows: list[dict] = []

    for text_configuration in args.text_configurations:
        base_text = joined_text_for_configuration(df, text_columns, text_configuration)
        for alignment_mode in args.alignment_modes:
            aligned = apply_alignment(base_text, df, column_map.ticker, alignment_mode)
            for placebo_variant in args.placebo_variants:
                texts = apply_placebo(aligned, df, column_map.ticker, column_map.date, placebo_variant)
                identity = _row_identity(df, column_map.ticker, column_map.date, text_configuration, alignment_mode, placebo_variant)
                raw = encode_texts(
                    texts,
                    identity,
                    args.embedding_model,
                    args.embedding_device,
                    cache_dir,
                    text_configuration,
                    alignment_mode,
                    placebo_variant,
                )
                emb_df, emb_cols = train_only_pca(raw, train.index, args.pca_dim, f"{text_configuration}_{alignment_mode}_{placebo_variant}")
                work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
                train_work = work.loc[train.index]
                test_work = work.loc[test.index]
                feature_cols = ts_features + emb_cols

                for spec in specs:
                    tr = train_work.dropna(subset=feature_cols + [spec.column])
                    te = test_work.dropna(subset=feature_cols + [spec.column])
                    if tr.empty or te.empty:
                        continue

                    ts_pred = fit_predict_ridge(tr, te, ts_features, column_map.ticker, spec.column)
                    emb_pred = fit_predict_ridge(tr, te, feature_cols, column_map.ticker, spec.column)

                    for pct in SPIKE_PERCENTILES:
                        thresholds = tr.groupby(column_map.ticker)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
                        scored = _spike_scored_frame(te, column_map.ticker, column_map.date, spec.column, thresholds, ts_pred, emb_pred)
                        scored["classifier_score"] = np.nan

                        tr_cls = tr.merge(thresholds, left_on=column_map.ticker, right_index=True, how="left")
                        te_cls = te.merge(thresholds, left_on=column_map.ticker, right_index=True, how="left")
                        cls_score = _fit_classifier(tr_cls, te_cls, feature_cols, spec.column, "threshold")
                        if cls_score is not None:
                            scored.loc[cls_score.index, "classifier_score"] = cls_score
                            te_cls = te_cls.copy()
                            te_cls["is_spike"] = (te_cls[spec.column] > te_cls["threshold"]).astype(int)
                            te_cls["classifier_score"] = cls_score
                            for ticker, ticker_group in te_cls.groupby(column_map.ticker):
                                metrics = _classification_metrics(ticker_group["is_spike"], ticker_group["classifier_score"])
                                classifier_rows.append(
                                    {
                                        "ticker": ticker,
                                        "target_name": spec.target_name,
                                        "horizon": spec.horizon,
                                        "spike_percentile": pct,
                                        "text_configuration": text_configuration,
                                        "alignment_mode": alignment_mode,
                                        "placebo_variant": placebo_variant,
                                        "num_test_spikes": int(ticker_group["is_spike"].sum()),
                                        "num_test_samples": int(len(ticker_group)),
                                        **metrics,
                                    }
                                )

                        for ticker, ticker_group in scored.groupby(column_map.ticker):
                            spike = ticker_group[ticker_group["is_spike"]]
                            if spike.empty:
                                continue
                            ts_mae = float(spike["ts_abs_error"].mean())
                            emb_mae = float(spike["embedding_abs_error"].mean())
                            per_variant_rows.append(
                                {
                                    "ticker": ticker,
                                    "target_name": spec.target_name,
                                    "horizon": spec.horizon,
                                    "spike_percentile": pct,
                                    "text_configuration": text_configuration,
                                    "alignment_mode": alignment_mode,
                                    "placebo_variant": placebo_variant,
                                    "ts_only_mae": ts_mae,
                                    "embedding_mae": emb_mae,
                                    "mae_improvement_pct": (ts_mae - emb_mae) / ts_mae * 100 if ts_mae else np.nan,
                                    "underprediction_reduction": float(spike["underprediction_reduction"].mean()),
                                    "num_spike_samples": int(len(spike)),
                                }
                            )

                        if placebo_variant == "correct_text":
                            _append_event_rows(
                                event_rows,
                                scored,
                                te,
                                texts,
                                text_configuration,
                                alignment_mode,
                                placebo_variant,
                                spec.target_name,
                                spec.horizon,
                                pct,
                                column_map.ticker,
                                column_map.date,
                                spec.column,
                            )

    per_variant = pd.DataFrame(per_variant_rows)
    ranks = _summarize_placebo_ranks(per_variant)
    events = pd.DataFrame(event_rows)
    classifier = pd.DataFrame(classifier_rows)

    per_variant.to_csv(output_dir / "per_ticker_embedding_placebo_metrics.csv", index=False)
    ranks.to_csv(output_dir / "per_ticker_correct_vs_placebo.csv", index=False)
    classifier.to_csv(output_dir / "per_ticker_classifier_metrics.csv", index=False)

    if not events.empty:
        events.sort_values("abs_error_improvement", ascending=False).head(args.top_events).to_csv(
            output_dir / "top_event_level_improvements.csv", index=False
        )
        events.sort_values("abs_error_improvement", ascending=True).head(args.top_events).to_csv(
            output_dir / "top_event_level_failures.csv", index=False
        )
        events.to_csv(output_dir / "spike_event_level_predictions.csv", index=False)

    if not per_variant.empty:
        correct = per_variant[per_variant["placebo_variant"].eq("correct_text")]
        target_summary = (
            correct.groupby(["ticker", "target_name", "horizon", "spike_percentile", "text_configuration", "alignment_mode"], dropna=False)
            .agg(
                mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
                mean_underprediction_reduction=("underprediction_reduction", "mean"),
                total_spike_samples=("num_spike_samples", "sum"),
            )
            .reset_index()
        )
        target_summary.to_csv(output_dir / "target_config_spike_summary.csv", index=False)
        underprediction = (
            correct.groupby(["ticker", "target_name", "horizon", "text_configuration"], dropna=False)
            .agg(
                mean_underprediction_reduction=("underprediction_reduction", "mean"),
                mean_mae_improvement_pct=("mae_improvement_pct", "mean"),
                total_spike_samples=("num_spike_samples", "sum"),
            )
            .reset_index()
        )
        underprediction.to_csv(output_dir / "underprediction_by_ticker.csv", index=False)

    _write_report(output_dir, per_variant, ranks, events, classifier)
    print(f"Spike embedding diagnostics written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
