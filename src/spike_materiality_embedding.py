"""Materiality-weighted controlled embeddings for volatility spike analysis.

This module is separate from the existing pilot and controlled embedding
scripts. It tests whether explicit event-type weighting and materiality scores
make the text embedding signal more finance-relevant.

The script:

- builds target/sector text materiality scores from configured event keywords;
- applies target-company events more strongly than sector events;
- scales controlled embeddings by materiality;
- appends compact materiality/event-type features;
- evaluates the same spike-day regression/classification/placebo framework.

Example:

    python -m src.spike_materiality_embedding \
      --num_tickers 25 \
      --tickers BAC C AMGN ABT BX AXP AMD \
      --targets log_gk log_abs_return \
      --forecast_horizons 1 3 \
      --spike_percentiles 80 85 90 95 \
      --alignment_modes shifted_1_day_text same_day_text \
      --base_embedding_variants target_sector_weighted target_sector_event_weighted \
      --require_target_event_keyword \
      --output_dir outputs/spike_materiality_embedding_confirmatory_p80_p95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, EVENT_KEYWORDS
from .detailed_news_signal import split_by_time
from .spike_controlled_embedding import build_controlled_embedding, evaluate_variant, summarize_placebo
from .spike_news_embedding_signal import PLACEBO_VARIANTS, apply_alignment, apply_placebo, prepare_data, train_only_pca
from .text_features import add_event_keyword_features, add_text_presence_flags, joined_text_for_configuration


EVENT_TYPE_WEIGHTS = {
    "earnings": 2.0,
    "regulatory_legal": 2.0,
    "corporate_action": 1.5,
    "analyst_action": 1.25,
    "product_operations": 1.25,
    "macro_shock": 1.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run materiality-weighted embedding experiments on volatility spike days.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=["BAC", "C", "AMGN", "ABT", "BX", "AXP", "AMD"])
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "log_abs_return"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text", "same_day_text"])
    parser.add_argument("--base_embedding_variants", nargs="+", default=["target_sector_weighted", "target_sector_event_weighted"])
    parser.add_argument("--placebo_variants", nargs="+", default=PLACEBO_VARIANTS)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--output_dir", default="outputs/spike_materiality_embedding_confirmatory_p80_p95")
    parser.add_argument("--target_materiality_weight", type=float, default=2.0)
    parser.add_argument("--sector_materiality_weight", type=float, default=1.0)
    parser.add_argument("--target_weight", type=float, default=2.0, help="Base target embedding weight used by target_sector controlled variants.")
    parser.add_argument("--sector_weight", type=float, default=1.0, help="Base sector embedding weight used by target_sector controlled variants.")
    parser.add_argument("--materiality_scale", type=float, default=0.5)
    parser.add_argument("--require_target_event_keyword", action="store_true")
    return parser.parse_args()


def _event_pattern(keywords: list[str]) -> str:
    return "|".join(keyword.lower().replace(" ", r"\s+") for keyword in keywords)


def _event_group_hits(text: pd.Series, group: str) -> pd.Series:
    pattern = _event_pattern(EVENT_KEYWORDS[group])
    return text.str.lower().str.contains(pattern, regex=True, na=False).astype(float)


def _aligned_placebo_text(
    df: pd.DataFrame,
    text_columns: dict[str, list[str]],
    configuration: str,
    alignment_mode: str,
    placebo_variant: str,
    ticker_col: str,
    date_col: str,
) -> pd.Series:
    base = joined_text_for_configuration(df, text_columns, configuration)
    aligned = apply_alignment(base, df, ticker_col, alignment_mode)
    return apply_placebo(aligned, df, ticker_col, date_col, placebo_variant).fillna("")


def build_materiality_features(
    df: pd.DataFrame,
    text_columns: dict[str, list[str]],
    alignment_mode: str,
    placebo_variant: str,
    ticker_col: str,
    date_col: str,
    args: argparse.Namespace,
) -> pd.DataFrame:
    target_text = _aligned_placebo_text(df, text_columns, "target_only", alignment_mode, placebo_variant, ticker_col, date_col)
    sector_text = _aligned_placebo_text(df, text_columns, "sector_only", alignment_mode, placebo_variant, ticker_col, date_col)
    out = pd.DataFrame(index=df.index)

    target_score = pd.Series(0.0, index=df.index)
    sector_score = pd.Series(0.0, index=df.index)
    for group, weight in EVENT_TYPE_WEIGHTS.items():
        target_hit = _event_group_hits(target_text, group)
        sector_hit = _event_group_hits(sector_text, group)
        out[f"materiality_target_{group}"] = target_hit
        out[f"materiality_sector_{group}"] = sector_hit
        target_score += target_hit * weight
        sector_score += sector_hit * weight

    out["target_materiality_score"] = target_score
    out["sector_materiality_score"] = sector_score
    out["materiality_score"] = target_score * args.target_materiality_weight + sector_score * args.sector_materiality_weight
    out["materiality_score_log1p"] = np.log1p(out["materiality_score"])
    out["has_material_target_event"] = target_score.gt(0).astype(float)
    out["has_material_sector_event"] = sector_score.gt(0).astype(float)
    out["material_event_type_count"] = out[[f"materiality_target_{group}" for group in EVENT_TYPE_WEIGHTS]].sum(axis=1)
    return out


def _scale_embedding_by_materiality(raw: np.ndarray, materiality: pd.Series, scale: float) -> np.ndarray:
    score = materiality.astype(float).to_numpy()
    if np.nanmax(score) > 0:
        score = score / np.nanmax(score)
    multiplier = 1.0 + scale * np.nan_to_num(score, nan=0.0)
    return raw * multiplier.reshape(-1, 1)


def _materiality_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if col.startswith("materiality_") or col in {"target_materiality_score", "sector_materiality_score", "materiality_score", "materiality_score_log1p", "has_material_target_event", "has_material_sector_event", "material_event_type_count"}]


def write_summary(output_dir: Path, metrics: pd.DataFrame, ranks: pd.DataFrame, materiality_diag: pd.DataFrame) -> None:
    lines = ["# Materiality Embedding Summary", ""]
    if not metrics.empty:
        correct = metrics[metrics["placebo_variant"].eq("correct_text")]
        lines.append(f"- Correct rows: {len(correct)}")
        lines.append(f"- Positive MAE improvement rate: {(correct['mae_improvement_pct'] > 0).mean() * 100:.2f}%")
        lines.append(f"- Mean MAE improvement: {correct['mae_improvement_pct'].mean():.2f}%")
        lines.append(f"- Median MAE improvement: {correct['mae_improvement_pct'].median():.2f}%")
        lines.extend(["", "## By Variant", ""])
        by_variant = correct.groupby("embedding_variant")["mae_improvement_pct"].agg(["count", "mean", "median"]).sort_values("median", ascending=False)
        for variant, row in by_variant.iterrows():
            lines.append(f"- {variant}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
    if not ranks.empty:
        lines.extend(["", "## Placebo Rank", ""])
        lines.append(f"- Correct text ranked best in {(ranks['correct_text_rank'] == 1).sum()} of {len(ranks)} cases.")
        by_variant = ranks.assign(best=ranks["correct_text_rank"].eq(1)).groupby("embedding_variant")["best"].mean().sort_values(ascending=False)
        for variant, value in by_variant.items():
            lines.append(f"- {variant}: correct-best rate {value * 100:.2f}%")
    if not materiality_diag.empty:
        lines.extend(["", "## Materiality Diagnostics", ""])
        lines.append(f"- Mean materiality score: {materiality_diag['materiality_score'].mean():.3f}")
        lines.append(f"- Median materiality score: {materiality_diag['materiality_score'].median():.3f}")
        lines.append(f"- Rows with target material event: {materiality_diag['has_material_target_event'].mean() * 100:.2f}%")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Materiality scores are computed only from target/sector text available under the selected alignment and placebo mode.",
            "- The score is added as compact features and also scales the embedding before train-only PCA.",
            "- Treat positive results as predictive association, not causality.",
        ]
    )
    (output_dir / "materiality_embedding_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    materiality_diag_rows: list[pd.DataFrame] = []

    for base_variant in args.base_embedding_variants:
        for alignment_mode in args.alignment_modes:
            for placebo_variant in args.placebo_variants:
                materiality = build_materiality_features(df, text_columns, alignment_mode, placebo_variant, column_map.ticker, column_map.date, args)
                materiality_diag = materiality.copy()
                materiality_diag["base_embedding_variant"] = base_variant
                materiality_diag["alignment_mode"] = alignment_mode
                materiality_diag["placebo_variant"] = placebo_variant
                materiality_diag_rows.append(materiality_diag)

                raw = build_controlled_embedding(df, text_columns, base_variant, alignment_mode, placebo_variant, args)
                scaled = _scale_embedding_by_materiality(raw, materiality["materiality_score"], args.materiality_scale)
                embedding_variant = f"materiality_{base_variant}"
                emb_df, emb_cols = train_only_pca(scaled, train.index, args.pca_dim, f"{embedding_variant}_{alignment_mode}_{placebo_variant}")

                compact_materiality = materiality[_materiality_feature_columns(materiality)].reset_index(drop=True)
                work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True), compact_materiality], axis=1)
                materiality_cols = list(compact_materiality.columns)
                train_work = work.loc[train.index]
                test_work = work.loc[test.index]
                rows, cls_rows, ev_rows = evaluate_variant(
                    work,
                    train_work,
                    test_work,
                    specs,
                    ts_features,
                    emb_cols + materiality_cols,
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
    materiality_diag = pd.concat(materiality_diag_rows, ignore_index=True) if materiality_diag_rows else pd.DataFrame()

    metrics.to_csv(output_dir / "materiality_embedding_metrics.csv", index=False)
    ranks.to_csv(output_dir / "materiality_embedding_placebo_ranks.csv", index=False)
    classifier.to_csv(output_dir / "materiality_embedding_classifier_metrics.csv", index=False)
    events.to_csv(output_dir / "materiality_embedding_event_examples.csv", index=False)
    materiality_diag.to_csv(output_dir / "materiality_score_diagnostics.csv", index=False)
    if not events.empty:
        events.sort_values("abs_error_improvement", ascending=False).head(300).to_csv(output_dir / "materiality_embedding_top_improvements.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=True).head(300).to_csv(output_dir / "materiality_embedding_top_failures.csv", index=False)
    write_summary(output_dir, metrics, ranks, materiality_diag)
    print(f"Materiality embedding outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
