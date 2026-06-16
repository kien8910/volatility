"""Create a unique event-day audit from controlled spike embedding outputs.

The controlled embedding outputs intentionally contain repeated event rows
across percentiles, horizons, embedding variants, and alignment modes. This
script collapses those rows into an event-day level audit without changing any
modeling pipeline.

Example:

    python -m src.spike_event_day_audit \
      --input_dir outputs/spike_controlled_embedding_confirmatory_p80_p95 \
      --output_dir outputs/spike_event_day_audit_confirmatory_p80_p95
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


EVENT_GROUP_KEYWORDS = {
    "earnings": ["earnings", "revenue", "eps", "profit", "loss", "guidance", "forecast", "outlook", "quarter"],
    "analyst": ["analyst", "rating", "upgrade", "downgrade", "target price", "price target"],
    "corporate_action": ["acquire", "acquisition", "merger", "buyback", "repurchase", "dividend", "spin"],
    "legal_regulatory": ["lawsuit", "antitrust", "regulator", "regulatory", "sec ", "justice department", "fda"],
    "product_operations": ["launch", "product", "supply", "shortage", "delay", "recall", "contract"],
    "macro_market": ["fed", "federal reserve", "inflation", "rates", "cpi", "employment", "bank stress", "deposit"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collapse controlled embedding spike rows into a unique event-day audit.")
    parser.add_argument("--input_dir", default="outputs/spike_controlled_embedding_confirmatory_p80_p95")
    parser.add_argument("--output_dir", default="outputs/spike_event_day_audit_confirmatory_p80_p95")
    parser.add_argument("--min_positive_config_rate", type=float, default=0.5)
    parser.add_argument("--min_mean_improvement", type=float, default=0.0)
    parser.add_argument("--top_n", type=int, default=100)
    return parser.parse_args()


def _read_required(input_dir: Path, name: str) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)


def _event_tags(text: str) -> str:
    text_low = str(text).lower()
    tags = [group for group, keywords in EVENT_GROUP_KEYWORDS.items() if any(keyword in text_low for keyword in keywords)]
    return ",".join(tags)


def _best_nonempty(series: pd.Series) -> str:
    for value in series.dropna().astype(str):
        value = value.strip()
        if value:
            return value
    return ""


def _unique_event_audit(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    work = events.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["positive_improvement"] = work["abs_error_improvement"] > 0
    work["positive_underprediction_reduction"] = work["underprediction_reduction"] > 0
    group_cols = ["ticker", "date", "target_name", "horizon"]
    rows = []
    for key, group in work.groupby(group_cols, dropna=False):
        ordered = group.sort_values("abs_error_improvement", ascending=False)
        best = ordered.iloc[0]
        worst = group.sort_values("abs_error_improvement", ascending=True).iloc[0]
        tags = _event_tags(_best_nonempty(group["text_preview"]))
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "num_rows": int(len(group)),
                "num_spike_percentiles": int(group["spike_percentile"].nunique()),
                "num_embedding_variants": int(group["embedding_variant"].nunique()),
                "num_alignment_modes": int(group["alignment_mode"].nunique()),
                "mean_abs_error_improvement": float(group["abs_error_improvement"].mean()),
                "median_abs_error_improvement": float(group["abs_error_improvement"].median()),
                "max_abs_error_improvement": float(group["abs_error_improvement"].max()),
                "min_abs_error_improvement": float(group["abs_error_improvement"].min()),
                "positive_config_rate": float(group["positive_improvement"].mean()),
                "mean_underprediction_reduction": float(group["underprediction_reduction"].mean()),
                "median_underprediction_reduction": float(group["underprediction_reduction"].median()),
                "underprediction_positive_rate": float(group["positive_underprediction_reduction"].mean()),
                "best_spike_percentile": best["spike_percentile"],
                "best_embedding_variant": best["embedding_variant"],
                "best_alignment_mode": best["alignment_mode"],
                "best_abs_error_improvement": best["abs_error_improvement"],
                "worst_spike_percentile": worst["spike_percentile"],
                "worst_embedding_variant": worst["embedding_variant"],
                "worst_alignment_mode": worst["alignment_mode"],
                "worst_abs_error_improvement": worst["abs_error_improvement"],
                "actual_target_at_best": best["actual_target"],
                "threshold_at_best": best["threshold"],
                "ts_only_prediction_at_best": best["ts_only_prediction"],
                "embedding_prediction_at_best": best["embedding_prediction"],
                "has_target_text": int(group["has_target_text"].max()) if "has_target_text" in group else np.nan,
                "has_sector_text": int(group["has_sector_text"].max()) if "has_sector_text" in group else np.nan,
                "max_event_keyword_count": int(group["event_keyword_count"].max()) if "event_keyword_count" in group else np.nan,
                "max_target_event_keyword_count": int(group["target_event_keyword_count"].max()) if "target_event_keyword_count" in group else np.nan,
                "event_tags": tags,
                "text_preview": _best_nonempty(group["text_preview"]),
            }
        )
    return pd.DataFrame(rows)


def _merge_placebo_context(audit: pd.DataFrame, ranks: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or ranks.empty:
        return audit
    rank_summary = (
        ranks.assign(correct_best=ranks["correct_text_rank"].eq(1))
        .groupby(["ticker", "target_name", "horizon"], dropna=False)
        .agg(
            placebo_rank_rows=("correct_best", "size"),
            correct_best_rate=("correct_best", "mean"),
            mean_correct_text_improvement=("correct_text_mae_improvement_pct", "mean"),
            median_correct_text_improvement=("correct_text_mae_improvement_pct", "median"),
            mean_correct_underprediction_reduction=("correct_text_underprediction_reduction", "mean"),
        )
        .reset_index()
    )
    return audit.merge(rank_summary, on=["ticker", "target_name", "horizon"], how="left")


def _summary_tables(audit: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if audit.empty:
        return {}
    aggs = {
        "num_events": ("mean_abs_error_improvement", "size"),
        "mean_improvement": ("mean_abs_error_improvement", "mean"),
        "median_improvement": ("median_abs_error_improvement", "median"),
        "positive_event_rate": ("positive_config_rate", lambda s: (s >= 0.5).mean()),
        "mean_positive_config_rate": ("positive_config_rate", "mean"),
        "mean_underprediction_reduction": ("mean_underprediction_reduction", "mean"),
        "mean_correct_best_rate": ("correct_best_rate", "mean"),
    }
    return {
        "by_ticker": audit.groupby("ticker", dropna=False).agg(**aggs).reset_index(),
        "by_target_horizon": audit.groupby(["target_name", "horizon"], dropna=False).agg(**aggs).reset_index(),
        "by_event_tags": audit.assign(event_tags=audit["event_tags"].replace("", "unclassified")).groupby("event_tags", dropna=False).agg(**aggs).reset_index(),
        "by_best_variant": audit.groupby("best_embedding_variant", dropna=False).agg(**aggs).reset_index(),
        "by_best_alignment": audit.groupby("best_alignment_mode", dropna=False).agg(**aggs).reset_index(),
    }


def _write_report(output_dir: Path, audit: pd.DataFrame, summaries: dict[str, pd.DataFrame], args: argparse.Namespace) -> None:
    lines = ["# Spike Event-Day Audit Report", ""]
    if audit.empty:
        lines.append("No event rows were available for audit.")
        (output_dir / "event_day_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    robust = audit[
        (audit["positive_config_rate"] >= args.min_positive_config_rate)
        & (audit["mean_abs_error_improvement"] > args.min_mean_improvement)
        & (audit["mean_underprediction_reduction"] > 0)
    ]
    lines.append(f"- Unique event rows: {len(audit)}")
    lines.append(f"- Robust positive events: {len(robust)} ({len(robust) / len(audit) * 100:.2f}%)")
    lines.append(f"- Mean event improvement: {audit['mean_abs_error_improvement'].mean():.4f}")
    lines.append(f"- Median event improvement: {audit['median_abs_error_improvement'].median():.4f}")
    lines.append(f"- Mean positive-config rate: {audit['positive_config_rate'].mean() * 100:.2f}%")
    lines.append(f"- Mean underprediction reduction: {audit['mean_underprediction_reduction'].mean():.4f}")

    for title, name in [
        ("Best Tickers", "by_ticker"),
        ("Best Target/Horizon", "by_target_horizon"),
        ("Best Event Tags", "by_event_tags"),
        ("Best Variants", "by_best_variant"),
        ("Best Alignments", "by_best_alignment"),
    ]:
        table = summaries.get(name)
        if table is None or table.empty:
            continue
        lines.extend(["", f"## {title}", ""])
        for _, row in table.sort_values(["median_improvement", "mean_improvement"], ascending=False).head(8).iterrows():
            label_cols = [col for col in table.columns if col not in {"num_events", "mean_improvement", "median_improvement", "positive_event_rate", "mean_positive_config_rate", "mean_underprediction_reduction", "mean_correct_best_rate"}]
            label = " ".join(str(row[col]) for col in label_cols)
            lines.append(
                f"- {label}: events={int(row['num_events'])}, median={row['median_improvement']:.4f}, "
                f"positive_event_rate={row['positive_event_rate'] * 100:.2f}%"
            )

    lines.extend(["", "## Top Robust Events", ""])
    for _, row in robust.sort_values("mean_abs_error_improvement", ascending=False).head(10).iterrows():
        lines.append(
            f"- {row['ticker']} {row['date'].date() if pd.notna(row['date']) else row['date']} "
            f"{row['target_name']} h={row['horizon']}: mean={row['mean_abs_error_improvement']:.4f}, "
            f"positive_config_rate={row['positive_config_rate'] * 100:.1f}%, tags={row['event_tags'] or 'unclassified'}"
        )

    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This audit deduplicates event rows for interpretation only; it does not retrain models.",
            "- Repeated rows across percentiles, variants, and alignments are summarized into robustness rates.",
            "- Treat the result as predictive association, not causality.",
        ]
    )
    (output_dir / "event_day_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = _read_required(input_dir, "controlled_embedding_event_examples.csv")
    ranks = _read_required(input_dir, "controlled_embedding_placebo_ranks.csv")
    audit = _unique_event_audit(events)
    audit = _merge_placebo_context(audit, ranks)
    audit["robust_positive_event"] = (
        audit["positive_config_rate"].ge(args.min_positive_config_rate)
        & audit["mean_abs_error_improvement"].gt(args.min_mean_improvement)
        & audit["mean_underprediction_reduction"].gt(0)
    )

    audit.to_csv(output_dir / "unique_event_day_audit.csv", index=False)
    audit[audit["robust_positive_event"]].sort_values("mean_abs_error_improvement", ascending=False).head(args.top_n).to_csv(
        output_dir / "top_robust_positive_events.csv", index=False
    )
    audit.sort_values("mean_abs_error_improvement", ascending=False).head(args.top_n).to_csv(output_dir / "top_unique_helpful_events.csv", index=False)
    audit.sort_values("mean_abs_error_improvement", ascending=True).head(args.top_n).to_csv(output_dir / "top_unique_harmful_events.csv", index=False)

    summaries = _summary_tables(audit)
    for name, table in summaries.items():
        table.to_csv(output_dir / f"event_audit_summary_{name}.csv", index=False)

    _write_report(output_dir, audit, summaries, args)
    print(f"Spike event-day audit outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
