"""Build presentation tables for text-correction models, optionally including LSTM.

This script reads one temporal-text experiment output directory and optionally
one LSTM text-correction output directory. It writes combined text-correction
tables that exclude early-fusion/full-row text models.

Example:

    python -m src.text_correction_result_summary \
      --temporal_dir outputs/spike_temporal_text_experiment_v2 \
      --lstm_dir outputs/spike_lstm_text_correction \
      --output_dir outputs/text_correction_combined
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


TEMPORAL_TEXT_CORRECTION_MODELS = {
    "HAR_TextResidual_Gated": "HAR_Ridge",
    "LookbackGKReturn_TextResidual_Gated": "LookbackGKReturn_Ridge",
    "TemporalStats_TextResidual_Gated": "TemporalStats_Ridge",
}
LSTM_TEXT_CORRECTION_MODELS = {
    "LSTM_TextResidual_Gated": "LSTM_Temporal",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize text-correction-only results, with optional LSTM.")
    parser.add_argument("--temporal_dir", required=True, help="Directory containing temporal_text_metrics.csv.")
    parser.add_argument("--lstm_dir", default=None, help="Optional directory containing lstm_text_correction_metrics.csv.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument(
        "--conservative_embedding_variant",
        default="target_sector_event_weighted",
        help="Embedding variant used for the conservative view.",
    )
    parser.add_argument(
        "--conservative_alignment_mode",
        default="shifted_1_day_text",
        help="Alignment mode used for the conservative view.",
    )
    return parser.parse_args()


def _load_temporal_metrics(path: Path) -> pd.DataFrame:
    metrics_path = path / "temporal_text_metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing temporal metrics file: {metrics_path}")
    metrics = pd.read_csv(metrics_path)
    out = metrics[metrics["model"].isin(TEMPORAL_TEXT_CORRECTION_MODELS)].copy()
    out["base_model"] = out["model"].map(TEMPORAL_TEXT_CORRECTION_MODELS)
    out["source_experiment"] = "temporal_text"
    return out


def _load_lstm_metrics(path: Path | None) -> pd.DataFrame:
    if path is None:
        return pd.DataFrame()
    metrics_path = path / "lstm_text_correction_metrics.csv"
    if not metrics_path.exists():
        return pd.DataFrame()
    metrics = pd.read_csv(metrics_path)
    out = metrics[metrics["model"].isin(LSTM_TEXT_CORRECTION_MODELS)].copy()
    out["base_model"] = out["model"].map(LSTM_TEXT_CORRECTION_MODELS)
    out["source_experiment"] = "lstm_text"
    return out


def _standardize(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    out = metrics.copy()
    out["text_integration_type"] = "gated_residual_text_correction"
    out["text_used_rate"] = out["gate_active_rate"]
    front = [
        "source_experiment",
        "text_integration_type",
        "base_model",
        "model",
        "ticker",
        "target_name",
        "horizon",
        "spike_percentile",
        "segment",
        "embedding_variant",
        "alignment_mode",
        "gate_type",
        "text_used_rate",
        "gate_active_rate",
        "num_samples",
        "num_spikes",
    ]
    return out[[col for col in front if col in out.columns] + [col for col in out.columns if col not in front]]


def _summary_by_segment(metrics: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        "mae_improvement_vs_har_pct",
        "rmse_improvement_vs_har_pct",
        "underprediction_improvement_vs_har",
        "qlike_improvement_vs_har",
        "mae",
        "rmse",
        "underprediction_loss",
        "asymmetric_under_2x_loss",
        "qlike_proxy_loss",
        "spike_f1",
    ]
    available = [col for col in value_cols if col in metrics.columns]
    summary = (
        metrics.groupby(
            ["source_experiment", "base_model", "model", "embedding_variant", "alignment_mode", "gate_type", "segment"],
            dropna=False,
        )[available]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    summary.columns = [
        "_".join([str(part) for part in col if str(part)])
        if isinstance(col, tuple)
        else str(col)
        for col in summary.columns
    ]
    return summary


def _ticker_percentile_segment(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby(
            [
                "source_experiment",
                "base_model",
                "model",
                "embedding_variant",
                "alignment_mode",
                "ticker",
                "spike_percentile",
                "segment",
            ],
            dropna=False,
        )
        .agg(
            cases=("mae_improvement_vs_har_pct", "size"),
            total_samples=("num_samples", "sum"),
            total_spikes=("num_spikes", "sum"),
            text_used_rate=("text_used_rate", "mean"),
            mae_improvement_vs_har_median=("mae_improvement_vs_har_pct", "median"),
            rmse_improvement_vs_har_median=("rmse_improvement_vs_har_pct", "median"),
            underprediction_improvement_vs_har_median=("underprediction_improvement_vs_har", "median"),
            qlike_improvement_vs_har_median=("qlike_improvement_vs_har", "median"),
        )
        .reset_index()
    )


def _write_report(output_dir: Path, metrics: pd.DataFrame, conservative: pd.DataFrame) -> None:
    lines = ["# Combined Text Correction Results", ""]
    lines.append("Included models:")
    for model in sorted(metrics["model"].dropna().unique()):
        base = metrics.loc[metrics["model"].eq(model), "base_model"].dropna().iloc[0]
        lines.append(f"- `{model}`: base model `{base}` plus gated residual text correction.")
    lines.extend(["", "## Overall By Segment", ""])
    overall = (
        metrics.groupby(["model", "segment"])
        .agg(
            cases=("mae_improvement_vs_har_pct", "size"),
            text_used_rate=("text_used_rate", "mean"),
            mae_median=("mae_improvement_vs_har_pct", "median"),
            rmse_median=("rmse_improvement_vs_har_pct", "median"),
            under_median=("underprediction_improvement_vs_har", "median"),
            qlike_median=("qlike_improvement_vs_har", "median"),
        )
        .reset_index()
        .sort_values(["segment", "mae_median"], ascending=[True, False])
    )
    lines.append(_to_markdown(overall))
    if not conservative.empty:
        lines.extend(["", "## Conservative View By Model And Segment", ""])
        view = (
            conservative.groupby(["model", "segment"])
            .agg(
                cases=("mae_improvement_vs_har_median", "size"),
                text_used_rate=("text_used_rate", "mean"),
                mae_median=("mae_improvement_vs_har_median", "median"),
                rmse_median=("rmse_improvement_vs_har_median", "median"),
                under_median=("underprediction_improvement_vs_har_median", "median"),
                qlike_median=("qlike_improvement_vs_har_median", "median"),
            )
            .reset_index()
            .sort_values(["segment", "mae_median"], ascending=[True, False])
        )
        lines.append(_to_markdown(view))
    (output_dir / "text_correction_combined_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _to_markdown(df: pd.DataFrame) -> str:
    view = df.copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: "" if pd.isna(value) else f"{value:.2f}")
    return view.to_markdown(index=False)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    temporal = _load_temporal_metrics(Path(args.temporal_dir))
    lstm = _load_lstm_metrics(Path(args.lstm_dir)) if args.lstm_dir else pd.DataFrame()
    combined = _standardize(pd.concat([temporal, lstm], ignore_index=True))
    if combined.empty:
        raise ValueError("No text-correction metrics found.")

    combined.to_csv(output_dir / "text_correction_combined_metrics.csv", index=False)
    _summary_by_segment(combined).to_csv(output_dir / "text_correction_combined_summary_by_segment.csv", index=False)
    ticker_summary = _ticker_percentile_segment(combined)
    ticker_summary.to_csv(output_dir / "text_correction_combined_ticker_percentile_segment.csv", index=False)

    conservative = ticker_summary[
        ticker_summary["embedding_variant"].eq(args.conservative_embedding_variant)
        & ticker_summary["alignment_mode"].eq(args.conservative_alignment_mode)
    ].copy()
    conservative.to_csv(output_dir / "text_correction_combined_conservative_view.csv", index=False)
    _write_report(output_dir, combined, conservative)
    print(f"Combined text correction outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
