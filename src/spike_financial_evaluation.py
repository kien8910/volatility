"""Financial-style evaluation for controlled spike embedding outputs.

This module is read-only with respect to model pipelines: it consumes existing
controlled embedding outputs and writes additional volatility-forecast
diagnostics commonly used in finance.

It focuses on event/elevated-volatility rows available in
``controlled_embedding_event_examples.csv``:

- QLIKE-style variance loss from log-volatility forecasts;
- asymmetric underprediction loss;
- Diebold-Mariano tests on event-row losses;
- ticker/date block bootstrap confidence intervals;
- classifier summaries for PR-AUC/recall/F1;
- optional event-type summaries from ``unique_event_day_audit.csv``.

Example:

    python -m src.spike_financial_evaluation \
      --input_dir outputs/spike_controlled_embedding_confirmatory_p80_p95 \
      --audit_dir outputs/spike_event_day_audit_confirmatory_p80_p95 \
      --output_dir outputs/spike_financial_evaluation_confirmatory_p80_p95
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-12
LOSS_NAMES = ["absolute_error", "squared_error", "qlike", "underprediction_loss"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run financial evaluation on controlled spike embedding outputs.")
    parser.add_argument("--input_dir", default="outputs/spike_controlled_embedding_confirmatory_p80_p95")
    parser.add_argument("--audit_dir", default="outputs/spike_event_day_audit_confirmatory_p80_p95")
    parser.add_argument("--output_dir", default="outputs/spike_financial_evaluation_confirmatory_p80_p95")
    parser.add_argument("--bootstrap_iterations", type=int, default=1000)
    parser.add_argument("--block_size", type=int, default=5)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--top_risk_quantile", type=float, default=0.9)
    return parser.parse_args()


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _normal_two_sided_pvalue(statistic: float) -> float:
    if not np.isfinite(statistic):
        return np.nan
    return float(math.erfc(abs(statistic) / math.sqrt(2.0)))


def _log_forecast_to_variance(values: pd.Series) -> pd.Series:
    clipped = values.astype(float).clip(lower=-30, upper=30)
    return np.exp(2.0 * clipped).clip(lower=EPS)


def _qlike(actual_log_vol: pd.Series, forecast_log_vol: pd.Series) -> pd.Series:
    actual_var = _log_forecast_to_variance(actual_log_vol)
    forecast_var = _log_forecast_to_variance(forecast_log_vol)
    return np.log(forecast_var) + actual_var / forecast_var


def _add_losses(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for prefix, pred_col in [("ts", "ts_only_prediction"), ("embedding", "embedding_prediction")]:
        error = work[pred_col] - work["actual_target"]
        work[f"{prefix}_absolute_error"] = error.abs()
        work[f"{prefix}_squared_error"] = error.pow(2)
        work[f"{prefix}_underprediction_loss"] = (work["actual_target"] - work[pred_col]).clip(lower=0)
        work[f"{prefix}_qlike"] = _qlike(work["actual_target"], work[pred_col])
    for loss in LOSS_NAMES:
        work[f"{loss}_improvement"] = work[f"ts_{loss}"] - work[f"embedding_{loss}"]
        denom = work[f"ts_{loss}"].replace(0, np.nan)
        work[f"{loss}_improvement_pct"] = work[f"{loss}_improvement"] / denom * 100.0
    work["embedding_predicted_risk"] = work["embedding_prediction"]
    work["ts_predicted_risk"] = work["ts_only_prediction"]
    return work


def _safe_mean(series: pd.Series) -> float:
    return float(series.mean()) if len(series) else np.nan


def _loss_comparison(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ticker", "target_name", "horizon", "spike_percentile", "embedding_variant", "alignment_mode"]
    rows = []
    for key, group in events.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, key))
        row["num_event_rows"] = int(len(group))
        row["num_unique_dates"] = int(group["date"].nunique())
        row["positive_abs_error_improvement_rate"] = float(group["absolute_error_improvement"].gt(0).mean())
        for loss in LOSS_NAMES:
            row[f"ts_{loss}"] = _safe_mean(group[f"ts_{loss}"])
            row[f"embedding_{loss}"] = _safe_mean(group[f"embedding_{loss}"])
            row[f"{loss}_improvement"] = _safe_mean(group[f"{loss}_improvement"])
            row[f"{loss}_improvement_pct"] = _safe_mean(group[f"{loss}_improvement_pct"])
        rows.append(row)
    return pd.DataFrame(rows)


def _newey_west_variance(values: np.ndarray, max_lag: int | None = None) -> float:
    clean = values[np.isfinite(values)]
    n = len(clean)
    if n < 2:
        return np.nan
    centered = clean - clean.mean()
    if max_lag is None:
        max_lag = min(10, int(np.floor(n ** 0.25)))
    gamma0 = float(np.dot(centered, centered) / n)
    var = gamma0
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        weight = 1.0 - lag / (max_lag + 1.0)
        gamma = float(np.dot(centered[lag:], centered[:-lag]) / n)
        var += 2.0 * weight * gamma
    return var / n


def _dm_tests(events: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["ticker", "target_name", "horizon", "embedding_variant", "alignment_mode"]
    rows = []
    for key, group in events.sort_values("date").groupby(group_cols, dropna=False):
        base = dict(zip(group_cols, key))
        for loss in LOSS_NAMES:
            # Positive improvement means text has lower loss. DM loss differential
            # is TS loss minus embedding loss, so positive statistic favors text.
            diff = group[f"{loss}_improvement"].to_numpy(dtype=float)
            mean_diff = float(np.nanmean(diff)) if len(diff) else np.nan
            var = _newey_west_variance(diff)
            stat = mean_diff / math.sqrt(var) if var and var > 0 else np.nan
            rows.append(
                {
                    **base,
                    "test_name": f"dm_{loss}",
                    "num_rows": int(np.isfinite(diff).sum()),
                    "mean_loss_improvement": mean_diff,
                    "statistic": stat,
                    "pvalue": _normal_two_sided_pvalue(stat),
                    "favors_embedding": bool(mean_diff > 0) if np.isfinite(mean_diff) else False,
                }
            )
    return pd.DataFrame(rows)


def _make_blocks(events: pd.DataFrame, block_size: int) -> list[pd.DataFrame]:
    blocks: list[pd.DataFrame] = []
    for _, ticker_group in events.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
        unique_dates = ticker_group["date"].drop_duplicates().tolist()
        for start in range(0, len(unique_dates), block_size):
            dates = set(unique_dates[start : start + block_size])
            block = ticker_group[ticker_group["date"].isin(dates)]
            if not block.empty:
                blocks.append(block)
    return blocks


def _block_bootstrap(events: pd.DataFrame, iterations: int, block_size: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    blocks = _make_blocks(events, block_size)
    if not blocks:
        return pd.DataFrame()
    rows = []
    for loss in LOSS_NAMES:
        samples = []
        for _ in range(iterations):
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            sample = pd.concat([blocks[i] for i in chosen], ignore_index=True)
            samples.append(float(sample[f"{loss}_improvement"].mean()))
        values = np.asarray(samples, dtype=float)
        observed = float(events[f"{loss}_improvement"].mean())
        rows.append(
            {
                "loss_name": loss,
                "observed_mean_improvement": observed,
                "bootstrap_mean": float(np.mean(values)),
                "confidence_interval_low": float(np.quantile(values, 0.025)),
                "confidence_interval_high": float(np.quantile(values, 0.975)),
                "probability_improvement_positive": float(np.mean(values > 0)),
                "bootstrap_iterations": int(iterations),
                "block_size": int(block_size),
            }
        )
    return pd.DataFrame(rows)


def _top_risk_proxy(events: pd.DataFrame, quantile: float) -> pd.DataFrame:
    rows = []
    group_cols = ["target_name", "horizon", "embedding_variant", "alignment_mode"]
    for key, group in events.groupby(group_cols, dropna=False):
        if len(group) < 5:
            continue
        threshold = group["embedding_predicted_risk"].quantile(quantile)
        top = group[group["embedding_predicted_risk"] >= threshold]
        rest = group[group["embedding_predicted_risk"] < threshold]
        if top.empty or rest.empty:
            continue
        rows.append(
            {
                **dict(zip(group_cols, key)),
                "risk_quantile": quantile,
                "num_top_risk_rows": int(len(top)),
                "num_other_rows": int(len(rest)),
                "top_mean_actual_target": float(top["actual_target"].mean()),
                "other_mean_actual_target": float(rest["actual_target"].mean()),
                "top_mean_abs_return_proxy": float(np.exp(top["actual_target"]).mean()),
                "other_mean_abs_return_proxy": float(np.exp(rest["actual_target"]).mean()),
                "top_mean_abs_error_improvement": float(top["absolute_error_improvement"].mean()),
                "other_mean_abs_error_improvement": float(rest["absolute_error_improvement"].mean()),
                "top_underprediction_reduction": float(top["underprediction_loss_improvement"].mean()),
                "other_underprediction_reduction": float(rest["underprediction_loss_improvement"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _classifier_summary(classifier: pd.DataFrame) -> pd.DataFrame:
    if classifier.empty:
        return pd.DataFrame()
    correct = classifier[classifier["placebo_variant"].eq("correct_text")].copy()
    group_cols = ["target_name", "horizon", "spike_percentile", "embedding_variant", "alignment_mode"]
    return (
        correct.groupby(group_cols, dropna=False)
        .agg(
            rows=("pr_auc", "size"),
            mean_pr_auc=("pr_auc", "mean"),
            median_pr_auc=("pr_auc", "median"),
            mean_roc_auc=("roc_auc", "mean"),
            mean_recall=("recall", "mean"),
            mean_f1=("f1", "mean"),
            mean_num_test_spikes=("num_test_spikes", "mean"),
        )
        .reset_index()
    )


def _event_type_summary(audit: pd.DataFrame) -> pd.DataFrame:
    if audit.empty or "event_tags" not in audit.columns:
        return pd.DataFrame()
    work = audit.copy()
    work["event_tags"] = work["event_tags"].fillna("").replace("", "unclassified")
    return (
        work.groupby("event_tags", dropna=False)
        .agg(
            events=("mean_abs_error_improvement", "size"),
            robust_positive_rate=("robust_positive_event", "mean"),
            mean_event_improvement=("mean_abs_error_improvement", "mean"),
            median_event_improvement=("median_abs_error_improvement", "median"),
            mean_positive_config_rate=("positive_config_rate", "mean"),
            mean_underprediction_reduction=("mean_underprediction_reduction", "mean"),
            mean_correct_best_rate=("correct_best_rate", "mean"),
        )
        .reset_index()
        .sort_values(["median_event_improvement", "events"], ascending=False)
    )


def _write_report(
    output_dir: Path,
    events: pd.DataFrame,
    loss_summary: pd.DataFrame,
    dm_tests: pd.DataFrame,
    bootstrap: pd.DataFrame,
    top_risk: pd.DataFrame,
    classifier_summary: pd.DataFrame,
    event_type_summary: pd.DataFrame,
) -> None:
    lines = ["# Spike Financial Evaluation Report", ""]
    lines.append(f"- Event rows evaluated: {len(events)}")
    lines.append(f"- Unique tickers: {events['ticker'].nunique() if 'ticker' in events else 0}")
    lines.append(f"- Unique dates: {events['date'].nunique() if 'date' in events else 0}")
    lines.append(f"- Mean absolute-error improvement: {events['absolute_error_improvement'].mean():.4f}")
    lines.append(f"- Median absolute-error improvement: {events['absolute_error_improvement'].median():.4f}")
    lines.append(f"- Mean QLIKE improvement: {events['qlike_improvement'].mean():.4f}")
    lines.append(f"- Mean underprediction loss improvement: {events['underprediction_loss_improvement'].mean():.4f}")

    if not bootstrap.empty:
        lines.extend(["", "## Block Bootstrap", ""])
        for _, row in bootstrap.iterrows():
            lines.append(
                f"- {row['loss_name']}: observed={row['observed_mean_improvement']:.4f}, "
                f"95% CI=[{row['confidence_interval_low']:.4f}, {row['confidence_interval_high']:.4f}], "
                f"P(improvement>0)={row['probability_improvement_positive']:.3f}"
            )

    if not dm_tests.empty:
        significant = dm_tests[(dm_tests["pvalue"] < 0.05) & (dm_tests["favors_embedding"])]
        lines.extend(["", "## Diebold-Mariano Style Tests", ""])
        lines.append(f"- Text-favoring tests with p<0.05: {len(significant)} / {len(dm_tests)}")
        for _, row in significant.sort_values("mean_loss_improvement", ascending=False).head(8).iterrows():
            lines.append(
                f"- {row['ticker']} {row['target_name']} h={row['horizon']} {row['embedding_variant']} "
                f"{row['alignment_mode']} {row['test_name']}: improvement={row['mean_loss_improvement']:.4f}, p={row['pvalue']:.4f}"
            )

    if not loss_summary.empty:
        lines.extend(["", "## Best Loss Groups", ""])
        best = loss_summary.sort_values("absolute_error_improvement", ascending=False).head(8)
        for _, row in best.iterrows():
            lines.append(
                f"- {row['ticker']} {row['target_name']} h={row['horizon']} p{row['spike_percentile']} "
                f"{row['embedding_variant']} {row['alignment_mode']}: abs improvement={row['absolute_error_improvement']:.4f}, "
                f"QLIKE improvement={row['qlike_improvement']:.4f}"
            )

    if not top_risk.empty:
        lines.extend(["", "## Top Predicted-Risk Proxy", ""])
        top = top_risk.sort_values("top_mean_abs_error_improvement", ascending=False).head(6)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['target_name']} h={row['horizon']} {row['embedding_variant']} {row['alignment_mode']}: "
                f"top-risk abs improvement={row['top_mean_abs_error_improvement']:.4f}, "
                f"other={row['other_mean_abs_error_improvement']:.4f}"
            )

    if not classifier_summary.empty:
        lines.extend(["", "## Classifier Summary", ""])
        top = classifier_summary.sort_values("mean_pr_auc", ascending=False).head(8)
        for _, row in top.iterrows():
            lines.append(
                f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']} "
                f"{row['embedding_variant']} {row['alignment_mode']}: PR-AUC={row['mean_pr_auc']:.4f}, recall={row['mean_recall']:.4f}"
            )

    if not event_type_summary.empty:
        lines.extend(["", "## Event Type Summary", ""])
        filtered = event_type_summary[event_type_summary["events"] >= 5].head(8)
        for _, row in filtered.iterrows():
            lines.append(
                f"- {row['event_tags']}: events={int(row['events'])}, median={row['median_event_improvement']:.4f}, "
                f"robust={row['robust_positive_rate'] * 100:.1f}%"
            )

    lines.extend(
        [
            "",
            "## Limitations",
            "",
            "- QLIKE is computed from log-volatility-style targets by exponentiating to a variance proxy.",
            "- Top-risk proxy is computed within available event/elevated-volatility rows, not the full test universe.",
            "- DM tests are exploratory and use Newey-West style variance on event rows.",
            "- Block bootstrap resamples ticker/date blocks to reduce iid assumptions, but it remains a lightweight diagnostic.",
        ]
    )
    (output_dir / "financial_evaluation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    input_dir = Path(args.input_dir)
    audit_dir = Path(args.audit_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events = _read_csv_if_exists(input_dir / "controlled_embedding_event_examples.csv")
    if events.empty:
        raise FileNotFoundError(f"No controlled_embedding_event_examples.csv found in {input_dir}")
    events = _add_losses(events)

    classifier = _read_csv_if_exists(input_dir / "controlled_embedding_classifier_metrics.csv")
    audit = _read_csv_if_exists(audit_dir / "unique_event_day_audit.csv")

    loss_summary = _loss_comparison(events)
    dm_tests = _dm_tests(events)
    bootstrap = _block_bootstrap(events, args.bootstrap_iterations, args.block_size, args.random_seed)
    top_risk = _top_risk_proxy(events, args.top_risk_quantile)
    classifier_summary = _classifier_summary(classifier)
    event_type_summary = _event_type_summary(audit)

    events.to_csv(output_dir / "financial_event_losses.csv", index=False)
    loss_summary.to_csv(output_dir / "financial_forecast_metrics.csv", index=False)
    dm_tests.to_csv(output_dir / "dm_tests_financial_losses.csv", index=False)
    bootstrap.to_csv(output_dir / "block_bootstrap_confidence_intervals.csv", index=False)
    top_risk.to_csv(output_dir / "top_risk_lift_proxy.csv", index=False)
    classifier_summary.to_csv(output_dir / "classifier_financial_summary.csv", index=False)
    event_type_summary.to_csv(output_dir / "event_type_financial_summary.csv", index=False)

    _write_report(output_dir, events, loss_summary, dm_tests, bootstrap, top_risk, classifier_summary, event_type_summary)
    print(f"Financial evaluation outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
