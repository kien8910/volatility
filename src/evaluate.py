"""Evaluation helpers and output writers."""

from __future__ import annotations

from pathlib import Path
import math

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _paired_ttest(errors_a: np.ndarray, errors_b: np.ndarray) -> float:
    diff = errors_a - errors_b
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2 or np.isclose(diff.std(ddof=1), 0):
        return np.nan
    t_stat = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff)))
    try:
        from scipy import stats

        return float(stats.ttest_rel(errors_a, errors_b, nan_policy="omit").pvalue)
    except Exception:
        # Normal approximation, kept dependency-free because scipy is optional.
        return float(2 * (1 - 0.5 * (1 + math.erf(abs(t_stat) / np.sqrt(2)))))


def summarize_predictions(predictions: pd.DataFrame, target_col: str = "target", baseline_model: str = "TS_only_lags") -> pd.DataFrame:
    rows = []
    baseline = predictions[predictions["model"] == baseline_model].set_index("row_id")
    baseline_abs = (baseline["prediction"] - baseline[target_col]).abs()

    for model, group in predictions.groupby("model"):
        y_true = group[target_col].to_numpy()
        y_pred = group["prediction"].to_numpy()
        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_true, y_pred) if len(group) > 1 else np.nan
        improvement = np.nan
        p_value = np.nan
        if model != baseline_model and not baseline.empty:
            aligned = group.set_index("row_id").join(baseline_abs.rename("baseline_abs_error"), how="inner")
            model_abs = (aligned["prediction"] - aligned[target_col]).abs()
            baseline_mae = aligned["baseline_abs_error"].mean()
            improvement = (baseline_mae - model_abs.mean()) / baseline_mae * 100 if baseline_mae else np.nan
            p_value = _paired_ttest(model_abs.to_numpy(), aligned["baseline_abs_error"].to_numpy())
        rows.append({"model": model, "MAE": mae, "RMSE": rmse, "MSE": mse, "R2": r2, "MAE_improvement_vs_TS_only_lags_pct": improvement, "paired_ttest_abs_error_pvalue": p_value})
    return pd.DataFrame(rows).sort_values("RMSE")


def error_by_ticker(predictions: pd.DataFrame, ticker_col: str, target_col: str = "target") -> pd.DataFrame:
    rows = []
    for (model, ticker), group in predictions.groupby(["model", ticker_col]):
        mse = mean_squared_error(group[target_col], group["prediction"])
        rows.append({"model": model, ticker_col: ticker, "MAE": mean_absolute_error(group[target_col], group["prediction"]), "RMSE": np.sqrt(mse), "MSE": mse, "count": len(group)})
    return pd.DataFrame(rows)


def write_evaluation_outputs(predictions: pd.DataFrame, output_dir: str | Path, ticker_col: str) -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results = summarize_predictions(predictions)
    results.to_csv(output / "results.csv", index=False)
    predictions.to_csv(output / "predictions.csv", index=False)
    error_by_ticker(predictions, ticker_col).to_csv(output / "error_by_ticker.csv", index=False)
    return results
