"""Time-series feature engineering for Garman-Klass volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import EPSILON
from .load_data import ColumnMap


def add_volatility_features(
    df: pd.DataFrame,
    column_map: ColumnMap,
    forecast_horizon: int = 1,
    max_lag: int = 22,
) -> pd.DataFrame:
    """Create leakage-safe volatility, HAR, lag, and future target features."""
    out = df.copy()
    ticker = column_map.ticker

    open_px = out[column_map.open].clip(lower=EPSILON)
    high_px = out[column_map.high].clip(lower=EPSILON)
    low_px = out[column_map.low].clip(lower=EPSILON)
    close_px = out[column_map.close].clip(lower=EPSILON)

    out["log_return"] = np.log(close_px / close_px.groupby(out[ticker]).shift(1))
    gk_var = 0.5 * np.square(np.log(high_px / low_px)) - (2 * np.log(2) - 1) * np.square(np.log(close_px / open_px))
    out["GKVar"] = np.maximum(gk_var, EPSILON)
    out["GKVol"] = np.sqrt(out["GKVar"])
    out["logGKVol"] = np.log(out["GKVol"] + EPSILON)

    grouped = out.groupby(ticker, sort=False)["logGKVol"]
    out["har_daily"] = out["logGKVol"]
    out["har_weekly"] = grouped.transform(lambda s: s.rolling(window=5, min_periods=5).mean())
    out["har_monthly"] = grouped.transform(lambda s: s.rolling(window=22, min_periods=22).mean())

    for lag in range(1, max_lag + 1):
        out[f"logGKVol_lag_{lag}"] = grouped.shift(lag)

    out["target"] = grouped.shift(-forecast_horizon)
    return out


def add_detailed_volatility_targets(df: pd.DataFrame, column_map: ColumnMap, horizons: list[int]) -> pd.DataFrame:
    """Add direct, cumulative, overnight, and return-based future volatility targets.

    All targets use only future prices relative to the row date and are meant to be
    consumed after a chronological train/test split.
    """
    out = df.copy()
    ticker = column_map.ticker
    close_px = out[column_map.close].clip(lower=EPSILON)
    open_px = out[column_map.open].clip(lower=EPSILON)
    grouped_gk = out.groupby(ticker, sort=False)["GKVar"]
    grouped_close = close_px.groupby(out[ticker], sort=False)
    grouped_open = open_px.groupby(out[ticker], sort=False)

    for horizon in sorted(set(horizons)):
        future_gk = grouped_gk.shift(-horizon)
        out[f"GKVar_t_plus_{horizon}"] = future_gk
        out[f"logGKVol_t_plus_{horizon}"] = np.log(np.sqrt(future_gk.clip(lower=EPSILON)) + EPSILON)

        future_close = grouped_close.shift(-horizon)
        ret = np.log(future_close / close_px)
        out[f"squared_return_t_plus_{horizon}"] = np.square(ret)
        out[f"log_abs_return_t_plus_{horizon}"] = np.log(ret.abs() + EPSILON)

        future_open = grouped_open.shift(-horizon)
        overnight = np.log(future_open / close_px)
        total_var = np.square(overnight) + future_gk
        out[f"overnight_return_t_plus_{horizon}"] = overnight
        out[f"overnight_variance_t_plus_{horizon}"] = np.square(overnight)
        out[f"total_variance_t_plus_{horizon}"] = total_var
        out[f"log_total_volatility_t_plus_{horizon}"] = np.log(np.sqrt(total_var.clip(lower=EPSILON)) + EPSILON)

    for horizon in [h for h in [3, 5] if h in set(horizons)]:
        future_sum = None
        for step in range(1, horizon + 1):
            shifted = grouped_gk.shift(-step)
            future_sum = shifted if future_sum is None else future_sum + shifted
        out[f"GKVarFuture_{horizon}"] = future_sum
        out[f"FutureVol_{horizon}"] = np.sqrt(future_sum.clip(lower=EPSILON))
        out[f"logFutureVol_{horizon}"] = np.log(out[f"FutureVol_{horizon}"] + EPSILON)
    return out


def target_column_for(target_name: str, horizon: int) -> str:
    mapping = {
        "log_gk": f"logGKVol_t_plus_{horizon}",
        "total_volatility": f"log_total_volatility_t_plus_{horizon}",
        "log_abs_return": f"log_abs_return_t_plus_{horizon}",
        "squared_return": f"squared_return_t_plus_{horizon}",
        "future_volatility": f"logFutureVol_{horizon}",
    }
    if target_name not in mapping:
        raise ValueError(f"Unknown target name: {target_name}")
    return mapping[target_name]


def time_series_feature_columns(max_lag: int = 22) -> list[str]:
    return (
        ["log_return", "har_daily", "har_weekly", "har_monthly"]
        + [f"logGKVol_lag_{lag}" for lag in range(1, max_lag + 1)]
    )
