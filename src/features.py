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


def time_series_feature_columns(max_lag: int = 22) -> list[str]:
    return (
        ["log_return", "har_daily", "har_weekly", "har_monthly"]
        + [f"logGKVol_lag_{lag}" for lag in range(1, max_lag + 1)]
    )
