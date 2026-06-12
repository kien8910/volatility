from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_detailed_volatility_targets, add_volatility_features, target_column_for
from src.load_data import ColumnMap
from src.text_features import train_only_pca_features


def _sample_prices() -> tuple[pd.DataFrame, ColumnMap]:
    df = pd.DataFrame(
        {
            "ticker": ["A"] * 8,
            "date": pd.date_range("2024-01-01", periods=8),
            "open": [10, 11, 12, 11, 13, 14, 15, 16],
            "high": [11, 12, 13, 12, 14, 15, 16, 17],
            "low": [9, 10, 11, 10, 12, 13, 14, 15],
            "close": [10.5, 11.5, 11.2, 12.8, 13.7, 14.2, 15.5, 16.1],
        }
    )
    return df, ColumnMap("ticker", "date", "open", "high", "low", "close")


def test_garman_klass_calculation_matches_formula() -> None:
    df, cols = _sample_prices()
    out = add_volatility_features(df, cols, forecast_horizon=1, max_lag=1)
    expected = 0.5 * np.log(11 / 9) ** 2 - (2 * np.log(2) - 1) * np.log(10.5 / 10) ** 2
    assert np.isclose(out.loc[0, "GKVar"], expected)
    assert np.isclose(out.loc[0, "target"], out.loc[1, "logGKVol"])


def test_overnight_and_horizon_targets_align_to_future_rows() -> None:
    df, cols = _sample_prices()
    out = add_volatility_features(df, cols, forecast_horizon=1, max_lag=1)
    out = add_detailed_volatility_targets(out, cols, [1, 3])
    overnight = np.log(df.loc[1, "open"] / df.loc[0, "close"])
    assert np.isclose(out.loc[0, "overnight_return_t_plus_1"], overnight)
    assert np.isclose(out.loc[0, "logGKVol_t_plus_3"], out.loc[3, "logGKVol"])
    assert target_column_for("total_volatility", 1) == "log_total_volatility_t_plus_1"


def test_cumulative_future_gk_uses_only_future_window() -> None:
    df, cols = _sample_prices()
    out = add_volatility_features(df, cols, forecast_horizon=1, max_lag=1)
    out = add_detailed_volatility_targets(out, cols, [3])
    expected = out.loc[1, "GKVar"] + out.loc[2, "GKVar"] + out.loc[3, "GKVar"]
    assert np.isclose(out.loc[0, "GKVarFuture_3"], expected)


def test_train_only_pca_uses_train_rows_for_component_count() -> None:
    matrix = np.arange(30, dtype=float).reshape(10, 3)
    features, names = train_only_pca_features(matrix, pd.Index([0, 1]), pca_dim=3, prefix="x")
    assert features.shape == (10, 2)
    assert names == ["x_emb_0", "x_emb_1"]


def test_text_shift_alignment_can_be_computed_without_future_rows() -> None:
    df = pd.DataFrame({"ticker": ["A", "A", "A"], "has_text_target": [1, 0, 1]})
    shifted = df.groupby("ticker", sort=False)[["has_text_target"]].shift(1).fillna(0)
    assert shifted["has_text_target"].tolist() == [0, 1, 0]
