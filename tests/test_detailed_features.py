from __future__ import annotations

import numpy as np
import pandas as pd

from src.features import add_detailed_volatility_targets, add_volatility_features, target_column_for
from src.load_data import ColumnMap
from src.detailed_news_signal import placebo_tests, TargetSpec
from src.text_features import (
    combine_text_columns,
    joined_text_for_configuration,
    normalize_text_value,
    text_columns_for_configuration,
    train_only_pca_features,
)


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


def test_normalize_text_value_removes_missing_literals() -> None:
    assert normalize_text_value(None) == ""
    assert normalize_text_value(np.nan) == ""
    assert normalize_text_value(pd.NA) == ""
    assert normalize_text_value(" None ") == ""
    assert normalize_text_value("nan") == ""
    assert normalize_text_value("NULL") == ""
    assert normalize_text_value("  real news  ") == "real news"


def test_combine_text_columns_drops_empty_and_exact_duplicates() -> None:
    row = pd.Series({"a": " headline ", "b": None, "c": "headline", "d": "None", "e": "second"})
    assert combine_text_columns(row, ["a", "b", "c", "d", "e"], separator="|") == "headline|second"


def test_news_all_excludes_filing_columns() -> None:
    text_columns = {
        "macro": ["macro_category1"],
        "sector": ["sector_category1"],
        "related": ["relatedCompany_category1"],
        "target": ["targetCompany_category1"],
        "filing": ["filing_financialStatement"],
    }
    assert text_columns_for_configuration(text_columns, "news_all") == [
        "macro_category1",
        "sector_category1",
        "relatedCompany_category1",
        "targetCompany_category1",
    ]
    assert text_columns_for_configuration(text_columns, "news_plus_filing")[-1] == "filing_financialStatement"


def test_joined_text_configuration_does_not_emit_none_tokens() -> None:
    df = pd.DataFrame(
        {
            "macro_category1": ["None"],
            "sector_category1": [None],
            "relatedCompany_category1": ["nan"],
            "targetCompany_category1": ["target news"],
            "filing_financialStatement": ["filing context"],
        }
    )
    text_columns = {
        "macro": ["macro_category1"],
        "sector": ["sector_category1"],
        "related": ["relatedCompany_category1"],
        "target": ["targetCompany_category1"],
        "filing": ["filing_financialStatement"],
    }
    assert joined_text_for_configuration(df, text_columns, "news_all").iloc[0] == "target news"
    assert "filing context" not in joined_text_for_configuration(df, text_columns, "news_all").iloc[0]
    assert "filing context" in joined_text_for_configuration(df, text_columns, "news_plus_filing").iloc[0]


def test_placebo_cross_ticker_variant_preserves_feature_shape(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "A", "B", "A", "B"],
            "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"]),
            "row_id": range(6),
            "har_daily": [1, 1, 1, 1, 1, 1],
            "har_weekly": [1, 1, 1, 1, 1, 1],
            "har_monthly": [1, 1, 1, 1, 1, 1],
            "log_return": [0, 0, 0, 0, 0, 0],
            "logGKVol_lag_1": [1, 1, 1, 1, 1, 1],
            "target": [1, 2, 1, 2, 1, 2],
            "has_text_macro": [1, 0, 1, 0, 1, 0],
            "has_text_sector": [0, 1, 0, 1, 0, 1],
            "has_text_related": [0, 0, 0, 0, 0, 0],
            "has_text_target": [1, 1, 1, 1, 1, 1],
            "text_length_macro": [10, 0, 10, 0, 10, 0],
            "text_length_sector": [0, 8, 0, 8, 0, 8],
            "text_length_related": [0, 0, 0, 0, 0, 0],
            "text_length_target": [5, 5, 5, 5, 5, 5],
            "total_text_length": [15, 13, 15, 13, 15, 13],
            "num_text_levels_present": [2, 2, 2, 2, 2, 2],
        }
    )
    train = df.iloc[:4]
    test = df.iloc[4:]
    specs = [TargetSpec("log_gk", 1, "target")]
    result = placebo_tests(df, train, test, specs, "ticker", "date", ["har_daily", "har_weekly", "har_monthly", "log_return", "logGKVol_lag_1"], None, tmp_path)
    assert set(result["placebo_variant"]) == {"correct_text", "no_text", "shuffled_text", "cross_ticker_text", "stale_text"}
