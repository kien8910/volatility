"""Model definitions for the volatility pilot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class FittedModel:
    name: str
    estimator: object | None
    prediction_column: str


def make_tree_regressor():
    try:
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=42,
            n_jobs=-1,
        )
    except Exception:
        try:
            return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
        except Exception:
            return RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)


def make_ridge_model(numeric_features: list[str], categorical_features: list[str] | None = None) -> Pipeline:
    categorical_features = categorical_features or []
    transformer = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", transformer), ("model", Ridge(alpha=1.0))])


def make_one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_supervised_model(feature_columns: list[str], ticker_column: str | None = None):
    categorical = [ticker_column] if ticker_column else []
    transformer = ColumnTransformer(
        transformers=[
            ("num", "passthrough", feature_columns),
            ("cat", make_one_hot_encoder(), categorical),
        ],
        remainder="drop",
    )
    return Pipeline([("prep", transformer), ("model", make_tree_regressor())])


def fit_and_predict(name: str, estimator, train_df, test_df, feature_columns: list[str], target_col: str) -> np.ndarray:
    estimator.fit(train_df[feature_columns], train_df[target_col])
    return estimator.predict(test_df[feature_columns])
