"""Temporal volatility/return models with gated text integration.

This standalone experiment compares strong time-series baselines against text
integration schemes that only adjust predictions on event-gated days. It is
designed for a quick check of whether target/sector text can improve spike-day
volatility forecasts without harming ordinary non-spike days too much.

Example:

    python -m src.spike_temporal_text_experiment \
      --num_tickers 25 \
      --tickers C BAC AMGN AMD AXP \
      --look_back 22 \
      --targets log_gk log_abs_return \
      --forecast_horizons 1 3 \
      --spike_percentiles 80 85 90 95 \
      --embedding_variants target_sector_weighted target_sector_event_weighted \
      --alignment_modes shifted_1_day_text same_day_text \
      --output_dir outputs/spike_temporal_text_experiment
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, RANDOM_SEED
from .detailed_news_signal import split_by_time
from .models import make_ridge_model
from .spike_controlled_embedding import build_controlled_embedding
from .spike_news_embedding_signal import prepare_data, train_only_pca
from .text_features import add_event_keyword_features, add_text_presence_flags, normalize_text_value


DEFAULT_TEMPORAL_MODELS = [
    "Naive",
    "HAR_Ridge",
    "LookbackGK_Ridge",
    "LookbackGKReturn_Ridge",
    "TemporalStats_Ridge",
    "TemporalStats_ElasticNet",
    "TemporalStats_HGBR",
]

DEFAULT_TEXT_MODELS = [
    "LookbackGKReturn_TextEarlyFusion",
    "TemporalStats_TextEarlyFusion",
    "HAR_TextResidual_Gated",
    "LookbackGKReturn_TextResidual_Gated",
    "TemporalStats_TextResidual_Gated",
]

DEFAULT_EMBEDDING_VARIANTS = ["target_sector_weighted", "target_sector_event_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run temporal volatility/return plus gated text experiments.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=["C", "BAC", "AMGN", "AMD", "AXP"])
    parser.add_argument("--look_back", type=int, default=22)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "log_abs_return"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--temporal_models", nargs="+", default=DEFAULT_TEMPORAL_MODELS)
    parser.add_argument("--text_models", nargs="+", default=DEFAULT_TEXT_MODELS)
    parser.add_argument("--embedding_variants", nargs="+", default=DEFAULT_EMBEDDING_VARIANTS)
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text", "same_day_text"])
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--delta_window", type=int, default=20)
    parser.add_argument("--target_weight", type=float, default=2.0)
    parser.add_argument("--sector_weight", type=float, default=1.0)
    parser.add_argument("--max_lag", type=int, default=None)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--gate_type", choices=["target_event_keyword", "any_event_keyword", "always"], default="target_event_keyword")
    parser.add_argument("--output_dir", default="outputs/spike_temporal_text_experiment")
    return parser.parse_args()


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def _make_elastic_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [("num", StandardScaler(), numeric_features), ("cat", _one_hot_encoder(), categorical_features)],
        remainder="drop",
    )
    model = ElasticNet(alpha=0.001, l1_ratio=0.2, max_iter=10000, random_state=RANDOM_SEED)
    return Pipeline([("prep", prep), ("model", model)])


def _make_hgbr_model(numeric_features: list[str], categorical_features: list[str]) -> Pipeline:
    prep = ColumnTransformer(
        [("num", "passthrough", numeric_features), ("cat", _one_hot_encoder(), categorical_features)],
        remainder="drop",
    )
    model = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.1, random_state=RANDOM_SEED)
    return Pipeline([("prep", prep), ("model", model)])


def _fit_predict(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    ticker_col: str,
    target_col: str,
    kind: str,
) -> tuple[pd.Series, pd.Series]:
    if kind == "ridge":
        model = make_ridge_model(features, [ticker_col])
    elif kind == "elastic":
        model = _make_elastic_model(features, [ticker_col])
    elif kind == "hgbr":
        model = _make_hgbr_model(features, [ticker_col])
    else:
        raise ValueError(f"Unknown model kind: {kind}")
    model.fit(train[features + [ticker_col]], train[target_col])
    train_pred = pd.Series(model.predict(train[features + [ticker_col]]), index=train.index)
    test_pred = pd.Series(model.predict(test[features + [ticker_col]]), index=test.index)
    return train_pred, test_pred


def _add_sequence_windows(df: pd.DataFrame, ticker_col: str, date_col: str, look_back: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.sort_values([ticker_col, date_col]).copy()
    grouped = work.groupby(ticker_col, sort=False)
    gk_cols: list[str] = []
    ret_cols: list[str] = []
    for offset in range(look_back):
        gk_col = f"seq_logGKVol_t_minus_{offset}"
        ret_col = f"seq_log_return_t_minus_{offset}"
        work[gk_col] = grouped["logGKVol"].shift(offset)
        work[ret_col] = grouped["log_return"].shift(offset)
        gk_cols.append(gk_col)
        ret_cols.append(ret_col)
    work["sequence_complete"] = work[gk_cols + ret_cols].notna().all(axis=1)
    return work.sort_index(), gk_cols, ret_cols


def _row_slope(values: np.ndarray) -> np.ndarray:
    x = np.arange(values.shape[1], dtype=float)
    x = x - x.mean()
    denom = float(np.square(x).sum())
    centered = values - values.mean(axis=1, keepdims=True)
    return centered @ x / denom


def _add_temporal_stats(df: pd.DataFrame, gk_cols: list[str], ret_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    stats_cols: list[str] = []
    gk_ordered = list(reversed(gk_cols))
    ret_ordered = list(reversed(ret_cols))
    gk = out[gk_ordered].to_numpy(dtype=float)
    ret = out[ret_ordered].to_numpy(dtype=float)

    for window in [3, 5, 10, 22]:
        if window > len(gk_ordered):
            continue
        g = gk[:, -window:]
        r = ret[:, -window:]
        created = {
            f"gk_mean_{window}": np.mean(g, axis=1),
            f"gk_std_{window}": np.std(g, axis=1),
            f"gk_min_{window}": np.min(g, axis=1),
            f"gk_max_{window}": np.max(g, axis=1),
            f"gk_slope_{window}": _row_slope(g),
            f"gk_last_minus_mean_{window}": g[:, -1] - np.mean(g, axis=1),
            f"ret_mean_{window}": np.mean(r, axis=1),
            f"ret_std_{window}": np.std(r, axis=1),
            f"abs_ret_sum_{window}": np.abs(r).sum(axis=1),
            f"neg_ret_sum_{window}": np.minimum(r, 0.0).sum(axis=1),
            f"pos_ret_sum_{window}": np.maximum(r, 0.0).sum(axis=1),
            f"max_abs_ret_{window}": np.abs(r).max(axis=1),
        }
        for name, values in created.items():
            out[name] = values
            stats_cols.append(name)
    out["ret_gk_interaction_5"] = out["abs_ret_sum_5"] * out["gk_mean_5"]
    out["ret_gk_interaction_22"] = out["abs_ret_sum_22"] * out["gk_mean_22"]
    stats_cols.extend(["ret_gk_interaction_5", "ret_gk_interaction_22"])
    return out, stats_cols


def _gate_mask(df: pd.DataFrame, gate_type: str) -> pd.Series:
    if gate_type == "always":
        return pd.Series(True, index=df.index)
    if gate_type == "any_event_keyword":
        return df["event_keyword_count"].fillna(0).gt(0)
    if gate_type == "target_event_keyword":
        return df["target_event_keyword_count"].fillna(0).gt(0)
    raise ValueError(f"Unknown gate_type: {gate_type}")


def _prediction_metrics(actual: pd.Series, pred: pd.Series) -> dict[str, float]:
    mae = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    under = float((actual - pred).clip(lower=0).mean())
    return {"mae": mae, "rmse": rmse, "underprediction_loss": under}


def _evaluate_predictions(
    scored: pd.DataFrame,
    prediction_cols: dict[str, pd.Series],
    target_col: str,
    har_pred: pd.Series | None,
    ticker_col: str,
    target_name: str,
    horizon: int,
    spike_percentile: float,
    extra: dict,
) -> list[dict]:
    rows: list[dict] = []
    segments = {
        "all": pd.Series(True, index=scored.index),
        "non_spike": ~scored["is_spike"],
        "spike": scored["is_spike"],
        "event_spike": scored["is_spike"] & scored["gate_active"],
    }
    for model_name, pred in prediction_cols.items():
        for segment, mask in segments.items():
            subset = scored[mask]
            if subset.empty:
                continue
            for ticker, group in subset.groupby(ticker_col, sort=False):
                idx = group.index
                model_pred = pred.loc[idx]
                valid = group[target_col].notna() & model_pred.notna()
                if not valid.any():
                    continue
                group = group.loc[valid]
                idx = group.index
                model_pred = model_pred.loc[idx]
                metrics = _prediction_metrics(group[target_col], model_pred)
                har_mae = np.nan
                if har_pred is not None:
                    har_values = har_pred.loc[idx]
                    har_valid = har_values.notna() & group[target_col].notna()
                    if har_valid.any():
                        har_mae = float(mean_absolute_error(group.loc[har_valid, target_col], har_values.loc[har_valid]))
                rows.append(
                    {
                        "ticker": ticker,
                        "target_name": target_name,
                        "horizon": horizon,
                        "spike_percentile": spike_percentile,
                        "segment": segment,
                        "model": model_name,
                        "num_samples": int(len(group)),
                        "num_spikes": int(group["is_spike"].sum()),
                        "gate_active_rate": float(group["gate_active"].mean()),
                        **extra,
                        **metrics,
                        "har_mae": har_mae,
                        "mae_improvement_vs_har_pct": (har_mae - metrics["mae"]) / har_mae * 100 if har_mae else np.nan,
                    }
                )
    return rows


def _fit_text_residual(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_train_pred: pd.Series,
    base_test_pred: pd.Series,
    target_col: str,
    text_features: list[str],
    ticker_col: str,
    gate_type: str,
) -> pd.Series:
    event_features = [
        col
        for col in [
            "event_keyword_count",
            "target_event_keyword_count",
            "news_text_count_total",
            "news_text_length_total",
            "has_target_text",
            "has_sector_text",
        ]
        if col in train.columns
    ]
    features = text_features + event_features
    tr = train.dropna(subset=features + [target_col]).copy()
    te = test.dropna(subset=features + [target_col]).copy()
    train_gate = _gate_mask(tr, gate_type)
    if train_gate.sum() >= 20:
        tr_fit = tr[train_gate].copy()
    else:
        tr_fit = tr
    tr_fit = tr_fit[base_train_pred.loc[tr_fit.index].notna()].copy()
    if tr_fit.empty:
        return base_test_pred.copy()
    residual = tr_fit[target_col] - base_train_pred.loc[tr_fit.index]
    model = make_ridge_model(features, [ticker_col])
    model.fit(tr_fit[features + [ticker_col]], residual)
    correction = pd.Series(0.0, index=test.index)
    if not te.empty:
        raw = pd.Series(model.predict(te[features + [ticker_col]]), index=te.index)
        gate = _gate_mask(te, gate_type)
        correction.loc[te.index] = raw.where(gate, 0.0)
    return base_test_pred + correction


def _fit_early_fusion(
    train: pd.DataFrame,
    test: pd.DataFrame,
    base_features: list[str],
    text_features: list[str],
    ticker_col: str,
    target_col: str,
) -> pd.Series:
    features = base_features + text_features
    tr = train.dropna(subset=features + [target_col]).copy()
    te = test.dropna(subset=features + [target_col]).copy()
    pred = pd.Series(np.nan, index=test.index)
    if tr.empty or te.empty:
        return pred
    _, te_pred = _fit_predict(tr, te, features, ticker_col, target_col, "ridge")
    pred.loc[te.index] = te_pred
    return pred


def _event_preview(df: pd.DataFrame, text_columns: dict[str, list[str]], idx: int) -> str:
    parts: list[str] = []
    for group in ["target", "sector"]:
        for column in text_columns.get(group, []):
            if column in df.columns:
                text = normalize_text_value(df.loc[idx, column])
                if text:
                    parts.append(text.replace("\n", " ")[:160])
    return " | ".join(parts)[:320]


def _write_summary(output_dir: Path, metrics: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = ["# Temporal Text Spike Experiment Summary", ""]
    lines.append(f"- look_back: {args.look_back}")
    lines.append(f"- gate_type: {args.gate_type}")
    if metrics.empty:
        lines.append("- No metrics were produced.")
    else:
        spike = metrics[metrics["segment"].eq("spike")]
        all_days = metrics[metrics["segment"].eq("all")]
        lines.extend(["", "## Best Spike Models vs HAR", ""])
        best = (
            spike.groupby("model")["mae_improvement_vs_har_pct"]
            .agg(["count", "mean", "median"])
            .sort_values("median", ascending=False)
            .head(12)
        )
        for model, row in best.iterrows():
            lines.append(f"- {model}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
        lines.extend(["", "## All-Day Guardrail vs HAR", ""])
        guard = (
            all_days.groupby("model")["mae_improvement_vs_har_pct"]
            .agg(["count", "mean", "median"])
            .sort_values("median", ascending=False)
            .head(12)
        )
        for model, row in guard.iterrows():
            lines.append(f"- {model}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- Spike thresholds are computed on train rows per ticker.",
            "- Text residual correction is applied only when the configured gate is active.",
            "- PCA is fit on train rows only before transforming test rows.",
            "- same_day_text should be treated as an upper-bound unless intraday timestamps confirm availability before forecast cutoff.",
        ]
    )
    (output_dir / "temporal_text_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.max_lag = args.max_lag or args.look_back
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, _, _, specs, column_map, _, text_columns = prepare_data(args)
    if args.tickers:
        df = df[df[column_map.ticker].isin(set(args.tickers))].copy()
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    df, gk_cols, ret_cols = _add_sequence_windows(df, column_map.ticker, column_map.date, args.look_back)
    df = df[df["sequence_complete"]].copy()
    df, temporal_stats = _add_temporal_stats(df, gk_cols, ret_cols)
    df = df.reset_index(drop=True)

    train_part, val_part, test = split_by_time(df, column_map.ticker, column_map.date)
    train = pd.concat([train_part, val_part], ignore_index=False)
    args._ticker_col = column_map.ticker
    args._date_col = column_map.date

    feature_sets = {
        "Naive": [],
        "HAR_Ridge": ["har_daily", "har_weekly", "har_monthly"],
        "LookbackGK_Ridge": gk_cols,
        "LookbackGKReturn_Ridge": gk_cols + ret_cols,
        "TemporalStats_Ridge": temporal_stats,
        "TemporalStats_ElasticNet": temporal_stats,
        "TemporalStats_HGBR": temporal_stats,
    }
    model_kinds = {
        "HAR_Ridge": "ridge",
        "LookbackGK_Ridge": "ridge",
        "LookbackGKReturn_Ridge": "ridge",
        "TemporalStats_Ridge": "ridge",
        "TemporalStats_ElasticNet": "elastic",
        "TemporalStats_HGBR": "hgbr",
    }
    text_base_features = {
        "HAR_TextResidual_Gated": "HAR_Ridge",
        "LookbackGKReturn_TextResidual_Gated": "LookbackGKReturn_Ridge",
        "TemporalStats_TextResidual_Gated": "TemporalStats_Ridge",
        "LookbackGKReturn_TextEarlyFusion": "LookbackGKReturn_Ridge",
        "TemporalStats_TextEarlyFusion": "TemporalStats_Ridge",
    }

    metric_rows: list[dict] = []
    event_rows: list[dict] = []
    for embedding_variant in args.embedding_variants:
        for alignment_mode in args.alignment_modes:
            raw = build_controlled_embedding(df, text_columns, embedding_variant, alignment_mode, "correct_text", args)
            emb_df, emb_cols = train_only_pca(raw, train.index, args.pca_dim, f"{embedding_variant}_{alignment_mode}")
            work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
            train_work = work.loc[train.index]
            test_work = work.loc[test.index]

            for spec in specs:
                base_train_pred: dict[str, pd.Series] = {}
                base_test_pred: dict[str, pd.Series] = {}
                for model_name in args.temporal_models:
                    if model_name == "Naive":
                        base_train_pred[model_name] = train_work["logGKVol"].copy()
                        base_test_pred[model_name] = test_work["logGKVol"].copy()
                        continue
                    features = feature_sets.get(model_name)
                    if not features:
                        continue
                    tr = train_work.dropna(subset=features + [spec.column]).copy()
                    te = test_work.dropna(subset=features + [spec.column]).copy()
                    if tr.empty or te.empty:
                        continue
                    train_pred, test_pred = _fit_predict(tr, te, features, column_map.ticker, spec.column, model_kinds[model_name])
                    base_train_pred[model_name] = train_pred.reindex(train_work.index)
                    base_test_pred[model_name] = test_pred.reindex(test_work.index)

                text_test_pred: dict[str, pd.Series] = {}
                for text_model in args.text_models:
                    base_name = text_base_features.get(text_model)
                    if base_name not in base_train_pred or base_name not in base_test_pred:
                        continue
                    if text_model.endswith("TextEarlyFusion"):
                        base_features = feature_sets[base_name]
                        text_test_pred[text_model] = _fit_early_fusion(
                            train_work,
                            test_work,
                            base_features,
                            emb_cols,
                            column_map.ticker,
                            spec.column,
                        )
                    else:
                        text_test_pred[text_model] = _fit_text_residual(
                            train_work,
                            test_work,
                            base_train_pred[base_name],
                            base_test_pred[base_name],
                            spec.column,
                            emb_cols,
                            column_map.ticker,
                            args.gate_type,
                        )

                usable_pred = {**base_test_pred, **text_test_pred}
                har_pred = base_test_pred.get("HAR_Ridge")
                for pct in args.spike_percentiles:
                    tr_for_threshold = train_work.dropna(subset=[spec.column])
                    thresholds = tr_for_threshold.groupby(column_map.ticker)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
                    scored = test_work[[column_map.ticker, column_map.date, spec.column, "target_event_keyword_count", "event_keyword_count"]].copy()
                    scored = scored.merge(thresholds, left_on=column_map.ticker, right_index=True, how="left")
                    scored["is_spike"] = scored[spec.column] > scored["threshold"]
                    scored["gate_active"] = _gate_mask(scored, args.gate_type)
                    metric_rows.extend(
                        _evaluate_predictions(
                            scored,
                            usable_pred,
                            spec.column,
                            har_pred,
                            column_map.ticker,
                            spec.target_name,
                            spec.horizon,
                            pct,
                            {"embedding_variant": embedding_variant, "alignment_mode": alignment_mode, "gate_type": args.gate_type},
                        )
                    )
                    for model_name, pred in text_test_pred.items():
                        rows = scored[scored["is_spike"] & scored["gate_active"]].copy()
                        if rows.empty:
                            continue
                        base_name = text_base_features[model_name]
                        base_pred = base_test_pred[base_name]
                        rows["base_prediction"] = base_pred.loc[rows.index]
                        rows["text_prediction"] = pred.loc[rows.index]
                        rows["abs_error_improvement"] = (rows["base_prediction"] - rows[spec.column]).abs() - (rows["text_prediction"] - rows[spec.column]).abs()
                        for idx, row in rows.iterrows():
                            event_rows.append(
                                {
                                    "ticker": row[column_map.ticker],
                                    "date": row[column_map.date],
                                    "target_name": spec.target_name,
                                    "horizon": spec.horizon,
                                    "spike_percentile": pct,
                                    "model": model_name,
                                    "base_model": base_name,
                                    "embedding_variant": embedding_variant,
                                    "alignment_mode": alignment_mode,
                                    "actual_target": row[spec.column],
                                    "threshold": row["threshold"],
                                    "base_prediction": row["base_prediction"],
                                    "text_prediction": row["text_prediction"],
                                    "abs_error_improvement": row["abs_error_improvement"],
                                    "target_event_keyword_count": int(row["target_event_keyword_count"]),
                                    "event_keyword_count": int(row["event_keyword_count"]),
                                    "text_preview": _event_preview(work, text_columns, idx),
                                }
                            )

    metrics = pd.DataFrame(metric_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(output_dir / "temporal_text_metrics.csv", index=False)
    if not events.empty:
        events.to_csv(output_dir / "temporal_text_event_examples.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=False).head(300).to_csv(output_dir / "temporal_text_top_improvements.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=True).head(300).to_csv(output_dir / "temporal_text_top_failures.csv", index=False)
    _write_summary(output_dir, metrics, args)
    print(f"Temporal text experiment outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
