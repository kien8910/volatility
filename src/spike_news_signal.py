"""Focused tests for FinTexTS news signal on volatility spike days.

This module is intentionally separate from the full detailed analysis pipeline. It
keeps the test light-weight by using time-series features plus explicit news
metadata/keyword features, then evaluates only spike-day behavior.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_EMBEDDING_MODEL, RANDOM_SEED
from .detailed_news_signal import TargetSpec, split_by_time
from .features import add_detailed_volatility_targets, add_volatility_features, target_column_for, time_series_feature_columns
from .load_data import load_fintexts
from .models import make_ridge_model
from .text_features import add_event_keyword_features, add_text_presence_flags, detect_text_columns, joined_text_for_configuration, write_text_schema_validation


NEWS_LEVELS = ["macro", "sector", "related", "target"]
SPIKE_PERCENTILES = [85, 90, 95, 97.5]


@dataclass(frozen=True)
class SpikeRunConfig:
    num_tickers: int
    forecast_horizons: list[int]
    targets: list[str]
    max_lag: int
    output_dir: Path
    spike_weight: float = 5.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run focused FinTexTS news signal tests on volatility spike days.")
    parser.add_argument("--num_tickers", type=int, default=5)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3, 5])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "total_volatility", "log_abs_return"])
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--output_dir", default="outputs/spike_signal_tests")
    parser.add_argument("--spike_weight", type=float, default=5.0)
    return parser.parse_args()


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ticker_col: str, target_col: str, sample_weight: np.ndarray | None = None) -> np.ndarray:
    model = make_ridge_model(features, [ticker_col])
    fit_kwargs = {"model__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(train[features + [ticker_col]], train[target_col], **fit_kwargs)
    return model.predict(test[features + [ticker_col]])


def _classification_scores(y_true: pd.Series, score: pd.Series) -> dict[str, float]:
    if y_true.nunique() < 2:
        return {"precision": np.nan, "recall": np.nan, "f1": np.nan, "roc_auc": np.nan, "pr_auc": np.nan}
    pred = score >= 0.5
    out = {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": np.nan,
        "pr_auc": np.nan,
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, score))
        out["pr_auc"] = float(average_precision_score(y_true, score))
    except Exception:
        pass
    return out


def _news_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for level in NEWS_LEVELS:
        cols.extend([f"has_text_{level}", f"text_length_{level}", f"{level}_text_count"])
    cols.extend(
        [
            "has_any_news_text",
            "news_text_count_total",
            "news_text_length_total",
            "num_text_levels_present",
            "event_keyword_count",
            "target_event_keyword_count",
            "has_event_keyword",
            "has_target_event_keyword",
            "filing_context_changed",
        ]
    )
    return [col for col in cols if col in df.columns]


def _variant_frame(df: pd.DataFrame, news_features: list[str], ticker_col: str, date_col: str, variant: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    frame = df[news_features].copy()
    if variant == "correct_news":
        return frame
    if variant == "no_news":
        return pd.DataFrame(0, index=df.index, columns=news_features)
    if variant == "shuffled_within_ticker":
        out = frame.copy()
        for _, idx in df.groupby(ticker_col, sort=False).groups.items():
            idx_list = list(idx)
            out.loc[idx_list, news_features] = frame.loc[idx_list, news_features].iloc[rng.permutation(len(idx_list))].to_numpy()
        return out
    if variant == "cross_ticker_same_day":
        out = frame.copy()
        for _, idx in df.groupby(date_col, sort=False).groups.items():
            idx_list = list(idx)
            if len(idx_list) <= 1:
                continue
            perm = rng.permutation(len(idx_list))
            if np.array_equal(perm, np.arange(len(idx_list))):
                perm = np.roll(perm, 1)
            out.loc[idx_list, news_features] = frame.loc[idx_list, news_features].iloc[perm].to_numpy()
        return out
    if variant == "stale_20d":
        return df.groupby(ticker_col, sort=False)[news_features].shift(20).fillna(0)
    raise ValueError(f"Unknown news variant: {variant}")


def _alignment_frame(df: pd.DataFrame, news_features: list[str], ticker_col: str, mode: str) -> pd.DataFrame:
    if mode == "same_day_text":
        return df[news_features].copy()
    if mode == "shifted_1_day_text":
        return df.groupby(ticker_col, sort=False)[news_features].shift(1).fillna(0)
    if mode == "shifted_2_day_text":
        return df.groupby(ticker_col, sort=False)[news_features].shift(2).fillna(0)
    if mode == "window_3d_text":
        return df.groupby(ticker_col, sort=False)[news_features].rolling(3, min_periods=1).max().reset_index(level=0, drop=True)
    if mode == "window_5d_text":
        return df.groupby(ticker_col, sort=False)[news_features].rolling(5, min_periods=1).max().reset_index(level=0, drop=True)
    raise ValueError(f"Unknown alignment mode: {mode}")


def prepare_data(config: SpikeRunConfig):
    raw, column_map = load_fintexts(num_tickers=config.num_tickers)
    df = add_volatility_features(raw, column_map, forecast_horizon=1, max_lag=config.max_lag)
    df = add_detailed_volatility_targets(df, column_map, config.forecast_horizons)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    write_text_schema_validation(df, config.output_dir)
    text_columns = detect_text_columns(df)
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    ts_features = time_series_feature_columns(config.max_lag)
    df = df.dropna(subset=ts_features + ["logGKVol"]).copy().reset_index(drop=False).rename(columns={"index": "row_id"})
    df = df.reset_index(drop=True)
    train, val, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train, val], ignore_index=False)
    specs = [TargetSpec(target, horizon, target_column_for(target, horizon)) for target in config.targets for horizon in config.forecast_horizons]
    return df, train_full, test, specs, column_map, ts_features, text_columns


def spike_threshold_sensitivity(train: pd.DataFrame, test: pd.DataFrame, specs: list[TargetSpec], ticker_col: str) -> tuple[pd.DataFrame, dict[tuple[str, int, float], pd.DataFrame]]:
    rows = []
    thresholds_by_spec: dict[tuple[str, int, float], pd.DataFrame] = {}
    for spec in specs:
        for pct in SPIKE_PERCENTILES:
            thresholds = train.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            thresholds_by_spec[(spec.target_name, spec.horizon, pct)] = thresholds
            enriched = test[[ticker_col, spec.column]].merge(thresholds, left_on=ticker_col, right_index=True, how="left")
            enriched["is_spike"] = enriched[spec.column] > enriched["threshold"]
            by_ticker = enriched.groupby(ticker_col)["is_spike"].agg(["sum", "count", "mean"]).reset_index()
            for _, row in by_ticker.iterrows():
                rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "ticker": row[ticker_col],
                        "spike_count": int(row["sum"]),
                        "test_count": int(row["count"]),
                        "spike_rate": float(row["mean"]),
                    }
                )
    return pd.DataFrame(rows), thresholds_by_spec


def run_spike_regression_tests(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    date_col: str,
    ts_features: list[str],
    news_features: list[str],
) -> pd.DataFrame:
    rows = []
    variants = ["correct_news", "no_news", "shuffled_within_ticker", "cross_ticker_same_day", "stale_20d"]
    for variant in variants:
        news_frame = _variant_frame(df, news_features, ticker_col, date_col, variant)
        work = pd.concat([df.drop(columns=news_features, errors="ignore"), news_frame], axis=1)
        features = ts_features + news_features
        for spec in specs:
            tr = work.loc[train.index].dropna(subset=features + [spec.column])
            te = work.loc[test.index].dropna(subset=features + [spec.column])
            if tr.empty or te.empty:
                continue
            ts_pred = _fit_predict(tr, te, ts_features, ticker_col, spec.column)
            news_pred = _fit_predict(tr, te, features, ticker_col, spec.column)
            for pct in SPIKE_PERCENTILES:
                thresholds = tr.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
                scored = te[[ticker_col, spec.column]].copy()
                scored["ts_pred"] = ts_pred
                scored["news_pred"] = news_pred
                scored = scored.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                scored["is_spike"] = scored[spec.column] > scored["threshold"]
                for segment, mask in {"all": pd.Series(True, index=scored.index), "spike": scored["is_spike"], "non_spike": ~scored["is_spike"]}.items():
                    subset = scored[mask]
                    if subset.empty:
                        continue
                    ts_abs = (subset["ts_pred"] - subset[spec.column]).abs()
                    news_abs = (subset["news_pred"] - subset[spec.column]).abs()
                    rows.append(
                        {
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "news_variant": variant,
                            "segment": segment,
                            "ts_only_mae": float(ts_abs.mean()),
                            "news_mae": float(news_abs.mean()),
                            "mae_improvement_pct": (float(ts_abs.mean()) - float(news_abs.mean())) / float(ts_abs.mean()) * 100 if float(ts_abs.mean()) else np.nan,
                            "ts_underprediction": float((subset[spec.column] - subset["ts_pred"]).clip(lower=0).mean()),
                            "news_underprediction": float((subset[spec.column] - subset["news_pred"]).clip(lower=0).mean()),
                            "num_samples": int(len(subset)),
                        }
                    )
    return pd.DataFrame(rows)


def run_spike_classifier_tests(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    ts_features: list[str],
    news_features: list[str],
) -> pd.DataFrame:
    rows = []
    features_by_model = {"ts_only": ts_features, "ts_plus_news_meta": ts_features + news_features}
    for spec in specs:
        for pct in SPIKE_PERCENTILES:
            thresholds = train.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            for model_name, features in features_by_model.items():
                tr = train.dropna(subset=features + [spec.column]).merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                te = test.dropna(subset=features + [spec.column]).merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                tr["is_spike"] = (tr[spec.column] > tr["threshold"]).astype(int)
                te["is_spike"] = (te[spec.column] > te["threshold"]).astype(int)
                if tr["is_spike"].nunique() < 2 or te["is_spike"].nunique() < 2:
                    continue
                model = Pipeline(
                    [
                        ("scale", StandardScaler()),
                        (
                            "logit",
                            LogisticRegression(
                                max_iter=5000,
                                class_weight="balanced",
                                random_state=RANDOM_SEED,
                                solver="liblinear",
                            ),
                        ),
                    ]
                )
                # Keep ticker out of the classifier to avoid sparse one-hot plumbing; this is a lightweight signal test.
                model.fit(tr[features], tr["is_spike"])
                score = pd.Series(model.predict_proba(te[features])[:, 1], index=te.index)
                metrics = _classification_scores(te["is_spike"], score)
                rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "model": model_name,
                        "num_train_spikes": int(tr["is_spike"].sum()),
                        "num_test_spikes": int(te["is_spike"].sum()),
                        "num_test_samples": int(len(te)),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def run_weighted_spike_regression(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    ts_features: list[str],
    news_features: list[str],
    spike_weight: float,
) -> pd.DataFrame:
    rows = []
    features_by_model = {"ts_only": ts_features, "ts_plus_news_meta": ts_features + news_features}
    for spec in specs:
        for pct in SPIKE_PERCENTILES:
            thresholds = train.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            for model_name, features in features_by_model.items():
                tr = train.dropna(subset=features + [spec.column]).merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                te = test.dropna(subset=features + [spec.column]).merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                tr["is_spike"] = tr[spec.column] > tr["threshold"]
                te["is_spike"] = te[spec.column] > te["threshold"]
                weights = np.where(tr["is_spike"], spike_weight, 1.0)
                pred = _fit_predict(tr, te, features, ticker_col, spec.column, sample_weight=weights)
                te = te.copy()
                te["prediction"] = pred
                for segment, mask in {"all": pd.Series(True, index=te.index), "spike": te["is_spike"], "non_spike": ~te["is_spike"]}.items():
                    subset = te[mask]
                    if subset.empty:
                        continue
                    rows.append(
                        {
                            "target_name": spec.target_name,
                            "horizon": spec.horizon,
                            "spike_percentile": pct,
                            "model": model_name,
                            "segment": segment,
                            "weighted_spike_factor": spike_weight,
                            "mae": float(mean_absolute_error(subset[spec.column], subset["prediction"])),
                            "underprediction": float((subset[spec.column] - subset["prediction"]).clip(lower=0).mean()),
                            "num_samples": int(len(subset)),
                        }
                    )
    return pd.DataFrame(rows)


def run_alignment_window_tests(
    df: pd.DataFrame,
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    ts_features: list[str],
    news_features: list[str],
) -> pd.DataFrame:
    rows = []
    modes = ["same_day_text", "shifted_1_day_text", "shifted_2_day_text", "window_3d_text", "window_5d_text"]
    for mode in modes:
        news_frame = _alignment_frame(df, news_features, ticker_col, mode)
        work = pd.concat([df.drop(columns=news_features, errors="ignore"), news_frame], axis=1)
        features = ts_features + news_features
        for spec in specs:
            tr = work.loc[train.index].dropna(subset=features + [spec.column])
            te = work.loc[test.index].dropna(subset=features + [spec.column])
            if tr.empty or te.empty:
                continue
            pred = _fit_predict(tr, te, features, ticker_col, spec.column)
            for pct in SPIKE_PERCENTILES:
                thresholds = tr.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
                scored = te[[ticker_col, spec.column]].copy()
                scored["prediction"] = pred
                scored = scored.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                scored["is_spike"] = scored[spec.column] > scored["threshold"]
                spike = scored[scored["is_spike"]]
                rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "alignment_mode": mode,
                        "all_mae": float(mean_absolute_error(scored[spec.column], scored["prediction"])),
                        "spike_mae": float(mean_absolute_error(spike[spec.column], spike["prediction"])) if len(spike) else np.nan,
                        "spike_underprediction": float((spike[spec.column] - spike["prediction"]).clip(lower=0).mean()) if len(spike) else np.nan,
                        "num_spike_samples": int(len(spike)),
                    }
                )
    return pd.DataFrame(rows)


def spike_event_attribution(
    train: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    date_col: str,
    ts_features: list[str],
    news_features: list[str],
    text_columns: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    features = ts_features + news_features
    for spec in specs:
        tr = train.dropna(subset=features + [spec.column])
        te = test.dropna(subset=features + [spec.column])
        if tr.empty or te.empty:
            continue
        ts_pred = _fit_predict(tr, te, ts_features, ticker_col, spec.column)
        news_pred = _fit_predict(tr, te, features, ticker_col, spec.column)
        for pct in SPIKE_PERCENTILES:
            thresholds = tr.groupby(ticker_col)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
            scored = te[[ticker_col, date_col, "row_id", spec.column] + news_features].copy()
            scored["ts_only_prediction"] = ts_pred
            scored["news_prediction"] = news_pred
            scored = scored.merge(thresholds, left_on=ticker_col, right_index=True, how="left")
            scored["is_spike"] = scored[spec.column] > scored["threshold"]
            spikes = scored[scored["is_spike"]].copy()
            if spikes.empty:
                continue
            for _, row in spikes.iterrows():
                keyword_cols = [col for col in row.index if col.startswith("event_keyword_") and bool(row[col])]
                rows.append(
                    {
                        "ticker": row[ticker_col],
                        "date": row[date_col],
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_percentile": pct,
                        "actual_target": row[spec.column],
                        "threshold": row["threshold"],
                        "ts_only_prediction": row["ts_only_prediction"],
                        "news_prediction": row["news_prediction"],
                        "underprediction_reduction": max(row[spec.column] - row["ts_only_prediction"], 0) - max(row[spec.column] - row["news_prediction"], 0),
                        "has_macro_text": int(row.get("has_text_macro", 0)),
                        "has_sector_text": int(row.get("has_text_sector", 0)),
                        "has_related_text": int(row.get("has_text_related", 0)),
                        "has_target_text": int(row.get("has_text_target", 0)),
                        "has_any_news_text": int(row.get("has_any_news_text", 0)),
                        "event_keyword_count": int(row.get("event_keyword_count", 0)),
                        "target_event_keyword_count": int(row.get("target_event_keyword_count", 0)),
                        "top_keyword_group": ",".join(keyword_cols) if keyword_cols else "",
                        "target_text_preview": joined_text_for_configuration(test.loc[[row.name]], text_columns, "target_only").iloc[0][:240],
                    }
                )
    return pd.DataFrame(rows)


def write_summary(output_dir: Path, regression: pd.DataFrame, classifier: pd.DataFrame, alignment: pd.DataFrame, attribution: pd.DataFrame) -> None:
    lines = ["# Spike News Signal Test Summary", ""]
    if not regression.empty:
        spike = regression[(regression["segment"] == "spike") & (regression["news_variant"] == "correct_news")]
        best = spike.sort_values("mae_improvement_pct", ascending=False).head(8)
        lines.extend(["## Best Correct-News Spike MAE Improvements", ""])
        for _, row in best.iterrows():
            lines.append(f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']}: {row['mae_improvement_pct']:.2f}% on {int(row['num_samples'])} spike samples")
    if not classifier.empty:
        pivot = classifier.pivot_table(index=["target_name", "horizon", "spike_percentile"], columns="model", values="pr_auc").reset_index()
        if {"ts_only", "ts_plus_news_meta"}.issubset(pivot.columns):
            pivot["pr_auc_delta"] = pivot["ts_plus_news_meta"] - pivot["ts_only"]
            best_cls = pivot.sort_values("pr_auc_delta", ascending=False).head(5)
            lines.extend(["", "## Best PR-AUC Deltas", ""])
            for _, row in best_cls.iterrows():
                lines.append(f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']}: delta={row['pr_auc_delta']:.4f}")
    if not alignment.empty:
        best_align = alignment.sort_values("spike_mae").groupby(["target_name", "horizon", "spike_percentile"]).head(1).head(10)
        lines.extend(["", "## Best Alignment Modes By Spike MAE", ""])
        for _, row in best_align.iterrows():
            lines.append(f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']}: {row['alignment_mode']} spike_mae={row['spike_mae']:.4f}")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Treat this as predictive association, not causality.",
            "- Correct-news variants should beat shuffled/cross-ticker/stale placebo on spike samples before claiming news-content signal.",
            "- Shifted/window alignment should remain competitive to reduce same-day timestamp leakage concerns.",
            "- Spike sample counts are small at high percentiles; prefer PR-AUC and effect sizes over p-values alone.",
        ]
    )
    if not attribution.empty:
        lines.append(f"- Spike event attribution rows generated: {len(attribution)}.")
    (output_dir / "spike_news_signal_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_spike_news_signal_tests(args: argparse.Namespace | SpikeRunConfig) -> None:
    config = SpikeRunConfig(
        num_tickers=args.num_tickers,
        forecast_horizons=list(args.forecast_horizons),
        targets=list(args.targets),
        max_lag=args.max_lag,
        output_dir=Path(args.output_dir),
        spike_weight=getattr(args, "spike_weight", 5.0),
    )
    df, train, test, specs, column_map, ts_features, text_columns = prepare_data(config)
    news_features = _news_feature_columns(df)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    threshold_df, _ = spike_threshold_sensitivity(train, test, specs, column_map.ticker)
    threshold_df.to_csv(config.output_dir / "spike_threshold_sensitivity.csv", index=False)

    regression = run_spike_regression_tests(df, train, test, specs, column_map.ticker, column_map.date, ts_features, news_features)
    regression.to_csv(config.output_dir / "spike_news_placebo_regression.csv", index=False)

    classifier = run_spike_classifier_tests(df, train, test, specs, column_map.ticker, ts_features, news_features)
    classifier.to_csv(config.output_dir / "spike_news_classifier_metrics.csv", index=False)

    weighted = run_weighted_spike_regression(train, test, specs, column_map.ticker, ts_features, news_features, config.spike_weight)
    weighted.to_csv(config.output_dir / "spike_weighted_regression.csv", index=False)

    alignment = run_alignment_window_tests(df, train, test, specs, column_map.ticker, ts_features, news_features)
    alignment.to_csv(config.output_dir / "spike_news_alignment_windows.csv", index=False)

    attribution = spike_event_attribution(train, test, specs, column_map.ticker, column_map.date, ts_features, news_features, text_columns)
    attribution.to_csv(config.output_dir / "spike_day_event_attribution.csv", index=False)

    write_summary(config.output_dir, regression, classifier, alignment, attribution)
    print(f"Spike news signal test outputs written to: {config.output_dir.resolve()}")


def main() -> None:
    run_spike_news_signal_tests(parse_args())


if __name__ == "__main__":
    main()
