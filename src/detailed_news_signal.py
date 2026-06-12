"""Detailed FinTexTS news signal analysis.

This module keeps the original pilot intact and adds diagnostics for ticker-level,
event-day, horizon, target, ablation, placebo, and spike behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from .config import EVENT_KEYWORDS, RANDOM_SEED
from .evaluate import write_evaluation_outputs
from .features import add_detailed_volatility_targets, add_volatility_features, target_column_for, time_series_feature_columns
from .load_data import ColumnMap, load_fintexts
from .models import make_ridge_model
from .text_features import (
    add_event_keyword_features,
    add_text_presence_flags,
    detect_text_columns,
    encode_text_raw,
    filing_context_diagnostics,
    joined_text_for_configuration,
    keyword_filtered_text,
    normalize_text_value,
    text_duplicate_diagnostics,
    text_columns_for_configuration,
    train_only_pca_features,
    write_text_schema_validation,
)


NEWS_LEVELS = ["macro", "sector", "related", "target"]
FILING_LEVELS = NEWS_LEVELS + ["filing"]
DEFAULT_BASELINE = "TS_only_lags"


@dataclass(frozen=True)
class TargetSpec:
    target_name: str
    horizon: int
    column: str


def split_by_time(df: pd.DataFrame, ticker_col: str, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from .config import TEST_YEAR, TRAIN_YEARS, VAL_YEAR

    years = df[date_col].dt.year
    train = df[(years >= TRAIN_YEARS[0]) & (years <= TRAIN_YEARS[1])]
    val = df[years == VAL_YEAR]
    test = df[years == TEST_YEAR]
    if len(train) and len(val) and len(test):
        return train.copy(), val.copy(), test.copy()

    parts = []
    for _, group in df.sort_values([ticker_col, date_col]).groupby(ticker_col, sort=False):
        n = len(group)
        first = int(n * 0.70)
        second = int(n * 0.85)
        parts.append((group.iloc[:first], group.iloc[first:second], group.iloc[second:]))
    return (
        pd.concat([p[0] for p in parts]).copy(),
        pd.concat([p[1] for p in parts]).copy(),
        pd.concat([p[2] for p in parts]).copy(),
    )


def _safe_name(value: object) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in str(value))


def _row_identity(df: pd.DataFrame, args=None) -> pd.Series:
    preferred = [col for col in ["ticker", "symbol", "date", "datetime", "trading_date"] if col in df.columns]
    if len(preferred) >= 2:
        return df[preferred].astype(str).agg("|".join, axis=1)
    if {"row_id"}.issubset(df.columns):
        return df["row_id"].astype(str)
    return pd.Series([str(idx) for idx in df.index], index=df.index)


def _paired_tests(model_abs: np.ndarray, baseline_abs: np.ndarray) -> dict[str, float]:
    out = {"paired_test_stat": np.nan, "paired_test_pvalue": np.nan, "wilcoxon_pvalue": np.nan, "dm_pvalue": np.nan}
    mask = np.isfinite(model_abs) & np.isfinite(baseline_abs)
    model_abs = model_abs[mask]
    baseline_abs = baseline_abs[mask]
    diff = model_abs - baseline_abs
    if len(diff) < 2 or np.isclose(np.std(diff, ddof=1), 0):
        return out
    stat = float(np.mean(diff) / (np.std(diff, ddof=1) / np.sqrt(len(diff))))
    out["paired_test_stat"] = stat
    try:
        from scipy import stats

        out["paired_test_pvalue"] = float(stats.ttest_rel(model_abs, baseline_abs, nan_policy="omit").pvalue)
        out["wilcoxon_pvalue"] = float(stats.wilcoxon(model_abs, baseline_abs).pvalue) if len(diff) >= 6 else np.nan
    except Exception:
        out["paired_test_pvalue"] = float(2 * (1 - 0.5 * (1 + math.erf(abs(stat) / np.sqrt(2)))))
    out["dm_pvalue"] = out["paired_test_pvalue"]
    return out


def _bootstrap_ci(model_abs: np.ndarray, baseline_abs: np.ndarray, seed: int = RANDOM_SEED) -> tuple[float, float]:
    mask = np.isfinite(model_abs) & np.isfinite(baseline_abs)
    model_abs = model_abs[mask]
    baseline_abs = baseline_abs[mask]
    if len(model_abs) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    values = []
    n = len(model_abs)
    for _ in range(300):
        idx = rng.integers(0, n, n)
        base_mae = float(np.mean(baseline_abs[idx]))
        if base_mae:
            values.append((base_mae - float(np.mean(model_abs[idx]))) / base_mae * 100)
    return (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))) if values else (np.nan, np.nan)


def _bh_adjust(pvalues: Iterable[float]) -> list[float]:
    vals = np.array([np.nan if pd.isna(v) else float(v) for v in pvalues], dtype=float)
    adjusted = np.full(len(vals), np.nan)
    mask = np.isfinite(vals)
    if not mask.any():
        return adjusted.tolist()
    idx = np.where(mask)[0]
    order = idx[np.argsort(vals[mask])]
    ranked = vals[order]
    m = len(ranked)
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adjusted[order] = np.clip(adj, 0, 1)
    return adjusted.tolist()


def _metric_row(group: pd.DataFrame, target_col: str) -> dict[str, float]:
    y_true = group[target_col].to_numpy()
    y_pred = group["prediction"].to_numpy()
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "mse": mse,
        "r2": float(r2_score(y_true, y_pred)) if len(group) > 1 else np.nan,
        "num_test_samples": int(len(group)),
    }


def _prediction_frame(test: pd.DataFrame, ticker_col: str, date_col: str, target_col: str, target_name: str, horizon: int) -> pd.DataFrame:
    out = test[["row_id", ticker_col, date_col, target_col]].copy()
    out = out.rename(columns={target_col: "target"})
    out["target_name"] = target_name
    out["horizon"] = horizon
    return out


def _fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, features: list[str], ticker_col: str, target_col: str) -> np.ndarray:
    model = make_ridge_model(features, [ticker_col])
    model.fit(train[features + [ticker_col]], train[target_col])
    return model.predict(test[features + [ticker_col]])


def _build_embedding_columns(
    df: pd.DataFrame,
    train_index: pd.Index,
    text_columns: dict[str, list[str]],
    configuration: str,
    args,
) -> tuple[pd.DataFrame, list[str], str | None]:
    if configuration == "no_text":
        return df.copy(), [], None
    try:
        if configuration in {"levelwise_news", "levelwise_news_plus_filing"}:
            frames = []
            names: list[str] = []
            levels = NEWS_LEVELS if configuration == "levelwise_news" else FILING_LEVELS
            identity = _row_identity(df, args)
            for level in levels:
                text = joined_text_for_configuration(df, text_columns, f"{level}_only")
                raw = encode_text_raw(text, args.embedding_model, suffix=f"{configuration}_{level}", row_identity=identity)
                emb_df, emb_names = train_only_pca_features(raw, train_index, args.pca_dim, f"{level}", RANDOM_SEED)
                frames.append(emb_df)
                names.extend(emb_names)
            emb = pd.concat(frames, axis=1)
        else:
            text = joined_text_for_configuration(df, text_columns, configuration)
            raw = encode_text_raw(text, args.embedding_model, suffix=configuration, row_identity=_row_identity(df, args))
            emb, names = train_only_pca_features(raw, train_index, args.pca_dim, configuration, RANDOM_SEED)
        out = pd.concat([df.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
        return out, names, None
    except Exception as exc:
        return df.copy(), [], f"Embedding unavailable for {configuration}: {exc}"


def _text_meta_features(configuration: str) -> list[str]:
    if configuration == "no_text":
        return []
    if configuration in {"news_all", "levelwise_news"}:
        levels = NEWS_LEVELS
    elif configuration in {"news_plus_filing", "levelwise_news_plus_filing"}:
        levels = FILING_LEVELS
    else:
        cols = {
            "macro_only": ["macro"],
            "sector_only": ["sector"],
            "related_only": ["related"],
            "target_only": ["target"],
            "target_sector": ["target", "sector"],
            "filing_only": ["filing"],
        }.get(configuration, [])
        levels = cols
    features = []
    for level in levels:
        if level == "filing":
            features.extend(["has_filing_context", "filing_context_changed", "filing_text_length_total"])
        else:
            features.extend([f"has_text_{level}", f"text_length_{level}"])
    base = ["num_text_levels_present", "news_text_length_total"]
    return list(dict.fromkeys(features + base))


def build_predictions_for_specs(
    df: pd.DataFrame,
    train_full: pd.DataFrame,
    test: pd.DataFrame,
    specs: list[TargetSpec],
    ticker_col: str,
    date_col: str,
    ts_features: list[str],
    text_columns: dict[str, list[str]],
    args,
) -> tuple[pd.DataFrame, list[str]]:
    records = []
    notes: list[str] = []
    configs = [
        "no_text",
        "macro_only",
        "sector_only",
        "related_only",
        "target_only",
        "target_sector",
        "news_all",
        "filing_only",
        "news_plus_filing",
        "levelwise_news",
        "levelwise_news_plus_filing",
    ]
    if getattr(args, "quick_mode", False):
        configs = [
            "no_text",
            "macro_only",
            "sector_only",
            "related_only",
            "target_only",
            "target_sector",
            "news_all",
            "filing_only",
            "news_plus_filing",
            "levelwise_news",
            "levelwise_news_plus_filing",
        ]
    embedded: dict[str, tuple[pd.DataFrame, list[str]]] = {}

    for config in configs:
        emb_df, emb_cols, note = _build_embedding_columns(df, train_full.index, text_columns, config, args)
        if note:
            notes.append(note)
        embedded[config] = (emb_df, emb_cols)

    for spec in specs:
        valid_train = train_full.dropna(subset=ts_features + [spec.column]).copy()
        valid_test = test.dropna(subset=ts_features + [spec.column]).copy()
        if valid_train.empty or valid_test.empty:
            notes.append(f"Skipped {spec.target_name} horizon {spec.horizon}: insufficient train/test rows.")
            continue
        base = _prediction_frame(valid_test, ticker_col, date_col, spec.column, spec.target_name, spec.horizon)

        naive = base.copy()
        naive["model"] = "Naive_logGKVol_t"
        naive["text_configuration"] = "no_text"
        naive["prediction"] = valid_test["logGKVol"].to_numpy()
        records.append(naive)

        har_cols = ["har_daily", "har_weekly", "har_monthly"]
        har = base.copy()
        har["model"] = "HAR_Ridge"
        har["text_configuration"] = "no_text"
        har["prediction"] = _fit_predict_ridge(valid_train, valid_test, har_cols, ticker_col, spec.column)
        records.append(har)

        ts = base.copy()
        ts["model"] = DEFAULT_BASELINE
        ts["text_configuration"] = "no_text"
        ts["prediction"] = _fit_predict_ridge(valid_train, valid_test, ts_features, ticker_col, spec.column)
        records.append(ts)

        for config, (emb_df, emb_cols) in embedded.items():
            if config == "no_text":
                continue
            meta_cols = _text_meta_features(config)
            feature_cols = ts_features + meta_cols + emb_cols
            work_train = emb_df.loc[valid_train.index].dropna(subset=feature_cols + [spec.column])
            work_test = emb_df.loc[valid_test.index].dropna(subset=feature_cols + [spec.column])
            if work_train.empty or work_test.empty:
                continue
            pred = _fit_predict_ridge(work_train, work_test, feature_cols, ticker_col, spec.column)
            model_df = _prediction_frame(work_test, ticker_col, date_col, spec.column, spec.target_name, spec.horizon)
            model_df["model"] = f"TS_plus_{config}"
            model_df["text_configuration"] = config
            model_df["prediction"] = pred
            records.append(model_df)

    return pd.concat(records, ignore_index=True) if records else pd.DataFrame(), notes


def ticker_metrics(predictions: pd.DataFrame, ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (target_name, horizon, ticker), target_group in predictions.groupby(["target_name", "horizon", ticker_col]):
        baseline = target_group[target_group["model"] == DEFAULT_BASELINE].set_index("row_id")
        if baseline.empty:
            continue
        baseline_abs = (baseline["prediction"] - baseline["target"]).abs()
        baseline_metrics = _metric_row(baseline.reset_index(), "target")
        for model, group in target_group.groupby("model"):
            row = {ticker_col: ticker, "target_name": target_name, "horizon": horizon, "model": model}
            row.update(_metric_row(group, "target"))
            row["baseline_mae"] = baseline_metrics["mae"]
            row["baseline_rmse"] = baseline_metrics["rmse"]
            row["mae_improvement_pct"] = (row["baseline_mae"] - row["mae"]) / row["baseline_mae"] * 100 if row["baseline_mae"] else np.nan
            row["rmse_improvement_pct"] = (row["baseline_rmse"] - row["rmse"]) / row["baseline_rmse"] * 100 if row["baseline_rmse"] else np.nan
            aligned = group.set_index("row_id").join(baseline_abs.rename("baseline_abs_error"), how="inner")
            tests = _paired_tests((aligned["prediction"] - aligned["target"]).abs().to_numpy(), aligned["baseline_abs_error"].to_numpy())
            row.update(tests)
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "news_signal_by_ticker.csv", index=False)
    return result


def ticker_improvement_summary(by_ticker: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    text_rows = by_ticker[by_ticker["model"].str.startswith("TS_plus_")].copy()
    rows = []
    for (target_name, horizon), group in text_rows.groupby(["target_name", "horizon"]):
        best = group.loc[group["mae_improvement_pct"].idxmax()] if len(group) else None
        worst = group.loc[group["mae_improvement_pct"].idxmin()] if len(group) else None
        rows.append(
            {
                "target_name": target_name,
                "horizon": horizon,
                "tickers_text_improved": int((group["mae_improvement_pct"] > 0).sum()),
                "tickers_text_worse": int((group["mae_improvement_pct"] < 0).sum()),
                "best_ticker": best.get("ticker", best.iloc[0]) if best is not None else np.nan,
                "worst_ticker": worst.get("ticker", worst.iloc[0]) if worst is not None else np.nan,
                "mean_mae_improvement_pct": float(group["mae_improvement_pct"].mean()) if len(group) else np.nan,
                "median_mae_improvement_pct": float(group["mae_improvement_pct"].median()) if len(group) else np.nan,
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "ticker_improvement_summary.csv", index=False)
    return result


def text_level_ablation(by_ticker: pd.DataFrame, ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for _, row in by_ticker[by_ticker["model"].str.startswith("TS_plus_")].iterrows():
        config = str(row["model"]).replace("TS_plus_", "")
        rows.append(
            {
                ticker_col: row[ticker_col],
                "target_name": row["target_name"],
                "horizon": row["horizon"],
                "text_configuration": config,
                "mae": row["mae"],
                "rmse": row["rmse"],
                "r2": row["r2"],
                "improvement_vs_ts_only_pct": row["mae_improvement_pct"],
                "pvalue_vs_ts_only": row["paired_test_pvalue"],
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "text_level_ablation.csv", index=False)
    return result


def corrected_text_ablation(predictions: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (target_name, horizon), target_group in predictions.groupby(["target_name", "horizon"]):
        baseline = target_group[target_group["model"] == DEFAULT_BASELINE].set_index("row_id")
        if baseline.empty:
            continue
        baseline_abs = (baseline["prediction"] - baseline["target"]).abs()
        baseline_mse = mean_squared_error(baseline["target"], baseline["prediction"])
        baseline_mae = mean_absolute_error(baseline["target"], baseline["prediction"])
        baseline_rmse = float(np.sqrt(baseline_mse))
        for model, group in target_group.groupby("model"):
            if model in {"Naive_logGKVol_t", "HAR_Ridge"}:
                continue
            config = "no_text" if model == DEFAULT_BASELINE else str(model).replace("TS_plus_", "")
            mse = mean_squared_error(group["target"], group["prediction"])
            mae = mean_absolute_error(group["target"], group["prediction"])
            rmse = float(np.sqrt(mse))
            aligned = group.set_index("row_id").join(baseline_abs.rename("baseline_abs_error"), how="inner")
            tests = _paired_tests((aligned["prediction"] - aligned["target"]).abs().to_numpy(), aligned["baseline_abs_error"].to_numpy()) if not aligned.empty else {}
            rows.append(
                {
                    "target_name": target_name,
                    "horizon": horizon,
                    "text_configuration": config,
                    "mae": float(mae),
                    "rmse": rmse,
                    "r2": float(r2_score(group["target"], group["prediction"])) if len(group) > 1 else np.nan,
                    "baseline_mae": float(baseline_mae),
                    "baseline_rmse": baseline_rmse,
                    "improvement_vs_ts_only_pct": (baseline_mae - mae) / baseline_mae * 100 if baseline_mae else np.nan,
                    "paired_test_pvalue": tests.get("paired_test_pvalue", np.nan),
                    "num_test_samples": int(len(group)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "corrected_text_column_ablation.csv", index=False)
    return result


def event_day_comparison(predictions: pd.DataFrame, df: pd.DataFrame, ticker_col: str, output_dir: Path) -> pd.DataFrame:
    flags = ["has_any_news_text", "has_text_macro", "has_text_sector", "has_text_related", "has_text_target"]
    aux = df[["row_id", "num_text_levels_present", "news_text_length_total"] + flags].copy()
    work = predictions.merge(aux, on="row_id", how="left")
    text_model = "TS_plus_target_sector" if "TS_plus_target_sector" in set(work["model"]) else "TS_plus_news_all"
    rows = []
    groups = {
        "no_text": work["has_any_news_text"].eq(False),
        "macro_or_sector_only": (work["has_text_macro"] | work["has_text_sector"]) & ~(work["has_text_related"] | work["has_text_target"]),
        "related_company_text": work["has_text_related"],
        "target_company_text": work["has_text_target"],
        "two_or_more_levels": work["num_text_levels_present"].fillna(0).ge(2),
    }
    for (target_name, horizon), target_group in work.groupby(["target_name", "horizon"]):
        for group_name, mask in groups.items():
            subset = target_group[mask.loc[target_group.index]]
            base = subset[subset["model"] == DEFAULT_BASELINE].set_index("row_id")
            comp = subset[subset["model"] == text_model].set_index("row_id")
            aligned = comp.join(base[["prediction"]].rename(columns={"prediction": "baseline_prediction"}), how="inner")
            if aligned.empty:
                continue
            base_abs = (aligned["baseline_prediction"] - aligned["target"]).abs().to_numpy()
            comp_abs = (aligned["prediction"] - aligned["target"]).abs().to_numpy()
            rows.append(
                {
                    "target_name": target_name,
                    "horizon": horizon,
                    "event_group": group_name,
                    "comparison_model": text_model,
                    "ts_only_mae": float(np.mean(base_abs)),
                    "text_mae": float(np.mean(comp_abs)),
                    "error_difference": float(np.mean(comp_abs - base_abs)),
                    "improvement_pct": (float(np.mean(base_abs)) - float(np.mean(comp_abs))) / float(np.mean(base_abs)) * 100 if float(np.mean(base_abs)) else np.nan,
                    "num_samples": int(len(aligned)),
                    "paired_pvalue": _paired_tests(comp_abs, base_abs)["paired_test_pvalue"],
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "event_day_forecast_comparison.csv", index=False)
    return result


def event_day_volatility_stats(df: pd.DataFrame, specs: list[TargetSpec], output_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for level in NEWS_LEVELS:
            flag = f"has_text_{level}"
            present = df[df[flag].astype(bool)][spec.column].dropna()
            absent = df[~df[flag].astype(bool)][spec.column].dropna()
            row = {
                "target_name": spec.target_name,
                "horizon": spec.horizon,
                "news_level": level,
                "mean_target_when_text_present": float(present.mean()) if len(present) else np.nan,
                "mean_target_when_text_absent": float(absent.mean()) if len(absent) else np.nan,
                "present_count": int(len(present)),
                "absent_count": int(len(absent)),
                "welch_pvalue": np.nan,
                "mannwhitney_pvalue": np.nan,
            }
            try:
                from scipy import stats

                if len(present) > 1 and len(absent) > 1:
                    row["welch_pvalue"] = float(stats.ttest_ind(present, absent, equal_var=False, nan_policy="omit").pvalue)
                    row["mannwhitney_pvalue"] = float(stats.mannwhitneyu(present, absent, alternative="two-sided").pvalue)
            except Exception:
                pass
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "event_day_volatility_statistics.csv", index=False)
    return result


def spike_analysis(predictions: pd.DataFrame, train: pd.DataFrame, df: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    aux_cols = ["row_id", ticker_col, "has_any_news_text"] + [f"has_text_{level}" for level in NEWS_LEVELS]
    work = predictions.merge(df[aux_cols], on=["row_id", ticker_col], how="left")
    for spec in specs:
        thresholds = train.groupby(ticker_col)[spec.column].quantile([0.90, 0.95]).unstack()
        thresholds.columns = ["spike_90_threshold", "spike_95_threshold"]
        target_group = work[(work["target_name"] == spec.target_name) & (work["horizon"] == spec.horizon)]
        for spike_name, threshold_col in [("spike_90", "spike_90_threshold"), ("spike_95", "spike_95_threshold")]:
            enriched = target_group.merge(thresholds[[threshold_col]], left_on=ticker_col, right_index=True, how="left")
            enriched["is_spike"] = enriched["target"] > enriched[threshold_col]
            for model, group in enriched.groupby("model"):
                for flag in ["all", "spike", "non_spike"]:
                    subset = group if flag == "all" else group[group["is_spike"].eq(flag == "spike")]
                    if subset.empty:
                        continue
                    abs_err = (subset["prediction"] - subset["target"]).abs()
                    under = (subset["target"] - subset["prediction"]).clip(lower=0)
                    pred_spike = subset["prediction"] > subset[threshold_col]
                    y_spike = subset["is_spike"].astype(int)
                    row = {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "spike_definition": spike_name,
                        "model": model,
                        "segment": flag,
                        "mae": float(abs_err.mean()),
                        "rmse": float(np.sqrt(mean_squared_error(subset["target"], subset["prediction"]))),
                        "mean_underprediction": float(under.mean()),
                        "spike_rate": float(y_spike.mean()),
                        "text_day_rate": float(subset["has_any_news_text"].mean()),
                        "precision": np.nan,
                        "recall": np.nan,
                        "f1": np.nan,
                        "roc_auc": np.nan,
                        "pr_auc": np.nan,
                        "num_samples": int(len(subset)),
                    }
                    if flag == "all" and y_spike.nunique() > 1:
                        row["precision"] = float(precision_score(y_spike, pred_spike, zero_division=0))
                        row["recall"] = float(recall_score(y_spike, pred_spike, zero_division=0))
                        row["f1"] = float(f1_score(y_spike, pred_spike, zero_division=0))
                        try:
                            row["roc_auc"] = float(roc_auc_score(y_spike, subset["prediction"]))
                            row["pr_auc"] = float(average_precision_score(y_spike, subset["prediction"]))
                        except Exception:
                            pass
                    rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "volatility_spike_analysis.csv", index=False)
    return result


def lead_lag_analysis(df: pd.DataFrame, by_ticker: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for spec in specs:
        for ticker, group in df.groupby(ticker_col):
            for level in NEWS_LEVELS:
                x = group[f"has_text_{level}"].astype(float)
                y = group[spec.column]
                valid = x.notna() & y.notna()
                corr = float(x[valid].corr(y[valid])) if valid.sum() > 2 and x[valid].nunique() > 1 else np.nan
                coef = np.nan
                pvalue = np.nan
                if valid.sum() > 2 and x[valid].nunique() > 1:
                    lr = LinearRegression().fit(x[valid].to_frame(), y[valid])
                    coef = float(lr.coef_[0])
                model = "TS_plus_target_sector" if level in ["target", "sector"] else f"TS_plus_{level}_only"
                metric = by_ticker[
                    (by_ticker[ticker_col] == ticker)
                    & (by_ticker["target_name"] == spec.target_name)
                    & (by_ticker["horizon"] == spec.horizon)
                    & (by_ticker["model"] == model)
                ]
                rows.append(
                    {
                        ticker_col: ticker,
                        "news_level": level,
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "correlation": corr,
                        "regression_coefficient": coef,
                        "regression_pvalue": pvalue,
                        "mae_improvement_pct": float(metric["mae_improvement_pct"].iloc[0]) if len(metric) else np.nan,
                        "rmse_improvement_pct": float(metric["rmse_improvement_pct"].iloc[0]) if len(metric) else np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "news_lead_lag_analysis.csv", index=False)
    return result


def signal_regression(df: pd.DataFrame, train: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    features = ["har_daily", "har_weekly", "har_monthly"] + [f"has_text_{level}" for level in NEWS_LEVELS] + ["total_text_length", "num_text_levels_present"]
    for spec in specs:
        data = train.dropna(subset=features + [spec.column]).copy()
        if data.empty:
            continue
        data[[f"has_text_{level}" for level in NEWS_LEVELS]] = data[[f"has_text_{level}" for level in NEWS_LEVELS]].astype(float)
        try:
            import statsmodels.api as sm

            x = pd.get_dummies(data[features + [ticker_col]], columns=[ticker_col], drop_first=True, dtype=float)
            x = sm.add_constant(x, has_constant="add")
            model = sm.OLS(data[spec.column].astype(float), x.astype(float)).fit()
            for feature in model.params.index:
                rows.append(
                    {
                        "ticker": "pooled",
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "feature": feature,
                        "coefficient": float(model.params[feature]),
                        "standard_error": float(model.bse[feature]),
                        "t_value": float(model.tvalues[feature]),
                        "p_value": float(model.pvalues[feature]),
                    }
                )
        except Exception:
            x = pd.get_dummies(data[features + [ticker_col]], columns=[ticker_col], drop_first=True, dtype=float)
            lr = LinearRegression().fit(x, data[spec.column])
            for feature, coef in zip(x.columns, lr.coef_):
                rows.append(
                    {
                        "ticker": "pooled",
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "feature": feature,
                        "coefficient": float(coef),
                        "standard_error": np.nan,
                        "t_value": np.nan,
                        "p_value": np.nan,
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "text_signal_regression.csv", index=False)
    return result


def keyword_filter_results(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, date_col: str, ts_features: list[str], text_columns: dict[str, list[str]], args, output_dir: Path) -> pd.DataFrame:
    rows = []
    configs = {
        "event_keyword_all": (None, False),
        "earnings_only": ("earnings", False),
        "regulatory_legal_only": ("regulatory_legal", False),
        "corporate_action_only": ("corporate_action", False),
        "macro_shock_only": ("macro_shock", False),
        "target_event_keyword_only": (None, True),
    }
    for config, (group_name, target_only) in configs.items():
        text = keyword_filtered_text(df, text_columns, group_name, target_only)
        try:
            raw = encode_text_raw(text, args.embedding_model, suffix=f"keyword_{config}", row_identity=_row_identity(df, args))
            emb, emb_cols = train_only_pca_features(raw, train.index, args.pca_dim, config, RANDOM_SEED)
            work = pd.concat([df.reset_index(drop=True), emb.reset_index(drop=True)], axis=1)
        except Exception:
            emb_cols = []
            work = df.copy()
        features = ts_features + emb_cols + ["event_keyword_count", "target_event_keyword_count"]
        for spec in specs:
            tr = work.loc[train.index].dropna(subset=features + [spec.column])
            te = work.loc[test.index].dropna(subset=features + [spec.column])
            if tr.empty or te.empty:
                continue
            pred = _fit_predict_ridge(tr, te, features, ticker_col, spec.column)
            rows.append(
                {
                    "target_name": spec.target_name,
                    "horizon": spec.horizon,
                    "text_configuration": config,
                    "mae": float(mean_absolute_error(te[spec.column], pred)),
                    "rmse": float(np.sqrt(mean_squared_error(te[spec.column], pred))),
                    "r2": float(r2_score(te[spec.column], pred)) if len(te) > 1 else np.nan,
                    "num_test_samples": int(len(te)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "event_keyword_filter_results.csv", index=False)
    return result


def text_intensity_analysis(predictions: pd.DataFrame, df: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    aux = df[["row_id", "total_text_length", "num_text_levels_present", "event_keyword_count"]].copy()
    aux["text_length_quartile"] = pd.qcut(aux["total_text_length"].rank(method="first"), 4, labels=["q1", "q2", "q3", "q4"])
    work = predictions.merge(aux, on="row_id", how="left")
    rows = []
    text_model = "TS_plus_target_sector" if "TS_plus_target_sector" in set(work["model"]) else "TS_plus_news_all"
    for (target_name, horizon, bucket), group in work.groupby(["target_name", "horizon", "text_length_quartile"], observed=False):
        base = group[group["model"] == DEFAULT_BASELINE].set_index("row_id")
        comp = group[group["model"] == text_model].set_index("row_id")
        aligned = comp.join(base[["prediction"]].rename(columns={"prediction": "baseline_prediction"}), how="inner")
        if aligned.empty:
            continue
        base_abs = (aligned["baseline_prediction"] - aligned["target"]).abs()
        comp_abs = (aligned["prediction"] - aligned["target"]).abs()
        rows.append(
            {
                "target_name": target_name,
                "horizon": horizon,
                "intensity_group": str(bucket),
                "comparison_model": text_model,
                "baseline_mae": float(base_abs.mean()),
                "text_mae": float(comp_abs.mean()),
                "mae_improvement_pct": (float(base_abs.mean()) - float(comp_abs.mean())) / float(base_abs.mean()) * 100 if float(base_abs.mean()) else np.nan,
                "num_samples": int(len(aligned)),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "text_intensity_analysis.csv", index=False)
    return result


def placebo_tests(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, ts_features: list[str], args, output_dir: Path) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    base_cols = [f"has_text_{level}" for level in NEWS_LEVELS] + [f"text_length_{level}" for level in NEWS_LEVELS] + ["total_text_length", "num_text_levels_present"]
    variants = {"correct_text": df[base_cols].copy(), "no_text": pd.DataFrame(0, index=df.index, columns=base_cols)}
    shuffled = df[base_cols + [ticker_col]].copy()
    for _, idx in shuffled.groupby(ticker_col).groups.items():
        shuffled.loc[idx, base_cols] = shuffled.loc[idx, base_cols].iloc[rng.permutation(len(idx))].to_numpy()
    variants["shuffled_text"] = shuffled[base_cols]
    cross = df[[ticker_col] + base_cols].copy()
    cross[base_cols] = cross.groupby(df.index).transform(lambda x: x)
    variants["cross_ticker_text"] = df.groupby(df.index % max(1, df[ticker_col].nunique()))[base_cols].transform("mean")
    stale = df.groupby(ticker_col, sort=False)[base_cols].shift(20).fillna(0)
    variants["stale_text"] = stale
    for variant, frame in variants.items():
        work = pd.concat([df.drop(columns=[col for col in base_cols if col in df.columns]), frame], axis=1)
        features = ts_features + base_cols
        for spec in specs:
            tr = work.loc[train.index].dropna(subset=features + [spec.column])
            te = work.loc[test.index].dropna(subset=features + [spec.column])
            if tr.empty or te.empty:
                continue
            pred = _fit_predict_ridge(tr, te, features, ticker_col, spec.column)
            rows.append(
                {
                    "target_name": spec.target_name,
                    "horizon": spec.horizon,
                    "placebo_variant": variant,
                    "mae": float(mean_absolute_error(te[spec.column], pred)),
                    "rmse": float(np.sqrt(mean_squared_error(te[spec.column], pred))),
                    "r2": float(r2_score(te[spec.column], pred)) if len(te) > 1 else np.nan,
                    "num_test_samples": int(len(te)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "text_placebo_tests.csv", index=False)
    return result


def alignment_sensitivity(by_ticker: pd.DataFrame, df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, ts_features: list[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    text_cols = [f"has_text_{level}" for level in NEWS_LEVELS] + [f"text_length_{level}" for level in NEWS_LEVELS] + ["total_text_length", "num_text_levels_present"]
    shifted = df.copy()
    shifted[text_cols] = shifted.groupby(ticker_col, sort=False)[text_cols].shift(1).fillna(0)
    for label, source in [("text_same_day", df), ("text_shifted_1_day", shifted)]:
        features = ts_features + text_cols
        for spec in specs:
            tr = source.loc[train.index].dropna(subset=features + [spec.column])
            te = source.loc[test.index].dropna(subset=features + [spec.column])
            if tr.empty or te.empty:
                continue
            pred = _fit_predict_ridge(tr, te, features, ticker_col, spec.column)
            rows.append(
                {
                    "target_name": spec.target_name,
                    "horizon": spec.horizon,
                    "alignment_mode": label,
                    "mae": float(mean_absolute_error(te[spec.column], pred)),
                    "rmse": float(np.sqrt(mean_squared_error(te[spec.column], pred))),
                    "num_test_samples": int(len(te)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "text_alignment_sensitivity.csv", index=False)
    return result


def statistical_tests(by_ticker: pd.DataFrame, predictions: pd.DataFrame, ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    for (target_name, horizon, ticker), group in predictions.groupby(["target_name", "horizon", ticker_col]):
        baseline = group[group["model"] == DEFAULT_BASELINE].set_index("row_id")
        if baseline.empty:
            continue
        baseline_abs = (baseline["prediction"] - baseline["target"]).abs()
        for model, comp in group[group["model"].str.startswith("TS_plus_")].groupby("model"):
            aligned = comp.set_index("row_id").join(baseline_abs.rename("baseline_abs_error"), how="inner")
            if aligned.empty:
                continue
            comp_abs = (aligned["prediction"] - aligned["target"]).abs().to_numpy()
            base_abs = aligned["baseline_abs_error"].to_numpy()
            tests = _paired_tests(comp_abs, base_abs)
            ci_low, ci_high = _bootstrap_ci(comp_abs, base_abs)
            for test_name, p_col in [("paired_ttest_abs_error", "paired_test_pvalue"), ("wilcoxon_abs_error", "wilcoxon_pvalue"), ("diebold_mariano_abs_error", "dm_pvalue")]:
                rows.append(
                    {
                        ticker_col: ticker,
                        "target_name": target_name,
                        "horizon": horizon,
                        "baseline_model": DEFAULT_BASELINE,
                        "comparison_model": model,
                        "test_name": test_name,
                        "statistic": tests["paired_test_stat"],
                        "raw_pvalue": tests[p_col],
                        "effect_size": float(np.mean(base_abs) - np.mean(comp_abs)),
                        "confidence_interval_low": ci_low,
                        "confidence_interval_high": ci_high,
                    }
                )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["adjusted_pvalue"] = _bh_adjust(result["raw_pvalue"])
        result["significant_at_005"] = result["adjusted_pvalue"].lt(0.05)
    result.to_csv(output_dir / "statistical_tests.csv", index=False)
    return result


def walk_forward_results(df: pd.DataFrame, specs: list[TargetSpec], ticker_col: str, date_col: str, ts_features: list[str], output_dir: Path) -> pd.DataFrame:
    rows = []
    dates = sorted(df[date_col].dropna().unique())
    if len(dates) < 8:
        result = pd.DataFrame(rows)
        result.to_csv(output_dir / "walk_forward_results.csv", index=False)
        return result
    cutoffs = np.array_split(dates, 4)[1:]
    text_cols = [f"has_text_{level}" for level in NEWS_LEVELS] + ["total_text_length", "num_text_levels_present"]
    for spec in specs:
        for block in cutoffs:
            start, end = block[0], block[-1]
            train = df[df[date_col] < start].dropna(subset=ts_features + text_cols + [spec.column])
            test = df[(df[date_col] >= start) & (df[date_col] <= end)].dropna(subset=ts_features + text_cols + [spec.column])
            if len(train) < 20 or test.empty:
                continue
            for label, features in [("TS_only_lags", ts_features), ("TS_plus_text_meta", ts_features + text_cols)]:
                pred = _fit_predict_ridge(train, test, features, ticker_col, spec.column)
                rows.append(
                    {
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "block_start": start,
                        "block_end": end,
                        "model": label,
                        "mae": float(mean_absolute_error(test[spec.column], pred)),
                        "rmse": float(np.sqrt(mean_squared_error(test[spec.column], pred))),
                        "num_test_samples": int(len(test)),
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(output_dir / "walk_forward_results.csv", index=False)
    return result


def plot_outputs(predictions: pd.DataFrame, by_ticker: pd.DataFrame, ablation: pd.DataFrame, spike: pd.DataFrame, lead_lag: pd.DataFrame, placebo: pd.DataFrame, df: pd.DataFrame, ticker_col: str, date_col: str, output_dir: Path) -> None:
    if by_ticker.empty:
        return
    first_target = by_ticker.iloc[0]["target_name"]
    first_horizon = by_ticker.iloc[0]["horizon"]
    subset = by_ticker[(by_ticker["target_name"] == first_target) & (by_ticker["horizon"] == first_horizon) & by_ticker["model"].str.startswith("TS_plus_")]
    plt.figure(figsize=(11, 5))
    for model, group in subset.groupby("model"):
        plt.plot(group[ticker_col].astype(str), group["mae_improvement_pct"], marker="o", label=model)
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel("MAE improvement vs TS-only (%)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "news_signal_by_ticker_mae.png", dpi=150)
    plt.close()

    if not ablation.empty:
        pivot = ablation[(ablation["target_name"] == first_target) & (ablation["horizon"] == first_horizon)].pivot_table(index=ticker_col, columns="text_configuration", values="improvement_vs_ts_only_pct")
        plt.figure(figsize=(12, 5))
        plt.imshow(pivot.fillna(0), aspect="auto", cmap="RdYlGn")
        plt.colorbar(label="MAE improvement (%)")
        plt.yticks(range(len(pivot.index)), pivot.index.astype(str))
        plt.xticks(range(len(pivot.columns)), pivot.columns, rotation=35, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "text_level_ablation_heatmap.png", dpi=150)
        plt.close()

    event_path = output_dir / "event_day_forecast_comparison.csv"
    if event_path.exists():
        event = pd.read_csv(event_path)
        if not event.empty:
            plt.figure(figsize=(10, 5))
            event.head(20).plot.bar(x="event_group", y="improvement_pct", ax=plt.gca(), legend=False)
            plt.ylabel("MAE improvement (%)")
            plt.xticks(rotation=35, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / "event_day_error_comparison.png", dpi=150)
            plt.close()

    if not spike.empty:
        err = spike[(spike["segment"].isin(["spike", "non_spike"])) & (spike["model"].isin([DEFAULT_BASELINE, "TS_plus_target_sector", "TS_plus_news_all"]))]
        if not err.empty:
            plt.figure(figsize=(10, 5))
            labels = err["model"] + " / " + err["segment"]
            plt.bar(labels, err["mae"])
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("MAE")
            plt.tight_layout()
            plt.savefig(output_dir / "spike_error_comparison.png", dpi=150)
            plt.close()
        rates = spike[(spike["segment"] == "all") & (spike["model"] == DEFAULT_BASELINE)]
        if not rates.empty:
            plt.figure(figsize=(9, 4))
            plt.bar(rates["spike_definition"], rates["spike_rate"])
            plt.ylabel("Spike rate")
            plt.tight_layout()
            plt.savefig(output_dir / "spike_rate_by_text_level.png", dpi=150)
            plt.close()
        metrics = spike[(spike["segment"] == "all") & spike["model"].isin([DEFAULT_BASELINE, "TS_plus_target_sector", "TS_plus_news_all"])]
        if not metrics.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(metrics["model"] + " " + metrics["spike_definition"], metrics["recall"].fillna(0))
            plt.xticks(rotation=35, ha="right")
            plt.ylabel("Spike recall")
            plt.tight_layout()
            plt.savefig(output_dir / "spike_prediction_metrics.png", dpi=150)
            plt.close()

    if not lead_lag.empty:
        pivot = lead_lag.pivot_table(index="news_level", columns="horizon", values="correlation", aggfunc="mean")
        plt.figure(figsize=(8, 4))
        plt.imshow(pivot.fillna(0), aspect="auto", cmap="coolwarm")
        plt.colorbar(label="Correlation")
        plt.yticks(range(len(pivot.index)), pivot.index)
        plt.xticks(range(len(pivot.columns)), pivot.columns)
        plt.tight_layout()
        plt.savefig(output_dir / "news_lead_lag_heatmap.png", dpi=150)
        plt.close()

    if not placebo.empty:
        plt.figure(figsize=(10, 5))
        plt.bar(placebo["placebo_variant"], placebo["mae"])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel("MAE")
        plt.tight_layout()
        plt.savefig(output_dir / "correct_vs_placebo_text.png", dpi=150)
        plt.close()

    target_summary = predictions.groupby(["target_name", "model"])["target"].mean().reset_index()
    if not target_summary.empty:
        plt.figure(figsize=(10, 5))
        for model, group in target_summary.groupby("model"):
            plt.plot(group["target_name"], group["target"], marker="o", label=model)
        plt.xticks(rotation=30, ha="right")
        plt.ylabel("Mean target")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(output_dir / "target_comparison.png", dpi=150)
        plt.close()

    if not predictions.empty:
        ticker = predictions[ticker_col].iloc[0]
        pred = predictions[predictions[ticker_col] == ticker]
        models = [DEFAULT_BASELINE, "TS_plus_target_sector", "TS_plus_news_all"]
        pred = pred[pred["model"].isin(models)]
        if not pred.empty:
            merged = pred.merge(df[["row_id", "has_text_target", "has_any_news_text"]], on="row_id", how="left")
            plt.figure(figsize=(12, 5))
            actual = merged.drop_duplicates("row_id").sort_values(date_col)
            plt.plot(actual[date_col], actual["target"], label="actual", linewidth=2)
            for model, group in merged.groupby("model"):
                group = group.sort_values(date_col)
                plt.plot(group[date_col], group["prediction"], label=model, alpha=0.8)
            events = actual[actual["has_text_target"].astype(bool) | actual["has_any_news_text"].astype(bool)].head(20)
            plt.scatter(events[date_col], events["target"], color="black", s=18, label="text/event day")
            plt.legend(fontsize=8)
            plt.xticks(rotation=30, ha="right")
            plt.tight_layout()
            plt.savefig(output_dir / f"example_event_predictions_{_safe_name(ticker)}.png", dpi=150)
            plt.close()


def write_report(output_dir: Path, notes: list[str]) -> None:
    def read(name: str) -> pd.DataFrame:
        path = output_dir / name
        if not path.exists() or path.stat().st_size == 0:
            return pd.DataFrame()
        try:
            return pd.read_csv(path)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()

    by_ticker = read("news_signal_by_ticker.csv")
    ablation = read("text_level_ablation.csv")
    event = read("event_day_forecast_comparison.csv")
    spike = read("volatility_spike_analysis.csv")
    placebo = read("text_placebo_tests.csv")
    stats = read("statistical_tests.csv")
    keyword = read("event_keyword_filter_results.csv")

    best_model = "not available"
    if not by_ticker.empty:
        best = by_ticker.sort_values("mae").iloc[0]
        best_model = f"{best['model']} on {best['target_name']} h={best['horizon']}"
    improved_tickers = 0
    if not by_ticker.empty:
        improved_tickers = int(by_ticker[by_ticker["model"].str.startswith("TS_plus_")].groupby("ticker")["mae_improvement_pct"].max().gt(0).sum())
    best_level = "not available"
    if not ablation.empty:
        best_level = str(ablation.groupby("text_configuration")["improvement_vs_ts_only_pct"].median().idxmax())
    event_gain = float(event["improvement_pct"].max()) if not event.empty else np.nan
    spike_gain = np.nan
    if not spike.empty:
        base_spike = spike[(spike["model"] == DEFAULT_BASELINE) & (spike["segment"] == "spike")]["mae"].mean()
        text_spike = spike[(spike["model"].str.startswith("TS_plus_")) & (spike["segment"] == "spike")]["mae"].min()
        spike_gain = (base_spike - text_spike) / base_spike * 100 if base_spike else np.nan
    placebo_ok = False
    if not placebo.empty and "correct_text" in set(placebo["placebo_variant"]):
        correct = placebo[placebo["placebo_variant"] == "correct_text"]["mae"].mean()
        others = placebo[placebo["placebo_variant"] != "correct_text"]["mae"].mean()
        placebo_ok = bool(correct < others)
    keyword_ok = False
    if not keyword.empty:
        keyword_ok = bool(keyword["mae"].min() < keyword["mae"].mean())
    stat_ok = bool((stats.get("adjusted_pvalue", pd.Series(dtype=float)) < 0.10).any()) if not stats.empty else False

    decision = "stop_agent_development"
    if (event_gain >= 2 or spike_gain >= 2) and placebo_ok and improved_tickers >= 2 and stat_ok:
        decision = "proceed_with_agents"
    elif improved_tickers >= 1 or (not np.isnan(event_gain) and event_gain > 0) or (not np.isnan(spike_gain) and spike_gain > 0):
        decision = "run_larger_pilot"
    if not event.empty and "total_volatility" in set(event["target_name"]) and event[event["target_name"] == "total_volatility"]["improvement_pct"].max() > event["improvement_pct"].median():
        decision = "revise_target_or_alignment"

    lines = [
        "# FinTexTS News Signal Report",
        "",
        f"- Best model overall: {best_model}",
        f"- Tickers improved by at least one text model: {improved_tickers}",
        f"- Best text level/configuration by median improvement: {best_level}",
        f"- Best event-day improvement: {event_gain:.3f}%" if not np.isnan(event_gain) else "- Best event-day improvement: not available",
        f"- Estimated spike-day improvement: {spike_gain:.3f}%" if not np.isnan(spike_gain) else "- Estimated spike-day improvement: not available",
        f"- Correct text better than placebo on average: {placebo_ok}",
        f"- Keyword-event filtering shows a useful comparison signal: {keyword_ok}",
        f"- Any adjusted statistical test below 0.10: {stat_ok}",
        f"- Recommendation: `{decision}`",
        "",
        "## Interpretation",
        "",
        "This is predictive signal analysis, not a causal claim. Positive values indicate that some text configuration reduced error relative to the TS-only baseline under the same split and target.",
        "",
        "## Limitations",
        "",
        "- Intraday news timestamps are not available, so same-day text alignment depends on the conservative assumptions documented in `text_alignment_sensitivity.csv`.",
        "- Quick mode uses Ridge models and reduced diagnostics to keep CPU runtime manageable.",
        "- Statistical tests are exploratory and corrected p-values should be read together with effect sizes and bootstrap intervals.",
    ]
    if notes:
        lines.extend(["", "## Runtime Notes", ""])
        lines.extend([f"- {note}" for note in notes])
    (output_dir / "news_signal_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_text_column_usage_report(
    output_dir: Path,
    schema_report: dict,
    text_columns: dict[str, list[str]],
    df: pd.DataFrame,
    ablation: pd.DataFrame,
    filing_diag: pd.DataFrame,
) -> None:
    missing = schema_report.get("missing_columns", [])
    extra = schema_report.get("extra_possible_text_columns", [])
    counts = {
        "macro": int(df["has_macro_text"].sum()) if "has_macro_text" in df else 0,
        "sector": int(df["has_sector_text"].sum()) if "has_sector_text" in df else 0,
        "related": int(df["has_related_text"].sum()) if "has_related_text" in df else 0,
        "target": int(df["has_target_text"].sum()) if "has_target_text" in df else 0,
        "any_news": int(df["has_any_news_text"].sum()) if "has_any_news_text" in df else 0,
        "filing": int(df["has_filing_context"].sum()) if "has_filing_context" in df else 0,
        "filing_changed": int(df["filing_context_changed"].sum()) if "filing_context_changed" in df else 0,
    }
    length_rows = []
    for level in NEWS_LEVELS:
        col = f"text_length_{level}"
        length_rows.append(f"- {level}: mean={df[col].mean():.2f}, median={df[col].median():.2f}" if col in df else f"- {level}: unavailable")
    if "filing_text_length_total" in df:
        length_rows.append(f"- filing: mean={df['filing_text_length_total'].mean():.2f}, median={df['filing_text_length_total'].median():.2f}")

    best_config = "not available"
    filing_comment = "not available"
    if not ablation.empty:
        grouped = ablation.groupby("text_configuration")["improvement_vs_ts_only_pct"].median().sort_values(ascending=False)
        if len(grouped):
            best_config = f"{grouped.index[0]} ({grouped.iloc[0]:.3f}% median MAE improvement)"
        filing_rows = grouped[[idx for idx in grouped.index if "filing" in idx]] if any("filing" in idx for idx in grouped.index) else pd.Series(dtype=float)
        news_rows = grouped[[idx for idx in grouped.index if idx in {"news_all", "target_sector", "target_only", "levelwise_news"}]] if any(idx in {"news_all", "target_sector", "target_only", "levelwise_news"} for idx in grouped.index) else pd.Series(dtype=float)
        if not filing_rows.empty and not news_rows.empty:
            filing_comment = "filing helps more than news median" if filing_rows.max() > news_rows.max() else "filing does not beat the best news-only median"
    stable_filing = False
    if not filing_diag.empty and "median_same_content_run_days" in filing_diag:
        stable_filing = bool(filing_diag["median_same_content_run_days"].fillna(0).max() >= 5)

    lines = [
        "# Text Column Usage Report",
        "",
        "## Explicit Columns",
        "",
        f"- Expected text columns: {len(schema_report.get('expected_columns', []))}",
        f"- Found text columns: {len(schema_report.get('found_columns', []))}",
        f"- Missing configured columns: {missing if missing else 'none'}",
        f"- Extra possible text columns outside config: {extra if extra else 'none'}",
        "",
        "## Previous Behavior",
        "",
        "- Previous code used substring-based detection when `TEXT_COLUMNS` was empty, so it could miss camelCase FinTexTS columns and could include ambiguous text fields.",
        "- Previous joins used `fillna(\"\").astype(str)` in some paths. That avoided many nulls but still did not centralize handling of literal strings like `None`, `nan`, and `null`.",
        "- Previous `all_text` meant news columns detected as macro/sector/related/target; filing inclusion depended on detection, so the definition was not explicit enough.",
        "",
        "## Corrected Behavior",
        "",
        "- Explicit FinTexTS columns are now the primary source; fallback detection only emits warnings.",
        "- `news_all` is macro + sector + related + target only. It never includes filing.",
        "- `filing_only` and `news_plus_filing` are explicit filing-aware configurations.",
        "- Filing is treated as slower-moving company context, not daily news.",
        f"- Filing appears stable over longer runs: {stable_filing}.",
        "- Text cache version is `v2_explicit_fintexts_columns`, so old caches built from ambiguous joins are not reused.",
        "",
        "## Row Counts",
        "",
        *[f"- {key}: {value}" for key, value in counts.items()],
        "",
        "## Text Lengths",
        "",
        *length_rows,
        "",
        "## Ablation Summary",
        "",
        f"- Best text configuration: {best_config}",
        f"- Filing context result: {filing_comment}",
        "- News and filing should continue to be handled separately unless a larger pilot shows stable benefit from `news_plus_filing`.",
    ]
    (output_dir / "text_column_usage_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_detailed_news_signal(args) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading FinTexTS...")
    raw, column_map = load_fintexts(num_tickers=args.num_tickers)
    horizons = sorted(set(args.forecast_horizons or [1, 2, 3, 5]))
    targets = args.targets or ["log_gk", "total_volatility", "log_abs_return"]
    if getattr(args, "quick_mode", False):
        horizons = [h for h in horizons if h in {1, 3}]
        targets = [target for target in targets if target in {"log_gk", "total_volatility"}]
        args.pca_dim = min(args.pca_dim, 16)

    df = add_volatility_features(raw, column_map, forecast_horizon=1, max_lag=args.max_lag)
    df = add_detailed_volatility_targets(df, column_map, horizons)
    ts_features = time_series_feature_columns(args.max_lag)
    schema_report = write_text_schema_validation(df, output_dir)
    text_columns = detect_text_columns(df)
    duplicate_diag = text_duplicate_diagnostics(df, text_columns, column_map.ticker, column_map.date)
    duplicate_diag.to_csv(output_dir / "text_duplicate_diagnostics.csv", index=False)
    filing_diag = filing_context_diagnostics(df, text_columns, column_map.ticker, column_map.date)
    filing_diag.to_csv(output_dir / "filing_context_diagnostics.csv", index=False)
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    needed = ts_features + ["logGKVol"]
    df = df.dropna(subset=needed).copy().reset_index(drop=False).rename(columns={"index": "row_id"})
    df = df.reset_index(drop=True)

    specs = [TargetSpec(target, horizon, target_column_for(target, horizon)) for target in targets for horizon in horizons]
    train, val, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train, val], ignore_index=False)

    print(f"Detected text columns: {text_columns}")
    print(f"Train rows: {len(train_full)}, test rows: {len(test)}, tickers: {df[column_map.ticker].nunique()}")
    predictions, notes = build_predictions_for_specs(df, train_full, test, specs, column_map.ticker, column_map.date, ts_features, text_columns, args)
    if predictions.empty:
        raise RuntimeError("No detailed predictions were produced.")
    predictions.to_csv(output_dir / "detailed_predictions.csv", index=False)

    main_preds = predictions[(predictions["target_name"] == specs[0].target_name) & (predictions["horizon"] == specs[0].horizon)].copy()
    if not main_preds.empty:
        compat = main_preds.rename(columns={"target": "target"})
        write_evaluation_outputs(compat, output_dir, column_map.ticker)

    by_ticker = ticker_metrics(predictions, column_map.ticker, output_dir)
    summary = ticker_improvement_summary(by_ticker, output_dir)
    ablation = text_level_ablation(by_ticker, column_map.ticker, output_dir)
    corrected_ablation = corrected_text_ablation(predictions, output_dir)
    ablation.to_csv(output_dir / "corrected_text_column_ablation_by_ticker.csv", index=False)
    event_cmp = event_day_comparison(predictions, df, column_map.ticker, output_dir)
    event_stats = event_day_volatility_stats(df, specs, output_dir)
    spike = spike_analysis(predictions, train_full, df, specs, column_map.ticker, output_dir)
    lead_lag = lead_lag_analysis(df, by_ticker, specs, column_map.ticker, output_dir)
    regression = signal_regression(df, train_full, specs, column_map.ticker, output_dir)
    if args.run_event_keyword_filter:
        keyword_filter_results(df, train_full, test, specs, column_map.ticker, column_map.date, ts_features, text_columns, args, output_dir)
    else:
        pd.DataFrame().to_csv(output_dir / "event_keyword_filter_results.csv", index=False)
    intensity = text_intensity_analysis(predictions, df, output_dir)
    if args.run_placebo_tests:
        placebo = placebo_tests(df, train_full, test, specs, column_map.ticker, ts_features, args, output_dir)
    else:
        placebo = pd.DataFrame()
        placebo.to_csv(output_dir / "text_placebo_tests.csv", index=False)
    alignment_sensitivity(by_ticker, df, train_full, test, specs, column_map.ticker, ts_features, output_dir)
    if args.run_statistical_tests:
        statistical_tests(by_ticker, predictions, column_map.ticker, output_dir)
    else:
        pd.DataFrame().to_csv(output_dir / "statistical_tests.csv", index=False)
    walk_forward_results(df, specs, column_map.ticker, column_map.date, ts_features, output_dir)
    plot_outputs(predictions, by_ticker, ablation, spike, lead_lag, placebo, df, column_map.ticker, column_map.date, output_dir)
    write_report(output_dir, notes)
    write_text_column_usage_report(output_dir, schema_report, text_columns, df, corrected_ablation, filing_diag)

    print(f"Detailed outputs written to: {output_dir.resolve()}")
    print("Key files: news_signal_report.md, news_signal_by_ticker.csv, text_level_ablation.csv, volatility_spike_analysis.csv")
