"""Run the end-to-end FinTexTS volatility forecasting pilot."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import DEFAULT_EMBEDDING_MODEL, TEST_YEAR, TRAIN_YEARS, VAL_YEAR
from .evaluate import write_evaluation_outputs
from .features import add_volatility_features, time_series_feature_columns
from .load_data import load_fintexts
from .models import make_ridge_model, make_supervised_model
from .text_features import add_text_presence_flags, detect_text_columns, make_text_features, write_text_schema_validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FinTexTS volatility forecasting pilot.")
    parser.add_argument("--num_tickers", type=int, default=10)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--pca_dim", type=int, default=64)
    parser.add_argument("--forecast_horizon", type=int, default=1)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 2, 3, 5])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "total_volatility", "log_abs_return"])
    parser.add_argument("--max_lag", type=int, default=22)
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--analysis_mode", choices=["pilot", "detailed_news_signal", "spike_news_signal"], default="pilot")
    parser.add_argument("--evaluation_mode", choices=["fixed", "walk_forward"], default="fixed")
    parser.add_argument("--quick_mode", action="store_true")
    parser.add_argument("--run_placebo_tests", action="store_true")
    parser.add_argument("--run_event_keyword_filter", action="store_true")
    parser.add_argument("--run_statistical_tests", action="store_true")
    return parser.parse_args()


def split_by_time(df: pd.DataFrame, ticker_col: str, date_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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


def add_embedding_columns(df: pd.DataFrame, matrix: np.ndarray | None, names: list[str]) -> tuple[pd.DataFrame, list[str]]:
    if matrix is None:
        return df, []
    out = df.copy()
    emb_df = pd.DataFrame(matrix, columns=names, index=out.index)
    return pd.concat([out, emb_df], axis=1), names


def fit_predict_model(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], ticker_col: str, target_col: str) -> np.ndarray:
    model = make_supervised_model(feature_cols, ticker_col)
    model.fit(train[feature_cols + [ticker_col]], train[target_col])
    return model.predict(test[feature_cols + [ticker_col]])


def fit_predict_ridge(train: pd.DataFrame, test: pd.DataFrame, feature_cols: list[str], ticker_col: str, target_col: str) -> np.ndarray:
    model = make_ridge_model(feature_cols, [ticker_col])
    model.fit(train[feature_cols + [ticker_col]], train[target_col])
    return model.predict(test[feature_cols + [ticker_col]])


def build_predictions(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame, ticker_col: str, date_col: str, ts_features: list[str], args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    records = []
    base_cols = ["row_id", ticker_col, date_col, "target"]
    test_base = test[base_cols].copy()

    naive = test_base.copy()
    naive["model"] = "Naive_logGKVol_t"
    naive["prediction"] = test["logGKVol"].to_numpy()
    records.append(naive)

    har_cols = ["har_daily", "har_weekly", "har_monthly"]
    har_pred = fit_predict_ridge(train, test, har_cols, ticker_col, "target")
    har = test_base.copy()
    har["model"] = "HAR_Ridge"
    har["prediction"] = har_pred
    records.append(har)

    ts_pred = fit_predict_model(train, test, ts_features, ticker_col, "target")
    ts = test_base.copy()
    ts["model"] = "TS_only_lags"
    ts["prediction"] = ts_pred
    records.append(ts)

    text_columns_seen: dict[str, list[str]] = {}
    for mode in ["news_all", "target_sector", "levelwise_news"]:
        matrix, names, text_columns_seen = make_text_features(
            df,
            mode=mode,
            embedding_model=args.embedding_model,
            pca_dim=args.pca_dim,
        )
        with_embeddings, emb_cols = add_embedding_columns(df, matrix, names)
        train_emb = with_embeddings.loc[train.index]
        test_emb = with_embeddings.loc[test.index]
        pred = fit_predict_model(train_emb, test_emb, ts_features + emb_cols, ticker_col, "target")
        model_df = test_base.copy()
        model_df["model"] = f"TS_plus_{mode}"
        model_df["prediction"] = pred
        records.append(model_df)

    return pd.concat(records, ignore_index=True), text_columns_seen


def event_day_analysis(df: pd.DataFrame, ticker_col: str, output_dir: Path) -> pd.DataFrame:
    rows = []
    work = df.copy()
    work["spike"] = work.groupby(ticker_col)["target"].transform(lambda s: s >= s.quantile(0.90))
    for flag in ["has_text_macro", "has_text_sector", "has_text_related", "has_text_target"]:
        grouped = work.groupby(flag).agg(mean_future_logGKVol=("target", "mean"), spike_rate=("spike", "mean"), count=("target", "size")).reset_index()
        grouped["flag"] = flag
        grouped = grouped.rename(columns={flag: "flag_value"})
        rows.append(grouped)
    result = pd.concat(rows, ignore_index=True)
    result.to_csv(output_dir / "event_day_analysis.csv", index=False)
    return result


def plot_outputs(results: pd.DataFrame, predictions: pd.DataFrame, ticker_col: str, date_col: str, output_dir: Path) -> None:
    for metric in ["RMSE", "MAE"]:
        ordered = results.sort_values(metric)
        plt.figure(figsize=(10, 5))
        plt.bar(ordered["model"], ordered[metric])
        plt.xticks(rotation=35, ha="right")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(output_dir / f"model_comparison_{metric.lower()}.png", dpi=150)
        plt.close()

    if predictions.empty:
        return
    ticker = predictions[ticker_col].iloc[0]
    subset = predictions[predictions[ticker_col] == ticker].sort_values(date_col)
    plt.figure(figsize=(12, 5))
    actual = subset.drop_duplicates("row_id")
    plt.plot(actual[date_col], actual["target"], label="actual", linewidth=2)
    for model, group in subset.groupby("model"):
        plt.plot(group[date_col], group["prediction"], label=model, alpha=0.75)
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    safe_ticker = "".join(ch if ch.isalnum() else "_" for ch in str(ticker))
    plt.savefig(output_dir / f"example_predictions_{safe_ticker}.png", dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    if args.analysis_mode == "detailed_news_signal":
        from .detailed_news_signal import run_detailed_news_signal

        run_detailed_news_signal(args)
        return
    if args.analysis_mode == "spike_news_signal":
        from .spike_news_signal import run_spike_news_signal_tests

        run_spike_news_signal_tests(args)
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw, column_map = load_fintexts(num_tickers=args.num_tickers)
    df = add_volatility_features(raw, column_map, forecast_horizon=args.forecast_horizon, max_lag=args.max_lag)
    ts_features = time_series_feature_columns(args.max_lag)
    needed = ts_features + ["target", "logGKVol"]
    df = df.dropna(subset=needed).copy().reset_index(drop=False).rename(columns={"index": "row_id"})

    write_text_schema_validation(df, output_dir)
    text_columns = detect_text_columns(df)
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)

    train, val, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train, val], ignore_index=False)
    predictions, text_columns = build_predictions(df, train_full, test, column_map.ticker, column_map.date, ts_features, args)

    results = write_evaluation_outputs(predictions, output_dir, column_map.ticker)
    event_day_analysis(df, column_map.ticker, output_dir)
    plot_outputs(results, predictions, column_map.ticker, column_map.date, output_dir)

    print("\nModel results:")
    print(results.to_string(index=False))
    print("\nDetected text columns:")
    for level, cols in text_columns.items():
        print(f"  {level}: {cols}")
    print(f"\nOutputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
