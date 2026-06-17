"""Validate look-back window construction before running experiments.

This script is intentionally diagnostic-only. It does not train models and does
not modify the existing pipelines. It checks whether a proposed look_back setup
keeps time ordering intact for volatility windows, text alignment, future
targets, train-only spike thresholds, and target-event filters.

Example:

    python -m src.lookback_dataset_check \
      --num_tickers 25 \
      --tickers BAC C AMGN AXP AMD \
      --look_back 22 \
      --targets log_abs_return log_gk \
      --forecast_horizons 1 3 \
      --spike_percentiles 80 85 90 95 \
      --alignment_modes shifted_1_day_text same_day_text \
      --require_target_event_keyword \
      --output_dir outputs/lookback_dataset_check_candidate
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .detailed_news_signal import split_by_time
from .features import target_column_for
from .spike_news_embedding_signal import apply_alignment, prepare_data
from .text_features import add_event_keyword_features, add_text_presence_flags, joined_text_for_configuration


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate look-back volatility windows and text alignment.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=["BAC", "C", "AMGN", "AXP", "AMD"])
    parser.add_argument("--look_back", type=int, default=22)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_abs_return", "log_gk"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text", "same_day_text"])
    parser.add_argument("--text_configuration", default="target_sector")
    parser.add_argument("--require_target_event_keyword", action="store_true")
    parser.add_argument("--max_lag", type=int, default=None, help="Internal feature lag. Defaults to look_back.")
    parser.add_argument("--output_dir", default="outputs/lookback_dataset_check_candidate")
    return parser.parse_args()


def _build_windows(df: pd.DataFrame, ticker_col: str, date_col: str, look_back: int) -> pd.DataFrame:
    work = df.sort_values([ticker_col, date_col]).copy()
    grouped = work.groupby(ticker_col, sort=False)
    for offset in range(look_back):
        work[f"lookback_logGKVol_t_minus_{offset}"] = grouped["logGKVol"].shift(offset)
        work[f"lookback_date_t_minus_{offset}"] = grouped[date_col].shift(offset)
    window_cols = [f"lookback_logGKVol_t_minus_{offset}" for offset in range(look_back)]
    date_cols = [f"lookback_date_t_minus_{offset}" for offset in range(look_back)]
    work["lookback_nonmissing_count"] = work[window_cols].notna().sum(axis=1)
    work["lookback_complete"] = work["lookback_nonmissing_count"].eq(look_back)
    date_frame = work[date_cols].apply(pd.to_datetime, errors="coerce")
    anchor = pd.to_datetime(work[date_col], errors="coerce")
    work["lookback_uses_future_date"] = date_frame.gt(anchor, axis=0).any(axis=1)
    oldest_col = f"lookback_date_t_minus_{look_back - 1}"
    work["lookback_oldest_date"] = work[oldest_col]
    return work


def _add_target_dates(df: pd.DataFrame, ticker_col: str, date_col: str, horizons: list[int]) -> pd.DataFrame:
    work = df.sort_values([ticker_col, date_col]).copy()
    grouped = work.groupby(ticker_col, sort=False)
    for horizon in horizons:
        work[f"target_date_h{horizon}"] = grouped[date_col].shift(-horizon)
        work[f"target_date_h{horizon}_after_anchor"] = pd.to_datetime(work[f"target_date_h{horizon}"], errors="coerce").gt(
            pd.to_datetime(work[date_col], errors="coerce")
        )
    return work


def _text_alignment_summary(df: pd.DataFrame, ticker_col: str, date_col: str, text_columns: dict[str, list[str]], args: argparse.Namespace) -> pd.DataFrame:
    base_text = joined_text_for_configuration(df, text_columns, args.text_configuration)
    rows = []
    for mode in args.alignment_modes:
        aligned = apply_alignment(base_text, df, ticker_col, mode)
        nonempty = aligned.fillna("").astype(str).str.strip().ne("")
        if mode == "shifted_1_day_text":
            expected_source_date = df.groupby(ticker_col, sort=False)[date_col].shift(1)
        elif mode == "same_day_text":
            expected_source_date = df[date_col]
        else:
            expected_source_date = pd.Series(pd.NaT, index=df.index)
        source_after_anchor = pd.to_datetime(expected_source_date, errors="coerce").gt(pd.to_datetime(df[date_col], errors="coerce"))
        rows.append(
            {
                "alignment_mode": mode,
                "rows": int(len(df)),
                "nonempty_text_rows": int(nonempty.sum()),
                "nonempty_text_rate": float(nonempty.mean()),
                "source_after_anchor_rows": int(source_after_anchor.fillna(False).sum()),
                "target_event_keyword_rows": int(df["target_event_keyword_count"].fillna(0).gt(0).sum()) if "target_event_keyword_count" in df else 0,
            }
        )
    return pd.DataFrame(rows)


def _split_summary(df: pd.DataFrame, split_name: str, ticker_col: str, date_col: str, args: argparse.Namespace) -> dict:
    frame = df.copy()
    if args.require_target_event_keyword and "target_event_keyword_count" in frame.columns:
        frame = frame[frame["target_event_keyword_count"].fillna(0).gt(0)]
    return {
        "split": split_name,
        "rows": int(len(frame)),
        "tickers": int(frame[ticker_col].nunique()) if len(frame) else 0,
        "start_date": str(frame[date_col].min()) if len(frame) else "",
        "end_date": str(frame[date_col].max()) if len(frame) else "",
        "complete_lookback_rows": int(frame["lookback_complete"].sum()) if len(frame) else 0,
        "complete_lookback_rate": float(frame["lookback_complete"].mean()) if len(frame) else np.nan,
        "future_window_violations": int(frame["lookback_uses_future_date"].sum()) if len(frame) else 0,
        "target_event_keyword_rows": int(frame["target_event_keyword_count"].fillna(0).gt(0).sum()) if "target_event_keyword_count" in frame else 0,
    }


def _target_summary(df: pd.DataFrame, split_name: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    frame = df.copy()
    if args.require_target_event_keyword and "target_event_keyword_count" in frame.columns:
        frame = frame[frame["target_event_keyword_count"].fillna(0).gt(0)]
    for target in args.targets:
        for horizon in args.forecast_horizons:
            col = target_column_for(target, horizon)
            if col not in frame.columns:
                rows.append({"split": split_name, "target_name": target, "horizon": horizon, "target_column": col, "rows_with_target": 0})
                continue
            target_date_flag = f"target_date_h{horizon}_after_anchor"
            valid = frame[col].notna()
            rows.append(
                {
                    "split": split_name,
                    "target_name": target,
                    "horizon": horizon,
                    "target_column": col,
                    "rows": int(len(frame)),
                    "rows_with_target": int(valid.sum()),
                    "rows_with_target_and_complete_lookback": int((valid & frame["lookback_complete"]).sum()),
                    "target_after_anchor_rows": int(frame.loc[valid, target_date_flag].fillna(False).sum()) if target_date_flag in frame else 0,
                    "target_after_anchor_rate": float(frame.loc[valid, target_date_flag].fillna(False).mean()) if valid.any() and target_date_flag in frame else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _spike_summary(train: pd.DataFrame, test: pd.DataFrame, ticker_col: str, args: argparse.Namespace) -> pd.DataFrame:
    rows = []
    test_frame = test.copy()
    if args.require_target_event_keyword and "target_event_keyword_count" in test_frame.columns:
        test_frame = test_frame[test_frame["target_event_keyword_count"].fillna(0).gt(0)]
    for target in args.targets:
        for horizon in args.forecast_horizons:
            col = target_column_for(target, horizon)
            if col not in train.columns or col not in test_frame.columns:
                continue
            tr = train.dropna(subset=[col])
            te = test_frame.dropna(subset=[col])
            for pct in args.spike_percentiles:
                thresholds = tr.groupby(ticker_col)[col].quantile(pct / 100.0).rename("threshold")
                scored = te[[ticker_col, col, "lookback_complete"]].merge(thresholds, left_on=ticker_col, right_index=True, how="left")
                scored["is_spike"] = scored[col] > scored["threshold"]
                rows.append(
                    {
                        "target_name": target,
                        "horizon": horizon,
                        "spike_percentile": pct,
                        "test_rows": int(len(scored)),
                        "spike_rows": int(scored["is_spike"].sum()),
                        "spike_rate": float(scored["is_spike"].mean()) if len(scored) else np.nan,
                        "spike_rows_with_complete_lookback": int((scored["is_spike"] & scored["lookback_complete"]).sum()),
                    }
                )
    return pd.DataFrame(rows)


def _sample_audit(df: pd.DataFrame, ticker_col: str, date_col: str, args: argparse.Namespace) -> pd.DataFrame:
    cols = [
        ticker_col,
        date_col,
        "lookback_oldest_date",
        "lookback_nonmissing_count",
        "lookback_complete",
        "lookback_uses_future_date",
        "target_event_keyword_count",
    ]
    for horizon in args.forecast_horizons:
        cols.extend([f"target_date_h{horizon}", f"target_date_h{horizon}_after_anchor"])
    usable = [col for col in cols if col in df.columns]
    sample = df[df["lookback_complete"]].head(25)
    return sample[usable].copy()


def _write_report(output_dir: Path, split_summary: pd.DataFrame, target_summary: pd.DataFrame, spike_summary: pd.DataFrame, text_summary: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = ["# Look-Back Dataset Validation Report", ""]
    lines.append(f"- look_back: {args.look_back}")
    lines.append(f"- text_configuration: {args.text_configuration}")
    lines.append(f"- require_target_event_keyword: {args.require_target_event_keyword}")
    lines.append("")
    lines.append("## Split Summary")
    for _, row in split_summary.iterrows():
        lines.append(
            f"- {row['split']}: rows={int(row['rows'])}, complete_lookback={row['complete_lookback_rate']:.2%}, "
            f"future_window_violations={int(row['future_window_violations'])}, dates={row['start_date']}..{row['end_date']}"
        )
    violations = int(split_summary["future_window_violations"].sum()) if not split_summary.empty else 0
    target_bad = int((target_summary["target_after_anchor_rate"].fillna(1.0) < 1.0).sum()) if not target_summary.empty else 0
    lines.extend(["", "## Validation Checks", ""])
    lines.append(f"- Look-back future-date violations: {violations}")
    lines.append(f"- Target/date alignment rows with rate below 100%: {target_bad}")
    lines.append(f"- Text alignment source-after-anchor rows: {int(text_summary['source_after_anchor_rows'].sum()) if not text_summary.empty else 0}")
    if not spike_summary.empty:
        lines.extend(["", "## Spike Sample Summary", ""])
        for _, row in spike_summary.head(12).iterrows():
            lines.append(
                f"- {row['target_name']} h={row['horizon']} p{row['spike_percentile']}: "
                f"spikes={int(row['spike_rows'])}/{int(row['test_rows'])}, complete_lookback_spikes={int(row['spike_rows_with_complete_lookback'])}"
            )
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- This script validates dataset construction only; it does not train a model.",
            "- Spike thresholds are computed on train rows and applied to test rows.",
            "- A complete look-back window contains current-day logGKVol and prior look_back-1 observations only.",
        ]
    )
    (output_dir / "lookback_validation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> None:
    args.max_lag = args.max_lag or args.look_back
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, train, test, specs, column_map, ts_features, text_columns = prepare_data(args)
    if args.tickers:
        keep = set(args.tickers)
        df = df[df[column_map.ticker].isin(keep)].copy()
    df = add_text_presence_flags(df, text_columns, ticker_col=column_map.ticker)
    df = add_event_keyword_features(df, text_columns)
    df = _build_windows(df, column_map.ticker, column_map.date, args.look_back)
    df = _add_target_dates(df, column_map.ticker, column_map.date, args.forecast_horizons)
    df = df.reset_index(drop=True)

    train_part, val_part, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train_part, val_part], ignore_index=False)

    split_rows = [
        _split_summary(train_part, "train", column_map.ticker, column_map.date, args),
        _split_summary(val_part, "validation", column_map.ticker, column_map.date, args),
        _split_summary(train_full, "train_plus_validation", column_map.ticker, column_map.date, args),
        _split_summary(test, "test", column_map.ticker, column_map.date, args),
    ]
    split_summary = pd.DataFrame(split_rows)
    target_summary = pd.concat(
        [
            _target_summary(train_part, "train", args),
            _target_summary(val_part, "validation", args),
            _target_summary(test, "test", args),
        ],
        ignore_index=True,
    )
    spike_summary = _spike_summary(train_full, test, column_map.ticker, args)
    text_summary = _text_alignment_summary(df, column_map.ticker, column_map.date, text_columns, args)
    sample_audit = _sample_audit(test, column_map.ticker, column_map.date, args)

    split_summary.to_csv(output_dir / "lookback_split_summary.csv", index=False)
    target_summary.to_csv(output_dir / "lookback_target_alignment_summary.csv", index=False)
    spike_summary.to_csv(output_dir / "lookback_spike_summary.csv", index=False)
    text_summary.to_csv(output_dir / "lookback_text_alignment_summary.csv", index=False)
    sample_audit.to_csv(output_dir / "lookback_alignment_samples.csv", index=False)
    _write_report(output_dir, split_summary, target_summary, spike_summary, text_summary, args)
    print(f"Look-back validation outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
