"""LSTM temporal backbone with gated text residual correction.

This standalone module adds a sequence model that preserves the order of
logGKVol/log_return look-back windows. It then tests whether target/sector text
embeddings can correct the LSTM residual on event-gated rows.

Example:

    python -m src.spike_lstm_text_correction \
      --num_tickers 25 \
      --tickers C BAC AMGN AMD AXP \
      --look_back 22 \
      --targets log_gk log_abs_return \
      --forecast_horizons 1 3 \
      --spike_percentiles 80 85 90 95 \
      --embedding_variants target_sector_event_weighted \
      --alignment_modes shifted_1_day_text \
      --device cpu \
      --output_dir outputs/spike_lstm_text_correction
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, RANDOM_SEED
from .detailed_news_signal import split_by_time
from .models import make_ridge_model
from .spike_controlled_embedding import build_controlled_embedding
from .spike_news_embedding_signal import prepare_data, train_only_pca
from .spike_temporal_text_experiment import (
    _evaluate_predictions,
    _financial_metric_summary,
    _gate_mask,
    _prediction_metrics,
)
from .text_features import add_event_keyword_features, add_text_presence_flags, normalize_text_value


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


DEFAULT_EMBEDDING_VARIANTS = ["target_sector_event_weighted"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LSTM temporal model with gated text residual correction.")
    parser.add_argument("--num_tickers", type=int, default=25)
    parser.add_argument("--tickers", nargs="*", default=["C", "BAC", "AMGN", "AMD", "AXP"])
    parser.add_argument("--look_back", type=int, default=22)
    parser.add_argument("--forecast_horizons", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--targets", nargs="+", default=["log_gk", "log_abs_return"])
    parser.add_argument("--spike_percentiles", nargs="+", type=float, default=[80.0, 85.0, 90.0, 95.0])
    parser.add_argument("--embedding_variants", nargs="+", default=DEFAULT_EMBEDDING_VARIANTS)
    parser.add_argument("--alignment_modes", nargs="+", default=["shifted_1_day_text"])
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--embedding_device", default="auto")
    parser.add_argument("--pca_dim", type=int, default=32)
    parser.add_argument("--delta_window", type=int, default=20)
    parser.add_argument("--target_weight", type=float, default=2.0)
    parser.add_argument("--sector_weight", type=float, default=1.0)
    parser.add_argument("--max_lag", type=int, default=None)
    parser.add_argument("--cache_dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--gate_type", choices=["target_event_keyword", "any_event_keyword", "always"], default="target_event_keyword")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, etc. for PyTorch LSTM.")
    parser.add_argument("--hidden_size", type=int, default=32)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ticker_embedding_dim", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--output_dir", default="outputs/spike_lstm_text_correction")
    return parser.parse_args()


def _resolve_torch_device(requested: str):
    import torch

    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _add_sequence_windows(df: pd.DataFrame, ticker_col: str, date_col: str, look_back: int) -> tuple[pd.DataFrame, list[str], list[str]]:
    work = df.sort_values([ticker_col, date_col]).copy()
    grouped = work.groupby(ticker_col, sort=False)
    gk_cols: list[str] = []
    ret_cols: list[str] = []
    for offset in range(look_back):
        gk_col = f"lstm_logGKVol_t_minus_{offset}"
        ret_col = f"lstm_log_return_t_minus_{offset}"
        work[gk_col] = grouped["logGKVol"].shift(offset)
        work[ret_col] = grouped["log_return"].shift(offset)
        gk_cols.append(gk_col)
        ret_cols.append(ret_col)
    work["lstm_sequence_complete"] = work[gk_cols + ret_cols].notna().all(axis=1)
    return work.sort_index(), gk_cols, ret_cols


def _make_sequence_array(df: pd.DataFrame, gk_cols: list[str], ret_cols: list[str]) -> np.ndarray:
    gk = df[list(reversed(gk_cols))].to_numpy(dtype=np.float32)
    ret = df[list(reversed(ret_cols))].to_numpy(dtype=np.float32)
    return np.stack([gk, ret], axis=2)


def _fit_sequence_scaler(train: pd.DataFrame, gk_cols: list[str], ret_cols: list[str]) -> StandardScaler:
    scaler = StandardScaler()
    seq = _make_sequence_array(train, gk_cols, ret_cols)
    scaler.fit(seq.reshape(-1, seq.shape[-1]))
    return scaler


def _transform_sequence(df: pd.DataFrame, gk_cols: list[str], ret_cols: list[str], scaler: StandardScaler) -> np.ndarray:
    seq = _make_sequence_array(df, gk_cols, ret_cols)
    flat = seq.reshape(-1, seq.shape[-1])
    return scaler.transform(flat).reshape(seq.shape).astype(np.float32)


def _ticker_codes(train: pd.DataFrame, *frames: pd.DataFrame, ticker_col: str) -> tuple[dict[str, int], list[np.ndarray]]:
    tickers = sorted(train[ticker_col].astype(str).unique().tolist())
    mapping = {ticker: idx for idx, ticker in enumerate(tickers)}
    arrays = []
    for frame in frames:
        arrays.append(frame[ticker_col].astype(str).map(mapping).fillna(0).to_numpy(dtype=np.int64))
    return mapping, arrays


def _load_torch_model(input_dim: int, hidden_size: int, num_layers: int, dropout: float, n_tickers: int, ticker_embedding_dim: int):
    import torch

    class LSTMRegressor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.ticker_emb = torch.nn.Embedding(max(n_tickers, 1), ticker_embedding_dim)
            self.head = torch.nn.Sequential(
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size + ticker_embedding_dim, hidden_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hidden_size, 1),
            )

        def forward(self, x, ticker):
            _, (hidden, _) = self.lstm(x)
            temporal = hidden[-1]
            ticker_vec = self.ticker_emb(ticker)
            return self.head(torch.cat([temporal, ticker_vec], dim=1)).squeeze(1)

    return LSTMRegressor()


def _train_lstm(
    train_frame: pd.DataFrame,
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    gk_cols: list[str],
    ret_cols: list[str],
    ticker_col: str,
    target_col: str,
    args: argparse.Namespace,
) -> tuple[pd.Series, pd.Series, pd.Series, dict]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    tr = train_frame.dropna(subset=gk_cols + ret_cols + [target_col]).copy()
    va = val_frame.dropna(subset=gk_cols + ret_cols + [target_col]).copy()
    te = test_frame.dropna(subset=gk_cols + ret_cols + [target_col]).copy()
    if tr.empty or va.empty or te.empty:
        empty = pd.Series(np.nan, index=test_frame.index)
        return empty.reindex(train_frame.index), empty.reindex(val_frame.index), empty, {"status": "empty_split"}

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    device = _resolve_torch_device(args.device)

    seq_scaler = _fit_sequence_scaler(tr, gk_cols, ret_cols)
    x_train = _transform_sequence(tr, gk_cols, ret_cols, seq_scaler)
    x_val = _transform_sequence(va, gk_cols, ret_cols, seq_scaler)
    x_test = _transform_sequence(te, gk_cols, ret_cols, seq_scaler)
    x_full_train = _transform_sequence(train_frame.dropna(subset=gk_cols + ret_cols + [target_col]), gk_cols, ret_cols, seq_scaler)

    y_mean = float(tr[target_col].mean())
    y_std = float(tr[target_col].std(ddof=0) or 1.0)
    y_train = ((tr[target_col].to_numpy(dtype=np.float32) - y_mean) / y_std).astype(np.float32)
    y_val = ((va[target_col].to_numpy(dtype=np.float32) - y_mean) / y_std).astype(np.float32)

    mapping, [tr_code, va_code, te_code, full_code] = _ticker_codes(
        tr,
        tr,
        va,
        te,
        train_frame.dropna(subset=gk_cols + ret_cols + [target_col]),
        ticker_col=ticker_col,
    )
    model = _load_torch_model(
        input_dim=x_train.shape[2],
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        n_tickers=len(mapping),
        ticker_embedding_dim=args.ticker_embedding_dim,
    ).to(device)

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(tr_code, dtype=torch.long),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_x = torch.tensor(x_val, dtype=torch.float32, device=device)
    val_ticker = torch.tensor(va_code, dtype=torch.long, device=device)
    val_y = torch.tensor(y_val, dtype=torch.float32, device=device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = torch.nn.MSELoss()
    best_state = None
    best_val = float("inf")
    bad_epochs = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        for bx, bt, by in train_loader:
            bx = bx.to(device)
            bt = bt.to(device)
            by = by.to(device)
            opt.zero_grad()
            loss = loss_fn(model(bx, bt), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x, val_ticker), val_y).cpu())
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    def predict(frame: pd.DataFrame, x_values: np.ndarray, codes: np.ndarray) -> pd.Series:
        model.eval()
        pred = []
        ds = TensorDataset(torch.tensor(x_values, dtype=torch.float32), torch.tensor(codes, dtype=torch.long))
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for bx, bt in loader:
                raw = model(bx.to(device), bt.to(device)).detach().cpu().numpy()
                pred.append(raw)
        values = np.concatenate(pred) * y_std + y_mean
        return pd.Series(values, index=frame.index)

    full_train_frame = train_frame.dropna(subset=gk_cols + ret_cols + [target_col])
    train_full_pred = pd.Series(np.nan, index=train_frame.index)
    train_full_pred.loc[full_train_frame.index] = predict(full_train_frame, x_full_train, full_code)
    val_pred = pd.Series(np.nan, index=val_frame.index)
    val_pred.loc[va.index] = predict(va, x_val, va_code)
    test_pred = pd.Series(np.nan, index=test_frame.index)
    test_pred.loc[te.index] = predict(te, x_test, te_code)
    meta = {
        "status": "ok",
        "device": str(device),
        "epochs_trained": epoch,
        "best_val_mse_scaled": best_val,
        "train_rows": len(tr),
        "val_rows": len(va),
        "test_rows": len(te),
    }
    return train_full_pred, val_pred, test_pred, meta


def _fit_lstm_text_residual(
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
    tr = tr[base_train_pred.loc[tr.index].notna()].copy()
    if tr.empty or te.empty:
        return base_test_pred.copy()
    gate = _gate_mask(tr, gate_type)
    tr_fit = tr[gate] if gate.sum() >= 20 else tr
    residual = tr_fit[target_col] - base_train_pred.loc[tr_fit.index]
    model = make_ridge_model(features, [ticker_col])
    model.fit(tr_fit[features + [ticker_col]], residual)
    final = base_test_pred.copy()
    raw = pd.Series(model.predict(te[features + [ticker_col]]), index=te.index)
    test_gate = _gate_mask(te, gate_type)
    final.loc[te.index] = base_test_pred.loc[te.index] + raw.where(test_gate, 0.0)
    return final


def _event_preview(df: pd.DataFrame, text_columns: dict[str, list[str]], idx: int) -> str:
    parts: list[str] = []
    for group in ["target", "sector"]:
        for column in text_columns.get(group, []):
            if column in df.columns:
                text = normalize_text_value(df.loc[idx, column])
                if text:
                    parts.append(text.replace("\n", " ")[:160])
    return " | ".join(parts)[:320]


def _write_summary(output_dir: Path, metrics: pd.DataFrame, train_log: pd.DataFrame, args: argparse.Namespace) -> None:
    lines = ["# LSTM Text Correction Summary", ""]
    lines.append(f"- look_back: {args.look_back}")
    lines.append(f"- gate_type: {args.gate_type}")
    if not train_log.empty:
        ok = train_log[train_log["status"].eq("ok")]
        if not ok.empty:
            lines.append(f"- LSTM device: {ok['device'].mode().iloc[0]}")
            lines.append(f"- Median trained epochs: {ok['epochs_trained'].median():.0f}")
    if not metrics.empty:
        lines.extend(["", "## By Segment vs HAR", ""])
        view = (
            metrics.groupby(["model", "segment"])["mae_improvement_vs_har_pct"]
            .agg(["count", "mean", "median"])
            .reset_index()
            .sort_values(["segment", "median"], ascending=[True, False])
        )
        for _, row in view.iterrows():
            lines.append(f"- {row['model']} {row['segment']}: mean={row['mean']:.2f}%, median={row['median']:.2f}% over {int(row['count'])} cases")
        lines.extend(["", "## Spike Underprediction Reduction vs HAR", ""])
        under = (
            metrics[metrics["segment"].eq("spike")]
            .groupby("model")["underprediction_improvement_vs_har"]
            .agg(["count", "mean", "median"])
            .sort_values("median", ascending=False)
        )
        for model, row in under.iterrows():
            lines.append(f"- {model}: mean={row['mean']:.4f}, median={row['median']:.4f} over {int(row['count'])} cases")
    lines.extend(
        [
            "",
            "## Guardrails",
            "",
            "- LSTM uses ordered sequences of logGKVol and log_return.",
            "- Text correction is a Ridge residual model on text embedding/event features.",
            "- Text correction is applied only when the configured gate is active.",
            "- Spike thresholds are computed from train+validation rows and applied to test rows.",
        ]
    )
    (output_dir / "lstm_text_correction_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


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
    df = df[df["lstm_sequence_complete"]].copy().reset_index(drop=True)

    train_part, val_part, test = split_by_time(df, column_map.ticker, column_map.date)
    train_full = pd.concat([train_part, val_part], ignore_index=False)
    args._ticker_col = column_map.ticker
    args._date_col = column_map.date

    metric_rows: list[dict] = []
    event_rows: list[dict] = []
    train_log_rows: list[dict] = []
    for embedding_variant in args.embedding_variants:
        for alignment_mode in args.alignment_modes:
            raw = build_controlled_embedding(df, text_columns, embedding_variant, alignment_mode, "correct_text", args)
            emb_df, emb_cols = train_only_pca(raw, train_full.index, args.pca_dim, f"{embedding_variant}_{alignment_mode}")
            work = pd.concat([df.reset_index(drop=True), emb_df.reset_index(drop=True)], axis=1)
            tr_part = work.loc[train_part.index]
            va_part = work.loc[val_part.index]
            tr_full = work.loc[train_full.index]
            te = work.loc[test.index]

            for spec in specs:
                base_train_pred_part, base_val_pred, base_test_pred, meta = _train_lstm(
                    tr_part,
                    va_part,
                    te,
                    gk_cols,
                    ret_cols,
                    column_map.ticker,
                    spec.column,
                    args,
                )
                train_log_rows.append(
                    {
                        **meta,
                        "target_name": spec.target_name,
                        "horizon": spec.horizon,
                        "embedding_variant": embedding_variant,
                        "alignment_mode": alignment_mode,
                    }
                )
                base_train_full_pred = pd.concat([base_train_pred_part, base_val_pred]).reindex(tr_full.index)
                text_pred = _fit_lstm_text_residual(
                    tr_full,
                    te,
                    base_train_full_pred,
                    base_test_pred,
                    spec.column,
                    emb_cols,
                    column_map.ticker,
                    args.gate_type,
                )
                usable_pred = {
                    "LSTM_Temporal": base_test_pred,
                    "LSTM_TextResidual_Gated": text_pred,
                }
                for pct in args.spike_percentiles:
                    tr_for_threshold = tr_full.dropna(subset=[spec.column])
                    thresholds = tr_for_threshold.groupby(column_map.ticker)[spec.column].quantile(pct / 100.0).rename("threshold").to_frame()
                    scored = te[[column_map.ticker, column_map.date, spec.column, "target_event_keyword_count", "event_keyword_count"]].copy()
                    scored = scored.merge(thresholds, left_on=column_map.ticker, right_index=True, how="left")
                    scored["is_spike"] = scored[spec.column] > scored["threshold"]
                    scored["gate_active"] = _gate_mask(scored, args.gate_type)
                    metric_rows.extend(
                        _evaluate_predictions(
                            scored,
                            usable_pred,
                            spec.column,
                            base_test_pred,
                            column_map.ticker,
                            spec.target_name,
                            spec.horizon,
                            pct,
                            {"embedding_variant": embedding_variant, "alignment_mode": alignment_mode, "gate_type": args.gate_type},
                        )
                    )
                    rows = scored[scored["is_spike"] & scored["gate_active"]].copy()
                    if rows.empty:
                        continue
                    rows["base_prediction"] = base_test_pred.loc[rows.index]
                    rows["text_prediction"] = text_pred.loc[rows.index]
                    rows["abs_error_improvement"] = (rows["base_prediction"] - rows[spec.column]).abs() - (rows["text_prediction"] - rows[spec.column]).abs()
                    for idx, row in rows.iterrows():
                        event_rows.append(
                            {
                                "ticker": row[column_map.ticker],
                                "date": row[column_map.date],
                                "target_name": spec.target_name,
                                "horizon": spec.horizon,
                                "spike_percentile": pct,
                                "model": "LSTM_TextResidual_Gated",
                                "base_model": "LSTM_Temporal",
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
    train_log = pd.DataFrame(train_log_rows)
    events = pd.DataFrame(event_rows)
    metrics.to_csv(output_dir / "lstm_text_correction_metrics.csv", index=False)
    _financial_metric_summary(metrics).to_csv(output_dir / "lstm_text_correction_financial_summary.csv", index=False)
    train_log.to_csv(output_dir / "lstm_training_log.csv", index=False)
    if not events.empty:
        events.to_csv(output_dir / "lstm_text_correction_event_examples.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=False).head(300).to_csv(output_dir / "lstm_text_correction_top_improvements.csv", index=False)
        events.sort_values("abs_error_improvement", ascending=True).head(300).to_csv(output_dir / "lstm_text_correction_top_failures.csv", index=False)
    _write_summary(output_dir, metrics, train_log, args)
    print(f"LSTM text correction outputs written to: {output_dir.resolve()}")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
