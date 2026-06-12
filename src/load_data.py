"""Load and normalize the FinTexTS dataset."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
from datasets import load_dataset

from .config import COLUMN_MAPPING, DATASET_NAME


@dataclass(frozen=True)
class ColumnMap:
    ticker: str
    date: str
    open: str
    high: str
    low: str
    close: str


ALIASES = {
    "ticker": ["ticker", "symbol", "stock", "stock_code", "permno"],
    "date": ["date", "datetime", "trading_date", "time"],
    "open": ["open", "open_price", "o"],
    "high": ["high", "high_price", "h"],
    "low": ["low", "low_price", "l"],
    "close": ["close", "close_price", "adj_close", "adjusted_close", "c"],
}


def _normalize(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def _find_column(columns: Iterable[str], role: str) -> str | None:
    configured = COLUMN_MAPPING.get(role)
    if configured:
        return configured

    normalized = {_normalize(col): col for col in columns}
    for alias in ALIASES[role]:
        hit = normalized.get(_normalize(alias))
        if hit:
            return hit

    for col in columns:
        norm = _normalize(col)
        if any(_normalize(alias) in norm for alias in ALIASES[role]):
            return col
    return None


def infer_column_map(df: pd.DataFrame) -> ColumnMap:
    mapping = {role: _find_column(df.columns, role) for role in ALIASES}
    missing = [role for role, col in mapping.items() if col is None]
    if missing:
        columns = "\n".join(f"  - {col}" for col in df.columns)
        raise ValueError(
            "Could not infer required column(s): "
            + ", ".join(missing)
            + "\nAvailable columns:\n"
            + columns
            + "\n\nEdit COLUMN_MAPPING in src/config.py to map these roles explicitly."
        )
    return ColumnMap(**mapping)  # type: ignore[arg-type]


def load_fintexts(num_tickers: int | None = None, split: str | None = None) -> tuple[pd.DataFrame, ColumnMap]:
    """Load FinTexTS from Hugging Face and return a sorted pandas DataFrame."""
    dataset = load_dataset(DATASET_NAME)
    if split is None:
        split = "train" if "train" in dataset else next(iter(dataset.keys()))

    df = dataset[split].to_pandas()
    column_map = infer_column_map(df)

    df = df.copy()
    df[column_map.date] = pd.to_datetime(df[column_map.date], errors="coerce")
    for col in [column_map.open, column_map.high, column_map.low, column_map.close]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=[column_map.ticker, column_map.date, column_map.open, column_map.high, column_map.low, column_map.close])
    df = df.sort_values([column_map.ticker, column_map.date]).reset_index(drop=True)

    if num_tickers:
        tickers = df[column_map.ticker].drop_duplicates().head(num_tickers)
        df = df[df[column_map.ticker].isin(tickers)].copy()

    return df.reset_index(drop=True), column_map
