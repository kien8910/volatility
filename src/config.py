"""Configuration for the FinTexTS volatility pilot."""

from pathlib import Path

DATASET_NAME = "EXAONE-BI/FinTexTS"

EPSILON = 1e-12
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("cache")

# Override these if automatic column detection does not match FinTexTS columns.
COLUMN_MAPPING = {
    "ticker": None,
    "date": None,
    "open": None,
    "high": None,
    "low": None,
    "close": None,
}

# Optional explicit text columns. Leave empty to auto-detect.
TEXT_COLUMNS = {
    "macro": [],
    "sector": [],
    "related": [],
    "target": [],
}

TEXT_KEYWORDS = {
    "macro": ["macro", "market", "economy", "economic"],
    "sector": ["sector", "industry"],
    "related": ["related", "peer", "supply", "customer", "competitor"],
    "target": ["target", "company", "firm", "stock", "ticker", "news", "text"],
}

TRAIN_YEARS = (2019, 2021)
VAL_YEAR = 2022
TEST_YEAR = 2023
