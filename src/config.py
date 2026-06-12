"""Configuration for the FinTexTS volatility pilot."""

from pathlib import Path

DATASET_NAME = "EXAONE-BI/FinTexTS"

EPSILON = 1e-12
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("cache")
RANDOM_SEED = 42

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

EVENT_KEYWORDS = {
    "earnings": ["earnings", "revenue", "profit", "loss", "guidance", "forecast", "outlook"],
    "regulatory_legal": ["lawsuit", "investigation", "regulator", "SEC", "antitrust", "fine", "penalty"],
    "corporate_action": ["merger", "acquisition", "takeover", "buyback", "dividend", "bankruptcy"],
    "product_operations": ["recall", "disruption", "shortage", "launch", "delay", "supply chain"],
    "analyst_action": ["downgrade", "upgrade", "rating", "target price"],
    "macro_shock": ["Federal Reserve", "Fed", "interest rate", "inflation", "recession", "CPI", "employment"],
}

TEXT_CONFIGURATIONS = [
    "no_text",
    "macro_only",
    "sector_only",
    "related_only",
    "target_only",
    "macro_sector",
    "target_sector",
    "target_related",
    "all_text",
    "levelwise_text",
]

TRAIN_YEARS = (2019, 2021)
VAL_YEAR = 2022
TEST_YEAR = 2023
