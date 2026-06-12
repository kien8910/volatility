"""Configuration for the FinTexTS volatility pilot."""

from pathlib import Path

DATASET_NAME = "EXAONE-BI/FinTexTS"

EPSILON = 1e-12
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_CACHE_DIR = Path("cache")
RANDOM_SEED = 42
TEXT_CACHE_VERSION = "v2_explicit_fintexts_columns"

# Override these if automatic column detection does not match FinTexTS columns.
COLUMN_MAPPING = {
    "ticker": None,
    "date": None,
    "open": None,
    "high": None,
    "low": None,
    "close": None,
}

MACRO_TEXT_COLUMNS = [
    "macro_category1",
    "macro_category2",
    "macro_category3",
    "macro_category4",
    "macro_category5",
]

SECTOR_TEXT_COLUMNS = [
    "sector_category1",
    "sector_category2",
    "sector_category3",
    "sector_category4",
    "sector_category5",
]

TARGET_COMPANY_TEXT_COLUMNS = [
    "targetCompany_category1",
    "targetCompany_category2",
    "targetCompany_category3",
]

RELATED_COMPANY_TEXT_COLUMNS = [
    "relatedCompany_category1",
    "relatedCompany_category2",
    "relatedCompany_category3",
]

FILING_TEXT_COLUMNS = [
    "filing_financialStatement",
    "filing_governanceRisks",
    "filing_overviewProduct",
    "filing_recentEventCatalyst",
    "filing_strategyMarketOps",
]

NEWS_TEXT_GROUPS = {
    "macro": MACRO_TEXT_COLUMNS,
    "sector": SECTOR_TEXT_COLUMNS,
    "related": RELATED_COMPANY_TEXT_COLUMNS,
    "target": TARGET_COMPANY_TEXT_COLUMNS,
}

FILING_TEXT_GROUP = FILING_TEXT_COLUMNS

TEXT_COLUMNS = {**NEWS_TEXT_GROUPS, "filing": FILING_TEXT_GROUP}

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
    "target_sector",
    "news_all",
    "filing_only",
    "news_plus_filing",
    "levelwise_news",
    "levelwise_news_plus_filing",
]

TRAIN_YEARS = (2019, 2021)
VAL_YEAR = 2022
TEST_YEAR = 2023
