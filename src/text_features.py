"""Text column validation, normalization, diagnostics, and embeddings."""

from __future__ import annotations

import hashlib
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .config import (
    DEFAULT_CACHE_DIR,
    DEFAULT_EMBEDDING_MODEL,
    EVENT_KEYWORDS,
    FILING_TEXT_GROUP,
    NEWS_TEXT_GROUPS,
    TEXT_CACHE_VERSION,
    TEXT_COLUMNS,
    TEXT_KEYWORDS,
)

NEWS_LEVELS = ["macro", "sector", "related", "target"]
TEXT_MODES = ["no_text", "news_all", "target_sector", "levelwise_news"]
BACKWARD_COMPAT_TEXT_MODES = {
    "all_text": "news_all",
    "target_sector_text": "target_sector",
    "levelwise_text": "levelwise_news",
}


def normalize_text_value(value: Any) -> str:
    """Normalize FinTexTS text cells without creating fake None/nan/null tokens."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    text = value if isinstance(value, str) else str(value)
    text = text.strip()
    if not text or text.lower() in {"none", "nan", "null"}:
        return ""
    return text


def combine_text_columns(row: pd.Series, columns: list[str], separator: str = "\n") -> str:
    """Combine configured columns, dropping empty and exact duplicate text."""
    seen: set[str] = set()
    parts: list[str] = []
    for col in columns:
        if col not in row.index:
            continue
        text = normalize_text_value(row[col])
        if text and text not in seen:
            seen.add(text)
            parts.append(text)
    return separator.join(parts)


def _combine_frame(df: pd.DataFrame, columns: list[str], separator: str = "\n") -> pd.Series:
    if not columns:
        return pd.Series([""] * len(df), index=df.index)
    usable = [col for col in columns if col in df.columns]
    if not usable:
        return pd.Series([""] * len(df), index=df.index)
    return df[usable].apply(lambda row: combine_text_columns(row, usable, separator), axis=1)


def expected_text_columns() -> list[str]:
    return [col for cols in TEXT_COLUMNS.values() for col in cols]


def validate_text_schema(df: pd.DataFrame, config: dict[str, list[str]] | None = None) -> dict[str, list[Any]]:
    """Validate configured FinTexTS text columns and report fallback candidates."""
    config = config or TEXT_COLUMNS
    expected = [col for cols in config.values() for col in cols]
    found = [col for col in expected if col in df.columns]
    missing = [col for col in expected if col not in df.columns]
    configured = set(expected)
    object_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]
    keyword_pool = sorted({kw.lower() for kws in TEXT_KEYWORDS.values() for kw in kws} | {"filing", "category"})
    extra = [col for col in object_cols if col not in configured and any(kw in col.lower() for kw in keyword_pool)]
    report = {
        "expected_columns": expected,
        "found_columns": found,
        "missing_columns": missing,
        "extra_possible_text_columns": extra,
    }
    if missing:
        message = (
            "Configured FinTexTS text columns are missing. "
            f"Expected={expected}; found={found}; missing={missing}; extra_possible_text_columns={extra}"
        )
        if len(missing) > 5:
            raise ValueError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    if extra:
        warnings.warn(f"Possible text columns outside explicit config: {extra}", RuntimeWarning, stacklevel=2)
    return report


def write_text_schema_validation(df: pd.DataFrame, output_dir: str | Path, config: dict[str, list[str]] | None = None) -> dict[str, list[Any]]:
    report = validate_text_schema(df, config)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "text_schema_validation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def detect_text_columns(df: pd.DataFrame, allow_fallback: bool = True) -> dict[str, list[str]]:
    """Return explicit configured columns, falling back only when explicit columns are absent."""
    detected: dict[str, list[str]] = {}
    object_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]
    lower_cols = {col: col.lower() for col in object_cols}
    for level, configured in TEXT_COLUMNS.items():
        found = [col for col in configured if col in df.columns]
        if found:
            detected[level] = found
            continue
        if not allow_fallback:
            detected[level] = []
            continue
        keywords = TEXT_KEYWORDS.get(level, [level])
        fallback = [col for col, low in lower_cols.items() if any(keyword.lower() in low for keyword in keywords)]
        if fallback:
            warnings.warn(f"Using fallback text column detection for {level}: {fallback}", RuntimeWarning, stacklevel=2)
        detected[level] = fallback
    return detected


def text_columns_for_configuration(text_columns: dict[str, list[str]], configuration: str) -> list[str]:
    configuration = BACKWARD_COMPAT_TEXT_MODES.get(configuration, configuration)
    config_levels = {
        "macro_only": ["macro"],
        "sector_only": ["sector"],
        "related_only": ["related"],
        "target_only": ["target"],
        "target_sector": ["target", "sector"],
        "news_all": NEWS_LEVELS,
        "filing_only": ["filing"],
        "news_plus_filing": NEWS_LEVELS + ["filing"],
        "levelwise_news": NEWS_LEVELS,
        "levelwise_news_plus_filing": NEWS_LEVELS + ["filing"],
    }
    if configuration == "no_text":
        return []
    if configuration not in config_levels:
        raise ValueError(f"Unknown text configuration: {configuration}")
    cols: list[str] = []
    for level in config_levels[configuration]:
        cols.extend(text_columns.get(level, []))
    return [col for col in cols if col in set(cols)]


def joined_text_for_configuration(df: pd.DataFrame, text_columns: dict[str, list[str]], configuration: str) -> pd.Series:
    configuration = BACKWARD_COMPAT_TEXT_MODES.get(configuration, configuration)
    return _combine_frame(df, text_columns_for_configuration(text_columns, configuration))


def build_text_by_mode(df: pd.DataFrame, text_columns: dict[str, list[str]], mode: str) -> list[pd.Series]:
    mode = BACKWARD_COMPAT_TEXT_MODES.get(mode, mode)
    if mode == "no_text":
        return []
    if mode in {"news_all", "target_sector", "filing_only", "news_plus_filing"}:
        return [joined_text_for_configuration(df, text_columns, mode)]
    if mode == "levelwise_news":
        return [_combine_frame(df, text_columns.get(level, [])) for level in NEWS_LEVELS]
    if mode == "levelwise_news_plus_filing":
        return [_combine_frame(df, text_columns.get(level, [])) for level in NEWS_LEVELS + ["filing"]]
    if mode.endswith("_only"):
        return [joined_text_for_configuration(df, text_columns, mode)]
    raise ValueError(f"Unknown text mode: {mode}")


def _nonempty_count_frame(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    usable = [col for col in columns if col in df.columns]
    if not usable:
        return pd.Series(0, index=df.index)
    return df[usable].apply(lambda row: sum(1 for col in usable if normalize_text_value(row[col])), axis=1)


def add_text_presence_flags(df: pd.DataFrame, text_columns: dict[str, list[str]], ticker_col: str | None = None) -> pd.DataFrame:
    out = df.copy()
    for level in NEWS_LEVELS:
        cols = text_columns.get(level, [])
        combined = _combine_frame(out, cols)
        out[f"has_{level}_text"] = combined.ne("").astype(int)
        out[f"has_text_{level}"] = out[f"has_{level}_text"]
        out[f"{level}_text_count"] = _nonempty_count_frame(out, cols).astype(int)
        out[f"text_length_{level}"] = combined.str.len().astype(int)

    filing = _combine_frame(out, text_columns.get("filing", []))
    out["has_filing_context"] = filing.ne("").astype(int)
    out["filing_text_length_total"] = filing.str.len().astype(int)
    filing_hash = filing.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "")
    if ticker_col and ticker_col in out.columns:
        out["filing_context_changed"] = filing_hash.ne(filing_hash.groupby(out[ticker_col], sort=False).shift(1)).astype(int)
        out.loc[filing_hash.eq(""), "filing_context_changed"] = 0
    else:
        out["filing_context_changed"] = filing_hash.ne(filing_hash.shift(1)).astype(int)
        out.loc[filing_hash.eq(""), "filing_context_changed"] = 0

    out["has_any_news_text"] = out[[f"has_{level}_text" for level in NEWS_LEVELS]].any(axis=1).astype(int)
    out["has_any_text"] = out["has_any_news_text"]  # Backward compatible alias; filing intentionally excluded.
    out["news_text_count_total"] = out[[f"{level}_text_count" for level in NEWS_LEVELS]].sum(axis=1).astype(int)
    out["num_text_levels_present"] = out[[f"has_{level}_text" for level in NEWS_LEVELS]].sum(axis=1).astype(int)
    out["news_text_length_total"] = out[[f"text_length_{level}" for level in NEWS_LEVELS]].sum(axis=1).astype(int)
    out["total_text_length"] = out["news_text_length_total"]  # Backward compatible alias; filing intentionally excluded.
    return out


def add_event_keyword_features(df: pd.DataFrame, text_columns: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    news_text = joined_text_for_configuration(out, text_columns, "news_all").str.lower()
    target_text = joined_text_for_configuration(out, text_columns, "target_only").str.lower()
    total_hits = pd.Series(0, index=out.index)
    target_hits = pd.Series(0, index=out.index)
    for group, keywords in EVENT_KEYWORDS.items():
        pattern = "|".join([keyword.lower().replace(" ", r"\s+") for keyword in keywords])
        out[f"event_keyword_{group}"] = news_text.str.contains(pattern, regex=True, na=False)
        out[f"target_event_keyword_{group}"] = target_text.str.contains(pattern, regex=True, na=False)
        total_hits += out[f"event_keyword_{group}"].astype(int)
        target_hits += out[f"target_event_keyword_{group}"].astype(int)
    out["event_keyword_count"] = total_hits
    out["target_event_keyword_count"] = target_hits
    out["has_event_keyword"] = total_hits.gt(0)
    out["has_target_event_keyword"] = target_hits.gt(0)
    return out


def text_duplicate_diagnostics(df: pd.DataFrame, text_columns: dict[str, list[str]], ticker_col: str, date_col: str) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        for group, cols in text_columns.items():
            values = [normalize_text_value(row[col]) for col in cols if col in row.index]
            values = [value for value in values if value]
            rows.append(
                {
                    "ticker": row.get(ticker_col),
                    "date": row.get(date_col),
                    "group": group,
                    "raw_nonempty_count": len(values),
                    "unique_nonempty_count": len(set(values)),
                    "duplicate_count": len(values) - len(set(values)),
                }
            )
    return pd.DataFrame(rows)


def filing_context_diagnostics(df: pd.DataFrame, text_columns: dict[str, list[str]], ticker_col: str, date_col: str) -> pd.DataFrame:
    filing_cols = text_columns.get("filing", FILING_TEXT_GROUP)
    rows = []
    work = df.sort_values([ticker_col, date_col]).copy()
    combined = _combine_frame(work, filing_cols)
    combined_hash = combined.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "")
    for col in filing_cols:
        if col not in work.columns:
            rows.append({"filing_column": col, "missing_rate": 1.0, "content_change_count": 0, "mean_text_length": 0.0, "median_same_content_run_days": np.nan})
            continue
        normalized = work[col].map(normalize_text_value)
        col_hash = normalized.map(lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest() if text else "")
        changes = col_hash.groupby(work[ticker_col], sort=False).transform(lambda s: s.ne(s.shift(1))).astype(int)
        run_lengths = []
        for _, series in col_hash.groupby(work[ticker_col], sort=False):
            run_id = series.ne(series.shift(1)).cumsum()
            run_lengths.extend(series.groupby(run_id).size().tolist())
        rows.append(
            {
                "filing_column": col,
                "missing_rate": float(normalized.eq("").mean()),
                "content_change_count": int(changes[normalized.ne("")].sum()),
                "mean_text_length": float(normalized.str.len().mean()),
                "median_same_content_run_days": float(np.median(run_lengths)) if run_lengths else np.nan,
            }
        )
    rows.append(
        {
            "filing_column": "__combined_filing_context__",
            "missing_rate": float(combined.eq("").mean()),
            "content_change_count": int(combined_hash.groupby(work[ticker_col], sort=False).transform(lambda s: s.ne(s.shift(1))).sum()),
            "mean_text_length": float(combined.str.len().mean()),
            "median_same_content_run_days": np.nan,
        }
    )
    return pd.DataFrame(rows)


def _cache_path(
    cache_dir: Path,
    model_name: str,
    texts: pd.Series,
    pca_dim: int | None,
    suffix: str,
    row_identity: pd.Series | None = None,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    normalized = texts.map(normalize_text_value)
    identity = row_identity.fillna("").astype(str).tolist() if row_identity is not None else [str(idx) for idx in texts.index]
    payload = {
        "cache_version": TEXT_CACHE_VERSION,
        "model": model_name,
        "pca_dim": pca_dim,
        "text_configuration": suffix,
        "row_identity_hash": hashlib.sha256("\n".join(identity).encode("utf-8")).hexdigest(),
        "normalized_text_hash": hashlib.sha256("\n".join(normalized.tolist()).encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"embeddings_{TEXT_CACHE_VERSION}_{digest}.npy"


def _raw_cache_path(cache_dir: Path, model_name: str, texts: pd.Series, suffix: str, row_identity: pd.Series | None = None) -> Path:
    return _cache_path(cache_dir, model_name, texts, None, f"raw_{suffix}", row_identity)


def _encode_series(
    texts: pd.Series,
    model: SentenceTransformer,
    model_name: str,
    cache_dir: Path,
    pca_dim: int | None,
    suffix: str,
    row_identity: pd.Series | None = None,
) -> np.ndarray:
    path = _cache_path(cache_dir, model_name, texts, pca_dim, suffix, row_identity)
    if path.exists():
        return np.load(path)

    normalized = texts.map(normalize_text_value)
    raw = model.encode(normalized.tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    if pca_dim and 0 < pca_dim < raw.shape[1]:
        n_components = min(pca_dim, raw.shape[0], raw.shape[1])
        raw = PCA(n_components=n_components, random_state=42).fit_transform(raw)
    np.save(path, raw)
    return raw


def encode_text_raw(
    texts: pd.Series,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    suffix: str = "detail",
    row_identity: pd.Series | None = None,
) -> np.ndarray:
    """Encode normalized text with cache, without PCA, so PCA can be train-only."""
    cache_path = _raw_cache_path(Path(cache_dir), embedding_model, texts, suffix, row_identity)
    if cache_path.exists():
        return np.load(cache_path)
    model = SentenceTransformer(embedding_model)
    normalized = texts.map(normalize_text_value)
    matrix = model.encode(normalized.tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return matrix


def train_only_pca_features(
    raw_matrix: np.ndarray,
    train_index: pd.Index,
    pca_dim: int | None,
    prefix: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str]]:
    """Fit PCA on train rows only, then transform all rows."""
    if raw_matrix.size == 0:
        return pd.DataFrame(index=train_index), []
    n_features = raw_matrix.shape[1]
    if pca_dim and 0 < pca_dim < n_features:
        n_components = min(pca_dim, len(train_index), n_features)
        pca = PCA(n_components=n_components, random_state=random_state)
        pca.fit(raw_matrix[train_index.to_numpy()])
        matrix = pca.transform(raw_matrix)
    else:
        matrix = raw_matrix
    names = [f"{prefix}_emb_{i}" for i in range(matrix.shape[1])]
    return pd.DataFrame(matrix, columns=names), names


def keyword_filtered_text(
    df: pd.DataFrame,
    text_columns: dict[str, list[str]],
    keyword_group: str | None = None,
    target_only: bool = False,
) -> pd.Series:
    config = "target_only" if target_only else "news_all"
    text = joined_text_for_configuration(df, text_columns, config)
    if keyword_group is None:
        keywords = [kw for values in EVENT_KEYWORDS.values() for kw in values]
    else:
        keywords = EVENT_KEYWORDS.get(keyword_group, [])
    if not keywords:
        return pd.Series([""] * len(df), index=df.index)
    pattern = "|".join([keyword.lower().replace(" ", r"\s+") for keyword in keywords])
    mask = text.str.lower().str.contains(pattern, regex=True, na=False)
    return text.where(mask, "")


def make_text_features(
    df: pd.DataFrame,
    mode: str,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    pca_dim: int | None = 64,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
) -> tuple[np.ndarray | None, list[str], dict[str, list[str]]]:
    original_mode = mode
    mode = BACKWARD_COMPAT_TEXT_MODES.get(mode, mode)
    if original_mode != mode:
        warnings.warn(f"Text mode {original_mode!r} is deprecated; using {mode!r}. Filing is excluded unless requested.", RuntimeWarning, stacklevel=2)
    text_columns = detect_text_columns(df)
    parts = build_text_by_mode(df, text_columns, mode)
    if not parts:
        return None, [], text_columns

    identity = pd.Series([str(idx) for idx in df.index], index=df.index)
    model = SentenceTransformer(embedding_model)
    arrays = [
        _encode_series(part, model, embedding_model, Path(cache_dir), pca_dim, f"{mode}_{idx}", identity)
        for idx, part in enumerate(parts)
    ]
    matrix = np.concatenate(arrays, axis=1)
    names = [f"{mode}_emb_{i}" for i in range(matrix.shape[1])]
    return matrix, names, text_columns
