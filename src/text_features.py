"""Text column detection and embedding generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, EVENT_KEYWORDS, TEXT_COLUMNS, TEXT_KEYWORDS

TEXT_MODES = ["no_text", "all_text", "target_sector_text", "levelwise_text"]


def detect_text_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    detected: dict[str, list[str]] = {}
    object_cols = [col for col in df.columns if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col])]
    lower_cols = {col: col.lower() for col in object_cols}

    for level, configured in TEXT_COLUMNS.items():
        if configured:
            detected[level] = [col for col in configured if col in df.columns]
            continue
        keywords = TEXT_KEYWORDS[level]
        detected[level] = [col for col, low in lower_cols.items() if any(keyword in low for keyword in keywords)]
    return detected


def add_text_presence_flags(df: pd.DataFrame, text_columns: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    for level in ["macro", "sector", "related", "target"]:
        cols = text_columns.get(level, [])
        if not cols:
            out[f"has_text_{level}"] = False
            out[f"text_length_{level}"] = 0
            continue
        text = out[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip()
        out[f"has_text_{level}"] = text.ne("")
        out[f"text_length_{level}"] = text.str.len()
    out["has_any_text"] = out[[f"has_text_{level}" for level in ["macro", "sector", "related", "target"]]].any(axis=1)
    out["num_text_levels_present"] = out[[f"has_text_{level}" for level in ["macro", "sector", "related", "target"]]].sum(axis=1)
    out["total_text_length"] = out[[f"text_length_{level}" for level in ["macro", "sector", "related", "target"]]].sum(axis=1)
    return out


def _join_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1)


def text_columns_for_configuration(text_columns: dict[str, list[str]], configuration: str) -> list[str]:
    level_map = {
        "macro_only": ["macro"],
        "sector_only": ["sector"],
        "related_only": ["related"],
        "target_only": ["target"],
        "macro_sector": ["macro", "sector"],
        "target_sector": ["target", "sector"],
        "target_related": ["target", "related"],
        "all_text": ["macro", "sector", "related", "target"],
        "levelwise_text": ["macro", "sector", "related", "target"],
    }
    if configuration == "no_text":
        return []
    if configuration not in level_map:
        raise ValueError(f"Unknown text configuration: {configuration}")
    cols: list[str] = []
    for level in level_map[configuration]:
        cols.extend(text_columns.get(level, []))
    return sorted(set(cols))


def joined_text_for_configuration(df: pd.DataFrame, text_columns: dict[str, list[str]], configuration: str) -> pd.Series:
    return _join_columns(df, text_columns_for_configuration(text_columns, configuration))


def build_text_by_mode(df: pd.DataFrame, text_columns: dict[str, list[str]], mode: str) -> list[pd.Series]:
    if mode == "no_text":
        return []
    if mode == "all_text":
        cols = sorted({col for cols in text_columns.values() for col in cols})
        return [_join_columns(df, cols)]
    if mode == "target_sector_text":
        cols = text_columns.get("target", []) + text_columns.get("sector", [])
        return [_join_columns(df, cols)]
    if mode == "levelwise_text":
        return [_join_columns(df, text_columns.get(level, [])) for level in ["macro", "sector", "related", "target"]]
    raise ValueError(f"Unknown text mode: {mode}")


def add_event_keyword_features(df: pd.DataFrame, text_columns: dict[str, list[str]]) -> pd.DataFrame:
    out = df.copy()
    all_text = joined_text_for_configuration(out, text_columns, "all_text").str.lower()
    target_text = joined_text_for_configuration(out, text_columns, "target_only").str.lower()
    total_hits = pd.Series(0, index=out.index)
    target_hits = pd.Series(0, index=out.index)
    for group, keywords in EVENT_KEYWORDS.items():
        pattern = "|".join([keyword.lower().replace(" ", r"\s+") for keyword in keywords])
        out[f"event_keyword_{group}"] = all_text.str.contains(pattern, regex=True, na=False)
        out[f"target_event_keyword_{group}"] = target_text.str.contains(pattern, regex=True, na=False)
        total_hits += out[f"event_keyword_{group}"].astype(int)
        target_hits += out[f"target_event_keyword_{group}"].astype(int)
    out["event_keyword_count"] = total_hits
    out["target_event_keyword_count"] = target_hits
    out["has_event_keyword"] = total_hits.gt(0)
    out["has_target_event_keyword"] = target_hits.gt(0)
    return out


def _cache_path(cache_dir: Path, model_name: str, texts: pd.Series, pca_dim: int | None, suffix: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model_name,
        "pca_dim": pca_dim,
        "suffix": suffix,
        "texts_hash": hashlib.sha256("\n".join(texts.fillna("").astype(str).tolist()).encode("utf-8")).hexdigest(),
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"embeddings_{digest}.npy"


def _raw_cache_path(cache_dir: Path, model_name: str, texts: pd.Series, suffix: str) -> Path:
    return _cache_path(cache_dir, model_name, texts, None, f"raw_{suffix}")


def _encode_series(
    texts: pd.Series,
    model: SentenceTransformer,
    model_name: str,
    cache_dir: Path,
    pca_dim: int | None,
    suffix: str,
) -> np.ndarray:
    path = _cache_path(cache_dir, model_name, texts, pca_dim, suffix)
    if path.exists():
        return np.load(path)

    raw = model.encode(texts.fillna("").astype(str).tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
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
) -> np.ndarray:
    """Encode text with cache, without PCA, so PCA can be fit on train only."""
    cache_path = _raw_cache_path(Path(cache_dir), embedding_model, texts, suffix)
    if cache_path.exists():
        return np.load(cache_path)
    model = SentenceTransformer(embedding_model)
    matrix = model.encode(texts.fillna("").astype(str).tolist(), show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
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
    config = "target_only" if target_only else "all_text"
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
    text_columns = detect_text_columns(df)
    parts = build_text_by_mode(df, text_columns, mode)
    if not parts:
        return None, [], text_columns

    model = SentenceTransformer(embedding_model)
    arrays = [
        _encode_series(part, model, embedding_model, Path(cache_dir), pca_dim, f"{mode}_{idx}")
        for idx, part in enumerate(parts)
    ]
    matrix = np.concatenate(arrays, axis=1)
    names = [f"{mode}_emb_{i}" for i in range(matrix.shape[1])]
    return matrix, names, text_columns
