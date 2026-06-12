"""Text column detection and embedding generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

from .config import DEFAULT_CACHE_DIR, DEFAULT_EMBEDDING_MODEL, TEXT_COLUMNS, TEXT_KEYWORDS

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
            continue
        out[f"has_text_{level}"] = out[cols].fillna("").astype(str).agg(" ".join, axis=1).str.strip().ne("")
    return out


def _join_columns(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    if not columns:
        return pd.Series([""] * len(df), index=df.index)
    return df[columns].fillna("").astype(str).agg(" ".join, axis=1)


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
