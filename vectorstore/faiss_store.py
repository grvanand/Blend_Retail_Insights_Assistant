# vectorstore/faiss_store.py
# Builds and persists a FAISS index over CSV metadata descriptors.
# Used by query_resolver to route user queries to the correct CSV(s).

import os
import json
import faiss
import numpy as np
from pathlib import Path
from typing import List, Dict

from ingestion.embedder import embed_texts, embed_query
from config.settings import settings
from utils.logger import logger
from utils.retry import io_retry


# File paths for persisted index and metadata store
_INDEX_FILE = Path(settings.faiss_index_path) / "index.faiss"
_META_FILE  = Path(settings.faiss_index_path) / "metadata.json"


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

def build_index(descriptors: List[Dict]) -> None:
    """
    Build FAISS flat index from CSV metadata descriptors and persist to disk.

    Args:
        descriptors: Output of ingestion.loader.get_metadata_descriptors()
    """
    if not descriptors:
        raise ValueError("No descriptors provided to build_index.")

    # Extract embedding texts and metadata
    texts    = [d["embedding_text"] for d in descriptors]
    metadata = [
        {
            "filename":    d["filename"],
            "name":        d["name"],
            "description": d["description"],
            "key_columns": d["key_columns"],
            "row_count":   d["row_count"],
        }
        for d in descriptors
    ]

    # Generate embeddings
    vectors = embed_texts(texts)               # shape: (n, embedding_dim)
    dim     = vectors.shape[1]

    # Build FAISS flat L2 index (exact search — only 7 docs, no need for ANN)
    index = faiss.IndexFlatIP(dim)             # Inner Product = cosine sim (normalized vecs)
    index.add(vectors)
    logger.info(f"FAISS index built — {index.ntotal} vectors, dim={dim}")

    # Persist index and metadata
    _INDEX_FILE.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(_INDEX_FILE))
    _META_FILE.write_text(json.dumps(metadata, indent=2))
    logger.info(f"FAISS index persisted to: {_INDEX_FILE.parent}")


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

@io_retry
def load_index() -> tuple[faiss.Index, List[Dict]]:
    """
    Load persisted FAISS index and metadata from disk.

    Returns:
        (faiss.Index, list of metadata dicts)

    Raises:
        FileNotFoundError: if index files are missing (run build first)
    """
    if not _INDEX_FILE.exists() or not _META_FILE.exists():
        raise FileNotFoundError(
            f"FAISS index not found at '{_INDEX_FILE.parent}'. "
            "Run build_index() first or call ensure_index()."
        )

    index    = faiss.read_index(str(_INDEX_FILE))
    metadata = json.loads(_META_FILE.read_text())
    logger.info(f"FAISS index loaded — {index.ntotal} vectors")
    return index, metadata


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_index(user_query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve top-k most relevant CSV metadata entries for a user query.

    Args:
        user_query: Natural language question from the user
        top_k:      Number of results to return (defaults to settings.faiss_top_k)

    Returns:
        List of metadata dicts ranked by relevance, each containing:
        filename, name, description, key_columns, row_count, score
    """
    top_k = top_k or settings.faiss_top_k

    index, metadata = load_index()

    # Embed query — shape: (embedding_dim,) → reshape to (1, dim) for FAISS
    query_vector = embed_query(user_query).reshape(1, -1)

    # Search
    scores, indices = index.search(query_vector, min(top_k, index.ntotal))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:                          # FAISS returns -1 for empty slots
            continue
        entry = metadata[idx].copy()
        entry["score"] = float(score)
        results.append(entry)

    logger.info(
        f"FAISS query matched {len(results)} CSV(s): "
        f"{[r['name'] for r in results]}"
    )
    return results


# ---------------------------------------------------------------------------
# Utility — build only if index doesn't exist yet
# ---------------------------------------------------------------------------

def ensure_index(descriptors: List[Dict]) -> None:
    """
    Build index only if it doesn't already exist on disk.
    Called at app startup — safe to call every time.
    """
    if _INDEX_FILE.exists() and _META_FILE.exists():
        logger.info("FAISS index already exists — skipping rebuild.")
        return

    logger.info("FAISS index not found — building now...")
    build_index(descriptors)
