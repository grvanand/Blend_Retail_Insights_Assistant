# ingestion/embedder.py
# Embeds CSV metadata descriptors using HuggingFace sentence-transformers.
# Produces numpy vectors ready for FAISS indexing.

import os
import numpy as np
from typing import List
from langchain_huggingface import HuggingFaceEmbeddings
from config.settings import settings
from utils.logger import logger
from utils.retry import io_retry


# ---------------------------------------------------------------------------
# Embedding model — singleton to avoid reloading on every call
# ---------------------------------------------------------------------------

_embedder: HuggingFaceEmbeddings | None = None


def get_embedder() -> HuggingFaceEmbeddings:
    """
    Lazy singleton loader for HuggingFace embedding model.
    Sets HF_TOKEN in env so private models are accessible if needed.
    """
    global _embedder

    if _embedder is None:
        # Ensure HF token is available in environment
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.hf_token

        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        _embedder = HuggingFaceEmbeddings(
            model_name=settings.embedding_model_name,
            model_kwargs={"device": "cpu"},        # switch to "cuda" if GPU available
            encode_kwargs={
                "batch_size": settings.embedding_batch_size,
                "normalize_embeddings": True,      # cosine similarity friendly
            },
        )
        logger.info("Embedding model loaded successfully.")

    return _embedder


@io_retry
def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of text strings.
    Returns a 2D numpy array of shape (n_texts, embedding_dim).

    Args:
        texts: List of strings to embed (metadata descriptors)

    Returns:
        np.ndarray: float32 embedding matrix
    """
    if not texts:
        raise ValueError("embed_texts received an empty list.")

    embedder = get_embedder()
    logger.info(f"Embedding {len(texts)} text(s)...")

    vectors = embedder.embed_documents(texts)
    matrix = np.array(vectors, dtype=np.float32)

    logger.info(f"Embedding complete — shape: {matrix.shape}")
    return matrix


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string for FAISS similarity search.

    Args:
        query: Natural language user query

    Returns:
        np.ndarray: 1D float32 vector, shape (embedding_dim,)
    """
    if not query or not query.strip():
        raise ValueError("embed_query received empty query string.")

    embedder = get_embedder()
    vector = embedder.embed_query(query)
    return np.array(vector, dtype=np.float32)
