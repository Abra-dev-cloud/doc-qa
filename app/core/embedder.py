"""
app/core/embedder.py — Thin wrapper around OpenAI Embeddings API.

Features:
- Batched embedding calls (OpenAI allows up to 2048 inputs per request).
- Exponential-backoff retry via tenacity for transient API errors.
- Returns normalised float vectors ready for ChromaDB ingestion.
"""

import logging
from functools import lru_cache

from openai import OpenAI, APIError, RateLimitError, APITimeoutError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

from app.config import get_settings

logger = logging.getLogger(__name__)

# Maximum texts per API call (OpenAI hard limit is 2048)
_BATCH_SIZE = 512


@lru_cache(maxsize=1)
def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


@retry(
    retry=retry_if_exception_type((RateLimitError, APITimeoutError, APIError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(4),
    reraise=True,
)
def _embed_batch(texts: list[str], model: str, dimensions: int) -> list[list[float]]:
    """Call OpenAI embeddings API for a single batch of texts."""
    client = _get_client()
    response = client.embeddings.create(
        model=model,
        input=texts,
        dimensions=dimensions,
    )
    # Response items are ordered to match input order
    return [item.embedding for item in response.data]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Embed a list of texts using the configured OpenAI model.

    Args:
        texts: List of strings to embed. Empty strings are rejected.

    Returns:
        List of float vectors, same order and length as input.

    Raises:
        ValueError: If any text is empty.
        openai.APIError: If the API call fails after retries.
    """
    if not texts:
        return []

    # Validate — OpenAI will 400 on empty strings
    for i, t in enumerate(texts):
        if not t.strip():
            raise ValueError(f"Empty text at index {i}; cannot embed.")

    settings = get_settings()
    model = settings.embedding_model
    dimensions = settings.embedding_dimensions

    all_embeddings: list[list[float]] = []

    for batch_start in range(0, len(texts), _BATCH_SIZE):
        batch = texts[batch_start : batch_start + _BATCH_SIZE]
        logger.debug(
            "Embedding batch %d–%d / %d",
            batch_start,
            batch_start + len(batch) - 1,
            len(texts),
        )
        batch_embeddings = _embed_batch(batch, model, dimensions)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings


def embed_query(query: str) -> list[float]:
    """
    Embed a single query string.

    Args:
        query: The user's question or search phrase.

    Returns:
        Single float vector.
    """
    if not query.strip():
        raise ValueError("Query text cannot be empty.")
    return embed_texts([query])[0]
