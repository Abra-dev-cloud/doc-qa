"""
app/services/vector_store.py — ChromaDB operations: upsert, query, manage collections.

ChromaDB is embedded (no separate server needed) and persists to disk.
Each "collection" maps to one logical document set (e.g. one PDF or a topic).

Key design choices:
- Use cosine distance (most appropriate for normalised OpenAI embeddings).
- Metadata stored alongside vectors enables source citation in answers.
- Collection names are sanitised to avoid ChromaDB naming restrictions.
"""

import logging
import re
from dataclasses import dataclass
from functools import lru_cache

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import get_settings
from app.core.chunker import TextChunk

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source_file: str
    page_number: int
    chunk_index: int
    relevance_score: float   # 1 - cosine_distance → higher = more relevant


import os

def _get_chroma_client() -> chromadb.Client:
    """
    Return the appropriate ChromaDB client based on environment.

    - Locally: PersistentClient (survives restarts, data on disk)
    - Vercel / serverless: EphemeralClient (in-memory, per-request lifecycle)

    The CHROMA_EPHEMERAL env var is set automatically in vercel.json.
    For a production persistent store, point CHROMA_HTTP_HOST at a
    hosted ChromaDB instance (e.g. Railway, Render) instead.
    """
    if os.getenv("CHROMA_EPHEMERAL", "false").lower() == "true":
        return chromadb.EphemeralClient(
            settings=ChromaSettings(anonymized_telemetry=False)
        )
    settings = get_settings()
    return chromadb.PersistentClient(
        path=settings.chroma_persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _sanitise_collection_name(name: str) -> str:
    """
    ChromaDB collection names must be 3–63 chars, start/end with alphanumeric,
    contain only alphanumerics, underscores, or hyphens.
    """
    sanitised = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    sanitised = re.sub(r"_+", "_", sanitised).strip("_")
    sanitised = sanitised[:63]
    if len(sanitised) < 3:
        sanitised = sanitised.ljust(3, "x")
    return sanitised


def get_or_create_collection(collection_name: str) -> chromadb.Collection:
    """Return an existing ChromaDB collection or create it."""
    client = _get_chroma_client()
    safe_name = _sanitise_collection_name(collection_name)
    collection = client.get_or_create_collection(
        name=safe_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("Using collection '%s'", safe_name)
    return collection


def upsert_chunks(
    chunks: list[TextChunk],
    embeddings: list[list[float]],
    collection_name: str,
) -> int:
    """
    Upsert text chunks + their embeddings into ChromaDB.

    Args:
        chunks:          TextChunk objects from the chunker.
        embeddings:      Corresponding float vectors (same order/length).
        collection_name: Target collection name.

    Returns:
        Number of chunks upserted.

    Raises:
        ValueError: If chunks and embeddings lengths don't match.
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) must match"
        )
    if not chunks:
        return 0

    collection = get_or_create_collection(collection_name)

    ids = [chunk.chunk_id for chunk in chunks]
    documents = [chunk.text for chunk in chunks]
    metadatas = [
        {
            "source_file": chunk.source_file,
            "page_number": chunk.page_number,
            "chunk_index": chunk.chunk_index,
            "token_count": chunk.token_count,
        }
        for chunk in chunks
    ]

    # Upsert is idempotent — safe to re-ingest the same document
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    logger.info("Upserted %d chunks into '%s'", len(chunks), collection_name)
    return len(chunks)


def query_collection(
    query_embedding: list[float],
    collection_name: str,
    top_k: int = 5,
) -> list[RetrievedChunk]:
    """
    Retrieve the top-k most similar chunks for a query embedding.

    Args:
        query_embedding: Embedded question vector.
        collection_name: Collection to search.
        top_k:           Number of results to return.

    Returns:
        List of RetrievedChunk objects sorted by relevance (descending).

    Raises:
        ValueError: If the collection is empty or doesn't exist.
    """
    collection = get_or_create_collection(collection_name)

    count = collection.count()
    if count == 0:
        raise ValueError(
            f"Collection '{collection_name}' is empty. "
            "Upload and ingest a document first."
        )

    effective_k = min(top_k, count)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=effective_k,
        include=["documents", "metadatas", "distances"],
    )

    retrieved: list[RetrievedChunk] = []

    for i in range(len(results["ids"][0])):
        chunk_id = results["ids"][0][i]
        text = results["documents"][0][i]
        meta = results["metadatas"][0][i]
        # ChromaDB returns cosine *distance* (0=identical, 2=opposite)
        # Convert to similarity score in [0, 1]
        distance = results["distances"][0][i]
        relevance = max(0.0, 1.0 - distance)

        retrieved.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                text=text,
                source_file=meta.get("source_file", "unknown"),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                relevance_score=round(relevance, 4),
            )
        )

    # Sort descending by relevance (ChromaDB already does this, but be explicit)
    retrieved.sort(key=lambda c: c.relevance_score, reverse=True)
    return retrieved


def list_collections() -> list[dict]:
    """Return a list of all collection names and their document counts."""
    client = _get_chroma_client()
    collections = client.list_collections()
    return [
        {"name": col.name, "count": col.count()}
        for col in collections
    ]


def delete_collection(collection_name: str) -> bool:
    """
    Delete a collection and all its vectors.

    Returns:
        True if deleted, False if it didn't exist.
    """
    client = _get_chroma_client()
    safe_name = _sanitise_collection_name(collection_name)
    try:
        client.delete_collection(safe_name)
        logger.info("Deleted collection '%s'", safe_name)
        return True
    except Exception:
        return False
