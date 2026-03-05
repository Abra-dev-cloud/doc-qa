"""
app/services/qa_service.py — RAG pipeline: retrieve context, prompt LLM, return answer.

The pipeline:
1. Embed the user's question.
2. Retrieve top-k relevant chunks from ChromaDB.
3. Build a structured prompt with retrieved context.
4. Call the LLM and parse the response.
5. Return answer + source citations.

Prompt engineering notes:
- System prompt establishes the assistant's role and constraints.
- Context is injected as numbered passages for easy citation reference.
- Temperature is low (0.2) to reduce hallucination on factual Q&A.
- Explicit instruction: "If the answer is not in the context, say so."
"""

import logging
from dataclasses import dataclass
from functools import lru_cache

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from app.config import get_settings
from app.core.embedder import embed_query
from app.services.vector_store import RetrievedChunk, query_collection

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise document Q&A assistant.
You will be given a user question and a set of numbered context passages extracted from a document.

Your rules:
1. Answer ONLY based on the provided context passages. Do not use outside knowledge.
2. If the answer cannot be found in the context, respond: "I could not find an answer to that question in the provided document."
3. Be concise but complete. Prefer bullet points for multi-part answers.
4. When you use information from a specific passage, cite it as [Passage N].
5. Never fabricate facts, statistics, or quotes.
"""


@dataclass
class SourceCitation:
    chunk_id: str
    page: int
    text_preview: str      # First 200 chars of the chunk
    relevance_score: float


@dataclass
class QAResponse:
    answer: str
    sources: list[SourceCitation]
    collection: str
    chunks_used: int
    question: str


@lru_cache(maxsize=1)
def _get_openai_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key)


def _build_context_block(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks as numbered passages for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        parts.append(
            f"[Passage {i}] (Source: {chunk.source_file}, Page {chunk.page_number})\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


@retry(
    wait=wait_exponential(multiplier=1, min=2, max=20),
    stop=stop_after_attempt(3),
    reraise=True,
)
def _call_llm(messages: list[dict]) -> str:
    """Call OpenAI chat completions with retry logic."""
    settings = get_settings()
    client = _get_openai_client()

    response = client.chat.completions.create(
        model=settings.llm_model,
        messages=messages,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )
    return response.choices[0].message.content.strip()


def answer_question(
    question: str,
    collection_name: str,
    top_k: int | None = None,
) -> QAResponse:
    """
    Full RAG pipeline: embed question → retrieve → prompt → answer.

    Args:
        question:        The user's natural language question.
        collection_name: ChromaDB collection to search.
        top_k:           Override default number of chunks to retrieve.

    Returns:
        QAResponse with answer text and source citations.

    Raises:
        ValueError: If collection is empty or question is blank.
    """
    settings = get_settings()
    k = top_k if top_k is not None else settings.default_top_k

    if not question.strip():
        raise ValueError("Question cannot be empty.")

    logger.info("Q&A request | collection='%s' question='%s'", collection_name, question)

    # ── Step 1: Embed the question ───────────────────────────────────────────
    query_vector = embed_query(question)

    # ── Step 2: Retrieve relevant chunks ────────────────────────────────────
    chunks = query_collection(
        query_embedding=query_vector,
        collection_name=collection_name,
        top_k=k,
    )

    logger.info("Retrieved %d chunks (top relevance: %.3f)", len(chunks), chunks[0].relevance_score if chunks else 0)

    # ── Step 3: Build prompt ────────────────────────────────────────────────
    context_block = _build_context_block(chunks)

    user_message = (
        f"Context passages from the document:\n\n"
        f"{context_block}\n\n"
        f"---\n\n"
        f"Question: {question}"
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # ── Step 4: Call the LLM ────────────────────────────────────────────────
    answer = _call_llm(messages)

    # ── Step 5: Build citation objects ──────────────────────────────────────
    sources = [
        SourceCitation(
            chunk_id=chunk.chunk_id,
            page=chunk.page_number,
            text_preview=chunk.text[:200] + ("..." if len(chunk.text) > 200 else ""),
            relevance_score=chunk.relevance_score,
        )
        for chunk in chunks
    ]

    return QAResponse(
        answer=answer,
        sources=sources,
        collection=collection_name,
        chunks_used=len(chunks),
        question=question,
    )
