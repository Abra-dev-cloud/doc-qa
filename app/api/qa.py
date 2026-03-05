"""
app/api/qa.py — FastAPI router for question answering.

Endpoints:
  POST /qa/ask    Ask a question against an ingested collection
"""

import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services.qa_service import QAResponse, answer_question

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/qa", tags=["Question Answering"])


# ── Request / Response schemas ────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="The question to answer from the document.",
        examples=["What are the main conclusions of this report?"],
    )
    collection_name: str = Field(
        ...,
        min_length=1,
        max_length=63,
        description="The ChromaDB collection to search (must be already ingested).",
        examples=["annual_report_2024"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve as context.",
    )


class SourceCitationOut(BaseModel):
    chunk_id: str
    page: int
    text_preview: str
    relevance_score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceCitationOut]
    collection: str
    chunks_used: int
    question: str


# ── Endpoint ──────────────────────────────────────────────────────────────────

@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a question",
    description=(
        "Ask a natural language question against a previously ingested document collection. "
        "The system retrieves relevant chunks via semantic similarity, then uses an LLM "
        "to synthesize a grounded answer with source citations."
    ),
)
async def ask_question(request: AskRequest) -> AskResponse:
    settings = get_settings()
    effective_top_k = request.top_k or settings.default_top_k

    try:
        result: QAResponse = answer_question(
            question=request.question,
            collection_name=request.collection_name,
            top_k=effective_top_k,
        )
    except ValueError as exc:
        # Empty collection, empty question, etc.
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )
    except Exception as exc:
        logger.exception("QA pipeline error")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"QA pipeline error: {exc}",
        )

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceCitationOut(
                chunk_id=s.chunk_id,
                page=s.page,
                text_preview=s.text_preview,
                relevance_score=s.relevance_score,
            )
            for s in result.sources
        ],
        collection=result.collection,
        chunks_used=result.chunks_used,
        question=result.question,
    )
