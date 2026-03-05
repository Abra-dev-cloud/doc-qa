"""
app/api/documents.py — FastAPI router for PDF upload and collection management.

Endpoints:
  POST   /documents/upload               Upload and ingest a PDF
  GET    /documents/collections          List all collections
  DELETE /documents/collections/{name}   Delete a collection
"""

import logging
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from pydantic import BaseModel

from app.config import get_settings
from app.core.chunker import chunk_document
from app.core.embedder import embed_texts
from app.core.pdf_parser import parse_pdf_bytes
from app.services.vector_store import delete_collection, list_collections, upsert_chunks

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Documents"])


# ── Response schemas ──────────────────────────────────────────────────────────

class IngestResponse(BaseModel):
    message: str
    filename: str
    collection: str
    total_pages: int
    total_chunks: int
    total_tokens: int


class CollectionInfo(BaseModel):
    name: str
    count: int


class CollectionsResponse(BaseModel):
    collections: list[CollectionInfo]
    total: int


class DeleteResponse(BaseModel):
    message: str
    collection: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a PDF",
    description=(
        "Upload a PDF file. The system will parse it, chunk it into overlapping "
        "token windows, embed each chunk using OpenAI, and store the vectors in "
        "the specified ChromaDB collection."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    collection_name: str = Form(
        ...,
        description="Name of the collection to store the document in",
        min_length=1,
        max_length=63,
    ),
) -> IngestResponse:
    # ── Validate file type ────────────────────────────────────────────────────
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Only PDF files are supported.",
        )

    if file.content_type and file.content_type not in ("application/pdf", "application/octet-stream"):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid content type: {file.content_type}. Expected application/pdf.",
        )

    # ── Read file bytes ───────────────────────────────────────────────────────
    try:
        pdf_bytes = await file.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to read uploaded file: {exc}",
        )

    if len(pdf_bytes) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    logger.info(
        "Ingesting '%s' (%d bytes) into collection '%s'",
        file.filename,
        len(pdf_bytes),
        collection_name,
    )

    # ── Parse PDF ─────────────────────────────────────────────────────────────
    try:
        parsed = parse_pdf_bytes(pdf_bytes, filename=file.filename)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        )

    if not parsed.full_text.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="No text could be extracted from this PDF. It may be scanned/image-only.",
        )

    # ── Chunk ─────────────────────────────────────────────────────────────────
    settings = get_settings()
    chunking_result = chunk_document(
        document=parsed,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    if not chunking_result.chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Document produced no chunks after parsing.",
        )

    # ── Embed ─────────────────────────────────────────────────────────────────
    try:
        texts = [chunk.text for chunk in chunking_result.chunks]
        embeddings = embed_texts(texts)
    except Exception as exc:
        logger.exception("Embedding failed")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"OpenAI embedding failed: {exc}",
        )

    # ── Store in ChromaDB ─────────────────────────────────────────────────────
    try:
        upserted = upsert_chunks(
            chunks=chunking_result.chunks,
            embeddings=embeddings,
            collection_name=collection_name,
        )
    except Exception as exc:
        logger.exception("ChromaDB upsert failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Vector store write failed: {exc}",
        )

    total_tokens = sum(c.token_count for c in chunking_result.chunks)

    return IngestResponse(
        message="Document ingested successfully.",
        filename=file.filename,
        collection=collection_name,
        total_pages=parsed.total_pages,
        total_chunks=upserted,
        total_tokens=total_tokens,
    )


@router.get(
    "/collections",
    response_model=CollectionsResponse,
    summary="List all collections",
)
async def get_collections() -> CollectionsResponse:
    collections_raw = list_collections()
    collections = [CollectionInfo(**c) for c in collections_raw]
    return CollectionsResponse(collections=collections, total=len(collections))


@router.delete(
    "/collections/{collection_name}",
    response_model=DeleteResponse,
    summary="Delete a collection and all its embeddings",
)
async def remove_collection(collection_name: str) -> DeleteResponse:
    deleted = delete_collection(collection_name)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Collection '{collection_name}' not found.",
        )
    return DeleteResponse(
        message="Collection deleted successfully.",
        collection=collection_name,
    )
