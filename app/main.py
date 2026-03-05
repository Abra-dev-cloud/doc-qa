"""
app/main.py — FastAPI application factory and entry point.

Run with:
    uvicorn app.main:app --reload --port 8000

Swagger UI: http://localhost:8000/docs
Frontend:   http://localhost:8000/
"""

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.documents import router as documents_router
from app.api.qa import router as qa_router
from app.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    logger.info("DocQA API starting up")
    logger.info("  LLM model:       %s", settings.llm_model)
    logger.info("  Embedding model: %s", settings.embedding_model)
    logger.info("  ChromaDB dir:    %s", settings.chroma_persist_dir)
    yield
    logger.info("DocQA API shutting down")


def create_app() -> FastAPI:
    app = FastAPI(
        title="DocQA — PDF Question Answering API",
        description=(
            "Upload PDF documents and ask natural language questions. "
            "Powered by OpenAI embeddings + ChromaDB vector search + GPT-4o-mini."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    API_PREFIX = "/api/v1"
    app.include_router(documents_router, prefix=API_PREFIX)
    app.include_router(qa_router, prefix=API_PREFIX)

    @app.get("/health", tags=["Meta"], summary="Health check")
    async def health():
        return {"status": "ok", "version": "1.0.0"}

    if FRONTEND_DIR.exists():
        @app.get("/", include_in_schema=False)
        async def serve_frontend():
            return FileResponse(FRONTEND_DIR / "index.html")
    else:
        @app.get("/", include_in_schema=False)
        async def root():
            return JSONResponse({"message": "DocQA API", "docs": "/docs"})

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    s = get_settings()
    uvicorn.run("app.main:app", host=s.api_host, port=s.api_port, reload=True)
