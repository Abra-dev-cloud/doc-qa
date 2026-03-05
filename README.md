# 📄 DocQA — PDF Question Answering System

> Upload a PDF. Ask anything. Get grounded answers with source citations.

A production-grade **Retrieval-Augmented Generation (RAG)** pipeline built with Python, OpenAI, ChromaDB, and FastAPI — with a clean dark UI served from the same process. Deployable locally or to **Vercel** in one command.

**Author:** [Abra-dev-cloud](https://github.com/Abra-dev-cloud)

---

## Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [RAG Pipeline — Step by Step](#rag-pipeline--step-by-step)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Quickstart — Local](#quickstart--local)
- [Deploy to Vercel](#deploy-to-vercel)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Design Decisions](#design-decisions)
- [Chunking Strategy](#chunking-strategy)
- [Prompt Engineering](#prompt-engineering)
- [Vercel Constraints & Workarounds](#vercel-constraints--workarounds)
- [Running Tests](#running-tests)
- [Extending the Project](#extending-the-project)
- [Roadmap](#roadmap)

---

## Overview

DocQA solves a common problem: **your knowledge is locked in PDFs**. Annual reports, research papers, legal contracts, technical manuals — documents that are painful to search and impossible to query semantically.

This system lets you:

1. **Ingest** any PDF via drag-and-drop UI or REST API
2. **Store** its meaning (not just its text) in a vector database
3. **Query** it in plain English and receive a concise, cited answer

The system is designed around **correctness over creativity** — it will tell you when it doesn't know something rather than hallucinate.

```
 ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
 │  PDF Upload │────▶│ Text Extract │────▶│   Chunking   │────▶│  Embed Chunks │
 └─────────────┘     └──────────────┘     └──────────────┘     └───────┬───────┘
                                                                        │
                                                                        ▼
 ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌───────────────┐
 │   Answer +  │◀────│  GPT-4o-mini │◀────│  Top-K Chunks│◀────│   ChromaDB    │
 │  Citations  │     │  (prompted)  │     │  (retrieved) │     │  Vector Store │
 └─────────────┘     └──────────────┘     └──────────────┘     └───────────────┘
                                                 ▲
                                    ┌────────────┴────────────┐
                                    │  Embed User Question    │
                                    └─────────────────────────┘
```

---

## Live Demo

> 🔗 **[docqa.vercel.app](https://docqa.vercel.app)** *(replace with your deployed URL)*

---

## Architecture

The codebase is split into three distinct layers, each with a single responsibility:

### Layer 1 — Core (`app/core/`)
Pure Python business logic with **no framework dependencies**. Each module can be imported and tested in isolation.

| Module | Responsibility |
|---|---|
| `pdf_parser.py` | Extracts text from PDFs page-by-page using PyMuPDF |
| `chunker.py` | Splits text into overlapping token windows using `tiktoken` |
| `embedder.py` | Calls OpenAI Embeddings API with batching + retry logic |

### Layer 2 — Services (`app/services/`)
Orchestration logic that wires the core modules together.

| Module | Responsibility |
|---|---|
| `vector_store.py` | All ChromaDB interactions — upsert, query, manage collections |
| `qa_service.py` | The full RAG pipeline: embed → retrieve → prompt → answer |

### Layer 3 — API + UI (`app/api/`, `frontend/`)
FastAPI routers exposing the services as HTTP endpoints, plus a single-file HTML frontend served at `/`.

| Module | Responsibility |
|---|---|
| `documents.py` | PDF upload, ingest, collection management |
| `qa.py` | Question answering |
| `frontend/index.html` | Dark UI — drag-and-drop upload + chat interface |

This separation means you can swap any layer independently — replace ChromaDB with Pinecone by only touching `vector_store.py`, or replace FastAPI with a CLI by only touching the API layer.

---

## RAG Pipeline — Step by Step

### Ingestion (PDF → Vectors)

**Step 1 — PDF Parsing**

PyMuPDF (`fitz`) opens the PDF and extracts text page-by-page. Each page's text is cleaned: excessive whitespace collapsed, null bytes removed, line endings normalised. The output is a `ParsedDocument` object containing per-page `PageContent` items.

```python
parsed = parse_pdf_bytes(pdf_bytes, filename="report.pdf")
# → ParsedDocument(total_pages=42, pages=[PageContent(page=1, text="..."), ...])
```

**Step 2 — Token-Aware Chunking**

Text is tokenised using `tiktoken` (the same tokeniser OpenAI uses internally). A sliding window of `CHUNK_SIZE` tokens advances by `(CHUNK_SIZE - CHUNK_OVERLAP)` tokens per step. This ensures:
- Every chunk fits within the embedding model's context window
- Adjacent chunks share `CHUNK_OVERLAP` tokens, preserving cross-boundary context
- Each chunk carries metadata: `source_file`, `page_number`, `chunk_index`

```python
result = chunk_document(parsed, chunk_size=512, chunk_overlap=200)
# → ChunkingResult(total_chunks=87, chunks=[TextChunk(...), ...])
```

**Step 3 — Embedding**

Chunks are batched (up to 512 per API call) and sent to OpenAI's `text-embedding-3-small` model. Each chunk becomes a 1536-dimensional float vector. The embedder uses `tenacity` for automatic exponential-backoff retry on rate limit and timeout errors.

```python
embeddings = embed_texts([chunk.text for chunk in chunks])
# → [[0.021, -0.013, 0.087, ...], ...]  shape: (87, 1536)
```

**Step 4 — Vector Store Upsert**

Chunks + embeddings + metadata are upserted into ChromaDB. The collection uses cosine distance (best for normalised OpenAI embeddings). Upsert is idempotent — re-ingesting the same document doesn't create duplicates.

---

### Querying (Question → Answer)

**Step 1 — Embed Question**

The user's question is embedded with the same model used during ingestion. This is critical — using a different model would destroy semantic alignment.

**Step 2 — Similarity Search**

The question vector is compared against all chunk vectors using cosine similarity. ChromaDB returns the top-K most relevant chunks with distance scores, converted to relevance scores `(1 - distance)`.

**Step 3 — Prompt Construction**

Retrieved chunks are formatted as numbered passages and injected into the user message alongside a strict system prompt.

**Step 4 — LLM Generation**

GPT-4o-mini generates the final answer. Temperature `0.2` — low enough to prevent hallucination while still allowing natural synthesis.

**Step 5 — Response**

Answer returned with source citations: `chunk_id`, `page_number`, 200-char text preview, and `relevance_score`.

---

## Tech Stack

| Component | Technology | Why |
|---|---|---|
| **API Framework** | FastAPI 0.111 | Async, auto OpenAPI docs, Pydantic validation |
| **Embeddings** | OpenAI `text-embedding-3-small` | Best perf/cost; 1536 dims |
| **LLM** | OpenAI `gpt-4o-mini` | Fast, cheap, grounded Q&A |
| **Vector DB** | ChromaDB 0.5 (local/ephemeral) | Zero infra; cosine similarity built-in |
| **PDF Parsing** | PyMuPDF (fitz) | Fastest Python PDF library |
| **Tokenisation** | tiktoken | Exact token counts matching OpenAI |
| **Retry Logic** | tenacity | Handles OpenAI rate limits declaratively |
| **Frontend** | Vanilla HTML/CSS/JS | No build step; serves from FastAPI |
| **Testing** | pytest + httpx TestClient | Full coverage, no API key needed |
| **Deployment** | Vercel (serverless Python) | Free tier, zero config |

---

## Project Structure

```
doc-qa/
│
├── app/
│   ├── main.py                  # FastAPI app + frontend serving
│   ├── config.py                # Pydantic Settings — all config from env
│   │
│   ├── core/                    # Pure logic, no framework deps
│   │   ├── pdf_parser.py        # PDF → ParsedDocument (per-page text)
│   │   ├── chunker.py           # ParsedDocument → overlapping TextChunks
│   │   └── embedder.py          # texts → float vectors (batched, retry)
│   │
│   ├── services/
│   │   ├── vector_store.py      # ChromaDB: upsert, query, collections
│   │   └── qa_service.py        # Full RAG pipeline + prompt engineering
│   │
│   └── api/
│       ├── documents.py         # POST /upload, GET/DELETE /collections
│       └── qa.py                # POST /ask
│
├── frontend/
│   └── index.html               # Dark UI — drag-and-drop + chat
│
├── tests/
│   ├── conftest.py
│   ├── test_chunker.py          # Unit tests — no external calls
│   ├── test_qa_service.py       # Mocked OpenAI + ChromaDB
│   └── test_api.py              # Full endpoint tests via TestClient
│
├── vercel.json                  # Vercel serverless config
├── .gitignore
├── .env.example
├── requirements.txt
├── Makefile
└── README.md
```

---

## Quickstart — Local

### 1. Clone

```bash
git clone https://github.com/Abra-dev-cloud/doc-qa.git
cd doc-qa
```

### 2. Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 3. Install

```bash
pip install -r requirements.txt
```

### 4. Configure

```bash
cp .env.example .env
# Open .env and set OPENAI_API_KEY=sk-...
```

### 5. Run

```bash
uvicorn app.main:app --reload --port 8000
```

Open **http://localhost:8000** — the UI loads immediately. Swagger API docs at **http://localhost:8000/docs**.

---

## Deploy to Vercel

### Prerequisites

- [Vercel account](https://vercel.com) (free)
- [Vercel CLI](https://vercel.com/docs/cli): `npm i -g vercel`

### One-command deploy

```bash
vercel
```

Vercel detects `vercel.json` automatically and configures the Python runtime.

### Set environment variable

In the Vercel dashboard → Project → Settings → Environment Variables:

```
OPENAI_API_KEY = sk-...your-key...
```

Or via CLI:
```bash
vercel env add OPENAI_API_KEY
```

### Redeploy

```bash
vercel --prod
```

Your API and UI are now live at `https://your-project.vercel.app`.

> **Note on persistence:** Vercel is serverless — ChromaDB runs in ephemeral (in-memory) mode. Collections reset between cold starts. For persistent storage, point `CHROMA_HTTP_HOST` at a hosted ChromaDB on Railway or Render. See [Vercel Constraints](#vercel-constraints--workarounds).

---

## Configuration

All settings are read from environment variables or `.env`.

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | **required** | Your OpenAI secret key |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | OpenAI embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Vector dimensionality |
| `LLM_MODEL` | `gpt-4o-mini` | Chat completion model |
| `LLM_TEMPERATURE` | `0.2` | Lower = more factual |
| `LLM_MAX_TOKENS` | `1024` | Max answer length |
| `CHUNK_SIZE` | `512` | Tokens per chunk |
| `CHUNK_OVERLAP` | `200` | Overlapping tokens between chunks |
| `DEFAULT_TOP_K` | `5` | Chunks to retrieve per query |
| `CHROMA_PERSIST_DIR` | `./chroma_db` | Local ChromaDB path |
| `CHROMA_EPHEMERAL` | `false` | Set `true` for serverless (auto-set by vercel.json) |

Upgrade to a stronger model:
```env
LLM_MODEL=gpt-4o
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIMENSIONS=3072
```

---

## API Reference

All endpoints are under `/api/v1`. Full interactive docs at `/docs`.

---

### `POST /api/v1/documents/upload`

Upload and ingest a PDF. Parses → chunks → embeds → stores in ChromaDB.

**Request** — `multipart/form-data`

| Field | Type | Description |
|---|---|---|
| `file` | File | PDF file (`.pdf` required) |
| `collection_name` | string | Target collection name |

**Response — 201**
```json
{
  "message": "Document ingested successfully.",
  "filename": "annual_report_2024.pdf",
  "collection": "financials",
  "total_pages": 48,
  "total_chunks": 312,
  "total_tokens": 159744
}
```

**cURL example**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@report.pdf" \
  -F "collection_name=financials"
```

---

### `POST /api/v1/qa/ask`

Ask a natural language question against an ingested collection.

**Request**
```json
{
  "question": "What was the revenue growth rate in Q3?",
  "collection_name": "financials",
  "top_k": 5
}
```

| Field | Constraints | Description |
|---|---|---|
| `question` | 3–1000 chars | Natural language question |
| `collection_name` | 1–63 chars | Must already be ingested |
| `top_k` | 1–20, default 5 | Chunks to retrieve as context |

**Response — 200**
```json
{
  "answer": "Q3 revenue grew by 18.4% year-over-year [Passage 2]. The CFO attributed this to expanded enterprise contracts [Passage 4].",
  "sources": [
    {
      "chunk_id": "annual_report_2024.pdf_chunk_47",
      "page": 12,
      "text_preview": "Q3 2024 showed strong revenue performance with 18.4% YoY growth...",
      "relevance_score": 0.931
    }
  ],
  "collection": "financials",
  "chunks_used": 5,
  "question": "What was the revenue growth rate in Q3?"
}
```

**cURL example**
```bash
curl -X POST "http://localhost:8000/api/v1/qa/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key risks?", "collection_name": "financials"}'
```

---

### `GET /api/v1/documents/collections`

List all collections and chunk counts.

```json
{
  "collections": [
    { "name": "financials", "count": 312 },
    { "name": "contracts",  "count": 89  }
  ],
  "total": 2
}
```

---

### `DELETE /api/v1/documents/collections/{name}`

Delete a collection and all its embeddings.

```json
{ "message": "Collection deleted successfully.", "collection": "financials" }
```

---

### `GET /health`

```json
{ "status": "ok", "version": "1.0.0" }
```

---

## Design Decisions

**Why ChromaDB over Pinecone / Weaviate?**

ChromaDB runs embedded in-process — zero infra, zero signup, zero cost. The abstraction is clean enough that swapping to Pinecone requires changing only `vector_store.py`.

**Why `text-embedding-3-small` over `ada-002`?**

5× cheaper than `text-embedding-ada-002` and outperforms it on MTEB benchmarks. The 1536-dimension output is identical shape, so it's a drop-in replacement.

**Why token-based chunking over sentence splitting?**

Sentence splitters break on bullet points, tables, technical abbreviations, and figures with captions. Token-based windowing with overlap is deterministic, language-agnostic, and precisely aligned with how the embedding model was trained.

**Why `chunk_overlap=200` with `chunk_size=512`?**

A 39% overlap ratio is high but deliberate. Key information often straddles natural text boundaries. The cost is ~1.6× more chunks than non-overlapping — acceptable.

**Why `temperature=0.2`?**

Document Q&A is a factual retrieval task. High temperature increases hallucination probability. At 0.2 the model synthesises naturally but stays closely grounded in retrieved context.

**Why separate `core/` and `services/` layers?**

`core/` modules are pure functions — data in, data out, no I/O. Trivially unit-testable with zero mocks. `services/` modules own I/O (OpenAI, ChromaDB). This separation means you can test chunking logic with 0 mocks and retrieval logic with 2 targeted mocks.

---

## Chunking Strategy

The chunker operates on **token IDs**, not characters or words:

```
1. Tokenise all page text with tiktoken (cl100k_base encoding)
2. Build a flat token array across all pages
3. Maintain a parallel page-number map: token_index → page_number
4. Slide a window of CHUNK_SIZE tokens, advancing by (CHUNK_SIZE - CHUNK_OVERLAP)
5. Decode each window back to a string
6. Attach page_number of the window's first token as metadata
```

Token count per chunk is **always ≤ CHUNK_SIZE** — guaranteed, not approximate.

---

## Prompt Engineering

The system prompt prevents the three most common RAG failure modes:

**Hallucination**
> "Answer ONLY based on the provided context passages. Do not use outside knowledge."

**Silent ignorance** (making something up when it doesn't know)
> "If the answer cannot be found in the context, respond: 'I could not find an answer to that question in the provided document.'"

**Uncited claims** (user can't verify the source)
> "When you use information from a specific passage, cite it as [Passage N]."

Context is injected as numbered passages with source metadata, giving the LLM unambiguous references to cite.

---

## Vercel Constraints & Workarounds

Vercel's serverless functions have three key constraints that affect this project:

**1. No persistent disk**

ChromaDB's `PersistentClient` writes to disk — which doesn't survive between function invocations on Vercel. The code automatically switches to `EphemeralClient` (in-memory) when `CHROMA_EPHEMERAL=true` (set in `vercel.json`).

*Workaround for persistence:* Deploy a ChromaDB HTTP server on [Railway](https://railway.app) or [Render](https://render.com) (both have free tiers) and set `CHROMA_HTTP_HOST` to point at it.

**2. 50MB function size limit**

PyMuPDF and ChromaDB are large. `vercel.json` sets `"maxLambdaSize": "50mb"`. If you hit the limit, replace PyMuPDF with `pypdf` (~3MB vs ~20MB).

**3. 30-second execution limit**

Ingesting a large PDF (500+ pages) may exceed 30s on cold start. Vercel Pro extends this to 300s. For large documents, implement async ingestion with a background queue.

---

## Running Tests

Tests run entirely without an OpenAI API key — all external calls are mocked.

```bash
pytest tests/ -v
# or: make test
```

| File | What it tests |
|---|---|
| `test_chunker.py` | Pure unit tests: chunk sizing, overlap, edge cases |
| `test_qa_service.py` | RAG pipeline with mocked `embed_query`, `query_collection`, `_call_llm` |
| `test_api.py` | Full HTTP tests via FastAPI `TestClient` |

---

## Extending the Project

**Swap the vector database**

Implement these four functions in `app/services/vector_store.py` against any backend (Pinecone, Weaviate, pgvector, Qdrant):
- `upsert_chunks(chunks, embeddings, collection_name)`
- `query_collection(query_embedding, collection_name, top_k)`
- `list_collections()`
- `delete_collection(collection_name)`

Zero changes elsewhere.

**Add streaming responses**

Replace the LLM call with `client.chat.completions.create(stream=True)` and use FastAPI's `StreamingResponse` with `text/event-stream`.

**Add hybrid search**

Combine BM25 keyword search with semantic search, then merge results with Reciprocal Rank Fusion. Better recall for queries with specific names, dates, or technical terms.

**Add authentication**

```python
from fastapi.security import HTTPBearer
security = HTTPBearer()

@router.post("/ask")
async def ask(request: AskRequest, token: str = Depends(security)):
    verify_token(token)
    ...
```

**Persistent storage on Vercel**

Deploy ChromaDB as an HTTP server on Railway:
1. `railway new` → deploy `chromadb/chroma` Docker image
2. Set `CHROMA_HTTP_HOST=https://your-chroma.railway.app` in Vercel env vars
3. Update `vector_store.py` to use `chromadb.HttpClient` when the env var is set

---

## Roadmap

- [ ] Async ingestion with job status polling
- [ ] Streaming answer endpoint (`text/event-stream`)
- [ ] Hybrid search: BM25 + semantic (Reciprocal Rank Fusion)
- [ ] Conversation memory — multi-turn Q&A
- [ ] Persistent ChromaDB via Railway/Render
- [ ] Multi-file ingestion (zip → ingest all PDFs)
- [ ] Docker Compose setup
- [ ] OpenAI Batch API for large-scale ingestion (50% cost reduction)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Built by [Abra-dev-cloud](https://github.com/Abra-dev-cloud)*
