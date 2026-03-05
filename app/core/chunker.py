"""
app/core/chunker.py — Split parsed document text into overlapping token chunks.

Design goals:
- Token-aware (not character-aware) sizing, matching how OpenAI bills embeddings.
- Overlap between adjacent chunks preserves cross-boundary context.
- Each chunk carries back-reference metadata (source page, position).
"""

from dataclasses import dataclass, field

import tiktoken

from app.core.pdf_parser import ParsedDocument, PageContent


@dataclass
class TextChunk:
    chunk_id: str          # "{filename}_chunk_{index}"
    text: str
    token_count: int
    source_file: str
    page_number: int       # page where this chunk *starts*
    chunk_index: int       # global position in the document


@dataclass
class ChunkingResult:
    source_file: str
    chunks: list[TextChunk]
    total_chunks: int = field(init=False)

    def __post_init__(self):
        self.total_chunks = len(self.chunks)


# Reuse tokeniser across calls — initialisation is expensive
_TOKENISER_CACHE: dict[str, tiktoken.Encoding] = {}


def _get_tokeniser(model: str = "text-embedding-3-small") -> tiktoken.Encoding:
    if model not in _TOKENISER_CACHE:
        try:
            _TOKENISER_CACHE[model] = tiktoken.encoding_for_model(model)
        except KeyError:
            # Fall back to cl100k_base (used by all recent OpenAI models)
            _TOKENISER_CACHE[model] = tiktoken.get_encoding("cl100k_base")
    return _TOKENISER_CACHE[model]


def chunk_document(
    document: ParsedDocument,
    chunk_size: int = 512,
    chunk_overlap: int = 200,
    model: str = "text-embedding-3-small",
) -> ChunkingResult:
    """
    Chunk a ParsedDocument into overlapping token windows.

    Args:
        document:      Parsed PDF document.
        chunk_size:    Max tokens per chunk.
        chunk_overlap: How many tokens the next chunk shares with the previous.
        model:         Tokeniser model name (for accurate token counting).

    Returns:
        ChunkingResult containing all TextChunk objects.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    enc = _get_tokeniser(model)
    chunks: list[TextChunk] = []

    # Build a flat list of (token_ids, page_number) pairs
    # We process per-page text so we can track which page tokens came from
    token_pages: list[tuple[list[int], int]] = []

    for page in document.pages:
        if not page.text.strip():
            continue
        token_ids = enc.encode(page.text)
        if token_ids:
            token_pages.append((token_ids, page.page_number))

    if not token_pages:
        return ChunkingResult(source_file=document.filename, chunks=[])

    # Flatten tokens, keeping a page map: token index → page number
    all_tokens: list[int] = []
    token_page_map: list[int] = []

    for token_ids, page_num in token_pages:
        all_tokens.extend(token_ids)
        token_page_map.extend([page_num] * len(token_ids))

    # Slide a window across all_tokens
    stride = chunk_size - chunk_overlap
    chunk_index = 0
    pos = 0

    while pos < len(all_tokens):
        end = min(pos + chunk_size, len(all_tokens))
        window_tokens = all_tokens[pos:end]
        page_num = token_page_map[pos]  # page of chunk's first token

        text = enc.decode(window_tokens)

        chunk = TextChunk(
            chunk_id=f"{document.filename}_chunk_{chunk_index}",
            text=text,
            token_count=len(window_tokens),
            source_file=document.filename,
            page_number=page_num,
            chunk_index=chunk_index,
        )
        chunks.append(chunk)
        chunk_index += 1

        if end == len(all_tokens):
            break
        pos += stride

    return ChunkingResult(source_file=document.filename, chunks=chunks)


def count_tokens(text: str, model: str = "text-embedding-3-small") -> int:
    """Utility: count tokens in a string for a given model."""
    enc = _get_tokeniser(model)
    return len(enc.encode(text))
