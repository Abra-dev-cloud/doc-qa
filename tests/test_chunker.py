"""
tests/test_chunker.py — Unit tests for the text chunking logic.

These tests run without any API keys or external services.
"""

import pytest
from app.core.chunker import TextChunk, chunk_document, count_tokens
from app.core.pdf_parser import PageContent, ParsedDocument


def make_document(text: str, filename: str = "test.pdf") -> ParsedDocument:
    """Helper: create a single-page ParsedDocument from plain text."""
    return ParsedDocument(
        filename=filename,
        total_pages=1,
        pages=[
            PageContent(page_number=1, text=text, char_count=len(text))
        ],
    )


class TestCountTokens:
    def test_empty_string(self):
        assert count_tokens("") == 0

    def test_single_word(self):
        assert count_tokens("hello") > 0

    def test_longer_text(self):
        # rough sanity: ~1 token per word for English prose
        text = " ".join(["word"] * 100)
        tokens = count_tokens(text)
        assert 80 <= tokens <= 120


class TestChunkDocument:
    def test_empty_document_returns_no_chunks(self):
        doc = make_document("")
        result = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        assert result.total_chunks == 0
        assert result.chunks == []

    def test_short_document_produces_single_chunk(self):
        doc = make_document("This is a very short document.")
        result = chunk_document(doc, chunk_size=512, chunk_overlap=50)
        assert result.total_chunks == 1
        chunk = result.chunks[0]
        assert "short document" in chunk.text
        assert chunk.page_number == 1
        assert chunk.chunk_index == 0

    def test_long_document_produces_multiple_chunks(self):
        # ~1000 tokens of text
        long_text = ("The quick brown fox jumped over the lazy dog. " * 100).strip()
        doc = make_document(long_text)
        result = chunk_document(doc, chunk_size=100, chunk_overlap=20)
        assert result.total_chunks > 1

    def test_chunks_have_correct_structure(self):
        doc = make_document("Hello world. " * 200)
        result = chunk_document(doc, chunk_size=50, chunk_overlap=10)
        for i, chunk in enumerate(result.chunks):
            assert isinstance(chunk, TextChunk)
            assert chunk.chunk_index == i
            assert chunk.source_file == "test.pdf"
            assert chunk.token_count <= 50
            assert len(chunk.text) > 0
            assert chunk.chunk_id == f"test.pdf_chunk_{i}"

    def test_overlap_creates_shared_content(self):
        """Adjacent chunks should share tokens due to overlap."""
        # Build text with distinct 10-word sentences
        sentences = [f"sentence number {i} has some unique content here end." for i in range(50)]
        doc = make_document(" ".join(sentences))
        result = chunk_document(doc, chunk_size=40, chunk_overlap=15)
        
        if result.total_chunks >= 2:
            chunk0_words = set(result.chunks[0].text.lower().split())
            chunk1_words = set(result.chunks[1].text.lower().split())
            shared = chunk0_words & chunk1_words
            # There should be some shared words due to overlap
            assert len(shared) > 0, "Adjacent chunks should share content due to overlap"

    def test_invalid_overlap_raises(self):
        doc = make_document("Some text")
        with pytest.raises(ValueError, match="chunk_overlap"):
            chunk_document(doc, chunk_size=100, chunk_overlap=100)

    def test_chunk_result_metadata(self):
        doc = make_document("Hello " * 200, filename="my_report.pdf")
        result = chunk_document(doc)
        assert result.source_file == "my_report.pdf"
        assert result.total_chunks == len(result.chunks)
