"""
tests/test_qa_service.py — Unit tests for the QA pipeline with mocked external calls.

All OpenAI and ChromaDB calls are mocked — no API key required.
"""

import pytest
from unittest.mock import MagicMock, patch

from app.services.qa_service import (
    QAResponse,
    SourceCitation,
    _build_context_block,
    answer_question,
)
from app.services.vector_store import RetrievedChunk


def make_retrieved_chunk(
    chunk_id: str = "doc_chunk_0",
    text: str = "This is retrieved content about the topic.",
    page: int = 1,
    score: float = 0.85,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source_file="test.pdf",
        page_number=page,
        chunk_index=0,
        relevance_score=score,
    )


class TestBuildContextBlock:
    def test_single_chunk(self):
        chunks = [make_retrieved_chunk(text="Some important content.")]
        block = _build_context_block(chunks)
        assert "[Passage 1]" in block
        assert "Some important content." in block
        assert "test.pdf" in block
        assert "Page 1" in block

    def test_multiple_chunks_numbered(self):
        chunks = [
            make_retrieved_chunk(chunk_id=f"chunk_{i}", text=f"Content {i}")
            for i in range(3)
        ]
        block = _build_context_block(chunks)
        assert "[Passage 1]" in block
        assert "[Passage 2]" in block
        assert "[Passage 3]" in block

    def test_empty_chunks(self):
        block = _build_context_block([])
        assert block == ""


class TestAnswerQuestion:
    @patch("app.services.qa_service.query_collection")
    @patch("app.services.qa_service.embed_query")
    @patch("app.services.qa_service._call_llm")
    def test_successful_answer(self, mock_llm, mock_embed, mock_query):
        # Arrange
        mock_embed.return_value = [0.1] * 1536
        mock_query.return_value = [
            make_retrieved_chunk(text="The revenue was $5M in Q3.", score=0.9)
        ]
        mock_llm.return_value = "The revenue was $5M in Q3 [Passage 1]."

        # Act
        result = answer_question(
            question="What was the Q3 revenue?",
            collection_name="test_collection",
        )

        # Assert
        assert isinstance(result, QAResponse)
        assert "5M" in result.answer
        assert result.chunks_used == 1
        assert result.collection == "test_collection"
        assert result.question == "What was the Q3 revenue?"
        assert len(result.sources) == 1
        assert result.sources[0].page == 1
        assert result.sources[0].relevance_score == 0.9

    @patch("app.services.qa_service.embed_query")
    def test_empty_question_raises(self, mock_embed):
        with pytest.raises(ValueError, match="empty"):
            answer_question(question="   ", collection_name="col")

    @patch("app.services.qa_service.query_collection")
    @patch("app.services.qa_service.embed_query")
    @patch("app.services.qa_service._call_llm")
    def test_multiple_chunks_used(self, mock_llm, mock_embed, mock_query):
        mock_embed.return_value = [0.0] * 1536
        mock_query.return_value = [
            make_retrieved_chunk(chunk_id=f"chunk_{i}", score=0.9 - i * 0.1)
            for i in range(5)
        ]
        mock_llm.return_value = "Based on passages 1-3, the answer is X."

        result = answer_question("Tell me everything", "col", top_k=5)
        assert result.chunks_used == 5
        assert len(result.sources) == 5

    @patch("app.services.qa_service.query_collection")
    @patch("app.services.qa_service.embed_query")
    @patch("app.services.qa_service._call_llm")
    def test_source_preview_truncated(self, mock_llm, mock_embed, mock_query):
        long_text = "A" * 500
        mock_embed.return_value = [0.0] * 1536
        mock_query.return_value = [make_retrieved_chunk(text=long_text)]
        mock_llm.return_value = "Some answer."

        result = answer_question("Question?", "col")
        preview = result.sources[0].text_preview
        assert len(preview) <= 203  # 200 chars + "..."
        assert preview.endswith("...")
