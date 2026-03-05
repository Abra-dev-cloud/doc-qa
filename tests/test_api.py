"""
tests/test_api.py — FastAPI endpoint integration tests using TestClient.

These tests mock all external services so no API key is needed.
"""

import io
import pytest
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestDocumentUpload:
    def _make_pdf_bytes(self) -> bytes:
        """Return minimal valid PDF bytes for testing."""
        import fitz
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((72, 72), "Hello, this is a test document with some content for testing.")
        return doc.tobytes()

    @patch("app.api.documents.upsert_chunks", return_value=3)
    @patch("app.api.documents.embed_texts", return_value=[[0.1] * 1536] * 3)
    def test_upload_valid_pdf(self, mock_embed, mock_upsert):
        pdf_bytes = self._make_pdf_bytes()
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("test.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            data={"collection_name": "test_col"},
        )
        assert response.status_code == 201
        body = response.json()
        assert body["filename"] == "test.pdf"
        assert body["collection"] == "test_col"
        assert body["total_chunks"] == 3

    def test_upload_non_pdf_rejected(self):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("doc.txt", io.BytesIO(b"hello"), "text/plain")},
            data={"collection_name": "test_col"},
        )
        assert response.status_code == 422

    def test_upload_empty_file_rejected(self):
        response = client.post(
            "/api/v1/documents/upload",
            files={"file": ("empty.pdf", io.BytesIO(b""), "application/pdf")},
            data={"collection_name": "test_col"},
        )
        assert response.status_code in (400, 422)


class TestQAEndpoint:
    @patch("app.api.qa.answer_question")
    def test_ask_question_success(self, mock_answer):
        from app.services.qa_service import QAResponse, SourceCitation
        mock_answer.return_value = QAResponse(
            answer="The answer is 42.",
            sources=[
                SourceCitation(
                    chunk_id="doc_chunk_0",
                    page=2,
                    text_preview="Some relevant text...",
                    relevance_score=0.92,
                )
            ],
            collection="my_col",
            chunks_used=1,
            question="What is the answer?",
        )

        response = client.post(
            "/api/v1/qa/ask",
            json={
                "question": "What is the answer?",
                "collection_name": "my_col",
                "top_k": 5,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["answer"] == "The answer is 42."
        assert len(body["sources"]) == 1
        assert body["sources"][0]["page"] == 2

    def test_ask_empty_question_rejected(self):
        response = client.post(
            "/api/v1/qa/ask",
            json={"question": "ab", "collection_name": "col"},  # min_length=3
        )
        assert response.status_code == 422

    @patch("app.api.qa.answer_question", side_effect=ValueError("Collection is empty"))
    def test_ask_empty_collection_returns_422(self, mock_answer):
        response = client.post(
            "/api/v1/qa/ask",
            json={"question": "What is this?", "collection_name": "empty_col"},
        )
        assert response.status_code == 422
        assert "empty" in response.json()["detail"].lower()


class TestCollectionsEndpoint:
    @patch("app.api.documents.list_collections", return_value=[
        {"name": "col1", "count": 10},
        {"name": "col2", "count": 25},
    ])
    def test_list_collections(self, mock_list):
        response = client.get("/api/v1/documents/collections")
        assert response.status_code == 200
        body = response.json()
        assert body["total"] == 2
        assert body["collections"][0]["name"] == "col1"

    @patch("app.api.documents.delete_collection", return_value=True)
    def test_delete_existing_collection(self, mock_delete):
        response = client.delete("/api/v1/documents/collections/my_col")
        assert response.status_code == 200

    @patch("app.api.documents.delete_collection", return_value=False)
    def test_delete_missing_collection_returns_404(self, mock_delete):
        response = client.delete("/api/v1/documents/collections/nonexistent")
        assert response.status_code == 404
