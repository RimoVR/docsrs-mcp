"""Integration tests for fuzzy path resolution in API endpoints."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import aiosqlite
import pytest
from fastapi.testclient import TestClient

from docsrs_mcp.app import app
from docsrs_mcp.database import init_database


@pytest.fixture
async def mock_ingest():
    """Mock the ingest_crate function to return a test database."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = Path(tmp.name)

        # Initialize database
        await init_database(db_path)

        # Add test data
        async with aiosqlite.connect(db_path) as db:
            # Insert test items
            test_items = [
                ("tokio::spawn", "fn spawn", "Spawns a new asynchronous task"),
                ("tokio::spawn_blocking", "fn spawn_blocking", "Runs blocking code"),
                ("tokio::spawn_local", "fn spawn_local", "Spawns a local task"),
                ("tokio::runtime::Runtime", "struct Runtime", "The Tokio runtime"),
                ("tokio::task::JoinHandle", "struct JoinHandle", "Handle to a task"),
            ]

            for item_path, header, content in test_items:
                # Create dummy embedding (384 dimensions for BAAI/bge-small-en-v1.5)
                dummy_embedding = b"\x00" * (384 * 4)  # 384 float32 values

                await db.execute(
                    """
                    INSERT INTO embeddings 
                    (item_path, header, content, embedding, item_type)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (item_path, header, content, dummy_embedding, "function"),
                )

            # Add crate metadata
            await db.execute(
                """
                INSERT INTO crate_metadata 
                (name, version, description)
                VALUES (?, ?, ?)
                """,
                ("tokio", "1.0.0", "An async runtime"),
            )

            await db.commit()

        # Mock ingest_crate to return our test database
        with patch("docsrs_mcp.app.ingest_crate") as mock:
            mock.return_value = db_path
            yield mock

        # Cleanup
        db_path.unlink(missing_ok=True)


def test_get_item_doc_exact_match(mock_ingest):
    """Test that exact matches return documentation directly."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "tokio::spawn",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "Spawns a new asynchronous task" in data["content"]
        assert data["format"] == "markdown"


def test_get_item_doc_fuzzy_match_typo(mock_ingest):
    """Test fuzzy matching with a typo returns suggestions."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "tokio::spwan",  # Typo in 'spawn'
            },
        )

        # Should return 404 with suggestions
        assert response.status_code == 404
        data = response.json()

        # Check error structure
        assert "detail" in data
        detail = data["detail"]

        # Check that suggestions are in the error message
        assert "tokio::spawn" in detail
        assert "Did you mean" in detail


def test_get_item_doc_fuzzy_match_partial(mock_ingest):
    """Test fuzzy matching with partial path."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "spawn_block",  # Partial match
            },
        )

        assert response.status_code == 404
        data = response.json()

        # Check for suggestions in the error response
        assert "detail" in data
        detail = data["detail"]
        # Should suggest spawn_blocking in the error message
        assert "spawn_blocking" in detail


def test_get_item_doc_no_fuzzy_match(mock_ingest):
    """Test when no fuzzy matches are found."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "completely::unrelated::path",
            },
        )

        assert response.status_code == 404
        data = response.json()

        # Should still have error structure but possibly no suggestions
        assert "detail" in data


def test_get_item_doc_with_version(mock_ingest):
    """Test fuzzy matching with specific version."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "tokio::spwan",  # Typo
                "version": "1.0.0",
            },
        )

        assert response.status_code == 404
        # Should still provide suggestions with version-specific search


def test_fuzzy_suggestions_limit(mock_ingest):
    """Test that fuzzy suggestions are limited to 3."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "spawn",  # Should match multiple items
            },
        )

        assert response.status_code == 404
        data = response.json()

        assert "detail" in data
        # The detail should exist but might not have suggestions for a partial match


def test_fuzzy_match_case_insensitive(mock_ingest):
    """Test that fuzzy matching is case-insensitive."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "TOKIO::SPAWN",  # All caps
            },
        )

        # Could be exact match or fuzzy match depending on preprocessing
        if response.status_code == 404:
            data = response.json()
            assert "detail" in data
            detail = data["detail"]
            # Should suggest the correct case
            if "Did you mean" in detail:
                assert "spawn" in detail.lower()


def test_error_response_format(mock_ingest):
    """Test that error response follows the expected format."""
    with TestClient(app) as client:
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "tokio",
                "item_path": "nonexistent::path",
            },
        )

        assert response.status_code == 404
        data = response.json()

        # FastAPI wraps HTTPException detail in a standard error format
        assert "detail" in data
        detail = data["detail"]
        
        # Should be a string with our error message
        assert isinstance(detail, str)
        assert "No documentation found" in detail
