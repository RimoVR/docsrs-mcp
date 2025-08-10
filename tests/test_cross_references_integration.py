"""Integration tests for end-to-end cross-reference flow."""

import json
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from docsrs_mcp.app import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


def create_comprehensive_rustdoc_json():
    """Create a comprehensive test rustdoc JSON with cross-references."""
    return json.dumps({
        "format_version": 27,
        "root": "0:0",
        "crate": {
            "name": "test_crate",
            "version": "1.0.0",
        },
        "paths": {
            "0:0": {"path": ["test_crate"], "kind": "module"},
            "0:1": {"path": ["test_crate", "parse"], "kind": "function"},
            "0:2": {"path": ["test_crate", "Parser"], "kind": "struct"},
            "0:3": {"path": ["test_crate", "ParseError"], "kind": "enum"},
            "0:4": {"path": ["test_crate", "Parseable"], "kind": "trait"},
        },
        "index": {
            "0:0": {
                "name": "test_crate",
                "docs": "Test crate root module",
                "inner": {"Module": {}},
            },
            "0:1": {
                "name": "parse",
                "docs": "Parse input using [`Parser`] which implements [`Parseable`]. May return [`ParseError`].",
                "inner": {"Function": {"decl": {"inputs": [], "output": None}}},
                "links": {
                    "Parser": "0:2",
                    "Parseable": "0:4",
                    "ParseError": "0:3",
                },
            },
            "0:2": {
                "name": "Parser",
                "docs": "Main parser struct that implements [`Parseable`]",
                "inner": {"Struct": {"kind": "unit"}},
                "links": {
                    "Parseable": "0:4",
                },
            },
            "0:3": {
                "name": "ParseError",
                "docs": "Error type returned by [`parse`] function",
                "inner": {"Enum": {}},
                "links": {
                    "parse": "0:1",
                },
            },
            "0:4": {
                "name": "Parseable",
                "docs": "Trait implemented by [`Parser`]",
                "inner": {"Trait": {}},
                "links": {
                    "Parser": "0:2",
                },
            },
        },
    })


@pytest.mark.asyncio
async def test_end_to_end_crossref_flow(client):
    """Test the complete flow from ingestion to API response with cross-references."""

    # Mock the rustdoc download and decompression
    rustdoc_json = create_comprehensive_rustdoc_json()

    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch_info, \
         patch("docsrs_mcp.ingest.resolve_version") as mock_resolve_version, \
         patch("docsrs_mcp.ingest.download_rustdoc") as mock_download, \
         patch("docsrs_mcp.ingest.decompress_content") as mock_decompress, \
         patch("docsrs_mcp.ingest.generate_embeddings_streaming") as mock_embeddings:

        # Setup mocks
        mock_fetch_info.return_value = {
            "name": "test_crate",
            "description": "Test crate for cross-references",
            "max_stable_version": "1.0.0",
        }

        mock_resolve_version.return_value = ("1.0.0", "https://docs.rs/test_crate/1.0.0/test_crate.json")
        mock_download.return_value = (b"compressed_content", "https://docs.rs/test_crate/1.0.0/test_crate.json")
        mock_decompress.return_value = rustdoc_json

        # Mock embeddings to return simple values
        def mock_embedding_generator(chunks):
            for chunk in chunks:
                yield (chunk, [0.1] * 384)  # Mock embedding vector

        mock_embeddings.return_value = mock_embedding_generator([])

        # Test 1: Get item documentation with cross-references
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "test_crate",
                "item_path": "test_crate::parse",
                "version": "1.0.0",
            },
        )

        # Check response
        assert response.status_code == 200
        data = response.json()

        # Should have content
        assert "content" in data
        assert "format" in data
        assert data["format"] == "markdown"

        # Should have cross-references
        assert "cross_references" in data
        cross_refs = data["cross_references"]

        # Should have outgoing references from parse function
        assert "from" in cross_refs
        from_refs = cross_refs["from"]

        # Check that parse references Parser, Parseable, and ParseError
        target_paths = {ref["target_path"] for ref in from_refs}
        assert "test_crate::Parser" in target_paths
        assert "test_crate::Parseable" in target_paths
        assert "test_crate::ParseError" in target_paths

        # Test 2: Get Parser struct documentation
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "test_crate",
                "item_path": "test_crate::Parser",
                "version": "1.0.0",
            },
        )

        assert response.status_code == 200
        data = response.json()

        # Parser should have both incoming and outgoing references
        if "cross_references" in data:
            cross_refs = data["cross_references"]

            # Outgoing: Parser -> Parseable
            if "from" in cross_refs:
                assert any(
                    ref["target_path"] == "test_crate::Parseable"
                    for ref in cross_refs["from"]
                )

            # Incoming: parse -> Parser, Parseable -> Parser
            if "to" in cross_refs:
                source_paths = {ref["source_path"] for ref in cross_refs["to"]}
                assert "test_crate::parse" in source_paths or "test_crate::Parseable" in source_paths


@pytest.mark.asyncio
async def test_crossref_with_path_resolution(client):
    """Test that cross-references work with path alias resolution."""

    rustdoc_json = json.dumps({
        "format_version": 27,
        "root": "0:0",
        "crate": {
            "name": "test_crate",
            "version": "1.0.0",
        },
        "paths": {
            "0:0": {"path": ["test_crate"], "kind": "module"},
            "0:1": {"path": ["test_crate", "prelude"], "kind": "module"},
            "0:2": {"path": ["test_crate", "core", "Parser"], "kind": "struct"},
        },
        "index": {
            "0:0": {
                "name": "test_crate",
                "docs": "Root",
                "inner": {"Module": {}},
            },
            "0:1": {
                "name": "prelude",
                "docs": "Prelude with re-export",
                "inner": {"Module": {}},
                # Re-export (handled separately from cross-refs)
            },
            "0:2": {
                "name": "Parser",
                "docs": "Parser struct",
                "inner": {"Struct": {"kind": "unit"}},
                "links": {},
            },
        },
    })

    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch_info, \
         patch("docsrs_mcp.ingest.resolve_version") as mock_resolve_version, \
         patch("docsrs_mcp.ingest.download_rustdoc") as mock_download, \
         patch("docsrs_mcp.ingest.decompress_content") as mock_decompress, \
         patch("docsrs_mcp.ingest.generate_embeddings_streaming") as mock_embeddings:

        # Setup mocks
        mock_fetch_info.return_value = {
            "name": "test_crate",
            "description": "Test",
            "max_stable_version": "1.0.0",
        }

        mock_resolve_version.return_value = ("1.0.0", "https://docs.rs/test.json")
        mock_download.return_value = (b"compressed", "https://docs.rs/test.json")
        mock_decompress.return_value = rustdoc_json

        def mock_embedding_generator(chunks):
            for chunk in chunks:
                yield (chunk, [0.1] * 384)

        mock_embeddings.return_value = mock_embedding_generator([])

        # Test that path resolution works with cross-references
        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "test_crate",
                "item_path": "test_crate::core::Parser",
                "version": "1.0.0",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data


@pytest.mark.asyncio
async def test_missing_crossrefs_graceful_handling(client):
    """Test that missing cross-references are handled gracefully."""

    # Rustdoc without links field
    rustdoc_json = json.dumps({
        "format_version": 27,
        "root": "0:0",
        "crate": {
            "name": "test_crate",
            "version": "1.0.0",
        },
        "paths": {
            "0:0": {"path": ["test_crate"], "kind": "module"},
            "0:1": {"path": ["test_crate", "foo"], "kind": "function"},
        },
        "index": {
            "0:0": {
                "name": "test_crate",
                "docs": "Root",
                "inner": {"Module": {}},
            },
            "0:1": {
                "name": "foo",
                "docs": "Function without links",
                "inner": {"Function": {}},
                # No links field
            },
        },
    })

    with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch_info, \
         patch("docsrs_mcp.ingest.resolve_version") as mock_resolve_version, \
         patch("docsrs_mcp.ingest.download_rustdoc") as mock_download, \
         patch("docsrs_mcp.ingest.decompress_content") as mock_decompress, \
         patch("docsrs_mcp.ingest.generate_embeddings_streaming") as mock_embeddings:

        mock_fetch_info.return_value = {
            "name": "test_crate",
            "description": "Test",
            "max_stable_version": "1.0.0",
        }

        mock_resolve_version.return_value = ("1.0.0", "https://docs.rs/test.json")
        mock_download.return_value = (b"compressed", "https://docs.rs/test.json")
        mock_decompress.return_value = rustdoc_json

        def mock_embedding_generator(chunks):
            for chunk in chunks:
                yield (chunk, [0.1] * 384)

        mock_embeddings.return_value = mock_embedding_generator([])

        response = client.post(
            "/mcp/tools/get_item_doc",
            json={
                "crate_name": "test_crate",
                "item_path": "test_crate::foo",
                "version": "1.0.0",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data

        # Should not have cross_references field or it should be empty
        if "cross_references" in data:
            assert data["cross_references"] == {} or data["cross_references"] is None
