"""Tests for the FastAPI application."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from docsrs_mcp.app import app


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "docsrs-mcp"}


@pytest.mark.asyncio
async def test_root():
    """Test the root endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "docsrs-mcp"
        assert data["version"] == "0.1.0"
        assert "mcp_manifest" in data


@pytest.mark.asyncio
async def test_search_items_with_type_filter():
    """Test the search_items endpoint with type filter."""
    transport = ASGITransport(app=app)

    # Mock dependencies
    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                # Setup mocks
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.95, "tokio::spawn", "spawn", "Spawn a future onto the runtime")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "tokio",
                            "query": "spawn task",
                            "k": 5,
                            "item_type": "function",
                        },
                    )

                    assert response.status_code == 200

                    # Verify the search_embeddings was called with type filter
                    mock_search.assert_called_once()
                    call_args = mock_search.call_args
                    assert call_args.kwargs["type_filter"] == "function"
                    assert call_args.kwargs["crate_filter"] is None


@pytest.mark.asyncio
async def test_search_items_with_crate_filter():
    """Test the search_items endpoint with crate filter."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                # Setup mocks
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (
                        0.90,
                        "serde::Serialize",
                        "Serialize",
                        "A data structure that can be serialized",
                    )
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "serde",
                            "query": "serialize",
                            "k": 3,
                            "crate_filter": "serde",
                        },
                    )

                    assert response.status_code == 200

                    # Verify the search_embeddings was called with crate filter
                    mock_search.assert_called_once()
                    call_args = mock_search.call_args
                    assert call_args.kwargs["crate_filter"] == "serde"
                    assert call_args.kwargs["type_filter"] is None


@pytest.mark.asyncio
async def test_search_items_with_combined_filters():
    """Test the search_items endpoint with both type and crate filters."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                # Setup mocks
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.88, "tokio::sync::Mutex", "Mutex", "An asynchronous Mutex")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "tokio",
                            "query": "mutex",
                            "k": 10,
                            "item_type": "struct",
                            "crate_filter": "tokio",
                        },
                    )

                    assert response.status_code == 200

                    # Verify both filters were passed
                    mock_search.assert_called_once()
                    call_args = mock_search.call_args
                    assert call_args.kwargs["type_filter"] == "struct"
                    assert call_args.kwargs["crate_filter"] == "tokio"


@pytest.mark.asyncio
async def test_search_items_filter_validation():
    """Test that invalid item_type values are rejected."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": "test", "item_type": "invalid_type"},
        )

        assert response.status_code == 422  # Validation error
        data = response.json()
        assert "item_type must be one of" in str(data)


@pytest.mark.asyncio
async def test_query_normalization_integration():
    """Test that query normalization works correctly in the API endpoint."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Test whitespace normalization
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": "  async   runtime  "},
        )
        assert response.status_code == 200
        # The normalized query is used for search

        # Test Unicode normalization
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": "café"},
        )
        assert response.status_code == 200

        # Test special characters are preserved
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": "tokio::spawn"},
        )
        assert response.status_code == 200

        # Test empty query is rejected
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": ""},
        )
        assert response.status_code == 422
        error = response.json()
        assert "Query cannot be empty" in str(error)

        # Test whitespace-only query is rejected
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": "   \t\n   "},
        )
        assert response.status_code == 422
        error = response.json()
        assert "Query cannot be empty after normalization" in str(error)

        # Test query too long is rejected
        long_query = "a" * 501
        response = await client.post(
            "/mcp/tools/search_items",
            json={"crate_name": "tokio", "query": long_query},
        )
        assert response.status_code == 422
        error = response.json()
        assert "Query too long" in str(error)


@pytest.mark.asyncio
async def test_mcp_manifest_includes_filter_parameters():
    """Test that the MCP manifest includes the new filter parameters."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/mcp/manifest")
        assert response.status_code == 200

        data = response.json()

        # Find the search_items tool
        search_tool = None
        for tool in data["tools"]:
            if tool["name"] == "search_items":
                search_tool = tool
                break

        assert search_tool is not None
        schema = search_tool["input_schema"]

        # Check that filter parameters are present
        assert "item_type" in schema["properties"]
        assert schema["properties"]["item_type"]["type"] == "string"
        assert "enum" in schema["properties"]["item_type"]
        assert set(schema["properties"]["item_type"]["enum"]) == {
            "function",
            "struct",
            "trait",
            "enum",
            "module",
        }

        assert "crate_filter" in schema["properties"]
        assert schema["properties"]["crate_filter"]["type"] == "string"
