"""Integration tests for search result filtering functionality."""

from unittest.mock import patch

import pytest
from httpx import ASGITransport, AsyncClient

from docsrs_mcp.app import app


@pytest.mark.asyncio
async def test_search_with_all_filters():
    """Test search with all filter parameters."""
    transport = ASGITransport(app=app)

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
                            "crate_filter": "tokio",
                            "has_examples": True,
                            "min_doc_length": 200,
                            "visibility": "public",
                            "deprecated": False,
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "results" in data

                    # Verify search_embeddings was called with all filters
                    mock_search.assert_called_once()
                    call_args = mock_search.call_args
                    assert call_args.kwargs["type_filter"] == "function"
                    assert call_args.kwargs["crate_filter"] == "tokio"
                    assert call_args.kwargs["has_examples"] is True
                    assert call_args.kwargs["min_doc_length"] == 200
                    assert call_args.kwargs["visibility"] == "public"
                    assert call_args.kwargs["deprecated"] is False


@pytest.mark.asyncio
async def test_search_with_visibility_filter():
    """Test search with visibility filter."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.90, "internal::helper", "helper", "Internal helper function")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "helper",
                            "visibility": "private",
                        },
                    )

                    assert response.status_code == 200
                    mock_search.assert_called_once()
                    assert mock_search.call_args.kwargs["visibility"] == "private"


@pytest.mark.asyncio
async def test_search_with_deprecated_filter():
    """Test search with deprecated filter."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.85, "old::function", "function", "Deprecated function")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    # Test filtering for deprecated items only
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "function",
                            "deprecated": True,
                        },
                    )

                    assert response.status_code == 200
                    mock_search.assert_called_once()
                    assert mock_search.call_args.kwargs["deprecated"] is True

                    # Reset mock
                    mock_search.reset_mock()
                    mock_search.return_value = [
                        (0.90, "new::function", "function", "Active function")
                    ]

                    # Test filtering for non-deprecated items only
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "function",
                            "deprecated": False,
                        },
                    )

                    assert response.status_code == 200
                    mock_search.assert_called_once()
                    assert mock_search.call_args.kwargs["deprecated"] is False


@pytest.mark.asyncio
async def test_search_with_has_examples_filter():
    """Test search with has_examples filter."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.92, "example::function", "function", "Function with examples")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "example",
                            "has_examples": True,
                        },
                    )

                    assert response.status_code == 200
                    mock_search.assert_called_once()
                    assert mock_search.call_args.kwargs["has_examples"] is True


@pytest.mark.asyncio
async def test_search_with_min_doc_length_filter():
    """Test search with min_doc_length filter."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (
                        0.88,
                        "detailed::function",
                        "function",
                        "A very detailed and comprehensive documentation " * 50,
                    )
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "detailed",
                            "min_doc_length": 500,
                        },
                    )

                    assert response.status_code == 200
                    mock_search.assert_called_once()
                    assert mock_search.call_args.kwargs["min_doc_length"] == 500


@pytest.mark.asyncio
async def test_search_filter_combinations():
    """Test various combinations of filters."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = []

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    # Test combination 1: type + visibility
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "test",
                            "item_type": "struct",
                            "visibility": "public",
                        },
                    )
                    assert response.status_code == 200
                    assert mock_search.call_args.kwargs["type_filter"] == "struct"
                    assert mock_search.call_args.kwargs["visibility"] == "public"

                    # Test combination 2: examples + min_doc_length
                    mock_search.reset_mock()
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "test",
                            "has_examples": True,
                            "min_doc_length": 300,
                        },
                    )
                    assert response.status_code == 200
                    assert mock_search.call_args.kwargs["has_examples"] is True
                    assert mock_search.call_args.kwargs["min_doc_length"] == 300

                    # Test combination 3: deprecated + type + crate
                    mock_search.reset_mock()
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "test",
                            "deprecated": False,
                            "item_type": "function",
                            "crate_filter": "mylib",
                        },
                    )
                    assert response.status_code == 200
                    assert mock_search.call_args.kwargs["deprecated"] is False
                    assert mock_search.call_args.kwargs["type_filter"] == "function"
                    assert mock_search.call_args.kwargs["crate_filter"] == "mylib"


@pytest.mark.asyncio
async def test_search_with_no_filters():
    """Test search with no filters still works."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                mock_ingest.return_value = "/path/to/db"
                mock_model.return_value.embed.return_value = [[0.1] * 768]
                mock_search.return_value = [
                    (0.95, "result::item", "item", "Search result")
                ]

                async with AsyncClient(
                    transport=transport, base_url="http://test"
                ) as client:
                    response = await client.post(
                        "/mcp/tools/search_items",
                        json={
                            "crate_name": "mylib",
                            "query": "search query",
                        },
                    )

                    assert response.status_code == 200
                    data = response.json()
                    assert "results" in data

                    # Verify all filter parameters are None
                    call_args = mock_search.call_args
                    assert call_args.kwargs.get("type_filter") is None
                    assert call_args.kwargs.get("crate_filter") is None
                    assert call_args.kwargs.get("has_examples") is None
                    assert call_args.kwargs.get("min_doc_length") is None
                    assert call_args.kwargs.get("visibility") is None
                    assert call_args.kwargs.get("deprecated") is None
