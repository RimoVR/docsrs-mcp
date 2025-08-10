"""Tests for the FastAPI application."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from docsrs_mcp.app import app, extract_smart_snippet
from docsrs_mcp.models import ErrorResponse, SearchResult


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
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = True
                mock_ingest.return_value = mock_path
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
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = True
                mock_ingest.return_value = mock_path
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
                mock_path = MagicMock(spec=Path)
                mock_path.exists.return_value = True
                mock_ingest.return_value = mock_path
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
async def test_search_items_with_suggestions():
    """Test that search_items returns see-also suggestions."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                with patch(
                    "docsrs_mcp.app.get_see_also_suggestions"
                ) as mock_suggestions:
                    # Setup mocks
                    mock_path = MagicMock(spec=Path)
                    mock_path.exists.return_value = True
                    mock_ingest.return_value = mock_path
                    mock_model.return_value.embed.return_value = [[0.1] * 768]
                    mock_search.return_value = [
                        (
                            0.95,
                            "tokio::spawn",
                            "spawn",
                            "Spawn a future onto the runtime",
                        ),
                        (0.85, "tokio::task", "task", "Task utilities"),
                    ]
                    # Mock suggestions - related items but not in main results
                    mock_suggestions.return_value = [
                        "tokio::runtime::Runtime",
                        "tokio::task::JoinHandle",
                        "tokio::spawn_blocking",
                    ]

                    async with AsyncClient(
                        transport=transport, base_url="http://test"
                    ) as client:
                        response = await client.post(
                            "/mcp/tools/search_items",
                            json={
                                "crate_name": "tokio",
                                "query": "spawn task",
                                "k": 2,
                            },
                        )

                        assert response.status_code == 200
                        data = response.json()

                        # Check that results are returned
                        assert "results" in data
                        assert len(data["results"]) == 2

                        # Check that suggestions are only added to the first result
                        first_result = data["results"][0]
                        assert "suggestions" in first_result
                        assert first_result["suggestions"] == [
                            "tokio::runtime::Runtime",
                            "tokio::task::JoinHandle",
                            "tokio::spawn_blocking",
                        ]

                        # Second result should not have suggestions
                        second_result = data["results"][1]
                        assert (
                            "suggestions" not in second_result
                            or second_result["suggestions"] is None
                        )

                        # Verify get_see_also_suggestions was called with correct params
                        mock_suggestions.assert_called_once()
                        call_args = mock_suggestions.call_args
                        assert call_args.args[2] == {
                            "tokio::spawn",
                            "tokio::task",
                        }  # original_paths
                        assert call_args.kwargs["k"] == 10
                        assert call_args.kwargs["similarity_threshold"] == 0.7
                        assert call_args.kwargs["max_suggestions"] == 5


@pytest.mark.asyncio
async def test_search_items_no_suggestions_when_no_results():
    """Test that no suggestions are computed when there are no search results."""
    transport = ASGITransport(app=app)

    with patch("docsrs_mcp.app.ingest_crate") as mock_ingest:
        with patch("docsrs_mcp.app.get_embedding_model") as mock_model:
            with patch("docsrs_mcp.app.search_embeddings") as mock_search:
                with patch(
                    "docsrs_mcp.app.get_see_also_suggestions"
                ) as mock_suggestions:
                    # Setup mocks - no search results
                    mock_ingest.return_value = "/path/to/db"
                    mock_model.return_value.embed.return_value = [[0.1] * 768]
                    mock_search.return_value = []  # No results

                    async with AsyncClient(
                        transport=transport, base_url="http://test"
                    ) as client:
                        response = await client.post(
                            "/mcp/tools/search_items",
                            json={
                                "crate_name": "tokio",
                                "query": "nonexistent",
                                "k": 5,
                            },
                        )

                        assert response.status_code == 200
                        data = response.json()

                        # Check that results are empty
                        assert "results" in data
                        assert len(data["results"]) == 0

                        # Verify get_see_also_suggestions was NOT called
                        mock_suggestions.assert_not_called()


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


@pytest.mark.asyncio
async def test_mcp_manifest_includes_tutorials():
    """Test that MCP manifest includes tutorial fields when present."""
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/mcp/manifest")
        assert response.status_code == 200

        manifest = response.json()
        tools = manifest["tools"]

        # Check that all tools have tutorial fields
        expected_tools = [
            "get_crate_summary",
            "search_items",
            "get_item_doc",
            "search_examples",
            "get_module_tree",
        ]

        found_tools = {tool["name"] for tool in tools}
        for expected_tool in expected_tools:
            assert expected_tool in found_tools, (
                f"Tool {expected_tool} not found in manifest"
            )

        # Check each tool has tutorial fields
        for tool in tools:
            if tool["name"] in expected_tools:
                # Tutorial fields should be present
                assert "tutorial" in tool, f"Tool {tool['name']} missing tutorial field"
                assert "examples" in tool, f"Tool {tool['name']} missing examples field"
                assert "use_cases" in tool, (
                    f"Tool {tool['name']} missing use_cases field"
                )

                # Verify tutorial is a non-empty string
                assert isinstance(tool["tutorial"], str), (
                    f"Tool {tool['name']} tutorial should be a string"
                )
                assert len(tool["tutorial"]) > 0, (
                    f"Tool {tool['name']} tutorial should not be empty"
                )

                # Verify examples is a list of strings
                assert isinstance(tool["examples"], list), (
                    f"Tool {tool['name']} examples should be a list"
                )
                assert len(tool["examples"]) > 0, (
                    f"Tool {tool['name']} should have at least one example"
                )
                for example in tool["examples"]:
                    assert isinstance(example, str), (
                        f"Tool {tool['name']} examples should be strings"
                    )

                # Verify use_cases is a list of strings
                assert isinstance(tool["use_cases"], list), (
                    f"Tool {tool['name']} use_cases should be a list"
                )
                assert len(tool["use_cases"]) > 0, (
                    f"Tool {tool['name']} should have at least one use case"
                )
                for use_case in tool["use_cases"]:
                    assert isinstance(use_case, str), (
                        f"Tool {tool['name']} use_cases should be strings"
                    )

                # Verify backward compatibility - fields can be None in the model
                # but we're providing values in this implementation
                assert tool["tutorial"] is not None
                assert tool["examples"] is not None
                assert tool["use_cases"] is not None


@pytest.mark.asyncio
async def test_error_response_status_code_validation():
    """Test that ErrorResponse status_code accepts strings and validates bounds."""
    # Test valid integer status codes
    error = ErrorResponse(error="NotFound", status_code=404)
    assert error.status_code == 404

    error = ErrorResponse(error="ServerError", status_code=500)
    assert error.status_code == 500

    # Test string conversion
    error = ErrorResponse.model_validate({"error": "BadRequest", "status_code": "400"})
    assert error.status_code == 400

    error = ErrorResponse.model_validate(
        {"error": "TooManyRequests", "status_code": "429"}
    )
    assert error.status_code == 429

    # Test default value when None
    error = ErrorResponse.model_validate({"error": "UnknownError", "status_code": None})
    assert error.status_code == 500

    # Test out of range values with integers (Pydantic's built-in validation)
    with pytest.raises(ValidationError, match="greater than or equal to 400"):
        ErrorResponse(error="Invalid", status_code=399)

    with pytest.raises(ValidationError, match="less than or equal to 599"):
        ErrorResponse(error="Invalid", status_code=600)

    # Test out of range values with strings
    with pytest.raises(ValueError, match="must be at least 400"):
        ErrorResponse.model_validate({"error": "Invalid", "status_code": "300"})

    with pytest.raises(ValueError, match="cannot exceed 599"):
        ErrorResponse.model_validate({"error": "Invalid", "status_code": "700"})

    # Test invalid string
    with pytest.raises(ValueError, match="must be a valid HTTP error code"):
        ErrorResponse.model_validate({"error": "Invalid", "status_code": "invalid"})


@pytest.mark.asyncio
async def test_search_result_score_validation():
    """Test that SearchResult score accepts strings and validates bounds."""
    # Test valid float scores
    result = SearchResult(
        score=0.9,
        item_path="tokio::spawn",
        header="spawn function",
        snippet="Spawn a future",
    )
    assert result.score == 0.9

    # Test string conversion
    result = SearchResult.model_validate(
        {
            "score": "0.75",
            "item_path": "tokio::spawn",
            "header": "spawn function",
            "snippet": "Spawn a future",
        }
    )
    assert result.score == 0.75

    # Test integer to float conversion
    result = SearchResult.model_validate(
        {
            "score": 1,
            "item_path": "tokio::spawn",
            "header": "spawn function",
            "snippet": "Spawn a future",
        }
    )
    assert result.score == 1.0

    # Test boundary values
    result = SearchResult.model_validate(
        {
            "score": "0.0",
            "item_path": "tokio::spawn",
            "header": "spawn function",
            "snippet": "Spawn a future",
        }
    )
    assert result.score == 0.0

    result = SearchResult.model_validate(
        {
            "score": "1.0",
            "item_path": "tokio::spawn",
            "header": "spawn function",
            "snippet": "Spawn a future",
        }
    )
    assert result.score == 1.0

    # Test out of range values
    with pytest.raises(ValueError, match="must be at least 0.0"):
        SearchResult.model_validate(
            {
                "score": "-0.1",
                "item_path": "path",
                "header": "header",
                "snippet": "snippet",
            }
        )

    with pytest.raises(ValueError, match="cannot exceed 1.0"):
        SearchResult.model_validate(
            {
                "score": "1.5",
                "item_path": "path",
                "header": "header",
                "snippet": "snippet",
            }
        )

    # Test invalid string
    with pytest.raises(ValueError, match="must be a valid number"):
        SearchResult.model_validate(
            {
                "score": "invalid",
                "item_path": "path",
                "header": "header",
                "snippet": "snippet",
            }
        )

    # Test None is not allowed (score is required)
    with pytest.raises(ValueError, match="Score is required"):
        SearchResult.model_validate(
            {
                "score": None,
                "item_path": "path",
                "header": "header",
                "snippet": "snippet",
            }
        )


def test_extract_smart_snippet_short_content():
    """Test snippet extraction with content shorter than min_length."""
    # Content shorter than min_length (200) should be returned as-is
    short_content = "This is a short description."
    result = extract_smart_snippet(short_content)
    assert result == short_content

    # Content exactly at min_length
    content_200 = "x" * 200
    result = extract_smart_snippet(content_200)
    assert result == content_200


def test_extract_smart_snippet_sentence_boundaries():
    """Test snippet extraction with sentence boundary detection."""
    # Content with clear sentence boundaries
    content = (
        "This is the first sentence. This is the second sentence. "
        "This is the third sentence. This is the fourth sentence. "
        "This is the fifth sentence. This is the sixth sentence. "
        "This is the seventh sentence. This is the eighth sentence. "
        "This is the ninth sentence. This is the tenth sentence."
    )

    result = extract_smart_snippet(content)
    # Should extract complete sentences
    assert result.endswith(".") or result.endswith("...")
    assert 200 <= len(result) <= 400
    # Should not cut in the middle of a sentence
    assert "This is the" in result  # Complete sentence start

    # Test with exclamation and question marks
    content_mixed = (
        "What is Rust? Rust is a systems programming language! "
        "It focuses on safety. How does it achieve this? "
        "Through its ownership system! The borrow checker ensures memory safety. "
        "Is it fast? Yes, it's as fast as C++! "
        "Why use Rust? For safe systems programming."
    )

    result = extract_smart_snippet(content_mixed)
    assert 200 <= len(result) <= 400
    # Should preserve punctuation
    assert any(char in result for char in ["?", "!", "."])


def test_extract_smart_snippet_word_boundaries():
    """Test snippet extraction falling back to word boundaries."""
    # Content without clear sentence boundaries (long sentence) - make it longer to force truncation
    content = (
        "This is a very long sentence that continues for a while without any "
        "punctuation marks and keeps going with more words and more content "
        "that should be truncated at word boundaries when sentence detection "
        "fails and we need to fall back to word-level truncation to ensure "
        "that we don't cut words in the middle and maintain readability even "
        "when dealing with unusual content that lacks proper punctuation and "
        "continues even further with additional words to ensure we exceed the "
        "maximum length limit and trigger the truncation logic properly"
    )

    result = extract_smart_snippet(content)
    assert 200 <= len(result) <= 403  # +3 for ellipsis
    # Check if it was truncated (content is > 400 chars)
    if len(content) > 400:
        assert result.endswith("...")
    # The snippet should preserve complete words
    # Verify that we haven't cut a word in the middle by checking
    # that the text before ellipsis forms complete words
    if result.endswith("..."):
        text_part = result[:-3]
        # Split into words and verify the last word is complete
        # by checking it exists in the original content as a complete word
        words = text_part.split()
        if words:
            last_word = words[-1]
            # Check that this word appears in the original content
            assert last_word in content.split()


def test_extract_smart_snippet_character_truncation():
    """Test snippet extraction falling back to character truncation."""
    # Content with a single very long word
    content = "supercalifragilisticexpialidocious" * 20

    result = extract_smart_snippet(content)
    assert len(result) <= 400 + 3  # +3 for ellipsis
    assert result.endswith("...")


def test_extract_smart_snippet_empty_content():
    """Test snippet extraction with empty or None content."""
    # Empty string
    result = extract_smart_snippet("")
    assert result == ""

    # None should be handled gracefully if the function is updated to accept it
    # For now, test with empty string as the function expects str


def test_extract_smart_snippet_unicode_content():
    """Test snippet extraction with Unicode content."""
    # Content with Unicode characters
    content = (
        "Rust supports Unicode natively. You can use emojis 🦀 in your code. "
        "International characters like café, naïve, and 中文 are supported. "
        "This makes Rust great for internationalization. "
        "The string type in Rust is UTF-8 encoded by default. "
        "You can work with any language: العربية, ελληνικά, русский, 日本語. "
        "This is a powerful feature for global applications."
    )

    result = extract_smart_snippet(content)
    assert 200 <= len(result) <= 400
    # Should preserve Unicode characters
    if "🦀" in result[:400]:
        assert "🦀" in result
    if "café" in result[:400]:
        assert "café" in result


def test_extract_smart_snippet_edge_cases():
    """Test snippet extraction with edge cases."""
    # Content exactly at max_length
    content_400 = "x" * 400
    result = extract_smart_snippet(content_400)
    assert result == content_400

    # Content just over max_length
    content_401 = "x" * 401
    result = extract_smart_snippet(content_401)
    assert result == "x" * 400 + "..."

    # Content with only whitespace
    whitespace_content = "   \n\t   "
    result = extract_smart_snippet(whitespace_content)
    assert result == whitespace_content

    # Content with repeated periods
    dots_content = "..." * 100
    result = extract_smart_snippet(dots_content)
    assert len(result) <= 400


def test_extract_smart_snippet_preserves_context():
    """Test that snippet extraction preserves meaningful context."""
    # Technical documentation example
    content = (
        "The `tokio::spawn` function is used to spawn asynchronous tasks. "
        "It takes a future and runs it on the Tokio runtime. "
        "The spawned task runs concurrently with other tasks. "
        "You must ensure the future is `Send + 'static`. "
        "The function returns a `JoinHandle` that can be awaited. "
        "Error handling is important when working with spawned tasks. "
        "Tasks can be cancelled by dropping the JoinHandle."
    )

    result = extract_smart_snippet(content)
    assert 200 <= len(result) <= 400
    # Should include complete technical terms
    assert "`tokio::spawn`" in result or "JoinHandle" in result
    # Should maintain technical accuracy (not cut in middle of code blocks)
    if "`" in result:
        # Count backticks - should be even (paired)
        backtick_count = result.count("`")
        assert backtick_count % 2 == 0
