"""Tests for new filter validators in SearchItemsRequest."""

import unicodedata

import pytest
from pydantic import ValidationError

from docsrs_mcp.models import SearchItemsRequest


def test_has_examples_validator():
    """Test has_examples field validation."""
    # Test boolean values
    req = SearchItemsRequest(crate_name="tokio", query="spawn", has_examples=True)
    assert req.has_examples is True

    req = SearchItemsRequest(crate_name="tokio", query="spawn", has_examples=False)
    assert req.has_examples is False

    # Test string conversion
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "has_examples": "true"}
    )
    assert req.has_examples is True

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "has_examples": "1"}
    )
    assert req.has_examples is True

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "has_examples": "yes"}
    )
    assert req.has_examples is True

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "has_examples": "false"}
    )
    assert req.has_examples is False

    # Test empty string conversion to None
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "has_examples": ""}
    )
    assert req.has_examples is None

    # Test None value
    req = SearchItemsRequest(crate_name="tokio", query="spawn", has_examples=None)
    assert req.has_examples is None


def test_deprecated_validator():
    """Test deprecated field validation."""
    # Test boolean values
    req = SearchItemsRequest(crate_name="tokio", query="spawn", deprecated=True)
    assert req.deprecated is True

    req = SearchItemsRequest(crate_name="tokio", query="spawn", deprecated=False)
    assert req.deprecated is False

    # Test string conversion
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "deprecated": "true"}
    )
    assert req.deprecated is True

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "deprecated": "false"}
    )
    assert req.deprecated is False

    # Test empty string conversion to None
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "deprecated": ""}
    )
    assert req.deprecated is None


def test_visibility_validator():
    """Test visibility field validation."""
    # Test valid values
    req = SearchItemsRequest(crate_name="tokio", query="spawn", visibility="public")
    assert req.visibility == "public"

    req = SearchItemsRequest(crate_name="tokio", query="spawn", visibility="private")
    assert req.visibility == "private"

    req = SearchItemsRequest(crate_name="tokio", query="spawn", visibility="crate")
    assert req.visibility == "crate"

    # Test case normalization
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "visibility": "PUBLIC"}
    )
    assert req.visibility == "public"

    # Test empty string conversion to None
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "visibility": ""}
    )
    assert req.visibility is None

    # Test None value
    req = SearchItemsRequest(crate_name="tokio", query="spawn", visibility=None)
    assert req.visibility is None

    # Test invalid value
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest.model_validate(
            {"crate_name": "tokio", "query": "spawn", "visibility": "invalid"}
        )
    assert "visibility must be one of" in str(exc_info.value)


def test_min_doc_length_validator():
    """Test min_doc_length field validation."""
    # Test valid values
    req = SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=100)
    assert req.min_doc_length == 100

    req = SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=10000)
    assert req.min_doc_length == 10000

    # Test string conversion
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "min_doc_length": "100"}
    )
    assert req.min_doc_length == 100

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "min_doc_length": "5000"}
    )
    assert req.min_doc_length == 5000

    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "spawn", "min_doc_length": "10000"}
    )
    assert req.min_doc_length == 10000

    # Test None value
    req = SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=None)
    assert req.min_doc_length is None

    # Test out of range values with integers (Pydantic's built-in validation)
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=99)
    assert "greater than or equal to 100" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=10001)
    assert "less than or equal to 10000" in str(exc_info.value)

    # Test out of range values with strings
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest.model_validate(
            {"crate_name": "tokio", "query": "spawn", "min_doc_length": "99"}
        )
    assert "must be at least 100" in str(exc_info.value)

    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest.model_validate(
            {"crate_name": "tokio", "query": "spawn", "min_doc_length": "10001"}
        )
    assert "cannot exceed 10000" in str(exc_info.value)

    # Test invalid string
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest.model_validate(
            {"crate_name": "tokio", "query": "spawn", "min_doc_length": "invalid"}
        )
    assert "must be a valid integer" in str(exc_info.value)


def test_multiple_filters_together():
    """Test using multiple filters together."""
    req = SearchItemsRequest(
        crate_name="tokio",
        query="spawn",
        item_type="function",
        crate_filter="tokio",
        has_examples=True,
        min_doc_length=200,
        visibility="public",
        deprecated=False,
    )
    assert req.item_type == "function"
    assert req.crate_filter == "tokio"
    assert req.has_examples is True
    assert req.min_doc_length == 200
    assert req.visibility == "public"
    assert req.deprecated is False


def test_empty_string_handling():
    """Test that empty strings are properly converted to None."""
    req = SearchItemsRequest.model_validate(
        {
            "crate_name": "tokio",
            "query": "spawn",
            "item_type": "",
            "crate_filter": "",
            "has_examples": "",
            "visibility": "",
            "deprecated": "",
        }
    )
    assert req.item_type is None
    assert req.crate_filter is None
    assert req.has_examples is None
    assert req.visibility is None
    assert req.deprecated is None


def test_query_preprocessing():
    """Test query field preprocessing and normalization."""
    # Test basic query
    req = SearchItemsRequest(crate_name="tokio", query="async runtime")
    assert req.query == "async runtime"

    # Test whitespace normalization
    req = SearchItemsRequest(crate_name="tokio", query="  async   runtime  ")
    assert req.query == "async runtime"

    req = SearchItemsRequest(crate_name="tokio", query="async\t\nruntime")
    assert req.query == "async runtime"

    # Test Unicode normalization (NFKC)
    # Composed character é vs decomposed e + combining acute
    req = SearchItemsRequest(crate_name="tokio", query="café")
    normalized = unicodedata.normalize("NFKC", "café")
    assert req.query == normalized

    # Test compatibility characters
    req = SearchItemsRequest(crate_name="tokio", query="ﬀ ﬁ ﬂ")  # ff fi fl ligatures
    assert req.query == "ff fi fl"

    # Test special characters are preserved
    req = SearchItemsRequest(crate_name="tokio", query="tokio::spawn")
    assert req.query == "tokio::spawn"

    req = SearchItemsRequest(crate_name="tokio", query="Result<T, E>")
    assert req.query == "Result<T, E>"

    req = SearchItemsRequest(crate_name="tokio", query="#[derive(Debug)]")
    assert req.query == "#[derive(Debug)]"

    # Test length after normalization
    long_query = "a" * 499
    req = SearchItemsRequest(crate_name="tokio", query=long_query)
    assert len(req.query) == 499


def test_query_validation_errors():
    """Test query validation error cases."""
    # Test empty query
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="")
    assert "Query cannot be empty" in str(exc_info.value)

    # Test whitespace-only query
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="   \t\n   ")
    assert "Query cannot be empty after normalization" in str(exc_info.value)

    # Test query too long
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="a" * 501)
    assert "Query too long" in str(exc_info.value)
    assert "501 characters" in str(exc_info.value)

    # Test None query (should be caught by Pydantic as required field)
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query=None)
    # Pydantic will handle this as a missing required field


def test_query_unicode_edge_cases():
    """Test Unicode edge cases in query preprocessing."""
    # Test various Unicode normalization forms
    test_cases = [
        ("Å", "Å"),  # Angstrom symbol normalizes to regular A with ring
        ("½", "1⁄2"),  # Fraction normalization
        ("①", "1"),  # Circled number
        ("ℌ", "H"),  # Double-struck capital H
        ("ℍ", "H"),  # Double-struck capital H (different codepoint)
        ("㎡", "m2"),  # Square meter symbol
        ("℡", "TEL"),  # Telephone sign
    ]

    for input_str, expected in test_cases:
        req = SearchItemsRequest(crate_name="tokio", query=input_str)
        assert req.query == expected

    # Test that emoji are preserved (not normalized away)
    req = SearchItemsRequest(crate_name="tokio", query="rust 🦀 async")
    assert "🦀" in req.query
    assert req.query == "rust 🦀 async"

    # Test mixed scripts
    req = SearchItemsRequest(crate_name="tokio", query="async 异步 非同期")
    assert req.query == "async 异步 非同期"

    # Test bidirectional text
    req = SearchItemsRequest(crate_name="tokio", query="async مزامن غير")
    assert "async" in req.query
    assert "مزامن" in req.query


def test_query_mcp_compatibility():
    """Test query preprocessing with MCP client string inputs."""
    # Test that string inputs work as expected
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "  async  runtime  "}
    )
    assert req.query == "async runtime"

    # Test with various MCP-style inputs
    req = SearchItemsRequest.model_validate(
        {"crate_name": "tokio", "query": "tokio::spawn\n\n"}
    )
    assert req.query == "tokio::spawn"

    # Test with special characters from MCP clients
    req = SearchItemsRequest.model_validate(
        {"crate_name": "serde", "query": "Deserialize<'de>"}
    )
    assert req.query == "Deserialize<'de>"
