"""Tests for new filter validators in SearchItemsRequest."""

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

    # Test None value
    req = SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=None)
    assert req.min_doc_length is None

    # Test out of range values
    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=99)
    assert "greater than or equal to 100" in str(exc_info.value).lower()

    with pytest.raises(ValidationError) as exc_info:
        SearchItemsRequest(crate_name="tokio", query="spawn", min_doc_length=10001)
    assert "less than or equal to 10000" in str(exc_info.value).lower()


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
