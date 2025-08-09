"""
Tests for enhanced validation error messages.

This module tests the enhanced validation functions with contextual error messages,
ensuring that all error messages include examples, ranges, and actionable guidance.
"""

import pytest
from pydantic import ValidationError

from docsrs_mcp.models import (
    GetCrateSummaryRequest,
    SearchItemsRequest,
    StartPreIngestionRequest,
)
from docsrs_mcp.validation import (
    ERROR_TEMPLATES,
    coerce_to_float_with_bounds,
    coerce_to_int_with_bounds,
    format_error_message,
    format_examples,
    validate_crate_name,
    validate_rust_path,
    validate_version_string,
)


class TestErrorFormatting:
    """Test error message formatting utilities."""

    def test_format_examples(self):
        """Test that examples are formatted correctly."""
        assert format_examples([1, 2, 3]) == "1, 2, 3"
        assert format_examples(["tokio", "serde"]) == "'tokio', 'serde'"
        assert format_examples([]) == "No examples available"
        assert format_examples([1, 2, 3, 4, 5], max_examples=3) == "1, 2, 3"

    def test_format_error_message(self):
        """Test error message template formatting."""
        msg = format_error_message(
            "range", field="k", min=1, max=20, value=25, examples="1, 5, 10"
        )
        assert "k must be between 1 and 20" in msg
        assert "Got: 25" in msg
        assert "Examples: 1, 5, 10" in msg

    def test_custom_error_message(self):
        """Test custom error message template."""
        msg = format_error_message("custom", message="Custom error occurred")
        assert msg == "Custom error occurred"

        # Test default message when no message provided
        msg = format_error_message("custom")
        assert msg == "Validation failed"


class TestIntegerValidation:
    """Test integer validation with enhanced error messages."""

    def test_valid_integer(self):
        """Test valid integer values."""
        assert coerce_to_int_with_bounds(5, "test", 1, 10) == 5
        assert coerce_to_int_with_bounds("5", "test", 1, 10) == 5
        assert coerce_to_int_with_bounds(1, "test", 1, 10) == 1
        assert coerce_to_int_with_bounds(10, "test", 1, 10) == 10

    def test_none_value(self):
        """Test that None values raise error with examples."""
        with pytest.raises(ValueError) as exc_info:
            coerce_to_int_with_bounds(None, "test_field", 1, 10)
        error_msg = str(exc_info.value)
        assert "test_field is required" in error_msg
        assert "Examples:" in error_msg

    def test_string_conversion(self):
        """Test string to integer conversion."""
        assert coerce_to_int_with_bounds("42", "test", 1, 100) == 42

        # Test invalid string
        with pytest.raises(ValueError) as exc_info:
            coerce_to_int_with_bounds("not_a_number", "test_field", 1, 10)
        error_msg = str(exc_info.value)
        assert "test_field must be an integer" in error_msg
        assert "Got: string 'not_a_number'" in error_msg
        assert "Examples:" in error_msg

    def test_out_of_bounds(self):
        """Test out of bounds values."""
        with pytest.raises(ValueError) as exc_info:
            coerce_to_int_with_bounds(100, "k", 1, 20, examples=[1, 5, 10])
        error_msg = str(exc_info.value)
        assert "k must be between 1 and 20" in error_msg
        assert "Got: 100" in error_msg
        assert "Examples: 1, 5, 10" in error_msg

    def test_custom_examples(self):
        """Test custom examples are used."""
        with pytest.raises(ValueError) as exc_info:
            coerce_to_int_with_bounds(0, "count", 10, 500, examples=[100, 200, 500])
        error_msg = str(exc_info.value)
        assert "Examples: 100, 200, 500" in error_msg


class TestFloatValidation:
    """Test float validation with enhanced error messages."""

    def test_valid_float(self):
        """Test valid float values."""
        assert coerce_to_float_with_bounds(0.5, "score", 0.0, 1.0) == 0.5
        assert coerce_to_float_with_bounds("0.5", "score", 0.0, 1.0) == 0.5
        assert coerce_to_float_with_bounds(1, "score", 0.0, 1.0) == 1.0

    def test_none_value(self):
        """Test that None values raise error with examples."""
        with pytest.raises(ValueError) as exc_info:
            coerce_to_float_with_bounds(None, "score", 0.0, 1.0)
        error_msg = str(exc_info.value)
        assert "score is required" in error_msg
        assert "Examples:" in error_msg

    def test_string_conversion(self):
        """Test string to float conversion."""
        assert coerce_to_float_with_bounds("0.75", "weight", 0.0, 1.0) == 0.75

        # Test invalid string
        with pytest.raises(ValueError) as exc_info:
            coerce_to_float_with_bounds("not_a_float", "weight", 0.0, 1.0)
        error_msg = str(exc_info.value)
        assert "weight must be a decimal number" in error_msg
        assert "Got: string 'not_a_float'" in error_msg

    def test_out_of_bounds(self):
        """Test out of bounds values."""
        with pytest.raises(ValueError) as exc_info:
            coerce_to_float_with_bounds(
                1.5, "score", 0.0, 1.0, examples=[0.5, 0.7, 0.9]
            )
        error_msg = str(exc_info.value)
        assert "score must be between 0.0 and 1.0" in error_msg
        assert "Got: 1.5" in error_msg
        assert "Examples: 0.5, 0.7, 0.9" in error_msg


class TestStringValidation:
    """Test string validation with enhanced error messages."""

    def test_valid_crate_name(self):
        """Test valid crate names."""
        assert validate_crate_name("tokio") == "tokio"
        assert validate_crate_name("serde_json") == "serde_json"
        assert validate_crate_name("async-trait") == "async-trait"

    def test_invalid_crate_name(self):
        """Test invalid crate names with enhanced errors."""
        with pytest.raises(ValueError) as exc_info:
            validate_crate_name("InvalidCrate")
        error_msg = str(exc_info.value)
        assert "crate_name must match pattern" in error_msg
        assert "lowercase letters, numbers, hyphens, underscores only" in error_msg
        assert "Got: 'InvalidCrate'" in error_msg
        assert "Examples: 'tokio', 'serde_json', 'async-trait'" in error_msg

    def test_crate_name_too_long(self):
        """Test crate name length validation."""
        with pytest.raises(ValueError) as exc_info:
            validate_crate_name("a" * 100)
        error_msg = str(exc_info.value)
        assert "crate_name must be 64 characters or less" in error_msg
        assert "Got 100 characters" in error_msg

    def test_valid_version(self):
        """Test valid version strings."""
        assert validate_version_string("1.0.0") == "1.0.0"
        assert validate_version_string("2.1.3-alpha") == "2.1.3-alpha"
        assert validate_version_string("latest") == "latest"
        assert validate_version_string(None) is None

    def test_invalid_version(self):
        """Test invalid version strings."""
        with pytest.raises(ValueError) as exc_info:
            validate_version_string("v1.0.0")
        error_msg = str(exc_info.value)
        assert "version must match pattern" in error_msg
        assert "semantic version (MAJOR.MINOR.PATCH) or 'latest'" in error_msg
        assert "Examples:" in error_msg

    def test_valid_rust_path(self):
        """Test valid Rust paths."""
        assert validate_rust_path("tokio::spawn") == "tokio::spawn"
        assert validate_rust_path("std::vec::Vec") == "std::vec::Vec"
        assert validate_rust_path("crate") == "crate"

    def test_invalid_rust_path(self):
        """Test invalid Rust paths."""
        with pytest.raises(ValueError) as exc_info:
            validate_rust_path("invalid path with spaces")
        error_msg = str(exc_info.value)
        assert "item_path must match pattern" in error_msg
        assert "Rust identifiers separated by '::'" in error_msg
        assert "Examples:" in error_msg


class TestPydanticModelValidation:
    """Test Pydantic model validation with enhanced errors."""

    def test_search_items_k_validation(self):
        """Test SearchItemsRequest k parameter validation."""
        # Valid request
        request = SearchItemsRequest(crate_name="tokio", query="spawn task", k=5)
        assert request.k == 5

        # Test string conversion
        request = SearchItemsRequest(crate_name="tokio", query="spawn task", k="10")
        assert request.k == 10

        # Test out of bounds
        with pytest.raises(ValidationError) as exc_info:
            SearchItemsRequest(crate_name="tokio", query="spawn task", k=100)
        errors = exc_info.value.errors()
        assert any(
            "k (number of results) must be between 1 and 20" in str(e) for e in errors
        )

    def test_crate_name_validation(self):
        """Test crate name validation in models."""
        # Valid request
        request = GetCrateSummaryRequest(crate_name="serde")
        assert request.crate_name == "serde"

        # Invalid crate name
        with pytest.raises(ValidationError) as exc_info:
            GetCrateSummaryRequest(crate_name="Invalid-Crate-Name")
        errors = exc_info.value.errors()
        assert any(
            "lowercase letters, numbers, hyphens, underscores only" in str(e)
            for e in errors
        )

    def test_pre_ingestion_validation(self):
        """Test StartPreIngestionRequest validation."""
        # Valid request
        request = StartPreIngestionRequest(concurrency=3, count=100)
        assert request.concurrency == 3
        assert request.count == 100

        # Test string conversion
        request = StartPreIngestionRequest(concurrency="5", count="200")
        assert request.concurrency == 5
        assert request.count == 200

        # Test out of bounds concurrency
        with pytest.raises(ValidationError) as exc_info:
            StartPreIngestionRequest(concurrency=20, count=100)
        errors = exc_info.value.errors()
        assert any(
            "concurrency (parallel download workers) must be between 1 and 10" in str(e)
            for e in errors
        )

        # Test out of bounds count
        with pytest.raises(ValidationError) as exc_info:
            StartPreIngestionRequest(concurrency=3, count=1000)
        errors = exc_info.value.errors()
        assert any(
            "count (number of crates to pre-ingest) must be between 10 and 500"
            in str(e)
            for e in errors
        )


class TestErrorMessageConsistency:
    """Test that error messages are consistent across the application."""

    def test_all_templates_have_examples(self):
        """Test that all error templates include examples placeholder."""
        for key, template in ERROR_TEMPLATES.items():
            if key != "custom":  # Custom template is special
                assert "{examples}" in template or key == "enum", (
                    f"Template '{key}' should include examples placeholder"
                )

    def test_range_errors_include_bounds(self):
        """Test that range errors always include min and max."""
        template = ERROR_TEMPLATES["range"]
        assert "{min}" in template
        assert "{max}" in template
        assert "{value}" in template
        assert "{examples}" in template

    def test_type_errors_include_types(self):
        """Test that type errors include expected and actual types."""
        template = ERROR_TEMPLATES["type"]
        assert "{expected_type}" in template
        assert "{actual_type}" in template
        assert "{value}" in template


@pytest.mark.asyncio
async def test_api_error_response_format():
    """Test that API error responses follow the enhanced format."""
    from fastapi.testclient import TestClient

    from docsrs_mcp.app import app

    client = TestClient(app)

    # Test invalid k parameter
    response = client.post(
        "/mcp/tools/search_items", json={"crate_name": "tokio", "query": "test", "k": "invalid"}
    )

    assert response.status_code == 422
    error_data = response.json()
    assert "error" in error_data
    assert "details" in error_data
    assert "suggestion" in error_data
    assert error_data["error"] == "Validation failed"
    assert "Please check the field requirements" in error_data["suggestion"]
