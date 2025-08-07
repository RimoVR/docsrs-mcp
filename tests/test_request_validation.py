"""Tests for request model validation and validation utilities."""

import pytest
from pydantic import ValidationError

from docsrs_mcp.models import (
    GetCrateSummaryRequest,
    GetItemDocRequest,
    GetModuleTreeRequest,
    ListVersionsRequest,
)
from docsrs_mcp.validation import (
    coerce_to_int_with_bounds,
    validate_crate_name,
    validate_optional_path,
    validate_rust_path,
    validate_version_string,
)


class TestValidationUtilities:
    """Test the validation utility functions."""

    def test_validate_crate_name_valid(self):
        """Test valid crate names."""
        assert validate_crate_name("tokio") == "tokio"
        assert validate_crate_name("serde_json") == "serde_json"
        assert validate_crate_name("async-trait") == "async-trait"
        assert validate_crate_name("std") == "std"
        assert validate_crate_name("core") == "core"
        assert validate_crate_name("a123") == "a123"
        assert validate_crate_name("test_crate-123") == "test_crate-123"

    def test_validate_crate_name_invalid(self):
        """Test invalid crate names."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_crate_name(None)

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_crate_name("")

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_crate_name("  ")

        with pytest.raises(ValueError, match="must be 64 characters or less"):
            validate_crate_name("a" * 65)

        with pytest.raises(ValueError, match="must be a valid Rust crate name"):
            validate_crate_name("CamelCase")

        with pytest.raises(ValueError, match="must be a valid Rust crate name"):
            validate_crate_name("has spaces")

        with pytest.raises(ValueError, match="must be a valid Rust crate name"):
            validate_crate_name("special@chars")

    def test_validate_version_string_valid(self):
        """Test valid version strings."""
        assert validate_version_string("1.0.0") == "1.0.0"
        assert validate_version_string("2.1.3-alpha") == "2.1.3-alpha"
        assert validate_version_string("1.0.0-beta.1") == "1.0.0-beta.1"
        assert validate_version_string("1.0.0+build123") == "1.0.0+build123"
        assert validate_version_string("latest") == "latest"
        assert validate_version_string("LATEST") == "latest"
        assert validate_version_string(None) is None
        assert validate_version_string("") is None
        assert validate_version_string("  ") is None

    def test_validate_version_string_invalid(self):
        """Test invalid version strings."""
        with pytest.raises(ValueError, match="must be a valid semantic version"):
            validate_version_string("v1.0.0")  # v prefix not allowed

        with pytest.raises(ValueError, match="must be a valid semantic version"):
            validate_version_string("1.0")  # Missing patch version

        with pytest.raises(ValueError, match="must be a valid semantic version"):
            validate_version_string("not-a-version")

    def test_validate_rust_path_valid(self):
        """Test valid Rust paths."""
        assert validate_rust_path("tokio::spawn") == "tokio::spawn"
        assert validate_rust_path("std::vec::Vec") == "std::vec::Vec"
        assert validate_rust_path("crate") == "crate"
        assert validate_rust_path("my_module::MyStruct") == "my_module::MyStruct"
        assert validate_rust_path("a::b::c::d::e") == "a::b::c::d::e"
        assert validate_rust_path("_private::function") == "_private::function"

    def test_validate_rust_path_invalid(self):
        """Test invalid Rust paths."""
        with pytest.raises(ValueError, match="cannot be None"):
            validate_rust_path(None)

        with pytest.raises(ValueError, match="cannot be empty"):
            validate_rust_path("")

        with pytest.raises(ValueError, match="must be 256 characters or less"):
            validate_rust_path("a::" * 100)

        with pytest.raises(ValueError, match="must be a valid Rust path"):
            validate_rust_path("invalid path with spaces")

        with pytest.raises(ValueError, match="must be a valid Rust path"):
            validate_rust_path("123::starts_with_number")

        with pytest.raises(ValueError, match="must be a valid Rust path"):
            validate_rust_path("has-hyphen::path")

    def test_coerce_to_int_with_bounds(self):
        """Test integer coercion with bounds checking."""
        # Test normal integers
        assert coerce_to_int_with_bounds(5, "test", 1, 10) == 5
        assert coerce_to_int_with_bounds(1, "test", 1, 10) == 1
        assert coerce_to_int_with_bounds(10, "test", 1, 10) == 10

        # Test string conversion
        assert coerce_to_int_with_bounds("5", "test", 1, 10) == 5
        assert coerce_to_int_with_bounds("1", "test", 1, 10) == 1
        assert coerce_to_int_with_bounds("10", "test", 1, 10) == 10

        # Test bounds violations
        with pytest.raises(ValueError, match="must be between 1 and 10"):
            coerce_to_int_with_bounds(0, "test", 1, 10)

        with pytest.raises(ValueError, match="must be between 1 and 10"):
            coerce_to_int_with_bounds(11, "test", 1, 10)

        # Test invalid inputs
        with pytest.raises(ValueError, match="cannot be None"):
            coerce_to_int_with_bounds(None, "test", 1, 10)

        with pytest.raises(ValueError, match="cannot be converted"):
            coerce_to_int_with_bounds("not-a-number", "test", 1, 10)

    def test_validate_optional_path(self):
        """Test optional path validation."""
        assert validate_optional_path(None) is None
        assert validate_optional_path("") is None
        assert validate_optional_path("  ") is None
        assert validate_optional_path("src/lib") == "src/lib"
        assert validate_optional_path("tests/integration") == "tests/integration"
        assert validate_optional_path("module/submodule") == "module/submodule"

        # Test invalid paths
        with pytest.raises(ValueError, match="must be 256 characters or less"):
            validate_optional_path("a" * 257)

        with pytest.raises(ValueError, match="must be a valid module path"):
            validate_optional_path("has spaces/path")


class TestGetCrateSummaryRequest:
    """Test GetCrateSummaryRequest validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        req = GetCrateSummaryRequest(crate_name="tokio", version="1.35.1")
        assert req.crate_name == "tokio"
        assert req.version == "1.35.1"

        req = GetCrateSummaryRequest(crate_name="serde")
        assert req.crate_name == "serde"
        assert req.version is None

    def test_string_coercion(self):
        """Test MCP client string parameter handling."""
        # Test with dict (simulating JSON input)
        data = {"crate_name": "tokio", "version": "latest"}
        req = GetCrateSummaryRequest.model_validate(data)
        assert req.crate_name == "tokio"
        assert req.version == "latest"

    def test_invalid_crate_name(self):
        """Test invalid crate name validation."""
        with pytest.raises(ValidationError) as exc_info:
            GetCrateSummaryRequest(crate_name="Invalid-Name")
        assert "must be a valid Rust crate name" in str(exc_info.value)

    def test_invalid_version(self):
        """Test invalid version validation."""
        with pytest.raises(ValidationError) as exc_info:
            GetCrateSummaryRequest(crate_name="tokio", version="v1.0.0")
        assert "must be a valid semantic version" in str(exc_info.value)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            GetCrateSummaryRequest.model_validate(
                {"crate_name": "tokio", "version": "latest", "extra_field": "value"}
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestGetItemDocRequest:
    """Test GetItemDocRequest validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        req = GetItemDocRequest(
            crate_name="serde", item_path="serde::Deserialize", version="1.0.193"
        )
        assert req.crate_name == "serde"
        assert req.item_path == "serde::Deserialize"
        assert req.version == "1.0.193"

        req = GetItemDocRequest(crate_name="tokio", item_path="tokio::spawn")
        assert req.crate_name == "tokio"
        assert req.item_path == "tokio::spawn"
        assert req.version is None

    def test_crate_root_path(self):
        """Test special 'crate' path handling."""
        req = GetItemDocRequest(crate_name="serde", item_path="crate")
        assert req.item_path == "crate"

    def test_invalid_item_path(self):
        """Test invalid item path validation."""
        with pytest.raises(ValidationError) as exc_info:
            GetItemDocRequest(crate_name="tokio", item_path="invalid path with spaces")
        assert "must be a valid Rust path" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            GetItemDocRequest(crate_name="tokio", item_path="123::invalid")
        assert "must be a valid Rust path" in str(exc_info.value)

    def test_string_coercion(self):
        """Test MCP client string parameter handling."""
        data = {
            "crate_name": "tokio",
            "item_path": "tokio::spawn",
            "version": "1.35.1",
        }
        req = GetItemDocRequest.model_validate(data)
        assert req.crate_name == "tokio"
        assert req.item_path == "tokio::spawn"
        assert req.version == "1.35.1"


class TestGetModuleTreeRequest:
    """Test GetModuleTreeRequest validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        req = GetModuleTreeRequest(crate_name="tokio", version="1.35.1")
        assert req.crate_name == "tokio"
        assert req.version == "1.35.1"

        req = GetModuleTreeRequest(crate_name="actix-web")
        assert req.crate_name == "actix-web"
        assert req.version is None

    def test_invalid_crate_name(self):
        """Test invalid crate name validation."""
        with pytest.raises(ValidationError) as exc_info:
            GetModuleTreeRequest(crate_name="")
        assert "cannot be empty" in str(exc_info.value)

    def test_string_coercion(self):
        """Test MCP client string parameter handling."""
        data = {"crate_name": "serde", "version": "latest"}
        req = GetModuleTreeRequest.model_validate(data)
        assert req.crate_name == "serde"
        assert req.version == "latest"


class TestListVersionsRequest:
    """Test ListVersionsRequest validation."""

    def test_valid_request(self):
        """Test valid request creation."""
        req = ListVersionsRequest(crate_name="tokio")
        assert req.crate_name == "tokio"

        req = ListVersionsRequest(crate_name="serde_json")
        assert req.crate_name == "serde_json"

    def test_invalid_crate_name(self):
        """Test invalid crate name validation."""
        with pytest.raises(ValidationError) as exc_info:
            ListVersionsRequest(crate_name="Invalid Name")
        assert "must be a valid Rust crate name" in str(exc_info.value)

        with pytest.raises(ValidationError) as exc_info:
            ListVersionsRequest(crate_name="")
        assert "cannot be empty" in str(exc_info.value)

    def test_string_coercion(self):
        """Test MCP client string parameter handling."""
        data = {"crate_name": "async-trait"}
        req = ListVersionsRequest.model_validate(data)
        assert req.crate_name == "async-trait"

    def test_extra_fields_forbidden(self):
        """Test that extra fields are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            ListVersionsRequest.model_validate(
                {"crate_name": "tokio", "unexpected_field": "value"}
            )
        assert "Extra inputs are not permitted" in str(exc_info.value)


class TestMCPCompatibility:
    """Test MCP client compatibility scenarios."""

    def test_string_version_parameters(self):
        """Test that string versions are properly handled."""
        # Get crate summary with string version
        req = GetCrateSummaryRequest.model_validate(
            {"crate_name": "tokio", "version": "1.35.1"}
        )
        assert req.version == "1.35.1"

        # Get item doc with string version
        req = GetItemDocRequest.model_validate(
            {
                "crate_name": "serde",
                "item_path": "serde::Serialize",
                "version": "latest",
            }
        )
        assert req.version == "latest"

    def test_null_handling(self):
        """Test that null/None values are properly handled."""
        req = GetCrateSummaryRequest.model_validate(
            {"crate_name": "tokio", "version": None}
        )
        assert req.version is None

        req = GetModuleTreeRequest.model_validate(
            {"crate_name": "serde", "version": None}
        )
        assert req.version is None

    def test_whitespace_trimming(self):
        """Test that whitespace is properly trimmed."""
        req = GetCrateSummaryRequest.model_validate(
            {"crate_name": "  tokio  ", "version": "  latest  "}
        )
        assert req.crate_name == "tokio"
        assert req.version == "latest"

        req = GetItemDocRequest.model_validate(
            {
                "crate_name": "  serde  ",
                "item_path": "  serde::Deserialize  ",
                "version": "  1.0.0  ",
            }
        )
        assert req.crate_name == "serde"
        assert req.item_path == "serde::Deserialize"
        assert req.version == "1.0.0"
