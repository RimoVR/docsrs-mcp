"""
Reusable validation utilities for docsrs-mcp request models.

This module provides common validation functions used across multiple
request models to ensure consistency and reduce code duplication.
"""

import re
from typing import Any

# Precompiled regex patterns for performance
CRATE_NAME_PATTERN = re.compile(r"^[a-z0-9_-]+$")
VERSION_PATTERN = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(-((0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
    r"(\.(0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?"
    r"(\+([0-9a-zA-Z-]+(\.[0-9a-zA-Z-]+)*))?$"
)
RUST_PATH_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(::[a-zA-Z_][a-zA-Z0-9_]*)*$")


def validate_crate_name(value: Any, field_name: str = "crate_name") -> str:
    """
    Validate Rust crate naming conventions.

    Args:
        value: The value to validate (can be string or None)
        field_name: Name of the field for error messages

    Returns:
        Validated crate name string

    Raises:
        ValueError: If the crate name is invalid
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    # Check for empty string
    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    # Check length limits
    if len(value) > 64:
        raise ValueError(
            f"{field_name} must be 64 characters or less. "
            f"Got {len(value)} characters: {repr(value)[:100]}"
        )

    # Validate against Rust crate naming rules
    if not CRATE_NAME_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a valid Rust crate name "
            f"(lowercase, alphanumeric, hyphens, underscores). "
            f"Got: {repr(value)[:100]}. "
            f"Examples: 'tokio', 'serde_json', 'async-trait'"
        )

    return value


def validate_version_string(value: Any, field_name: str = "version") -> str | None:
    """
    Validate semantic version string or 'latest'.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        Validated version string, 'latest', or None

    Raises:
        ValueError: If the version string is invalid
    """
    if value is None:
        return None  # Preserve None value for default handling at app layer

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        return None  # Empty string treated as None

    # Check for special value
    if value.lower() == "latest":
        return "latest"

    # Validate semantic version format
    if not VERSION_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a valid semantic version or 'latest'. "
            f"Got: {repr(value)[:100]}. "
            f"Examples: '1.0.0', '2.1.3-alpha', 'latest'"
        )

    return value


def validate_rust_path(value: Any, field_name: str = "item_path") -> str:
    """
    Validate Rust item path syntax.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        Validated Rust path string

    Raises:
        ValueError: If the path is invalid
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        raise ValueError(f"{field_name} cannot be empty")

    # Special case for crate root
    if value == "crate":
        return value

    # Check length limits
    if len(value) > 256:
        raise ValueError(
            f"{field_name} must be 256 characters or less. "
            f"Got {len(value)} characters: {repr(value)[:100]}"
        )

    # Validate Rust path syntax
    if not RUST_PATH_PATTERN.match(value):
        raise ValueError(
            f"{field_name} must be a valid Rust path "
            f"(identifiers separated by '::'). "
            f"Got: {repr(value)[:100]}. "
            f"Examples: 'tokio::spawn', 'std::vec::Vec', 'crate'"
        )

    return value


def coerce_to_int_with_bounds(
    value: Any, field_name: str, min_val: int, max_val: int
) -> int:
    """
    Generic integer coercion with bounds checking.

    Handles MCP client compatibility by accepting string inputs.

    Args:
        value: The value to validate and coerce
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Returns:
        Validated integer within bounds

    Raises:
        ValueError: If the value cannot be converted or is out of bounds
    """
    if value is None:
        raise ValueError(f"{field_name} cannot be None")

    # Handle string inputs from MCP clients
    if isinstance(value, str):
        try:
            value = int(value)
        except ValueError as e:
            raise ValueError(
                f"{field_name} must be an integer. "
                f"Got string that cannot be converted: {repr(value)[:100]}"
            ) from e

    # Ensure we have an integer
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{field_name} must be an integer. "
                f"Got {type(value).__name__}: {repr(value)[:100]}"
            ) from e

    # Check bounds
    if value < min_val or value > max_val:
        raise ValueError(
            f"{field_name} must be between {min_val} and {max_val} (inclusive). "
            f"Got: {value}"
        )

    return value


def validate_optional_path(value: Any, field_name: str = "path") -> str | None:
    """
    Validate an optional module path.

    Args:
        value: The value to validate
        field_name: Name of the field for error messages

    Returns:
        Validated path string or None

    Raises:
        ValueError: If the path is invalid
    """
    if value is None:
        return None

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        return None

    # Validate similar to Rust paths but allow module paths
    if len(value) > 256:
        raise ValueError(
            f"{field_name} must be 256 characters or less. "
            f"Got {len(value)} characters: {repr(value)[:100]}"
        )

    # Module paths can contain :: and /
    if not re.match(r"^[a-zA-Z0-9_/:]+$", value):
        raise ValueError(
            f"{field_name} must be a valid module path. "
            f"Got: {repr(value)[:100]}. "
            f"Examples: 'src/lib', 'tests/integration', 'examples'"
        )

    return value
