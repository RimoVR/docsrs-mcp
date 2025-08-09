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


def validate_item_path_with_fallback(
    path: str | None, item_id: str | None, item_kind: str | None
) -> tuple[str, bool]:
    """
    Validate item_path and generate fallback if needed.

    This function ensures we always have a valid non-empty path for database
    insertion, preventing NOT NULL constraint violations.

    Args:
        path: The original path (may be None or empty)
        item_id: The item's ID for fallback generation
        item_kind: The item's kind/type for fallback generation

    Returns:
        tuple: (validated_path, used_fallback)
            - validated_path: A non-empty, valid path string
            - used_fallback: True if a fallback was generated
    """
    # Check for empty or whitespace-only path
    if not path or not path.strip():
        # Generate fallback path using item metadata
        if item_id and item_id.strip():
            # Use item_id as base for fallback
            if item_kind and item_kind.strip():
                fallback = f"{item_kind}::{item_id}"
            else:
                fallback = f"item::{item_id}"
        elif item_kind and item_kind.strip():
            # Use kind with hash for uniqueness
            fallback = f"{item_kind}::unknown_{abs(hash(str(path)))}"
        else:
            # Last resort: generic fallback
            fallback = f"unknown::item_{abs(hash(str((path, item_id, item_kind))))}"

        return fallback, True

    # Check if path is valid according to Rust path pattern
    path_stripped = path.strip()

    # Special case for crate root
    if path_stripped == "crate":
        return path_stripped, False

    # Truncate if too long
    if len(path_stripped) > 256:
        path_stripped = path_stripped[:256]
        # Ensure we don't cut in the middle of ::
        if path_stripped.endswith(":"):
            path_stripped = path_stripped[:-1]
        elif path_stripped.endswith("::"):
            path_stripped = path_stripped[:-2]

    # If it matches the pattern, return as-is
    if RUST_PATH_PATTERN.match(path_stripped):
        return path_stripped, False

    # Try to sanitize the path
    # Replace common invalid characters with underscores
    sanitized = path_stripped
    for char in [" ", "-", ".", "/", "\\", "(", ")", "[", "]", "{", "}", "<", ">", ","]:
        sanitized = sanitized.replace(char, "_")

    # Remove consecutive underscores
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")

    # Remove leading/trailing underscores from segments
    segments = sanitized.split("::")
    segments = [seg.strip("_") for seg in segments if seg.strip("_")]

    if segments:
        sanitized = "::".join(segments)
        # Ensure it starts with a valid identifier character
        if sanitized and not sanitized[0].isalpha() and sanitized[0] != "_":
            sanitized = "_" + sanitized

        # Check if sanitized version is valid
        if RUST_PATH_PATTERN.match(sanitized):
            return sanitized, True

    # If sanitization failed, generate a fallback
    if item_id and item_id.strip():
        fallback = f"sanitized::{item_id}"
    elif item_kind and item_kind.strip():
        fallback = f"{item_kind}::path_{abs(hash(path_stripped))}"
    else:
        fallback = f"invalid::path_{abs(hash(path_stripped))}"

    return fallback, True


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
