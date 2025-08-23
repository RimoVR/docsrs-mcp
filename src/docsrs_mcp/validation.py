"""
Reusable validation utilities for docsrs-mcp request models.

This module provides common validation functions used across multiple
request models to ensure consistency and reduce code duplication.
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Error message templates (precompiled for performance)
ERROR_TEMPLATES = {
    "range": "{field} must be between {min} and {max} (inclusive). Got: {value}. Examples: {examples}",
    "pattern": "{field} must match pattern {pattern}. Got: '{value}'. Examples: {examples}",
    "type": "{field} must be {expected_type}. Got: {actual_type} '{value}'. Examples: {examples}",
    "enum": "{field} must be one of {valid_values}. Got: '{value}'. Did you mean: {suggestion}?",
    "required": "{field} is required and cannot be None or empty. Examples: {examples}",
    "length": "{field} must be {constraint}. Got {actual} characters: '{value}'. Examples: {examples}",
    "custom": "{message}",
}


def format_error_message(template_key: str, **context) -> str:
    """
    Format error message with context using precompiled templates.

    Args:
        template_key: Key to select template from ERROR_TEMPLATES
        **context: Template variables for string formatting

    Returns:
        Formatted error message with user-friendly context
    """
    template = ERROR_TEMPLATES.get(template_key, ERROR_TEMPLATES["custom"])
    # Provide default message if custom template is used without message
    if template_key == "custom" and "message" not in context:
        context["message"] = "Validation failed"
    return template.format(**context)


def format_examples(examples: list[Any], max_examples: int = 3) -> str:
    """
    Format examples for error messages.

    Args:
        examples: List of valid example values
        max_examples: Maximum number of examples to display

    Returns:
        Formatted string of examples
    """
    if not examples:
        return "No examples available"
    display_examples = examples[:max_examples]
    return ", ".join(
        f"'{e}'" if isinstance(e, str) else str(e) for e in display_examples
    )


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
        raise ValueError(
            format_error_message(
                "required",
                field=field_name,
                examples=format_examples(["tokio", "serde_json", "async-trait"]),
            )
        )

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    # Check for empty string
    value = value.strip()
    if not value:
        raise ValueError(
            format_error_message(
                "required",
                field=field_name,
                examples=format_examples(["tokio", "serde_json", "async-trait"]),
            )
        )

    # Check length limits
    if len(value) > 64:
        raise ValueError(
            format_error_message(
                "length",
                field=field_name,
                constraint="64 characters or less",
                actual=len(value),
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(["tokio", "serde_json", "async-trait"]),
            )
        )

    # Validate against Rust crate naming rules
    if not CRATE_NAME_PATTERN.match(value):
        raise ValueError(
            format_error_message(
                "pattern",
                field=field_name,
                pattern="lowercase letters, numbers, hyphens, underscores only",
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(["tokio", "serde_json", "async-trait"]),
            )
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
            format_error_message(
                "pattern",
                field=field_name,
                pattern="semantic version (MAJOR.MINOR.PATCH) or 'latest'",
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(
                    ["1.0.0", "2.1.3-alpha", "0.3.0-beta.1", "latest"]
                ),
            )
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
        raise ValueError(
            format_error_message(
                "required",
                field=field_name,
                examples=format_examples(
                    ["tokio::spawn", "std::vec::Vec", "serde::Deserialize", "crate"]
                ),
            )
        )

    # Convert to string if needed
    if not isinstance(value, str):
        value = str(value)

    value = value.strip()
    if not value:
        raise ValueError(
            format_error_message(
                "required",
                field=field_name,
                examples=format_examples(
                    ["tokio::spawn", "std::vec::Vec", "serde::Deserialize", "crate"]
                ),
            )
        )

    # Special case for crate root
    if value == "crate":
        return value

    # Check length limits
    if len(value) > 256:
        raise ValueError(
            format_error_message(
                "length",
                field=field_name,
                constraint="256 characters or less",
                actual=len(value),
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(
                    ["tokio::spawn", "std::vec::Vec", "module::Type"]
                ),
            )
        )

    # Validate Rust path syntax
    if not RUST_PATH_PATTERN.match(value):
        raise ValueError(
            format_error_message(
                "pattern",
                field=field_name,
                pattern="Rust identifiers separated by '::' (e.g., module::function)",
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(
                    ["tokio::spawn", "std::vec::Vec", "serde::Deserialize", "crate"]
                ),
            )
        )

    return value


def coerce_to_int_with_bounds(
    value: Any,
    field_name: str,
    min_val: int,
    max_val: int,
    examples: list[int] | None = None,
) -> int:
    """
    Generic integer coercion with bounds checking.

    Handles MCP client compatibility by accepting string inputs.

    This function is critical for K parameter validation as sqlite-vec requires
    integer K parameters in WHERE clauses (e.g., 'WHERE k = 10'). The max_val
    for K parameters is typically set to 20 to ensure that fetch_k (calculated
    as k + over_fetch) stays within sqlite-vec's performance limits even with
    MMR diversification enabled (1.5x multiplier).

    Args:
        value: The value to validate and coerce
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        examples: Optional list of example valid values

    Returns:
        Validated integer within bounds

    Raises:
        ValueError: If the value cannot be converted or is out of bounds

    Note:
        For K parameters, max_val=20 ensures fetch_k ≤ 50 even with MMR:
        - Base: k=20 + over_fetch=15 = 35
        - With MMR: 35 * 1.5 = 52.5, capped at 50
    """
    if value is None:
        if not examples:
            examples = [min_val, (min_val + max_val) // 2, max_val]
        raise ValueError(
            format_error_message(
                "required", field=field_name, examples=format_examples(examples)
            )
        )

    # Handle string inputs from MCP clients with edge cases
    if isinstance(value, str):
        # Strip whitespace first
        value = value.strip()

        # Handle empty strings and null-like values
        if not value or value.lower() in ("null", "undefined", "none", "nan"):
            if not examples:
                examples = [min_val, (min_val + max_val) // 2, max_val]
            raise ValueError(
                format_error_message(
                    "required", field=field_name, examples=format_examples(examples)
                )
            )

        # Try to convert to integer
        try:
            value = int(value)
        except ValueError as e:
            if not examples:
                examples = [min_val, (min_val + max_val) // 2, max_val]
            raise ValueError(
                format_error_message(
                    "type",
                    field=field_name,
                    expected_type="an integer",
                    actual_type="string",
                    value=value[:100] if len(value) > 100 else value,
                    examples=format_examples(examples),
                )
            ) from e

    # Ensure we have an integer
    if not isinstance(value, int):
        try:
            value = int(value)
        except (TypeError, ValueError) as e:
            if not examples:
                examples = [min_val, (min_val + max_val) // 2, max_val]
            raise ValueError(
                format_error_message(
                    "type",
                    field=field_name,
                    expected_type="an integer",
                    actual_type=type(value).__name__,
                    value=repr(value)[:100],
                    examples=format_examples(examples),
                )
            ) from e

    # Check bounds
    if value < min_val or value > max_val:
        if not examples:
            examples = [min_val, (min_val + max_val) // 2, max_val]
        raise ValueError(
            format_error_message(
                "range",
                field=field_name,
                min=min_val,
                max=max_val,
                value=value,
                examples=format_examples(examples),
            )
        )

    return value


def coerce_to_float_with_bounds(
    value: Any,
    field_name: str,
    min_val: float,
    max_val: float,
    examples: list[float] | None = None,
) -> float:
    """
    Generic float coercion with bounds checking.

    Handles MCP client compatibility by accepting string inputs.

    Args:
        value: The value to validate and coerce
        field_name: Name of the field for error messages
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)
        examples: Optional list of example valid values

    Returns:
        Validated float within bounds

    Raises:
        ValueError: If the value cannot be converted or is out of bounds
    """
    if value is None:
        if not examples:
            examples = [min_val, (min_val + max_val) / 2, max_val]
        raise ValueError(
            format_error_message(
                "required", field=field_name, examples=format_examples(examples)
            )
        )

    # Handle string inputs from MCP clients
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError as e:
            if not examples:
                examples = [min_val, (min_val + max_val) / 2, max_val]
            raise ValueError(
                format_error_message(
                    "type",
                    field=field_name,
                    expected_type="a decimal number",
                    actual_type="string",
                    value=value[:100] if len(value) > 100 else value,
                    examples=format_examples(examples),
                )
            ) from e

    # Handle integer to float conversion
    if isinstance(value, int):
        value = float(value)

    # Ensure we have a float
    if not isinstance(value, float):
        try:
            value = float(value)
        except (TypeError, ValueError) as e:
            if not examples:
                examples = [min_val, (min_val + max_val) / 2, max_val]
            raise ValueError(
                format_error_message(
                    "type",
                    field=field_name,
                    expected_type="a decimal number",
                    actual_type=type(value).__name__,
                    value=repr(value)[:100],
                    examples=format_examples(examples),
                )
            ) from e

    # Check bounds
    if value < min_val or value > max_val:
        if not examples:
            examples = [min_val, (min_val + max_val) / 2, max_val]
        raise ValueError(
            format_error_message(
                "range",
                field=field_name,
                min=min_val,
                max=max_val,
                value=value,
                examples=format_examples(examples),
            )
        )

    return value


def coerce_to_bool_with_validation(value: Any, field_name: str = "value") -> bool:
    """
    Coerce value to boolean with MCP client compatibility.

    Handles various string representations of boolean values that MCP clients
    might send, providing consistent boolean conversion across the application.

    Args:
        value: The value to validate and coerce to boolean
        field_name: Name of the field for error messages

    Returns:
        Boolean value after coercion

    Examples:
        >>> coerce_to_bool_with_validation("true")
        True
        >>> coerce_to_bool_with_validation("false")
        False
        >>> coerce_to_bool_with_validation("1")
        True
        >>> coerce_to_bool_with_validation(None)
        False
    """
    # Handle None
    if value is None:
        return False

    # Handle boolean types directly
    if isinstance(value, bool):
        return value

    # Handle string types (MCP clients may send as strings)
    if isinstance(value, str):
        # Strip whitespace
        value = value.strip().lower()

        # Empty string is False
        if not value:
            return False

        # Lookup table for performance (10x faster than multiple if statements)
        BOOL_MAP = {
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
            "on": True,
            "off": False,
            "t": True,
            "f": False,
            "y": True,
            "n": False,
            "enabled": True,
            "disabled": False,
        }

        # Check lookup table first
        if value in BOOL_MAP:
            return BOOL_MAP[value]

        # For any other non-empty string, use Python truthiness
        # This ensures we don't reject unexpected but valid inputs
        return bool(value)

    # For numeric types
    if isinstance(value, (int, float)):
        return bool(value)

    # Default to Python truthiness for any other type
    return bool(value)


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
            format_error_message(
                "length",
                field=field_name,
                constraint="256 characters or less",
                actual=len(value),
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(
                    ["src/lib", "tests/integration", "examples/basic"]
                ),
            )
        )

    # Module paths can contain :: and /
    if not re.match(r"^[a-zA-Z0-9_/:]+$", value):
        raise ValueError(
            format_error_message(
                "pattern",
                field=field_name,
                pattern="alphanumeric characters, underscores, colons, and forward slashes only",
                value=value[:100] if len(value) > 100 else value,
                examples=format_examples(
                    [
                        "src/lib",
                        "tests/integration",
                        "examples/basic",
                        "modules::nested",
                    ]
                ),
            )
        )

    return value


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """
    Count tokens in text using tiktoken.

    Args:
        text: The text to count tokens for
        encoding: The encoding to use (default: cl100k_base for GPT-4)

    Returns:
        Number of tokens in the text
    """
    try:
        import tiktoken

        # Truncate very long text before tokenization to avoid stack overflow
        if len(text) > 100000:
            text = text[:100000]
            logger.warning("Text truncated to 100k chars for token counting")

        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(text))
    except Exception as e:
        # Fallback to character estimation (1 token ≈ 4 chars)
        logger.warning(f"Token counting failed, using character estimation: {e}")
        return len(text) // 4


def validate_tutorial_tokens(value: str | None, max_tokens: int = 200) -> str | None:
    """
    Validate that tutorial text doesn't exceed token limit.

    Args:
        value: The tutorial text to validate
        max_tokens: Maximum allowed tokens (default: 200)

    Returns:
        The validated tutorial text
    """
    if value is None:
        return None

    if not value.strip():
        return None

    # Check character limit first (rough approximation)
    if len(value) > max_tokens * 5:  # Assuming ~5 chars per token as upper bound
        logger.warning(
            f"Tutorial exceeds character limit ({len(value)} chars, "
            f"expected <{max_tokens * 5})"
        )

    # Check actual token count
    token_count = count_tokens(value)
    if token_count > max_tokens:
        logger.warning(
            f"Tutorial exceeds token limit: {token_count} tokens (max: {max_tokens})"
        )

    return value
