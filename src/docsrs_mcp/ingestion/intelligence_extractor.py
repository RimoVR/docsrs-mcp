"""Code intelligence extraction for Rust documentation.

This module provides specialized extraction functions for code intelligence features:
- Error type extraction from Result patterns
- Safety information detection (unsafe blocks and functions)
- Feature requirement extraction from cfg attributes
- Enhanced signature details with complete generic bounds
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Pre-compiled patterns for performance (10x speedup from research)
# These patterns are compiled at module level to avoid recompilation overhead

# Match Result<T, E> patterns to extract error types
# Handles nested generics like Result<Vec<T>, MyError<String>>
ERROR_TYPE_PATTERN = re.compile(
    r"Result<[^,>]*(?:<[^>]*>)?[^,>]*,\s*([^>]+(?:<[^>]*>)?)\s*>"
)

# Simple Result pattern for common cases
SIMPLE_ERROR_PATTERN = re.compile(r"Result<[^,]+,\s*([^>]+)>")

# Detect unsafe keywords in signatures and blocks
UNSAFE_PATTERN = re.compile(r"\bunsafe\b")

# Extract feature requirements from cfg attributes
CFG_FEATURE_PATTERN = re.compile(r'#\[cfg\(feature\s*=\s*"([^"]+)"\)\]')

# Extract any cfg condition (not just features)
CFG_ANY_PATTERN = re.compile(r"#\[cfg\(([^)]+)\)\]")

# Match lifetime parameters
LIFETIME_PATTERN = re.compile(r"'[a-zA-Z_][a-zA-Z0-9_]*")

# Match generic bounds in where clauses
WHERE_CLAUSE_PATTERN = re.compile(r"where\s+(.+?)(?:\s*\{|$)")


class IntelligenceCache:
    """Session-based cache for extraction patterns to improve performance."""

    def __init__(self):
        self.error_types = {}  # Cache parsed error types by signature
        self.features = set()  # Cache discovered feature flags
        self.unsafe_items = set()  # Cache unsafe item paths

    def clear_if_needed(self, item_count: int):
        """Clear cache periodically to manage memory."""
        if item_count % 100 == 0:
            self.error_types.clear()
            if item_count % 500 == 0:
                # Less frequent clearing for accumulated data
                self.features.clear()
                self.unsafe_items.clear()


# Global cache instance for the session
_cache = IntelligenceCache()


def extract_error_types(signature: str, item_type: str) -> list[str]:
    """Extract error types from Result<T, E> patterns in signatures.

    Args:
        signature: The function/method signature to analyze
        item_type: The type of item (function, method, etc.)

    Returns:
        List of error type names found in Result patterns
    """
    if not signature or "Result" not in signature:
        return []

    # Check cache first
    cache_key = f"{signature[:100]}_{item_type}"  # Truncate for reasonable key size
    if cache_key in _cache.error_types:
        return _cache.error_types[cache_key]

    error_types = []

    # Try complex pattern first for nested generics
    matches = ERROR_TYPE_PATTERN.findall(signature)
    if matches:
        error_types.extend(matches)
    else:
        # Fall back to simple pattern
        simple_matches = SIMPLE_ERROR_PATTERN.findall(signature)
        error_types.extend(simple_matches)

    # Clean up error types (remove extra whitespace, deduplicate)
    error_types = [e.strip() for e in error_types]
    error_types = list(dict.fromkeys(error_types))  # Preserve order while deduplicating

    # Cache the result
    _cache.error_types[cache_key] = error_types

    return error_types


def extract_safety_info(
    attrs: list[str], signature: str, docs: str | None = None
) -> dict[str, Any]:
    """Detect unsafe blocks and extract safety documentation.

    Args:
        attrs: List of attributes from rustdoc JSON
        signature: The function/method signature
        docs: Optional documentation string to search for safety info

    Returns:
        Dictionary with safety information:
        - is_safe: Boolean indicating if the item is safe
        - unsafe_reason: Reason for unsafe (if applicable)
        - safety_docs: Extracted safety documentation
    """
    safety_info = {"is_safe": True, "unsafe_reason": None, "safety_docs": None}

    # Check for unsafe in signature
    if signature and UNSAFE_PATTERN.search(signature):
        safety_info["is_safe"] = False
        safety_info["unsafe_reason"] = "unsafe keyword in signature"

    # Check for unsafe-related attributes
    unsafe_attrs = ["unsafe_precondition", "unsafe_op_in_unsafe_fn", "unsafe_code"]

    if attrs:
        for attr in attrs:
            attr_str = str(attr) if not isinstance(attr, str) else attr
            if any(ua in attr_str.lower() for ua in unsafe_attrs):
                safety_info["is_safe"] = False
                safety_info["unsafe_reason"] = f"unsafe attribute: {attr_str}"
                break

    # Extract safety documentation if present
    if docs:
        # Look for safety sections in documentation
        safety_patterns = [
            r"#\s*Safety\s*\n+(.*?)(?:\n#|$)",
            r"#\s*Unsafe\s*\n+(.*?)(?:\n#|$)",
            r"#\s*Preconditions?\s*\n+(.*?)(?:\n#|$)",
        ]

        for pattern in safety_patterns:
            match = re.search(pattern, docs, re.IGNORECASE | re.MULTILINE | re.DOTALL)
            if match:
                safety_info["safety_docs"] = match.group(1).strip()
                break

    return safety_info


def extract_feature_requirements(attrs: list[str]) -> list[str]:
    """Extract feature requirements from cfg attributes.

    Args:
        attrs: List of attributes from rustdoc JSON

    Returns:
        List of required feature names
    """
    if not attrs:
        return []

    features = []

    for attr in attrs:
        attr_str = str(attr) if not isinstance(attr, str) else attr

        # Extract feature-specific cfg attributes
        feature_matches = CFG_FEATURE_PATTERN.findall(attr_str)
        features.extend(feature_matches)

        # Also check for complex feature expressions
        # e.g., #[cfg(all(feature = "foo", feature = "bar"))]
        if "all(" in attr_str or "any(" in attr_str:
            # Extract all features from complex expressions
            inner_features = re.findall(r'feature\s*=\s*"([^"]+)"', attr_str)
            features.extend(inner_features)

    # Deduplicate while preserving order
    features = list(dict.fromkeys(features))

    # Add to global cache for summary
    _cache.features.update(features)

    return features


def enhance_signature_details(
    signature: str,
    generics: dict[str, Any] | None = None,
    where_clause: str | None = None,
) -> dict[str, Any]:
    """Enhance signature with complete generic bounds and lifetime information.

    Args:
        signature: The base signature
        generics: Generic parameters from rustdoc JSON
        where_clause: Optional where clause text

    Returns:
        Dictionary with enhanced signature details:
        - lifetimes: List of lifetime parameters
        - generic_params: Enhanced generic parameter info
        - trait_bounds: Complete trait bounds including where clauses
        - is_generic: Boolean indicating if signature has generics
    """
    enhanced = {
        "lifetimes": [],
        "generic_params": [],
        "trait_bounds": [],
        "is_generic": False,
    }

    # Extract lifetimes from signature
    if signature:
        lifetimes = LIFETIME_PATTERN.findall(signature)
        enhanced["lifetimes"] = list(dict.fromkeys(lifetimes))  # Deduplicate

    # Process generic parameters if provided
    if generics:
        enhanced["is_generic"] = True

        # Extract type parameters
        if isinstance(generics, dict):
            params = generics.get("params", [])
            for param in params:
                if isinstance(param, dict):
                    param_info = {
                        "name": param.get("name", ""),
                        "kind": param.get("kind", "type"),
                    }

                    # Add bounds if present
                    bounds = param.get("bounds", [])
                    if bounds:
                        param_info["bounds"] = bounds
                        enhanced["trait_bounds"].extend(bounds)

                    enhanced["generic_params"].append(param_info)

    # Extract where clause bounds
    if where_clause:
        where_match = WHERE_CLAUSE_PATTERN.search(where_clause)
        if where_match:
            where_bounds = where_match.group(1)
            # Split on commas not inside angle brackets
            bounds = re.split(r",(?![^<]*>)", where_bounds)
            enhanced["trait_bounds"].extend([b.strip() for b in bounds])
    elif signature and "where" in signature:
        # Try to extract from signature if where_clause not provided separately
        where_match = WHERE_CLAUSE_PATTERN.search(signature)
        if where_match:
            where_bounds = where_match.group(1)
            bounds = re.split(r",(?![^<]*>)", where_bounds)
            enhanced["trait_bounds"].extend([b.strip() for b in bounds])

    # Check if signature contains generic indicators
    if signature and ("<" in signature or "impl " in signature):
        enhanced["is_generic"] = True

    # Deduplicate trait bounds
    enhanced["trait_bounds"] = list(dict.fromkeys(enhanced["trait_bounds"]))

    return enhanced


def extract_all_intelligence(
    item: dict[str, Any],
    signature: str | None = None,
    attrs: list[str] | None = None,
    docs: str | None = None,
    item_count: int = 0,
) -> dict[str, Any]:
    """Extract all code intelligence from an item.

    Convenience function that runs all extractors and returns combined results.

    Args:
        item: The rustdoc item dictionary
        signature: Optional pre-extracted signature
        attrs: Optional pre-extracted attributes
        docs: Optional documentation string
        item_count: Current item count for cache management

    Returns:
        Dictionary with all extracted intelligence
    """
    # Clear cache if needed
    _cache.clear_if_needed(item_count)

    # Extract or get values from item
    if signature is None:
        signature = item.get("signature", "")
    if attrs is None:
        attrs = item.get("attrs", [])
    if docs is None:
        docs = item.get("docs", "")

    item_type = item.get("type", "unknown")
    generics = item.get("generics", {})

    # Run all extractors
    intelligence = {
        "error_types": extract_error_types(signature, item_type),
        "safety_info": extract_safety_info(attrs, signature, docs),
        "feature_requirements": extract_feature_requirements(attrs),
        "enhanced_signature": enhance_signature_details(signature, generics),
    }

    # Add summary flags for easy filtering
    intelligence["has_errors"] = bool(intelligence["error_types"])
    intelligence["is_safe"] = intelligence["safety_info"]["is_safe"]
    intelligence["requires_features"] = bool(intelligence["feature_requirements"])
    intelligence["is_generic"] = intelligence["enhanced_signature"]["is_generic"]

    return intelligence


def safe_extract(extractor_func, *args, default=None):
    """Wrapper for graceful degradation on extraction failures.

    Args:
        extractor_func: The extraction function to call
        *args: Arguments to pass to the function
        default: Default value to return on failure

    Returns:
        Extraction result or default value
    """
    try:
        return extractor_func(*args)
    except Exception as e:
        logger.warning(f"Extraction failed in {extractor_func.__name__}: {e}")
        return default
