"""
Request models for the docsrs-mcp API.

This module defines all request models used by the API endpoints,
including validation logic for MCP client compatibility.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from docsrs_mcp import config as app_config
from docsrs_mcp.validation import (
    coerce_to_float_with_bounds,
    coerce_to_int_with_bounds,
    validate_crate_name,
    validate_rust_path,
    validate_version_string,
)

from .base import strict_config


# Request Models
class GetCrateSummaryRequest(BaseModel):
    """
    Request for get_crate_summary tool.

    Supports both third-party crates and Rust standard library crates (std, core, alloc).

    Example:
        ```json
        {
            "crate_name": "tokio",
            "version": "1.35.1"
        }
        ```

    Standard library example:
        ```json
        {
            "crate_name": "std",
            "version": "latest"
        }
        ```
    """

    crate_name: str = Field(
        ...,
        description="Name of the Rust crate (e.g., 'tokio', 'serde') or stdlib crate ('std', 'core', 'alloc')",
        examples=["tokio", "serde", "std", "core"],
    )
    version: str | None = Field(
        None,
        description="Specific version or 'latest' (default: latest)",
        examples=["1.35.1", "latest", None],
    )

    @field_validator("crate_name", mode="before")
    @classmethod
    def validate_crate(cls, v: Any) -> str:
        """Validate crate name follows Rust naming conventions."""
        return validate_crate_name(v, field_name="crate_name")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> str | None:
        """Validate version string or preserve None."""
        return validate_version_string(v, field_name="version")

    model_config = strict_config


class SearchItemsRequest(BaseModel):
    """
    Request for search_items tool.

    Performs semantic search across crate documentation using vector embeddings.

    Example:
        ```json
        {
            "crate_name": "tokio",
            "query": "spawn async tasks",
            "k": 5,
            "item_type": "function",
            "crate_filter": "tokio"
        }
        ```
    """

    crate_name: str = Field(
        ...,
        description="Name of the crate to search within",
        examples=["tokio", "serde"],
    )
    query: str = Field(
        ...,
        description="Natural language search query",
        examples=["async runtime", "deserialize JSON", "spawn tasks"],
    )
    version: str | None = Field(
        None, description="Specific version or 'latest' (default: latest)"
    )
    k: int | None = Field(
        5, description="Number of results to return (1-20)", ge=1, le=20
    )
    item_type: str | None = Field(
        None,
        description="Filter by item type (function, struct, trait, enum, module)",
        examples=["function", "struct", "trait"],
    )
    crate_filter: str | None = Field(
        None,
        description="Filter results to specific crate",
        examples=["tokio", "serde"],
    )
    module_path: str | None = Field(
        None,
        description="Filter results to specific module path within the crate",
        examples=["runtime", "net::tcp", "sync::mpsc"],
    )
    has_examples: bool | None = Field(
        None,
        description="Filter to only items with code examples",
    )
    min_doc_length: int | None = Field(
        None,
        description="Minimum documentation length in characters",
        ge=100,
        le=10000,
    )
    visibility: Literal["public", "private", "crate"] | None = Field(
        None,
        description="Filter by item visibility",
    )
    deprecated: bool | None = Field(
        None,
        description="Filter by deprecation status (true=deprecated only, false=non-deprecated only)",
    )
    search_mode: str | None = Field(
        None,
        description="Search mode: 'vector' (default), 'fuzzy', 'regex', or 'hybrid'",
        examples=["vector", "fuzzy", "regex"],
    )
    fuzzy_tolerance: float | None = Field(
        None,
        description="Fuzzy match threshold (0.0-1.0, default: 0.7)",
        ge=0.0,
        le=1.0,
    )
    regex_pattern: str | None = Field(
        None,
        description="Regex pattern for pattern matching mode",
        examples=["async.*spawn", "tokio::.*::spawn", "^std::.*::Vec$"],
    )
    stability_filter: str | None = Field(
        None,
        description="Filter by stability level: 'stable', 'unstable', 'experimental', or 'all' (default)",
        examples=["stable", "unstable", "all"],
    )
    crates: list[str] | None = Field(
        None,
        description="List of crates for cross-crate search (max 5)",
        max_length=5,
        examples=[["tokio", "async-std"], ["serde", "serde_json", "serde_yaml"]],
    )

    @field_validator("query", mode="before")
    @classmethod
    def preprocess_query(cls, v: Any) -> str:
        """
        Preprocess and normalize search query for consistent matching.

        Applies Unicode normalization (NFKC) and whitespace normalization
        to improve search consistency and cache hit rates. Includes
        fuzzy normalization support for improved international queries.
        """
        # Step 1: Convert to string (but don't strip yet to detect whitespace-only)
        if v is None:
            raise ValueError(
                "Query cannot be empty. "
                "Please provide a search term, e.g., 'async runtime', 'spawn task', 'deserialize JSON'."
            )
        query = str(v)

        # Step 2: Check if query is only whitespace before stripping
        if not query.strip():
            if query:  # Has content but it's all whitespace
                raise ValueError(
                    "Query cannot be empty after normalization. "
                    "The query may have contained only whitespace or special characters."
                )
            else:  # Empty string
                raise ValueError(
                    "Query cannot be empty. "
                    "Please provide a search term, e.g., 'async runtime', 'spawn task', 'deserialize JSON'."
                )

        # Step 3: Strip leading/trailing whitespace
        query = query.strip()

        # Step 4: Length validation
        if len(query) > 500:
            raise ValueError(
                f"Query too long ({len(query)} characters). Maximum 500 characters allowed. "
                "Consider using more specific search terms to narrow your query."
            )

        # Step 5: Quick-check for already normalized text (optimization)
        # If text is already in NFKC form, skip normalization
        original_query = query
        normalized_query = unicodedata.normalize("NFKC", query)

        # Only apply full normalization if needed
        if original_query != normalized_query:
            query = normalized_query

        # Step 6: Whitespace normalization
        # Replace multiple spaces, tabs, newlines with single space
        query = " ".join(query.split())

        # Step 7: Fuzzy normalization for common variations
        # This handles common misspellings and international variations
        # e.g., "serialise" -> "serialize", "colour" -> "color"
        query = cls._apply_fuzzy_normalization(query)

        # Step 8: Rust-specific term expansion
        # Expand common Rust abbreviations to improve search coverage
        # e.g., "async" -> "async asynchronous", "impl" -> "impl implementation"
        query = cls._expand_rust_terms(query)

        # Step 9: Final validation after normalization
        if len(query) < 1:
            raise ValueError(
                "Query cannot be empty after normalization. "
                "The query may have contained only whitespace or special characters."
            )

        return query

    @classmethod
    def _apply_fuzzy_normalization(cls, query: str) -> str:
        """
        Apply fuzzy normalization for common spelling variations.

        This improves search for international English variations and
        common technical term misspellings.
        """
        # Common British/American English variations in technical terms
        replacements = {
            "serialise": "serialize",
            "deserialise": "deserialize",
            "synchronise": "synchronize",
            "initialise": "initialize",
            "optimise": "optimize",
            "finalise": "finalize",
            "normalise": "normalize",
            "authorise": "authorize",
            "colour": "color",
            "behaviour": "behavior",
            "catalogue": "catalog",
            "centre": "center",
            "defence": "defense",
            "licence": "license",
            "practise": "practice",
            # Additional programming-specific variations
            "analyse": "analyze",
            "tokenise": "tokenize",
            "randomise": "randomize",
            "maximise": "maximize",
            "minimise": "minimize",
            "visualise": "visualize",
            "utilise": "utilize",
            "customise": "customize",
            "prioritise": "prioritize",
            "standardise": "standardize",
            "summarise": "summarize",
            "parallelise": "parallelize",
        }

        # Apply replacements for whole words only (avoid partial replacements)
        words = query.split()
        normalized_words = []

        for word in words:
            # Check if word (case-insensitive) matches a replacement
            word_lower = word.lower()
            if word_lower in replacements:
                # Preserve original case pattern
                if word.isupper():
                    normalized_words.append(replacements[word_lower].upper())
                elif word[0].isupper():
                    normalized_words.append(replacements[word_lower].capitalize())
                else:
                    normalized_words.append(replacements[word_lower])
            else:
                normalized_words.append(word)

        return " ".join(normalized_words)

    @classmethod
    def _expand_rust_terms(cls, query: str) -> str:
        """
        Expand common Rust abbreviations and terms for better search coverage.

        This improves search by including both abbreviated and full forms
        of common Rust terminology.
        """
        # Use term expansions from config
        expansions = app_config.RUST_TERM_EXPANSIONS

        # Split query into words
        words = query.split()
        expanded_terms = []

        for word in words:
            word_lower = word.lower()

            # Check if this word has expansions
            if word_lower in expansions:
                # Add all expansion variants (including original)
                expanded_terms.extend(expansions[word_lower])
            else:
                # Keep original word
                expanded_terms.append(word)

        # Join and deduplicate while preserving order
        seen = set()
        result = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                result.append(term)

        return " ".join(result)

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return 5  # Default value for SearchItemsRequest

        # Handle empty strings and whitespace edge cases
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("null", "undefined", "none"):
                return 5  # Default for empty/null-like strings

        # Use enhanced validation with examples
        return coerce_to_int_with_bounds(
            value=v,
            field_name="k (number of results)",
            min_val=1,
            max_val=20,
            examples=[1, 5, 10],
        )

    @field_validator("min_doc_length", mode="before")
    @classmethod
    def coerce_min_doc_length_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        return coerce_to_int_with_bounds(
            value=v,
            field_name="min_doc_length (minimum documentation length)",
            min_val=100,
            max_val=10000,
            examples=[100, 500, 1000],
        )

    @field_validator("item_type", mode="before")
    @classmethod
    def coerce_item_type(cls, v):
        """Handle MCP string conversion and normalize item type."""
        if v is None or v == "":
            return None
        # Normalize to lowercase for consistency
        normalized = str(v).lower()
        # Validate against allowed types
        allowed_types = {"function", "struct", "trait", "enum", "module", "trait_impl"}
        if normalized not in allowed_types:
            # Provide helpful suggestions for common mistakes
            suggestions = []
            if "func" in normalized or "fn" in normalized:
                suggestions.append("Did you mean 'function'?")
            elif "class" in normalized:
                suggestions.append("Did you mean 'struct'?")
            elif "interface" in normalized:
                suggestions.append("Did you mean 'trait'?")

            suggestion_text = f" {suggestions[0]}" if suggestions else ""
            raise ValueError(
                f"item_type must be one of {sorted(allowed_types)}, got '{normalized}'.{suggestion_text} "
                f"Use 'function' for functions/methods, 'struct' for structs, 'trait' for traits."
            )
        return normalized

    @field_validator("crate_filter", mode="before")
    @classmethod
    def coerce_crate_filter(cls, v):
        """Handle MCP string conversion for crate filter."""
        if v is None or v == "":
            return None
        return str(v)

    @field_validator("module_path", mode="before")
    @classmethod
    def validate_module_path(cls, v):
        """Validate and normalize module path for filtering."""
        if v is None or v == "":
            return None

        # Convert to string and strip whitespace
        module_path = str(v).strip()

        # Store original for error messages
        original_path = module_path

        # Check for trailing :: which is invalid
        if module_path.endswith("::") and module_path != "::":
            raise ValueError(
                f"Invalid module path '{original_path}': trailing '::' is not allowed. "
                "Module paths should be like 'runtime' or 'sync::mpsc'."
            )

        # Check for leading :: which is invalid
        if module_path.startswith("::") and module_path != "::":
            raise ValueError(
                f"Invalid module path '{original_path}': leading '::' is not allowed. "
                "Module paths should be like 'runtime' or 'sync::mpsc'."
            )

        # Check for empty segments in the middle
        if ":::" in module_path:
            raise ValueError(
                f"Invalid module path '{original_path}': contains empty segments. "
                "Module paths should be like 'runtime' or 'sync::mpsc'."
            )

        # Remove leading/trailing :: if present
        module_path = module_path.strip(":")

        # Validate format: should not be empty after stripping
        if not module_path:
            raise ValueError(
                f"Invalid module path '{original_path}': cannot be empty. "
                "Examples: 'runtime', 'sync::mpsc', 'net::tcp'."
            )

        # Validate format: should not have empty segments
        segments = module_path.split("::")
        for segment in segments:
            if not segment:
                raise ValueError(
                    f"Invalid module path '{original_path}': contains empty segments. "
                    "Module paths should be like 'runtime' or 'sync::mpsc'."
                )
            # Basic validation: segment should be valid Rust identifier
            if not segment.replace("_", "").isalnum():
                raise ValueError(
                    f"Invalid module path '{original_path}': segment '{segment}' contains invalid characters. "
                    "Module names should only contain alphanumeric characters and underscores."
                )

        return module_path

    @field_validator("deprecated", "has_examples", mode="before")
    @classmethod
    def validate_boolean_filters(cls, v):
        """Convert various inputs to boolean for MCP compatibility.

        CRITICAL FOR CLAUDE CODE COMPATIBILITY:
        ----------------------------------------
        This validator handles the mismatch between:
        1. Schema declaration: Says 'string' (due to our override in mcp_server.py)
        2. Claude Code sends: Native boolean or integer
        3. Our internal needs: Actual boolean values

        The mode='before' is ESSENTIAL - it runs before Pydantic's type validation,
        allowing us to intercept and convert the raw input from Claude Code.

        Accepts multiple formats for maximum compatibility:
        - Native booleans: True/False (what Claude Code actually sends)
        - Strings: "true", "false", "1", "0", "yes", "no", "on", "off"
        - Numbers: 1/0, any non-zero = true
        - None/empty: Preserves as None (optional parameter)
        """
        if v is None or v == "":
            return None
            
        # Use centralized boolean coercion function
        from docsrs_mcp.validation import coerce_to_bool_with_validation
        return coerce_to_bool_with_validation(v)
    
    @field_validator("search_mode", mode="before")
    @classmethod
    def validate_search_mode(cls, v):
        """Validate and normalize search mode."""
        if v is None or v == "":
            return None
        normalized = str(v).lower().strip()
        allowed_modes = {"vector", "fuzzy", "regex", "hybrid"}
        if normalized not in allowed_modes:
            raise ValueError(
                f"search_mode must be one of {sorted(allowed_modes)}, got '{normalized}'. "
                f"Use 'vector' for semantic search, 'fuzzy' for typo-tolerant matching, "
                f"'regex' for pattern matching, 'hybrid' for combined approaches."
            )
        return normalized
    
    @field_validator("fuzzy_tolerance", mode="before")
    @classmethod
    def coerce_fuzzy_tolerance(cls, v):
        """Convert string to float and validate range."""
        if v is None or v == "":
            return None
        try:
            value = float(str(v))
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"fuzzy_tolerance must be between 0.0 and 1.0, got {value}. "
                    f"Use 0.7 for moderate tolerance, 0.9 for strict matching."
                )
            return value
        except (ValueError, TypeError) as e:
            raise ValueError(
                f"fuzzy_tolerance must be a number between 0.0 and 1.0, got '{v}'. "
                f"Examples: 0.7 (default), 0.5 (lenient), 0.9 (strict)."
            ) from e
    
    @field_validator("stability_filter", mode="before")
    @classmethod
    def validate_stability_filter(cls, v):
        """Validate and normalize stability filter."""
        if v is None or v == "":
            return None
        normalized = str(v).lower().strip()
        allowed_levels = {"stable", "unstable", "experimental", "all"}
        if normalized not in allowed_levels:
            raise ValueError(
                f"stability_filter must be one of {sorted(allowed_levels)}, got '{normalized}'. "
                f"Use 'stable' for production APIs, 'unstable' for APIs that may change, "
                f"'experimental' for new features, 'all' to include everything."
            )
        return normalized

    @field_validator("visibility", mode="before")
    @classmethod
    def validate_visibility(cls, v):
        """Validate and normalize visibility filter."""
        if v is None or v == "":
            return None
        normalized = str(v).lower()
        if normalized not in ["public", "private", "crate"]:
            # Common visibility terms mapping
            if normalized in ["pub", "exported"]:
                suggestion = " Did you mean 'public'?"
            elif normalized in ["priv", "internal"]:
                suggestion = " Did you mean 'private'?"
            elif normalized in ["pub(crate)", "crate-private"]:
                suggestion = " Did you mean 'crate'?"
            else:
                suggestion = ""

            raise ValueError(
                f"visibility must be one of ['public', 'private', 'crate'], got '{normalized}'.{suggestion} "
                f"Use 'public' for exported items, 'private' for module-private items, 'crate' for crate-visible items."
            )
        return normalized

    @model_validator(mode="after")
    def validate_filter_compatibility(self):
        """Check for conflicting or incompatible filter combinations."""
        # Validate module_path is only used with matching crate context
        if self.module_path:
            if self.crate_filter and self.crate_filter != self.crate_name:
                raise ValueError(
                    f"Module path '{self.module_path}' can only be used when searching "
                    f"within the same crate. Cannot use module_path with crate_filter='{self.crate_filter}' "
                    f"when searching in crate '{self.crate_name}'."
                )

        # Warn if searching for private items in a different crate
        if (
            self.visibility == "private"
            and self.crate_filter
            and self.crate_filter != self.crate_name
        ):
            raise ValueError(
                f"Cannot search for private items in crate '{self.crate_filter}' "
                f"when searching within crate '{self.crate_name}'. "
                "Private items are only visible within their own crate."
            )

        # Warn if min_doc_length is very high with has_examples=True
        if self.has_examples and self.min_doc_length and self.min_doc_length > 5000:
            raise ValueError(
                f"min_doc_length={self.min_doc_length} is very high when combined with has_examples=True. "
                "This may return no results. Consider using min_doc_length=1000 or lower for items with examples."
            )

        # Suggest optimization for common patterns
        if (
            self.deprecated is False
            and self.visibility == "public"
            and not self.item_type
        ):
            # This is fine, just a common pattern that benefits from our partial indexes
            pass

        return self

    model_config = strict_config


class GetItemDocRequest(BaseModel):
    """
    Request for get_item_doc tool.

    Retrieves complete documentation for a specific item by its path.

    Example:
        ```json
        {
            "crate_name": "serde",
            "item_path": "serde::Deserialize",
            "version": "1.0.193"
        }
        ```
    """

    crate_name: str = Field(..., description="Name of the crate containing the item")
    item_path: str = Field(
        ...,
        description="Full path to the item (e.g., 'tokio::spawn', 'std::vec::Vec')",
        examples=["tokio::spawn", "serde::Deserialize", "crate"],
    )
    version: str | None = Field(
        None, description="Specific version or 'latest' (default: latest)"
    )

    @field_validator("crate_name", mode="before")
    @classmethod
    def validate_crate(cls, v: Any) -> str:
        """Validate crate name follows Rust naming conventions."""
        return validate_crate_name(v, field_name="crate_name")

    @field_validator("item_path", mode="before")
    @classmethod
    def validate_path(cls, v: Any) -> str:
        """Validate Rust item path syntax."""
        return validate_rust_path(v, field_name="item_path")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> str | None:
        """Validate version string or preserve None."""
        return validate_version_string(v, field_name="version")

    model_config = strict_config


class GetModuleTreeRequest(BaseModel):
    """
    Request for get_module_tree tool.

    Retrieves the hierarchical module structure of a Rust crate.

    Example:
        ```json
        {
            "crate_name": "tokio",
            "version": "1.35.1"
        }
        ```
    """

    crate_name: str = Field(
        ...,
        description="Name of the Rust crate to get module tree for",
        examples=["tokio", "serde", "actix-web"],
    )
    version: str | None = Field(
        None,
        description="Specific version or 'latest' (default: latest)",
        examples=["1.35.1", "latest", None],
    )

    @field_validator("crate_name", mode="before")
    @classmethod
    def validate_crate(cls, v: Any) -> str:
        """Validate crate name follows Rust naming conventions."""
        return validate_crate_name(v, field_name="crate_name")

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> str | None:
        """Validate version string or preserve None."""
        return validate_version_string(v, field_name="version")

    model_config = strict_config


class ListVersionsRequest(BaseModel):
    """
    Request for list_versions resource.

    Lists all cached versions of a specific Rust crate.

    Example:
        ```json
        {
            "crate_name": "tokio"
        }
        ```
    """

    crate_name: str = Field(
        ...,
        description="Name of the Rust crate to list versions for",
        examples=["tokio", "serde", "actix-web"],
    )

    @field_validator("crate_name", mode="before")
    @classmethod
    def validate_crate(cls, v: Any) -> str:
        """Validate crate name follows Rust naming conventions."""
        return validate_crate_name(v, field_name="crate_name")

    model_config = strict_config


class SearchExamplesRequest(BaseModel):
    """Request model for searching code examples."""

    crate_name: str = Field(
        ...,
        description="Name of the crate to search within",
        examples=["tokio", "serde"],
    )
    query: str = Field(
        ...,
        description="Search query for finding relevant code examples",
        examples=["async runtime", "deserialize JSON"],
    )
    version: str | None = Field(
        None,
        description="Specific version to search (default: latest)",
        examples=["1.35.1", "latest"],
    )
    k: int = Field(default=5, ge=1, le=20, description="Number of examples to return")
    language: str | None = Field(
        None,
        description="Filter examples by programming language",
        examples=["rust", "bash", "toml"],
    )

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return 5  # Default value for SearchExamplesRequest

        # Handle empty strings and whitespace edge cases
        if isinstance(v, str):
            v = v.strip()
            if not v or v.lower() in ("null", "undefined", "none"):
                return 5  # Default for empty/null-like strings

        # Use enhanced validation with examples
        return coerce_to_int_with_bounds(
            value=v,
            field_name="k (number of examples)",
            min_val=1,
            max_val=20,
            examples=[5, 10, 20],
        )

    model_config = strict_config


# Pre-Ingestion Control Models
class StartPreIngestionRequest(BaseModel):
    """
    Request for start_pre_ingestion tool.

    Controls the pre-ingestion system that caches popular Rust crates
    to eliminate cold-start latency.

    Example:
        ```json
        {
            "force": false,
            "concurrency": 5,
            "count": 200
        }
        ```
    """

    force: bool = Field(
        default=False, description="Force restart if pre-ingestion is already running"
    )
    concurrency: int | None = Field(
        default=None,
        ge=1,
        le=10,
        description="Number of parallel download workers (1-10, default: 3)",
    )
    count: int | None = Field(
        default=None,
        ge=10,
        le=500,
        description="Number of crates to pre-ingest (10-500, default: 100)",
    )

    @field_validator("force", mode="before")
    @classmethod
    def validate_force(cls, v: Any) -> bool:
        """Validate and coerce force parameter to boolean."""
        # Use centralized boolean coercion function
        from docsrs_mcp.validation import coerce_to_bool_with_validation
        return coerce_to_bool_with_validation(v)

    @field_validator("concurrency", mode="before")
    @classmethod
    def coerce_concurrency_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        return coerce_to_int_with_bounds(
            value=v,
            field_name="concurrency (parallel download workers)",
            min_val=1,
            max_val=10,
            examples=[3, 5, 10],
        )

    @field_validator("count", mode="before")
    @classmethod
    def coerce_count_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        return coerce_to_int_with_bounds(
            value=v,
            field_name="count (number of crates to pre-ingest)",
            min_val=10,
            max_val=500,
            examples=[100, 200, 500],
        )

    model_config = strict_config


# Cargo File Ingestion Models
class IngestCargoFileRequest(BaseModel):
    """Request model for ingesting crates from Cargo files."""

    file_path: str = Field(..., description="Path to Cargo.toml or Cargo.lock file")
    concurrency: int | None = Field(
        default=3, ge=1, le=10, description="Number of parallel download workers (1-10)"
    )
    skip_existing: bool = Field(
        default=True, description="Skip already ingested crates"
    )
    resolve_versions: bool = Field(
        default=False, description="Resolve version specifications to concrete versions"
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, v: Any) -> str:
        """Validate and normalize file path."""
        # Convert any input to string first
        if v is None:
            raise ValueError("File path cannot be None")
        if not isinstance(v, str):
            v = str(v)

        path = Path(v).resolve()
        if not path.exists():
            raise ValueError(f"File not found: {v}")
        if path.name.lower() not in ["cargo.toml", "cargo.lock"]:
            raise ValueError(f"Must be Cargo.toml or Cargo.lock, got: {path.name}")
        return str(path)

    @field_validator("concurrency", mode="before")
    @classmethod
    def coerce_concurrency(cls, v):
        """Coerce concurrency to int."""
        if v is None:
            return 3
        return coerce_to_int_with_bounds(
            v, "concurrency", min_val=1, max_val=10, examples=[3, 5, 10]
        )

    @field_validator("skip_existing", mode="before")
    @classmethod
    def coerce_skip_existing(cls, v):
        """Coerce skip_existing to bool."""
        # Use centralized boolean coercion function with default
        from docsrs_mcp.validation import coerce_to_bool_with_validation
        if v is None:
            return True  # default
        return coerce_to_bool_with_validation(v)

    @field_validator("resolve_versions", mode="before")
    @classmethod
    def coerce_resolve_versions(cls, v):
        """Coerce resolve_versions to bool."""
        # Use centralized boolean coercion function
        from docsrs_mcp.validation import coerce_to_bool_with_validation
        return coerce_to_bool_with_validation(v)

    model_config = strict_config


# Pre-Ingestion Control Models
class PreIngestionControlRequest(BaseModel):
    """Request model for pre-ingestion control operations."""

    action: Literal["pause", "resume", "stop"] = Field(
        ..., description="Control action to perform on the pre-ingestion worker"
    )

    model_config = strict_config


# Ranking Config (used in responses but defined here for organization)
class RankingConfig(BaseModel):
    """
    Configuration for search result ranking algorithm.

    Defines weights for combining multiple scoring factors to produce
    a final relevance score for search results.
    """

    vector_weight: float = Field(
        0.7, ge=0.0, le=1.0, description="Weight for vector similarity score"
    )
    type_weight: float = Field(
        0.15, ge=0.0, le=1.0, description="Weight for item type boost"
    )
    quality_weight: float = Field(
        0.1, ge=0.0, le=1.0, description="Weight for documentation quality"
    )
    examples_weight: float = Field(
        0.05, ge=0.0, le=1.0, description="Weight for example presence"
    )

    @field_validator(
        "vector_weight",
        "type_weight",
        "quality_weight",
        "examples_weight",
        mode="before",
    )
    @classmethod
    def coerce_weights_to_float(cls, v):
        """Convert string numbers to float for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        return coerce_to_float_with_bounds(
            value=v,
            field_name="weight parameter",
            min_val=0.0,
            max_val=1.0,
            examples=[0.05, 0.15, 0.7],
        )

    @field_validator("examples_weight", mode="before")
    @classmethod
    def validate_weights_sum(cls, v, info):
        """Ensure all weights sum to 1.0 for normalized scoring."""
        if info.field_name == "examples_weight":
            # Check sum when all fields are set
            values = info.data
            total = (
                values.get("vector_weight", 0.7)
                + values.get("type_weight", 0.15)
                + values.get("quality_weight", 0.1)
                + v
            )
            if abs(total - 1.0) > 0.001:  # Allow small floating point errors
                raise ValueError(f"Weights must sum to 1.0, got {total}")
        return v

    model_config = strict_config

