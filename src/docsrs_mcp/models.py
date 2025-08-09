"""
Pydantic models for MCP protocol and API requests/responses.

This module defines all request and response models for the docsrs-mcp server,
following the Model Context Protocol (MCP) specification. All models use
strict validation with `extra="forbid"` to prevent injection attacks.
"""

import unicodedata
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .validation import validate_crate_name, validate_rust_path, validate_version_string


# MCP Manifest Models
class MCPTool(BaseModel):
    """
    MCP Tool definition for the manifest.

    Defines a tool that MCP clients can invoke, including its name,
    description, and JSON Schema for input validation.

    Example:
        ```python
        tool = MCPTool(
            name="search_items",
            description="Search crate documentation",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string"}
                },
                "required": ["query"]
            }
        )
        ```
    """

    name: str = Field(..., description="Unique tool identifier")
    description: str = Field(..., description="Human-readable tool description")
    input_schema: dict[str, Any] = Field(..., description="JSON Schema for tool inputs")

    # Tutorial fields (optional for backward compatibility)
    tutorial: str | None = Field(
        None,
        description="Concise tutorial with usage examples",
        json_schema_extra={"maxLength": 1000},  # ~200 tokens
    )
    examples: list[str] | None = Field(
        None, description="Quick example invocations", json_schema_extra={"maxItems": 3}
    )
    use_cases: list[str] | None = Field(
        None,
        description="Common use cases for this tool",
        json_schema_extra={"maxItems": 3},
    )

    model_config = ConfigDict(extra="forbid")


class MCPResource(BaseModel):
    """
    MCP Resource definition for the manifest.

    Defines a resource that MCP clients can access, typically for
    listing or browsing available data.
    """

    name: str = Field(..., description="Resource identifier")
    description: str = Field(..., description="Human-readable resource description")
    uri: str = Field(..., description="Resource endpoint URI")

    model_config = ConfigDict(extra="forbid")


class MCPManifest(BaseModel):
    """
    MCP Server manifest.

    Complete manifest describing all capabilities of the MCP server.
    This is returned by the /mcp/manifest endpoint and used by clients
    to discover available tools and resources.
    """

    tools: list[MCPTool] = Field(..., description="Available MCP tools")
    resources: list[MCPResource] = Field(..., description="Available MCP resources")

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")


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

    @field_validator("query", mode="before")
    @classmethod
    def preprocess_query(cls, v: Any) -> str:
        """
        Preprocess and normalize search query for consistent matching.

        Applies Unicode normalization (NFKC) and whitespace normalization
        to improve search consistency and cache hit rates.
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

        # Step 5: Unicode normalization (NFKC for search)
        # NFKC converts compatibility characters (e.g., ﬀ → ff) and normalizes
        # Unicode combining characters for consistent matching
        query = unicodedata.normalize("NFKC", query)

        # Step 6: Whitespace normalization
        # Replace multiple spaces, tabs, newlines with single space
        query = " ".join(query.split())

        # Step 7: Final validation after normalization
        if len(query) < 1:
            raise ValueError(
                "Query cannot be empty after normalization. "
                "The query may have contained only whitespace or special characters."
            )

        return query

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_int_with_bounds

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
        from docsrs_mcp.validation import coerce_to_int_with_bounds

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
        """Convert various inputs to boolean for MCP compatibility."""
        if v is None or v == "":
            return None
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.lower() in ["true", "1", "yes"]
        return None

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

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")


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

    model_config = ConfigDict(extra="forbid")


# Response Models
class CrateModule(BaseModel):
    """
    Crate module information with hierarchy support.

    Represents a module within a Rust crate, including its name, path,
    and hierarchical relationships.
    """

    name: str = Field(..., description="Module name")
    path: str = Field(..., description="Full module path within the crate")
    parent_id: int | None = Field(None, description="Parent module ID")
    depth: int = Field(0, description="Depth in module hierarchy (0 = root)")
    item_count: int = Field(0, description="Number of items in this module")

    model_config = ConfigDict(extra="forbid")


class ModuleTreeNode(BaseModel):
    """
    Hierarchical module tree node for tree structure responses.

    Represents a module with its children in a tree structure.
    """

    name: str = Field(..., description="Module name")
    path: str = Field(..., description="Full module path within the crate")
    depth: int = Field(0, description="Depth in module hierarchy (0 = root)")
    item_count: int = Field(0, description="Number of items in this module")
    children: list["ModuleTreeNode"] = Field(
        default_factory=list, description="Child modules"
    )

    model_config = ConfigDict(extra="forbid")


# Forward reference resolution for recursive model
ModuleTreeNode.model_rebuild()


class GetCrateSummaryResponse(BaseModel):
    """
    Response for get_crate_summary tool.

    Contains comprehensive metadata about a Rust crate including its
    modules, repository links, and documentation URLs.

    Example:
        ```json
        {
            "name": "tokio",
            "version": "1.35.1",
            "description": "An event-driven, non-blocking I/O platform",
            "modules": [
                {"name": "io", "path": "tokio::io"},
                {"name": "net", "path": "tokio::net"}
            ],
            "repository": "https://github.com/tokio-rs/tokio",
            "documentation": "https://docs.rs/tokio/1.35.1"
        }
        ```
    """

    name: str = Field(..., description="Crate name")
    version: str = Field(..., description="Crate version")
    description: str = Field(..., description="Crate description from Cargo.toml")
    modules: list[CrateModule] = Field(
        ..., description="Top-level modules in the crate"
    )
    repository: str | None = Field(None, description="Source repository URL")
    documentation: str | None = Field(None, description="Documentation URL")

    model_config = ConfigDict(extra="forbid")


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
        from docsrs_mcp.validation import coerce_to_float_with_bounds

        return coerce_to_float_with_bounds(
            value=v,
            field_name="weight parameter",
            min_val=0.0,
            max_val=1.0,
            examples=[0.05, 0.15, 0.7],
        )

    @field_validator(
        "vector_weight", "type_weight", "quality_weight", "examples_weight"
    )
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

    model_config = ConfigDict(extra="forbid")


class SearchResult(BaseModel):
    """
    Individual search result from vector similarity search.

    Each result includes a similarity score and documentation snippet.
    """

    score: float = Field(
        ..., ge=0.0, le=1.0, description="Similarity score (0.0-1.0, higher is better)"
    )
    item_path: str = Field(..., description="Full path to the documented item")
    header: str = Field(..., description="Item header/signature")
    snippet: str = Field(..., description="Documentation excerpt (max 200 chars)")
    suggestions: list[str] | None = None  # See-also suggestions for related items

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score_to_float(cls, v):
        """Convert string numbers to float for MCP client compatibility."""
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_float_with_bounds

        return coerce_to_float_with_bounds(
            value=v,
            field_name="score (similarity score)",
            min_val=0.0,
            max_val=1.0,
            examples=[0.5, 0.7, 0.9],
        )

    model_config = ConfigDict(extra="forbid")


class SearchItemsResponse(BaseModel):
    """
    Response for search_items tool.

    Contains a list of search results ranked by semantic similarity.
    Results are ordered from highest to lowest similarity score.
    """

    results: list[SearchResult] = Field(
        ..., description="Search results ordered by similarity score"
    )

    model_config = ConfigDict(extra="forbid")


class VersionInfo(BaseModel):
    """
    Crate version information.

    Represents a single version of a crate with its yanked status.
    """

    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    yanked: bool = Field(
        False, description="Whether this version has been yanked from crates.io"
    )

    model_config = ConfigDict(extra="forbid")


class ListVersionsResponse(BaseModel):
    """
    Response for list_versions resource.

    Lists all locally cached versions of a crate.
    Note: This only includes versions that have been previously ingested,
    not all versions available on crates.io.
    """

    crate_name: str = Field(..., description="Name of the crate")
    versions: list[VersionInfo] = Field(..., description="List of cached versions")

    model_config = ConfigDict(extra="forbid")


# Error Response
class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Used for all API errors to provide consistent error handling.

    Example:
        ```json
        {
            "error": "item_not_found",
            "detail": "No documentation found for 'tokio::spwan'",
            "status_code": 404,
            "suggestions": ["tokio::spawn", "tokio::spawn_blocking"]
        }
        ```
    """

    error: str = Field(..., description="Error type/category")
    detail: str | None = Field(None, description="Detailed error message for debugging")
    status_code: int = Field(500, description="HTTP status code", ge=400, le=599)
    suggestions: list[str] | None = Field(
        None, description="Suggested alternative paths for fuzzy matching"
    )

    @field_validator("status_code", mode="before")
    @classmethod
    def coerce_status_code_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return 500  # Default to 500 if None
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_int_with_bounds

        return coerce_to_int_with_bounds(
            value=v,
            field_name="status_code (HTTP error code)",
            min_val=400,
            max_val=599,
            examples=[400, 404, 500],
        )

    model_config = ConfigDict(extra="forbid")


class PopularCrate(BaseModel):
    """
    Model for a popular crate with metadata from crates.io API.

    Stores essential information about popular crates for pre-ingestion
    and caching purposes. Includes download statistics and version info
    to enable priority-based processing and cache management.

    Example:
        ```python
        crate = PopularCrate(
            name="tokio",
            downloads=150000000,
            description="An asynchronous runtime for Rust",
            version="1.35.1",
            last_updated=1704067200.0
        )
        ```
    """

    name: str = Field(
        ...,
        description="Crate name as it appears on crates.io",
        examples=["tokio", "serde", "clap"],
    )
    downloads: int = Field(
        ...,
        description="Total download count from crates.io",
        ge=0,
        examples=[150000000, 250000000],
    )
    description: str | None = Field(
        None,
        description="Brief description of the crate's functionality",
        examples=["An asynchronous runtime for Rust", "Serialization framework"],
    )
    version: str | None = Field(
        None,
        description="Latest stable version of the crate",
        examples=["1.35.1", "1.0.195"],
    )
    last_updated: float = Field(
        ...,
        description="Unix timestamp of when this data was fetched",
        examples=[1704067200.0],
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate crate name follows crates.io naming rules."""
        return validate_crate_name(v)

    @field_validator("version")
    @classmethod
    def validate_version(cls, v: str | None) -> str | None:
        """Validate version string if provided."""
        if v is not None:
            return validate_version_string(v)
        return v

    model_config = ConfigDict(extra="forbid")


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
            return 5  # Default value
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_int_with_bounds

        return coerce_to_int_with_bounds(
            value=v,
            field_name="k (number of examples)",
            min_val=1,
            max_val=20,
            examples=[5, 10, 20],
        )

    model_config = ConfigDict(extra="forbid")


class CodeExample(BaseModel):
    """Individual code example with metadata."""

    code: str = Field(..., description="The code example content")
    language: str = Field(..., description="Programming language of the example")
    detected: bool = Field(
        ..., description="Whether language was auto-detected or explicitly specified"
    )
    item_path: str = Field(..., description="Path to the item containing this example")
    context: str | None = Field(
        None, description="Additional context about the example"
    )
    score: float | None = Field(None, description="Relevance score for search results")


class SearchExamplesResponse(BaseModel):
    """Response model for code example search."""

    crate_name: str = Field(..., description="Name of the searched crate")
    version: str = Field(..., description="Version of the crate")
    query: str = Field(..., description="The search query used")
    examples: list[CodeExample] = Field(
        default_factory=list, description="List of matching code examples"
    )
    total_count: int = Field(..., description="Total number of examples found")


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

    @field_validator("concurrency", mode="before")
    @classmethod
    def coerce_concurrency_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_int_with_bounds

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
        from docsrs_mcp.validation import coerce_to_int_with_bounds

        return coerce_to_int_with_bounds(
            value=v,
            field_name="count (number of crates to pre-ingest)",
            min_val=10,
            max_val=500,
            examples=[100, 200, 500],
        )

    model_config = ConfigDict(extra="forbid")


class StartPreIngestionResponse(BaseModel):
    """Response model for start_pre_ingestion tool."""

    status: Literal["started", "already_running", "restarted"] = Field(
        ..., description="Current status of the pre-ingestion operation"
    )
    message: str = Field(..., description="Detailed message about the operation result")
    stats: dict[str, Any] | None = Field(
        default=None, description="Current ingestion statistics if available"
    )
    monitoring: dict[str, str] = Field(
        default_factory=lambda: {
            "health_endpoint": "/health",
            "detailed_status": "/health/pre-ingestion",
        },
        description="Endpoints for monitoring pre-ingestion progress",
    )

    model_config = ConfigDict(extra="forbid")
