"""
Pydantic models for MCP protocol and API requests/responses.

This module defines all request and response models for the docsrs-mcp server,
following the Model Context Protocol (MCP) specification. All models use
strict validation with `extra="forbid"` to prevent injection attacks.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


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
            "k": 5
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

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return v
        if isinstance(v, str):
            try:
                return int(v)
            except ValueError as err:
                raise ValueError(
                    f"k parameter must be a valid integer, got '{v}'"
                ) from err
        return v

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

    model_config = ConfigDict(extra="forbid")


# Response Models
class CrateModule(BaseModel):
    """
    Crate module information.

    Represents a module within a Rust crate, including its name and path.
    """

    name: str = Field(..., description="Module name")
    path: str = Field(..., description="Full module path within the crate")

    model_config = ConfigDict(extra="forbid")


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
            "error": "RateLimitExceeded",
            "detail": "Rate limit exceeded. Please retry after 1 second.",
            "status_code": 429
        }
        ```
    """

    error: str = Field(..., description="Error type/category")
    detail: str | None = Field(None, description="Detailed error message for debugging")
    status_code: int = Field(500, description="HTTP status code", ge=400, le=599)

    model_config = ConfigDict(extra="forbid")
