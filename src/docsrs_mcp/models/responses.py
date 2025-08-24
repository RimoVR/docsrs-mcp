"""
Response models for the docsrs-mcp API.

This module defines all response models returned by the API endpoints,
including data structures for crate modules, search results, and more.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

from docsrs_mcp.validation import (
    coerce_to_float_with_bounds,
    validate_crate_name,
    validate_version_string,
)

from .base import strict_config


# Data Models
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

    model_config = strict_config


class ModuleTreeNode(BaseModel):
    """
    Hierarchical module tree node for tree structure responses.

    Represents a module with its children in a tree structure.
    """

    name: str = Field(..., description="Module name")
    path: str = Field(..., description="Full module path within the crate")
    depth: int = Field(0, description="Depth in module hierarchy (0 = root)")
    item_count: int = Field(0, description="Number of items in this module")
    children: list[ModuleTreeNode] = Field(
        default_factory=list, description="Child modules"
    )

    model_config = strict_config


# Forward reference resolution for recursive model
ModuleTreeNode.model_rebuild()


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
    snippet: str = Field(
        ..., description="Documentation excerpt (200-400 chars with context)"
    )
    suggestions: list[str] | None = None  # See-also suggestions for related items
    is_stdlib: bool = Field(
        default=False, description="Whether this is a Rust standard library item"
    )
    is_dependency: bool = Field(
        default=False, description="Whether this item is from a dependency"
    )

    @field_validator("score", mode="before")
    @classmethod
    def coerce_score_to_float(cls, v):
        """Convert string numbers to float for MCP client compatibility."""
        # Use enhanced validation with examples
        return coerce_to_float_with_bounds(
            value=v,
            field_name="score (similarity score)",
            min_val=0.0,
            max_val=1.0,
            examples=[0.5, 0.7, 0.9],
        )

    model_config = strict_config


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


class VersionInfo(BaseModel):
    """
    Crate version information.

    Represents a single version of a crate with its yanked status.
    """

    version: str = Field(..., description="Version string (e.g., '1.0.0')")
    yanked: bool = Field(
        False, description="Whether this version has been yanked from crates.io"
    )

    model_config = strict_config


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

    @field_validator("name", mode="before")
    @classmethod
    def validate_name(cls, v: Any) -> str:
        """Validate crate name follows crates.io naming rules."""
        # Handle any input type by converting to string first
        if v is None:
            raise ValueError("Crate name cannot be None")
        if not isinstance(v, str):
            v = str(v)
        return validate_crate_name(v)

    @field_validator("version", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> str | None:
        """Validate version string if provided."""
        if v is None:
            return None
        if not isinstance(v, str):
            v = str(v)
        return validate_version_string(v)

    model_config = strict_config


# Response Models
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

    model_config = strict_config


class SearchItemsResponse(BaseModel):
    """
    Response for search_items tool.

    Contains a list of search results ranked by semantic similarity.
    Results are ordered from highest to lowest similarity score.
    """

    results: list[SearchResult] = Field(
        ..., description="Search results ordered by similarity score"
    )

    model_config = strict_config


class GetItemDocResponse(BaseModel):
    """
    Response for get_item_doc tool.

    Contains the complete documentation for a specific Rust item.
    This is typically used when displaying detailed documentation
    for a function, struct, trait, or other Rust item.
    """

    item_path: str = Field(..., description="Full path to the documented item")
    documentation: str = Field(..., description="Complete documentation in markdown")
    signature: str | None = Field(None, description="Item signature/declaration")
    examples: list[str] | None = Field(
        None, description="Code examples from documentation"
    )
    is_stdlib: bool = Field(
        default=False, description="Whether this is a Rust standard library item"
    )
    is_dependency: bool = Field(
        default=False,
        description="Whether this item is from a dependency (filtered by default)",
    )

    model_config = strict_config


class GetModuleTreeResponse(BaseModel):
    """
    Response for get_module_tree tool.

    Contains the hierarchical module structure of a crate represented
    as a tree of ModuleTreeNode objects.
    """

    crate_name: str = Field(..., description="Name of the crate")
    version: str = Field(..., description="Version of the crate")
    tree: ModuleTreeNode = Field(..., description="Root module tree node")

    model_config = strict_config


class ListVersionsResponse(BaseModel):
    """
    Response for list_versions resource.

    Lists all locally cached versions of a crate.
    Note: This only includes versions that have been previously ingested,
    not all versions available on crates.io.
    """

    crate_name: str = Field(..., description="Name of the crate")
    versions: list[VersionInfo] = Field(..., description="List of cached versions")

    model_config = strict_config


class SearchExamplesResponse(BaseModel):
    """Response model for code example search."""

    crate_name: str = Field(..., description="Name of the searched crate")
    version: str = Field(..., description="Version of the crate")
    query: str = Field(..., description="The search query used")
    examples: list[CodeExample] = Field(
        default_factory=list, description="List of matching code examples"
    )
    total_count: int = Field(..., description="Total number of examples found")


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

    model_config = strict_config


class IngestCargoFileResponse(BaseModel):
    """Response model for Cargo file ingestion."""

    status: Literal["started", "completed", "failed"] = Field(
        ..., description="Status of the ingestion operation"
    )
    message: str = Field(..., description="Detailed message about the operation")
    crates_found: int = Field(..., description="Total number of crates found in file")
    crates_queued: int = Field(..., description="Number of crates queued for ingestion")
    crates_skipped: int = Field(
        ..., description="Number of crates skipped (already exist)"
    )
    estimated_time_seconds: float | None = Field(
        default=None, description="Estimated time to complete ingestion"
    )

    model_config = strict_config


class PreIngestionControlResponse(BaseModel):
    """Response model for pre-ingestion control."""

    status: Literal["success", "failed", "no_change"] = Field(
        ..., description="Result of the control operation"
    )
    message: str = Field(..., description="Detailed message about the operation")
    worker_state: str | None = Field(
        default=None, description="Current state of the pre-ingestion worker"
    )
    current_stats: dict[str, Any] | None = Field(
        default=None, description="Current ingestion statistics if available"
    )

    model_config = strict_config
