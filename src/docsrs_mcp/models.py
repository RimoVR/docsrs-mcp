"""Pydantic models for MCP protocol and API requests/responses."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# MCP Manifest Models
class MCPTool(BaseModel):
    """MCP Tool definition."""

    name: str
    description: str
    input_schema: dict[str, Any]

    model_config = ConfigDict(extra="forbid")


class MCPResource(BaseModel):
    """MCP Resource definition."""

    name: str
    description: str
    uri: str

    model_config = ConfigDict(extra="forbid")


class MCPManifest(BaseModel):
    """MCP Server manifest."""

    tools: list[MCPTool]
    resources: list[MCPResource]

    model_config = ConfigDict(extra="forbid")


# Request Models
class GetCrateSummaryRequest(BaseModel):
    """Request for get_crate_summary tool."""

    crate_name: str = Field(..., description="Name of the crate to query")
    version: str | None = Field(None, description="Specific version (default: latest)")

    model_config = ConfigDict(extra="forbid")


class SearchItemsRequest(BaseModel):
    """Request for search_items tool."""

    crate_name: str = Field(..., description="Name of the crate to search in")
    query: str = Field(..., description="Search query text")
    version: str | None = Field(None, description="Specific version (default: latest)")
    k: int | None = Field(5, description="Number of results to return", ge=1, le=20)

    model_config = ConfigDict(extra="forbid")


class GetItemDocRequest(BaseModel):
    """Request for get_item_doc tool."""

    crate_name: str = Field(..., description="Name of the crate")
    item_path: str = Field(
        ..., description="Full path to the item (e.g., 'tokio::spawn')"
    )
    version: str | None = Field(None, description="Specific version (default: latest)")

    model_config = ConfigDict(extra="forbid")


# Response Models
class CrateModule(BaseModel):
    """Crate module information."""

    name: str
    path: str

    model_config = ConfigDict(extra="forbid")


class GetCrateSummaryResponse(BaseModel):
    """Response for get_crate_summary tool."""

    name: str
    version: str
    description: str
    modules: list[CrateModule]
    repository: str | None = None
    documentation: str | None = None

    model_config = ConfigDict(extra="forbid")


class SearchResult(BaseModel):
    """Individual search result."""

    score: float = Field(..., ge=0.0, le=1.0)
    item_path: str
    header: str
    snippet: str

    model_config = ConfigDict(extra="forbid")


class SearchItemsResponse(BaseModel):
    """Response for search_items tool."""

    results: list[SearchResult]

    model_config = ConfigDict(extra="forbid")


class VersionInfo(BaseModel):
    """Crate version information."""

    version: str
    yanked: bool = False

    model_config = ConfigDict(extra="forbid")


class ListVersionsResponse(BaseModel):
    """Response for list_versions resource."""

    crate_name: str
    versions: list[VersionInfo]

    model_config = ConfigDict(extra="forbid")


# Error Response
class ErrorResponse(BaseModel):
    """Standard error response."""

    error: str
    detail: str | None = None
    status_code: int = 500

    model_config = ConfigDict(extra="forbid")
