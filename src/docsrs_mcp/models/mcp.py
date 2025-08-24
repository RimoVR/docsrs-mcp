"""
MCP (Model Context Protocol) specific models.

This module defines the MCP manifest models including MCPTool, MCPResource,
and MCPManifest used for describing the capabilities of the MCP server.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from .base import strict_config


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
        json_schema_extra={"maxItems": 5},  # Increased to match actual usage
    )

    model_config = strict_config

    @field_validator("use_cases", "examples", mode="before")
    @classmethod
    def validate_string_arrays(cls, v: Any, info: ValidationInfo) -> list[str] | None:
        """Validate string arrays with length constraints."""
        if v is None:
            return None

        # Handle string input (split by newlines or commas)
        if isinstance(v, str):
            v = [
                item.strip() for item in v.replace("\n", ",").split(",") if item.strip()
            ]

        # Ensure list type
        if not isinstance(v, list):
            v = [v]

        # Validate constraints
        field_name = info.field_name
        max_items = 5 if field_name == "use_cases" else 3
        max_length = 200  # Reasonable limit for individual strings

        if len(v) > max_items:
            raise ValueError(
                f"{field_name} cannot exceed {max_items} items (got {len(v)})"
            )

        # Validate and truncate individual strings
        validated = []
        for i, item in enumerate(v):
            item_str = str(item).strip()
            if len(item_str) > max_length:
                # Truncate at word boundary with ellipsis
                truncated = item_str[: max_length - 3].rsplit(" ", 1)[0] + "..."
                validated.append(truncated)
            else:
                validated.append(item_str)

        return validated if validated else None


class MCPResource(BaseModel):
    """
    MCP Resource definition for the manifest.

    Defines a resource that MCP clients can access, typically for
    listing or browsing available data.
    """

    name: str = Field(..., description="Resource identifier")
    description: str = Field(..., description="Human-readable resource description")
    uri: str = Field(..., description="Resource endpoint URI")

    model_config = strict_config


class MCPManifest(BaseModel):
    """
    MCP Server manifest.

    Complete manifest describing all capabilities of the MCP server.
    This is returned by the /mcp/manifest endpoint and used by clients
    to discover available tools and resources.
    """

    tools: list[MCPTool] = Field(..., description="Available MCP tools")
    resources: list[MCPResource] = Field(..., description="Available MCP resources")

    model_config = strict_config


class TraitImplementationResponse(BaseModel):
    """Response model for trait implementor queries."""

    trait_path: str = Field(..., description="Full path to the trait")
    implementors: list[dict[str, Any]] = Field(
        ..., description="List of implementing types"
    )
    total_count: int = Field(..., description="Total number of implementors")

    model_config = strict_config


class TypeTraitsResponse(BaseModel):
    """Response model for type trait queries."""

    type_path: str = Field(..., description="Full path to the type")
    traits: list[dict[str, Any]] = Field(..., description="List of implemented traits")
    total_count: int = Field(..., description="Total number of traits")

    model_config = strict_config


class MethodSignatureResponse(BaseModel):
    """Response model for method resolution queries."""

    type_path: str = Field(..., description="Full path to the type")
    method_name: str = Field(..., description="Name of the method")
    candidates: list[dict[str, Any]] = Field(..., description="Method candidates")
    disambiguation_hints: list[str] = Field(
        default_factory=list, description="Hints for disambiguation"
    )
    resolution_status: str = Field(..., description="Resolution status")

    model_config = strict_config


class AssociatedItemResponse(BaseModel):
    """Response model for associated item queries."""

    container_path: str = Field(..., description="Path to the containing trait/type")
    associated_items: dict[str, list[dict[str, Any]]] = Field(
        ..., description="Associated items grouped by kind"
    )
    total_count: int = Field(..., description="Total number of items")

    model_config = strict_config


class GenericConstraintResponse(BaseModel):
    """Response model for generic constraint queries."""

    item_path: str = Field(..., description="Path to the item")
    generic_constraints: dict[str, list[dict[str, Any]]] = Field(
        ..., description="Generic constraints grouped by kind"
    )
    total_params: int = Field(..., description="Total number of parameters")

    model_config = strict_config
