"""
Response models for cross-reference operations.

This module defines response models for import resolution, dependency graphs,
migration suggestions, and re-export tracing.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, AliasChoices

from .base import strict_config


class ImportAlternative(BaseModel):
    """Alternative import path suggestion."""

    path: str = Field(..., description="Alternative import path")
    confidence: float = Field(..., description="Confidence score (0.0-1.0)")
    link_type: str = Field(
        ...,
        validation_alias=AliasChoices("link_type", "type"),
        serialization_alias="link_type",
        description="Type of link (reexport, crossref, etc.)",
    )
    alias: str | None = Field(None, description="Alias path if re-exported")

    model_config = strict_config


class ResolveImportResponse(BaseModel):
    """Response for import resolution requests."""

    resolved_path: str = Field(..., description="Resolved import path")
    confidence: float = Field(..., description="Resolution confidence (0.0-1.0)")
    alternatives: list[ImportAlternative] = Field(
        default_factory=list, description="Alternative import suggestions"
    )
    source_crate: str | None = Field(None, description="Source crate name")

    model_config = strict_config


class DependencyNode(BaseModel):
    """Node in a dependency graph."""

    name: str = Field(..., description="Dependency name")
    version: str | None = Field(None, description="Version constraint")
    dependencies: list[DependencyNode] = Field(
        default_factory=list, description="Child dependencies"
    )

    model_config = strict_config


# Enable forward reference for recursive model
DependencyNode.model_rebuild()


class DependencyGraphResponse(BaseModel):
    """Response for dependency graph requests."""

    root: DependencyNode = Field(..., description="Root node of dependency tree")
    total_nodes: int = Field(..., description="Total number of unique nodes")
    max_depth: int = Field(..., description="Maximum depth traversed")
    has_cycles: bool = Field(..., description="Whether cycles were detected")

    model_config = strict_config


class MigrationSuggestion(BaseModel):
    """Suggestion for migrating between versions."""

    old_path: str | None = Field(..., description="Path in old version")
    new_path: str | None = Field(..., description="Path in new version")
    change_type: str = Field(
        ..., description="Type of change (renamed, moved, removed, added)"
    )
    confidence: float = Field(..., description="Suggestion confidence (0.0-1.0)")
    notes: str | None = Field(None, description="Additional migration guidance")

    model_config = strict_config


class MigrationSuggestionsResponse(BaseModel):
    """Response for migration suggestions requests."""

    crate_name: str = Field(..., description="Name of the crate")
    from_version: str = Field(..., description="Starting version")
    to_version: str = Field(..., description="Target version")
    suggestions: list[MigrationSuggestion] = Field(
        default_factory=list, description="List of migration suggestions"
    )

    model_config = strict_config


class ReexportTrace(BaseModel):
    """Trace of re-export chain to original source."""

    chain: list[str] = Field(
        default_factory=list, description="Re-export chain (source -> target format)"
    )
    original_source: str = Field(..., description="Original source path")
    original_crate: str = Field(..., description="Original crate name")

    model_config = strict_config


class CrossReferenceInfo(BaseModel):
    """General cross-reference information."""

    source_path: str = Field(..., description="Source item path")
    target_path: str = Field(..., description="Target item path")
    link_type: str = Field(..., description="Type of cross-reference")
    confidence_score: float = Field(..., description="Confidence of the reference")
    source_crate: str | None = Field(None, description="Source crate name")
    target_crate: str | None = Field(None, description="Target crate name")

    model_config = strict_config


class CrossReferencesResponse(BaseModel):
    """Response for general cross-reference queries."""

    references: list[CrossReferenceInfo] = Field(
        default_factory=list, description="List of cross-references"
    )
    total_count: int = Field(..., description="Total number of references found")

    model_config = strict_config
