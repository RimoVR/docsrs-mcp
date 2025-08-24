"""Response models for workflow enhancement features.

This module defines Pydantic models for Phase 7 workflow features including
progressive detail levels, usage pattern extraction, and learning path generation.
"""

from typing import Any

from pydantic import BaseModel, Field

from .base import strict_config


class ProgressiveDetailResponse(BaseModel):
    """Response for progressive detail level documentation."""

    model_config = strict_config

    item_path: str = Field(description="Full path to the documentation item")
    detail_level: str = Field(
        description="Current detail level (summary/detailed/expert)"
    )
    available_levels: list[str] = Field(
        default=["summary", "detailed", "expert"],
        description="Available detail levels for this item",
    )

    # Summary level fields
    summary: str | None = Field(None, description="Brief summary of the item")
    signature: str | None = Field(None, description="Function/type signature")
    type: str | None = Field(None, description="Item type (function/struct/trait/etc)")
    visibility: str | None = Field(
        None, description="Visibility level (public/private/etc)"
    )
    deprecated: bool | None = Field(None, description="Whether the item is deprecated")

    # Detailed level fields
    documentation: str | None = Field(None, description="Full documentation content")
    examples: str | None = Field(None, description="Code examples")

    # Expert level fields
    generic_params: str | None = Field(None, description="Generic parameters")
    trait_bounds: str | None = Field(None, description="Trait bounds")
    related_items: list[dict[str, Any]] | None = Field(
        None, description="Related items in the same module"
    )

    # Error field
    error: str | None = Field(None, description="Error message if item not found")


class UsagePattern(BaseModel):
    """Individual usage pattern."""

    model_config = strict_config

    pattern: str = Field(description="The usage pattern")
    frequency: int = Field(description="How often this pattern appears")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence score for this pattern"
    )
    pattern_type: str = Field(
        description="Category of pattern (method_call/error_handling/etc)"
    )
    examples: list[dict[str, str]] = Field(
        default_factory=list, description="Example occurrences of this pattern"
    )


class UsagePatternResponse(BaseModel):
    """Response for usage pattern extraction."""

    model_config = strict_config

    crate_name: str = Field(description="Name of the analyzed crate")
    version: str = Field(description="Version of the crate")
    patterns: list[UsagePattern] = Field(
        default_factory=list, description="Extracted usage patterns"
    )
    total_patterns_found: int = Field(
        description="Total number of patterns found before filtering"
    )
    analysis_scope: str = Field(
        default="documentation_and_examples", description="Scope of pattern analysis"
    )


class LearningStep(BaseModel):
    """Individual step in a learning path."""

    model_config = strict_config

    order: int = Field(description="Step order in the learning path")
    title: str = Field(description="Step title")
    description: str = Field(description="What this step covers")
    items: list[dict[str, Any]] | None = Field(
        None, description="Specific items to learn in this step"
    )
    estimated_time: str = Field(description="Estimated time to complete this step")
    resources: list[str] | None = Field(
        None, description="Additional resources for this step"
    )


class LearningPathResponse(BaseModel):
    """Response for learning path generation."""

    model_config = strict_config

    crate_name: str = Field(description="Name of the crate")
    type: str = Field(description="Type of learning path (migration/onboarding)")
    from_version: str = Field(description="Starting version or 'new_user'")
    to_version: str = Field(description="Target version")
    steps: list[LearningStep] = Field(
        default_factory=list, description="Ordered learning steps"
    )
    estimated_effort: str = Field(description="Total estimated effort")
    focus_areas: list[str] | None = Field(
        None, description="Optional focus areas for the learning path"
    )
    prerequisites: list[str] | None = Field(
        None, description="Prerequisites for this learning path"
    )
    error: str | None = Field(None, description="Error message if generation failed")


class WorkflowMetrics(BaseModel):
    """Metrics for workflow operations."""

    model_config = strict_config

    cache_hit_rate: float = Field(
        ge=0.0, le=1.0, description="Cache hit rate for workflow operations"
    )
    average_response_time_ms: float = Field(
        description="Average response time in milliseconds"
    )
    patterns_cached: int = Field(description="Number of cached pattern analyses")
    learning_paths_cached: int = Field(description="Number of cached learning paths")
    detail_levels_cached: int = Field(
        description="Number of cached detail level responses"
    )
