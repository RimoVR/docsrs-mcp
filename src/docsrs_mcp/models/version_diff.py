"""
Version difference models for comparing crate versions.

This module defines models for tracking changes between versions of a crate,
including change categories, item signatures, and migration hints.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from .base import strict_config


# Enums for version diff
class ChangeCategory(str, Enum):
    """Categories of changes between versions."""

    BREAKING = "breaking"
    DEPRECATED = "deprecated"
    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"


class ItemKind(str, Enum):
    """Types of Rust items that can be compared."""

    FUNCTION = "function"
    STRUCT = "struct"
    ENUM = "enum"
    TRAIT = "trait"
    TYPE_ALIAS = "type"
    CONST = "const"
    STATIC = "static"
    MODULE = "module"
    MACRO = "macro"
    IMPL = "impl"


class ChangeType(str, Enum):
    """Types of changes that can occur."""

    ADDED = "added"
    REMOVED = "removed"
    MODIFIED = "modified"
    MOVED = "moved"
    DEPRECATED = "deprecated"


class Severity(str, Enum):
    """Severity levels for changes."""

    BREAKING = "breaking"
    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"


class IngestionTier(str, Enum):
    """Tier of documentation ingestion method used."""

    RUSTDOC_JSON = "rustdoc_json"  # Full rustdoc JSON with complete metadata
    SOURCE_EXTRACTION = "source_extraction"  # Fallback source extraction from CDN
    DESCRIPTION_ONLY = "description_only"  # Minimal description fallback
    RUST_LANG_STDLIB = (
        "rust_lang_stdlib"  # rust-lang.org standard library documentation
    )


# Version diff models
class ItemSignature(BaseModel):
    """Represents the signature of a Rust item."""

    raw_signature: str
    generics: str | None = None
    parameters: list[str] | None = None
    return_type: str | None = None
    visibility: str = "public"
    deprecated: bool = False
    deprecation_note: str | None = None


class ChangeDetails(BaseModel):
    """Detailed information about a change."""

    before: ItemSignature | None = None
    after: ItemSignature | None = None
    semantic_changes: list[str] = Field(default_factory=list)
    documentation_changes: bool = False


class ItemChange(BaseModel):
    """Represents a single item change between versions."""

    path: str = Field(..., description="Full path to the item (e.g., 'tokio::spawn')")
    kind: ItemKind
    change_type: ChangeType
    severity: Severity
    details: ChangeDetails
    module_path: str | None = None


class MigrationHint(BaseModel):
    """Suggestion for migrating code to handle a breaking change."""

    affected_path: str
    issue: str
    suggested_fix: str
    severity: Severity
    example_before: str | None = None
    example_after: str | None = None


class DiffSummary(BaseModel):
    """Summary statistics of the diff."""

    total_changes: int = 0
    breaking_changes: int = 0
    deprecated_items: int = 0
    added_items: int = 0
    removed_items: int = 0
    modified_items: int = 0
    migration_hints_available: int = 0


class CompareVersionsRequest(BaseModel):
    """Request to compare two versions of a crate."""

    crate_name: str = Field(..., description="Name of the Rust crate")
    version_a: str = Field(..., description="First version to compare")
    version_b: str = Field(..., description="Second version to compare")
    include_unchanged: bool = Field(
        default=False, description="Include unchanged items in response"
    )
    categories: list[ChangeCategory] = Field(
        default=[
            ChangeCategory.BREAKING,
            ChangeCategory.DEPRECATED,
            ChangeCategory.ADDED,
            ChangeCategory.REMOVED,
            ChangeCategory.MODIFIED,
        ],
        description="Categories of changes to include",
    )
    max_results: int = Field(
        default=1000, ge=1, le=5000, description="Maximum number of changes to return"
    )

    @field_validator("crate_name", mode="before")
    @classmethod
    def validate_crate_name(cls, v: Any) -> str:
        """Validate crate name format."""
        from docsrs_mcp.validation import validate_crate_name

        # Convert any input to string first
        if v is None:
            raise ValueError("Crate name cannot be None")
        if not isinstance(v, str):
            v = str(v)
        return validate_crate_name(v)

    @field_validator("version_a", "version_b", mode="before")
    @classmethod
    def validate_version(cls, v: Any) -> str:
        """Validate version string format."""
        from docsrs_mcp.validation import validate_version_string

        # Convert any input to string first
        if v is None:
            raise ValueError("Version cannot be None")
        if not isinstance(v, str):
            v = str(v)
        return validate_version_string(v)

    @field_validator("include_unchanged", mode="before")
    @classmethod
    def validate_include_unchanged(cls, v: Any) -> bool:
        """Validate and coerce include_unchanged to boolean.

        Handles multiple input formats for Claude Code compatibility:
        - Native booleans: True/False
        - Strings: "true"/"false", "1"/"0", "yes"/"no", "on"/"off", "t"/"f", "y"/"n"
        - Numbers: 0 = False, non-zero = True
        - None: defaults to False
        """
        # Fast path for native booleans
        if isinstance(v, bool):
            return v

        # Handle None (default to False)
        if v is None:
            return False

        # Handle string representations (case-insensitive)
        if isinstance(v, str):
            v_lower = v.lower().strip()
            # Check for truthy strings
            if v_lower in ("true", "1", "yes", "on", "t", "y"):
                return True
            # Check for falsy strings
            if v_lower in ("false", "0", "no", "off", "f", "n"):
                return False
            # Invalid string value
            raise ValueError(
                f"Invalid boolean string: '{v}'. "
                f"Use: true/false, 1/0, yes/no, on/off, t/f, y/n"
            )

        # Handle numeric values (0 = False, non-zero = True)
        if isinstance(v, (int, float)):
            return bool(v)

        # Default fallback for other types
        return bool(v)  # default

    @field_validator("max_results", mode="before")
    @classmethod
    def validate_max_results(cls, v: Any) -> int:
        """Validate and coerce max_results to integer with bounds."""
        from docsrs_mcp.validation import coerce_to_int_with_bounds

        return coerce_to_int_with_bounds(
            v, "max_results", min_val=1, max_val=5000, examples=["100", "1000", "5000"]
        )

    model_config = strict_config


class VersionDiffResponse(BaseModel):
    """Response containing the diff between two versions."""

    crate_name: str
    version_a: str
    version_b: str
    summary: DiffSummary
    changes: dict[str, list[ItemChange]] = Field(default_factory=dict)
    migration_hints: list[MigrationHint] = Field(default_factory=list)
    computation_time_ms: float | None = None
    cached: bool = False

    class Config:
        json_encoders = {Enum: lambda v: v.value}
