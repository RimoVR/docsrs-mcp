"""
Models package for docsrs-mcp.

This module re-exports all models to maintain backward compatibility
after splitting the original models.py into focused modules.

All existing imports like `from docsrs_mcp.models import ModelName`
will continue to work without modification.
"""

# Re-export from base module
from .base import ErrorResponse

# Re-export from MCP module
from .mcp import (
    AssociatedItemResponse,
    GenericConstraintResponse,
    MCPManifest,
    MCPResource,
    MCPTool,
    MethodSignatureResponse,
    TraitImplementationResponse,
    TypeTraitsResponse,
)

# Re-export from requests module
from .requests import (
    GetCrateSummaryRequest,
    GetItemDocRequest,
    GetModuleTreeRequest,
    IngestCargoFileRequest,
    ListVersionsRequest,
    PreIngestionControlRequest,
    RankingConfig,
    SearchExamplesRequest,
    SearchItemsRequest,
    StartPreIngestionRequest,
)

# Re-export from responses module
from .responses import (
    CodeExample,
    CrateModule,
    GetCrateSummaryResponse,
    GetItemDocResponse,
    GetModuleTreeResponse,
    IngestCargoFileResponse,
    ListVersionsResponse,
    ModuleTreeNode,
    PopularCrate,
    PreIngestionControlResponse,
    SearchExamplesResponse,
    SearchItemsResponse,
    SearchResult,
    StartPreIngestionResponse,
    VersionInfo,
)

# Re-export from version_diff module
from .version_diff import (
    ChangeCategory,
    ChangeDetails,
    ChangeType,
    CompareVersionsRequest,
    DiffSummary,
    IngestionTier,
    ItemChange,
    ItemKind,
    ItemSignature,
    MigrationHint,
    Severity,
    VersionDiffResponse,
)

# Re-export from cross_references module
from .cross_references import (
    CrossReferenceInfo,
    CrossReferencesResponse,
    DependencyGraphResponse,
    DependencyNode,
    ImportAlternative,
    MigrationSuggestion,
    MigrationSuggestionsResponse,
    ReexportTrace,
    ResolveImportResponse,
)

# Define __all__ for explicit exports
__all__ = [
    # Base models
    "ErrorResponse",
    # MCP models
    "MCPTool",
    "MCPResource",
    "MCPManifest",
    "AssociatedItemResponse",
    "GenericConstraintResponse",
    "MethodSignatureResponse",
    "TraitImplementationResponse",
    "TypeTraitsResponse",
    # Request models
    "GetCrateSummaryRequest",
    "SearchItemsRequest",
    "GetItemDocRequest",
    "GetModuleTreeRequest",
    "ListVersionsRequest",
    "SearchExamplesRequest",
    "StartPreIngestionRequest",
    "IngestCargoFileRequest",
    "PreIngestionControlRequest",
    "RankingConfig",
    # Response models
    "GetCrateSummaryResponse",
    "SearchItemsResponse",
    "GetItemDocResponse",
    "GetModuleTreeResponse",
    "ListVersionsResponse",
    "SearchExamplesResponse",
    "StartPreIngestionResponse",
    "IngestCargoFileResponse",
    "PreIngestionControlResponse",
    # Data models
    "CrateModule",
    "ModuleTreeNode",
    "SearchResult",
    "CodeExample",
    "VersionInfo",
    "PopularCrate",
    # Version diff models
    "CompareVersionsRequest",
    "VersionDiffResponse",
    "ChangeCategory",
    "ItemKind",
    "ChangeType",
    "Severity",
    "IngestionTier",
    "ItemSignature",
    "ChangeDetails",
    "ItemChange",
    "MigrationHint",
    "DiffSummary",
    # Cross-reference models
    "CrossReferenceInfo",
    "CrossReferencesResponse",
    "DependencyGraphResponse",
    "DependencyNode",
    "ImportAlternative",
    "MigrationSuggestion",
    "MigrationSuggestionsResponse",
    "ReexportTrace",
    "ResolveImportResponse",
]

