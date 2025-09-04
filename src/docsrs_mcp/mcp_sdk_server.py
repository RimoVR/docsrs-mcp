"""MCP SDK implementation for docsrs-mcp server."""

import asyncio
import json
import logging
import os
import random
import sys

from mcp import types
from mcp.server import Server
from mcp.server.lowlevel.helper_types import ReadResourceContents
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from pydantic import AnyUrl

from .models import (
    AssociatedItemResponse,
    DependencyGraphResponse,
    GenericConstraintResponse,
    GetCrateSummaryResponse,
    GetItemDocResponse,
    GetModuleTreeResponse,
    IngestCargoFileResponse,
    ListVersionsResponse,
    MethodSignatureResponse,
    MigrationSuggestionsResponse,
    PreIngestionControlResponse,
    ReexportTrace,
    ResolveImportResponse,
    SearchExamplesResponse,
    SearchItemsResponse,
    StartPreIngestionResponse,
    TraitImplementationResponse,
    TypeTraitsResponse,
    VersionDiffResponse,
)
from .models.workflow import (
    LearningPathResponse,
    ProgressiveDetailResponse,
    UsagePatternResponse,
)

# Configure logging to stderr to avoid STDIO corruption
# This is critical for STDIO transport to work properly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

# Services will be imported lazily in factory functions

logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("docsrs-mcp")

# Global service instances (singleton pattern for lazy loading)
_crate_service = None
_ingestion_service = None
_type_navigation_service = None
_workflow_service = None
_cross_reference_service = None


def get_crate_service():
    """Get or create the CrateService instance with lazy loading."""
    global _crate_service
    if _crate_service is None:
        from .services.crate_service import CrateService

        logger.info("Lazy loaded CrateService")
        _crate_service = CrateService()
    return _crate_service


def get_ingestion_service():
    """Get or create the IngestionService instance with lazy loading."""
    global _ingestion_service
    if _ingestion_service is None:
        from .services.ingestion_service import IngestionService

        logger.info("Lazy loaded IngestionService")
        _ingestion_service = IngestionService()
    return _ingestion_service


def get_type_navigation_service():
    """Get or create the TypeNavigationService instance with lazy loading."""
    global _type_navigation_service
    if _type_navigation_service is None:
        from .services.type_navigation_service import TypeNavigationService

        logger.info("Lazy loaded TypeNavigationService")
        _type_navigation_service = TypeNavigationService()
    return _type_navigation_service


def get_workflow_service():
    """Get or create the WorkflowService instance with lazy loading."""
    global _workflow_service
    if _workflow_service is None:
        from .services.workflow_service import WorkflowService

        logger.info("Lazy loaded WorkflowService")
        _workflow_service = WorkflowService()
    return _workflow_service


def get_cross_reference_service(db_path: str):
    """Get or create the CrossReferenceService instance with lazy loading."""
    # CrossReferenceService takes db_path parameter, so we can't use global singleton
    # Import lazily and create new instance each time
    from .services.cross_reference_service import CrossReferenceService

    logger.info("Lazy loaded CrossReferenceService")
    return CrossReferenceService(db_path)


# Parameter validation utilities
def validate_int_parameter(
    value: str, default: int = None, min_val: int = None, max_val: int = None
) -> int:
    """Validate and convert string to integer with bounds checking."""
    if not value and default is not None:
        return default
    try:
        result = int(value)
        if min_val is not None and result < min_val:
            raise ValueError(f"Value must be >= {min_val}")
        if max_val is not None and result > max_val:
            raise ValueError(f"Value must be <= {max_val}")
        return result
    except (ValueError, TypeError) as e:
        if default is not None:
            return default
        raise ValueError(f"Invalid integer value: {value}") from e


def validate_bool_parameter(value: str, default: bool = None) -> bool | None:
    """Validate and convert string to boolean."""
    if not value and default is not None:
        return default
    if value is None:
        return None
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    elif value.lower() in ("false", "0", "no", "off"):
        return False
    else:
        if default is not None:
            return default
        return None


# MCP Tool Implementations


async def get_crate_summary(crate_name: str, version: str = "latest") -> dict:
    """Get summary information about a Rust crate.

    Fetches crate metadata including name, version, description, repository URL,
    and module structure. If the crate hasn't been ingested yet, it will be
    downloaded and processed automatically.

    Args:
        crate_name: Name of the Rust crate (e.g., 'tokio', 'serde')
        version: Specific version or 'latest' (default: latest)

    Returns:
        Crate metadata and module structure
    """
    try:
        result = await get_crate_service().get_crate_summary(
            crate_name, version if version != "latest" else None
        )
        return GetCrateSummaryResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_crate_summary: {e}")
        raise


async def search_items(
    crate_name: str,
    query: str,
    version: str = "latest",
    k: str = "5",
    item_type: str | None = None,
    crate_filter: str | None = None,
    module_path: str | None = None,
    has_examples: str | None = None,
    min_doc_length: str | None = None,
    visibility: str | None = None,
    deprecated: str | None = None,
) -> dict:
    """Search for items in a crate's documentation using semantic similarity.

    Performs vector similarity search across all documentation in the specified crate.
    Results are ranked by semantic similarity to the query.

    Args:
        crate_name: Name of the crate to search within
        query: Natural language search query
        version: Specific version or 'latest' (default: latest)
        k: Number of results to return (1-20, default: 5)
        item_type: Filter by item type (function, struct, trait, enum, module)
        crate_filter: Filter results to specific crate
        module_path: Filter results to specific module path within the crate
        has_examples: Filter to only items with code examples ("true"/"false")
        min_doc_length: Minimum documentation length in characters
        visibility: Filter by item visibility (public, private, crate)
        deprecated: Filter by deprecation status ("true"/"false")

    Returns:
        Ranked search results with smart snippets
    """
    try:
        # Validate and convert parameters
        k_int = validate_int_parameter(k, default=5, min_val=1, max_val=20)
        has_examples_bool = validate_bool_parameter(has_examples)
        deprecated_bool = validate_bool_parameter(deprecated)
        min_doc_length_int = (
            validate_int_parameter(min_doc_length, min_val=100, max_val=10000)
            if min_doc_length
            else None
        )

        results = await get_crate_service().search_items(
            crate_name,
            query,
            version=version if version != "latest" else None,
            k=k_int,
            item_type=item_type,
            crate_filter=crate_filter,
            module_path=module_path,
            has_examples=has_examples_bool,
            min_doc_length=min_doc_length_int,
            visibility=visibility,
            deprecated=deprecated_bool,
        )
        return SearchItemsResponse(results=results).model_dump()
    except Exception as e:
        logger.error(f"Error in search_items: {e}")
        raise


async def get_item_doc(
    crate_name: str, item_path: str, version: str = "latest"
) -> dict:
    """Get complete documentation for a specific item in a crate.

    Retrieves the full documentation for a specific item identified by its path
    (e.g., 'tokio::spawn', 'serde::Deserialize'). Returns markdown-formatted
    documentation including examples and detailed descriptions.

    Args:
        crate_name: Name of the crate containing the item
        item_path: Full path to the item (e.g., 'tokio::spawn', 'std::vec::Vec')
        version: Specific version or 'latest' (default: latest)

    Returns:
        Complete item documentation in markdown format
    """
    try:
        result = await get_crate_service().get_item_doc(
            crate_name, item_path, version if version != "latest" else None
        )
        return GetItemDocResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_item_doc: {e}")
        raise


async def get_module_tree(crate_name: str, version: str = "latest") -> dict:
    """Get the module hierarchy tree for a Rust crate.

    Returns a hierarchical tree structure of all modules in the specified crate,
    including parent-child relationships, depth levels, and item counts per module.

    Args:
        crate_name: Name of the Rust crate to get module tree for
        version: Specific version or 'latest' (default: latest)

    Returns:
        Module hierarchy tree structure
    """
    try:
        result = await get_crate_service().get_module_tree(
            crate_name, version if version != "latest" else None
        )
        return GetModuleTreeResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_module_tree: {e}")
        raise


async def search_examples(
    crate_name: str,
    query: str,
    version: str = "latest",
    k: str = "5",
    language: str | None = None,
) -> dict:
    """Search for code examples in a crate's documentation.

    Searches through extracted code examples from documentation and returns
    matching examples with language detection and metadata.

    Args:
        crate_name: Name of the crate to search within
        query: Search query for finding relevant code examples
        version: Specific version to search (default: latest)
        k: Number of examples to return (default: 5)
        language: Filter examples by programming language (e.g., 'rust', 'bash', 'toml')

    Returns:
        Code examples from documentation with context
    """
    try:
        k_int = validate_int_parameter(k, default=5, min_val=1, max_val=20)

        result = await get_crate_service().search_examples(
            crate_name,
            query,
            version=version if version != "latest" else None,
            k=k_int,
            language=language,
        )
        return SearchExamplesResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in search_examples: {e}")
        raise


async def list_versions(crate_name: str) -> dict:
    """List all locally cached versions of a crate.

    Returns a list of all versions that have been previously ingested and cached.
    If the crate hasn't been ingested yet, it will be downloaded and processed
    automatically using the latest version.

    Args:
        crate_name: Name of the Rust crate to query

    Returns:
        Available crate versions
    """
    try:
        result = await get_crate_service().list_versions(crate_name)
        return ListVersionsResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in list_versions: {e}")
        raise


async def start_pre_ingestion(
    count: str = "100", concurrency: str = "3", force: str = "false"
) -> dict:
    """Start background pre-ingestion of popular Rust crates.

    This tool initiates the pre-ingestion system to cache the most popular
    crates, eliminating cold-start latency for common queries.

    Args:
        count: Number of crates to pre-ingest (10-500, default: 100)
        concurrency: Number of parallel download workers (1-10, default: 3)
        force: Force restart if pre-ingestion is already running ("true"/"false")

    Returns:
        Status of the pre-ingestion operation
    """
    try:
        count_int = validate_int_parameter(count, default=100, min_val=10, max_val=500)
        concurrency_int = validate_int_parameter(
            concurrency, default=3, min_val=1, max_val=10
        )
        force_bool = validate_bool_parameter(force, default=False)

        result = await get_ingestion_service().start_pre_ingestion(
            count=count_int, concurrency=concurrency_int, force=force_bool
        )
        return StartPreIngestionResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in start_pre_ingestion: {e}")
        raise


async def control_pre_ingestion(action: str) -> dict:
    """Control the pre-ingestion worker (pause/resume/stop).

    Allows runtime control of the background pre-ingestion process without
    requiring server restart.

    Args:
        action: Control action to perform on the pre-ingestion worker (pause/resume/stop)

    Returns:
        Result of the control operation
    """
    try:
        if action not in ["pause", "resume", "stop"]:
            raise ValueError(
                f"Invalid action: {action}. Must be one of: pause, resume, stop"
            )

        result = await get_ingestion_service().control_pre_ingestion(action)
        return PreIngestionControlResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in control_pre_ingestion: {e}")
        raise


async def ingest_cargo_file(
    file_path: str,
    concurrency: str = "3",
    skip_existing: str = "true",
    resolve_versions: str = "false",
) -> dict:
    """Ingest crates from a Cargo.toml or Cargo.lock file.

    Parses the specified Cargo file and queues all dependencies for ingestion.
    Skips crates that are already cached to avoid redundant downloads.

    Args:
        file_path: Path to Cargo.toml or Cargo.lock file
        concurrency: Number of parallel download workers (1-10, default: 3)
        skip_existing: Skip already ingested crates ("true"/"false", default: true)
        resolve_versions: Resolve version specifications to concrete versions ("true"/"false")

    Returns:
        Status of the cargo file ingestion
    """
    try:
        concurrency_int = validate_int_parameter(
            concurrency, default=3, min_val=1, max_val=10
        )
        skip_existing_bool = validate_bool_parameter(skip_existing, default=True)
        resolve_versions_bool = validate_bool_parameter(resolve_versions, default=False)

        result = await get_ingestion_service().ingest_cargo_file(
            file_path,
            concurrency=concurrency_int,
            skip_existing=skip_existing_bool,
            resolve_versions=resolve_versions_bool,
        )
        return IngestCargoFileResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in ingest_cargo_file: {e}")
        raise


async def get_trait_implementors(
    crate_name: str, trait_path: str, version: str = "latest"
) -> dict:
    """Find all types that implement a specific trait.

    Discovers types implementing a trait, including generic implementations,
    blanket implementations, and type-specific implementations.

    Args:
        crate_name: Name of the Rust crate
        trait_path: Full path to the trait (e.g., 'std::fmt::Debug')
        version: Specific version or 'latest' (default: latest)

    Returns:
        List of implementing types with details
    """
    try:
        result = await get_type_navigation_service().get_trait_implementors(
            crate_name, trait_path, version if version != "latest" else None
        )
        return TraitImplementationResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_trait_implementors: {e}")
        raise


async def get_type_traits(
    crate_name: str, type_path: str, version: str = "latest"
) -> dict:
    """Get all traits implemented by a specific type.

    Lists all traits that a type implements, including standard library traits,
    custom traits, and derived traits.

    Args:
        crate_name: Name of the Rust crate
        type_path: Full path to the type (e.g., 'std::vec::Vec')
        version: Specific version or 'latest' (default: latest)

    Returns:
        List of implemented traits with details
    """
    try:
        result = await get_type_navigation_service().get_type_traits(
            crate_name, type_path, version if version != "latest" else None
        )
        return TypeTraitsResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_type_traits: {e}")
        raise


async def resolve_method(
    crate_name: str, type_path: str, method_name: str, version: str = "latest"
) -> dict:
    """Resolve method calls to find the correct implementation.

    Disambiguates between inherent methods and trait methods, providing
    full signature information and disambiguation hints when multiple
    candidates exist.

    Args:
        crate_name: Name of the Rust crate
        type_path: Full path to the type
        method_name: Name of the method to resolve
        version: Specific version or 'latest' (default: latest)

    Returns:
        Method resolution with candidates and disambiguation hints
    """
    try:
        result = await get_type_navigation_service().resolve_method(
            crate_name, type_path, method_name, version if version != "latest" else None
        )
        return MethodSignatureResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in resolve_method: {e}")
        raise


async def get_associated_items(
    crate_name: str,
    container_path: str,
    item_kind: str | None = None,
    version: str = "latest",
) -> dict:
    """Get associated items (types, constants, functions) for a trait or type.

    Retrieves associated types, constants, and functions defined in traits
    or implemented for types.

    Args:
        crate_name: Name of the Rust crate
        container_path: Path to the containing trait/type
        item_kind: Optional filter by item kind ('type', 'const', 'function')
        version: Specific version or 'latest' (default: latest)

    Returns:
        Associated items grouped by kind
    """
    try:
        result = await get_type_navigation_service().get_associated_items(
            crate_name,
            container_path,
            item_kind,
            version if version != "latest" else None,
        )
        return AssociatedItemResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_associated_items: {e}")
        raise


async def get_generic_constraints(
    crate_name: str, item_path: str, version: str = "latest"
) -> dict:
    """Get generic constraints (type bounds, lifetime parameters) for an item.

    Retrieves all generic parameters, type bounds, lifetime constraints,
    and const generics for a function, struct, trait, or impl block.

    Args:
        crate_name: Name of the Rust crate
        item_path: Path to the item
        version: Specific version or 'latest' (default: latest)

    Returns:
        Generic constraints grouped by kind
    """
    try:
        result = await get_type_navigation_service().get_generic_constraints(
            crate_name, item_path, version if version != "latest" else None
        )
        return GenericConstraintResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_generic_constraints: {e}")
        raise


async def compare_versions(
    crate_name: str,
    version_a: str,
    version_b: str,
    categories: str | None = None,
    include_unchanged: str = "false",
    max_results: str = "1000",
) -> dict:
    """Compare two versions of a crate for API changes.

    Performs semantic diff between two crate versions, identifying breaking changes,
    deprecations, and providing migration hints.

    Args:
        crate_name: Name of the Rust crate
        version_a: First version to compare
        version_b: Second version to compare
        categories: Comma-separated categories of changes to include (breaking,deprecated,added,removed,modified)
        include_unchanged: Include unchanged items in response ("true"/"false")
        max_results: Maximum number of changes to return (1-5000, default: 1000)

    Returns:
        Version diff with breaking changes and migration hints
    """
    try:
        include_unchanged_bool = validate_bool_parameter(
            include_unchanged, default=False
        )
        max_results_int = validate_int_parameter(
            max_results, default=1000, min_val=1, max_val=5000
        )

        result = await get_crate_service().compare_versions(
            crate_name,
            version_a,
            version_b,
            categories=categories,  # Pass raw string - field validator will handle conversion
            include_unchanged=include_unchanged_bool,
            max_results=max_results_int,
        )
        return VersionDiffResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in compare_versions: {e}")
        raise


# Phase 7: Workflow Enhancement Tools


async def get_documentation_detail(
    crate_name: str,
    item_path: str,
    detail_level: str = "summary",
    version: str = "latest",
) -> dict:
    """Get documentation at specified detail level.

    Provides progressive detail levels for documentation to reduce cognitive load
    while enabling deep exploration. Switch between summary, detailed, and expert
    views in <50ms.

    Args:
        crate_name: Name of the Rust crate
        item_path: Path to the item
        detail_level: Level of detail ('summary', 'detailed', 'expert')
        version: Specific version or 'latest' (default: latest)

    Returns:
        Documentation with appropriate detail level
    """
    try:
        result = await get_workflow_service().get_documentation_with_detail_level(
            crate_name,
            item_path,
            detail_level,
            version if version != "latest" else None,
        )
        return ProgressiveDetailResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_documentation_detail: {e}")
        raise


async def extract_usage_patterns(
    crate_name: str,
    version: str = "latest",
    limit: str = "10",
    min_frequency: str = "2",
) -> dict:
    """Extract common usage patterns from documentation and examples.

    Analyzes documentation and code examples to identify common usage patterns
    and idioms. Helps understand how APIs are typically used in practice.

    Args:
        crate_name: Name of the Rust crate
        version: Specific version or 'latest' (default: latest)
        limit: Maximum number of patterns to return (1-50, default: 10)
        min_frequency: Minimum pattern frequency (1-100, default: 2)

    Returns:
        List of usage patterns with frequency and examples
    """
    try:
        limit_int = validate_int_parameter(limit, default=10, min_val=1, max_val=50)
        min_freq_int = validate_int_parameter(
            min_frequency, default=2, min_val=1, max_val=100
        )

        patterns = await get_workflow_service().extract_usage_patterns(
            crate_name,
            version if version != "latest" else None,
            limit=limit_int,
            min_frequency=min_freq_int,
        )

        return UsagePatternResponse(
            crate_name=crate_name,
            version=version,
            patterns=patterns,
            total_patterns_found=len(patterns),
        ).model_dump()
    except Exception as e:
        logger.error(f"Error in extract_usage_patterns: {e}")
        raise


async def generate_learning_path(
    crate_name: str,
    from_version: str = "",
    to_version: str = "latest",
    focus_areas: str = "",
) -> dict:
    """Generate learning path for API migration or onboarding.

    Creates structured learning paths for migrating between versions or
    onboarding to a new crate. Provides step-by-step guidance with time
    estimates.

    Args:
        crate_name: Name of the Rust crate
        from_version: Starting version (empty for new users)
        to_version: Target version (default: latest)
        focus_areas: Comma-separated focus areas (optional)

    Returns:
        Structured learning path with steps and resources
    """
    try:
        # Parse focus areas if provided
        focus_list = []
        if focus_areas:
            focus_list = [
                area.strip() for area in focus_areas.split(",") if area.strip()
            ]

        # Convert empty string to None for from_version
        from_ver = from_version if from_version else None
        to_ver = to_version if to_version != "latest" else None

        result = await get_workflow_service().generate_learning_path(
            crate_name,
            from_version=from_ver,
            to_version=to_ver,
            focus_areas=focus_list if focus_list else None,
        )

        return LearningPathResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in generate_learning_path: {e}")
        raise


async def resolve_import_handler(
    crate_name: str,
    import_path: str,
    include_alternatives: str = "false",
) -> dict:
    """Resolve import paths and suggest alternatives.

    Helps resolve Rust import paths to their actual locations and provides
    alternative suggestions when exact matches aren't found. Useful for
    understanding re-exports and finding the correct import statements.

    Args:
        crate_name: Name of the Rust crate
        import_path: Import path to resolve (e.g., 'Result', 'io::Error')
        include_alternatives: Include alternative import paths ("true"/"false")

    Returns:
        Resolved import path with confidence score and alternatives
    """
    try:
        from .ingest import ingest_crate

        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name)

        # Initialize service if needed
        cross_ref_service = get_cross_reference_service(db_path)

        include_alts = validate_bool_parameter(include_alternatives, default=False)

        result = await cross_ref_service.resolve_import(
            crate_name, import_path, include_alternatives=include_alts
        )
        return ResolveImportResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in resolve_import: {e}")
        raise


async def get_dependency_graph_handler(
    crate_name: str,
    max_depth: str = "3",
    include_versions: str = "true",
) -> dict:
    """Get dependency graph with version constraints.

    Builds a hierarchical dependency graph showing the relationships between
    crates and their dependencies. Includes cycle detection and version
    constraint information.

    Args:
        crate_name: Name of the Rust crate
        max_depth: Maximum depth to traverse (1-10, default: 3)
        include_versions: Include version information ("true"/"false")

    Returns:
        Dependency graph with hierarchy and cycle detection
    """
    try:
        from .ingest import ingest_crate

        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name)

        # Initialize service if needed
        cross_ref_service = get_cross_reference_service(db_path)

        max_depth_int = validate_int_parameter(
            max_depth, default=3, min_val=1, max_val=10
        )
        include_vers = validate_bool_parameter(include_versions, default=True)

        result = await cross_ref_service.get_dependency_graph(
            crate_name, max_depth=max_depth_int, include_versions=include_vers
        )
        return DependencyGraphResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_dependency_graph: {e}")
        raise


async def suggest_migrations_handler(
    crate_name: str,
    from_version: str,
    to_version: str,
) -> dict:
    """Suggest migration paths for breaking changes between versions.

    Analyzes differences between two versions of a crate and suggests
    migration paths for renamed, moved, or removed items. Helps developers
    upgrade their code when crate APIs change.

    Args:
        crate_name: Name of the Rust crate
        from_version: Starting version (e.g., '1.0.0')
        to_version: Target version (e.g., '2.0.0')

    Returns:
        Migration suggestions with confidence scores
    """
    try:
        from .ingest import ingest_crate

        # Ensure both versions are ingested
        await ingest_crate(crate_name, from_version)
        db_path_to = await ingest_crate(crate_name, to_version)

        # Use the newer version's database
        cross_ref_service = get_cross_reference_service(db_path_to)

        suggestions = await cross_ref_service.suggest_migrations(
            crate_name, from_version, to_version
        )

        result = {
            "crate_name": crate_name,
            "from_version": from_version,
            "to_version": to_version,
            "suggestions": suggestions,
        }
        return MigrationSuggestionsResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in suggest_migrations: {e}")
        raise


async def trace_reexports_handler(
    crate_name: str,
    item_path: str,
) -> dict:
    """Trace re-exported items to their original source.

    Follows the re-export chain to find where an item was originally defined.
    Useful for understanding the true source of re-exported types and functions.

    Args:
        crate_name: Name of the Rust crate
        item_path: Path of the item to trace (e.g., 'Result', 'io::Error')

    Returns:
        Re-export chain showing the path from re-export to original source
    """
    try:
        from .ingest import ingest_crate

        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name)

        # Initialize service if needed
        cross_ref_service = get_cross_reference_service(db_path)

        result = await cross_ref_service.trace_reexports(crate_name, item_path)
        return ReexportTrace(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in trace_reexports: {e}")
        raise


# Phase 5: Code Intelligence Tools
async def get_code_intelligence(
    crate_name: str, item_path: str, version: str = "latest"
) -> dict:
    """Get comprehensive code intelligence for a specific item.

    Args:
        crate_name: Name of the Rust crate
        item_path: Full path to the item (e.g., 'tokio::spawn')
        version: Specific version or 'latest'

    Returns:
        Dictionary with code intelligence data including safety info,
        error types, and feature requirements
    """
    try:
        # Use type_navigation_service for intelligence data
        result = await get_type_navigation_service().get_item_intelligence(
            crate_name, item_path, version
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_code_intelligence: {e}")
        raise


async def get_error_types(
    crate_name: str, pattern: str | None = None, version: str = "latest"
) -> dict:
    """List all error types in a crate or matching a pattern.

    Args:
        crate_name: Name of the Rust crate
        pattern: Optional pattern to filter error types
        version: Specific version or 'latest'

    Returns:
        Dictionary with error types found in the crate
    """
    try:
        # Use type_navigation_service for error catalog
        result = await get_type_navigation_service().get_error_catalog(
            crate_name, pattern, version
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_error_types: {e}")
        raise


async def get_unsafe_items(
    crate_name: str, include_reasons: str = "false", version: str = "latest"
) -> dict:
    """List all unsafe items in a crate with optional safety documentation.

    Args:
        crate_name: Name of the Rust crate
        include_reasons: Include detailed unsafe reasons and safety docs ('true'/'false')
        version: Specific version or 'latest'

    Returns:
        Dictionary with unsafe items and their safety information
    """
    try:
        # Convert string boolean to actual boolean
        include_reasons_bool = include_reasons.lower() == "true"

        # Use type_navigation_service for safety search
        result = await get_type_navigation_service().search_by_safety(
            crate_name,
            is_safe=False,
            include_reasons=include_reasons_bool,
            version=version,
        )
        return result
    except Exception as e:
        logger.error(f"Error in get_unsafe_items: {e}")
        raise


# Health Monitoring Tools
async def server_health(include_subsystems: str = "true", include_metrics: str = "false") -> dict:
    """Get comprehensive server health status including subsystems.
    
    Args:
        include_subsystems: Include detailed subsystem status ('true'/'false')
        include_metrics: Include performance metrics ('true'/'false')
    
    Returns:
        Dictionary with health status and optional subsystem details
    """
    from datetime import datetime
    import time
    from pathlib import Path
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "docsrs-mcp",
        "version": "0.1.0",
        "mode": "mcp_sdk",
    }
    
    if include_subsystems.lower() == "true":
        subsystems = {}
        
        # Database health - check cache directory
        try:
            from .config import CACHE_DIR
            cache_count = len(list(CACHE_DIR.glob("*/*.db")))
            subsystems["database"] = {
                "status": "healthy",
                "cache_count": cache_count,
                "cache_dir": str(CACHE_DIR),
            }
        except Exception as e:
            subsystems["database"] = {"status": "unhealthy", "error": str(e)}
        
        # Memory usage
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            subsystems["memory"] = {
                "status": "healthy",
                "rss_mb": memory_info.rss / 1024 / 1024,
                "vms_mb": memory_info.vms / 1024 / 1024,
            }
        except ImportError:
            subsystems["memory"] = {"status": "unavailable", "note": "psutil not installed"}
        except Exception as e:
            subsystems["memory"] = {"status": "unhealthy", "error": str(e)}
        
        # Pre-ingestion worker status
        try:
            from .popular_crates import get_popular_crates_status, _pre_ingestion_worker, WorkerState
            
            popular_status = get_popular_crates_status()
            if popular_status:
                subsystems["pre_ingestion"] = popular_status
                
                if _pre_ingestion_worker:
                    worker_state = _pre_ingestion_worker._state
                    if worker_state == WorkerState.RUNNING:
                        subsystems["pre_ingestion"]["status"] = "active"
                    elif worker_state == WorkerState.PAUSED:
                        subsystems["pre_ingestion"]["status"] = "paused"
                    else:
                        subsystems["pre_ingestion"]["status"] = "stopped"
        except Exception as e:
            subsystems["pre_ingestion"] = {"status": "unavailable", "error": str(e)}
        
        health_status["subsystems"] = subsystems
    
    if include_metrics.lower() == "true":
        metrics = {}
        
        # Basic performance metrics
        try:
            import time
            start = time.time()
            # Simple database query to measure responsiveness
            from .config import CACHE_DIR
            test_db = CACHE_DIR / "test" / "test.db"
            if test_db.exists():
                metrics["db_response_ms"] = (time.time() - start) * 1000
            else:
                metrics["db_response_ms"] = None
        except Exception:
            metrics["db_response_ms"] = None
        
        health_status["metrics"] = metrics
    
    return health_status


async def get_ingestion_status(include_progress: str = "true") -> dict:
    """Get detailed pre-ingestion progress and statistics.
    
    Args:
        include_progress: Include detailed progress breakdown ('true'/'false')
    
    Returns:
        Dictionary with ingestion status and progress details
    """
    from .popular_crates import _pre_ingestion_worker, WorkerState, get_popular_crates_status
    
    status = {
        "worker_running": False,
        "status": "not_initialized",
    }
    
    # Get popular crates status
    popular_status = get_popular_crates_status()
    if popular_status:
        status.update(popular_status)
    
    # Check worker state
    if _pre_ingestion_worker:
        worker_state = _pre_ingestion_worker._state
        status["worker_running"] = worker_state == WorkerState.RUNNING
        
        if worker_state == WorkerState.RUNNING:
            status["status"] = "active"
        elif worker_state == WorkerState.PAUSED:
            status["status"] = "paused"
        elif worker_state == WorkerState.STOPPED:
            status["status"] = "stopped"
        else:
            status["status"] = "idle"
        
        if include_progress.lower() == "true" and hasattr(_pre_ingestion_worker, 'crate_progress'):
            # Include detailed progress information
            progress = _pre_ingestion_worker.crate_progress
            status["progress"] = {
                "total": progress.get("total", 0),
                "completed": progress.get("completed", 0),
                "failed": progress.get("failed", 0),
                "skipped": progress.get("skipped", 0),
                "current": progress.get("current"),
                "percent": progress.get("percent", 0),
            }
            
            # Calculate ETA if possible
            if progress.get("completed", 0) > 0 and progress.get("total", 0) > 0:
                elapsed_per_crate = _pre_ingestion_worker._total_elapsed_time / progress["completed"]
                remaining = progress["total"] - progress["completed"]
                eta_seconds = elapsed_per_crate * remaining
                status["progress"]["eta_seconds"] = eta_seconds
    
    return status


# Server initialization
@server.list_tools()
async def handle_list_tools() -> list[types.Tool]:
    """List all available tools."""
    return [
        types.Tool(
            name="get_crate_summary",
            description="Get summary information about a Rust crate including metadata and module structure",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate (e.g., 'tokio', 'serde')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="search_items",
            description="Search for items in a crate's documentation using semantic similarity",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the crate to search within",
                    },
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                    "k": {
                        "type": "string",
                        "default": "5",
                        "description": "Number of results to return (1-20)",
                    },
                    "item_type": {
                        "type": "string",
                        "description": "Filter by item type (function, struct, trait, enum, module)",
                    },
                    "crate_filter": {
                        "type": "string",
                        "description": "Filter results to specific crate",
                    },
                    "module_path": {
                        "type": "string",
                        "description": "Filter results to specific module path",
                    },
                    "has_examples": {
                        "type": "string",
                        "description": "Filter to only items with code examples ('true'/'false')",
                    },
                    "min_doc_length": {
                        "type": "string",
                        "description": "Minimum documentation length in characters",
                    },
                    "visibility": {
                        "type": "string",
                        "description": "Filter by item visibility (public, private, crate)",
                    },
                    "deprecated": {
                        "type": "string",
                        "description": "Filter by deprecation status ('true'/'false')",
                    },
                },
                "required": ["crate_name", "query"],
            },
        ),
        types.Tool(
            name="get_item_doc",
            description="Get complete documentation for a specific item in a crate",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the crate containing the item",
                    },
                    "item_path": {
                        "type": "string",
                        "description": "Full path to the item (e.g., 'tokio::spawn', 'std::vec::Vec')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "item_path"],
            },
        ),
        types.Tool(
            name="get_module_tree",
            description="Get the module hierarchy tree for a Rust crate",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate to get module tree for",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="search_examples",
            description="Search for code examples in a crate's documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the crate to search within",
                    },
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant code examples",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version to search",
                    },
                    "k": {
                        "type": "string",
                        "default": "5",
                        "description": "Number of examples to return (1-20)",
                    },
                    "language": {
                        "type": "string",
                        "description": "Filter examples by programming language (e.g., 'rust', 'bash', 'toml')",
                    },
                },
                "required": ["crate_name", "query"],
            },
        ),
        types.Tool(
            name="list_versions",
            description="List all locally cached versions of a crate",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate to query",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="start_pre_ingestion",
            description="Start background pre-ingestion of popular Rust crates",
            inputSchema={
                "type": "object",
                "properties": {
                    "count": {
                        "type": "string",
                        "default": "100",
                        "description": "Number of crates to pre-ingest (10-500)",
                    },
                    "concurrency": {
                        "type": "string",
                        "default": "3",
                        "description": "Number of parallel download workers (1-10)",
                    },
                    "force": {
                        "type": "string",
                        "default": "false",
                        "description": "Force restart if pre-ingestion is already running ('true'/'false')",
                    },
                },
                "required": [],
            },
        ),
        types.Tool(
            name="control_pre_ingestion",
            description="Control the pre-ingestion worker (pause/resume/stop)",
            inputSchema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Control action to perform (pause/resume/stop)",
                    },
                },
                "required": ["action"],
            },
        ),
        types.Tool(
            name="ingest_cargo_file",
            description="Ingest crates from a Cargo.toml or Cargo.lock file",
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to Cargo.toml or Cargo.lock file",
                    },
                    "concurrency": {
                        "type": "string",
                        "default": "3",
                        "description": "Number of parallel download workers (1-10)",
                    },
                    "skip_existing": {
                        "type": "string",
                        "default": "true",
                        "description": "Skip already ingested crates ('true'/'false')",
                    },
                    "resolve_versions": {
                        "type": "string",
                        "default": "false",
                        "description": "Resolve version specifications to concrete versions ('true'/'false')",
                    },
                },
                "required": ["file_path"],
            },
        ),
        types.Tool(
            name="get_trait_implementors",
            description="Find all types that implement a specific trait",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "trait_path": {
                        "type": "string",
                        "description": "Full path to the trait (e.g., 'std::fmt::Debug')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "trait_path"],
            },
        ),
        types.Tool(
            name="get_type_traits",
            description="Get all traits implemented by a specific type",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "type_path": {
                        "type": "string",
                        "description": "Full path to the type (e.g., 'std::vec::Vec')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "type_path"],
            },
        ),
        types.Tool(
            name="resolve_method",
            description="Resolve method calls to find the correct implementation",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "type_path": {
                        "type": "string",
                        "description": "Full path to the type",
                    },
                    "method_name": {
                        "type": "string",
                        "description": "Name of the method to resolve",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "type_path", "method_name"],
            },
        ),
        types.Tool(
            name="get_associated_items",
            description="Get associated items (types, constants, functions) for a trait or type",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "container_path": {
                        "type": "string",
                        "description": "Path to the containing trait/type",
                    },
                    "item_kind": {
                        "type": "string",
                        "description": "Optional filter by item kind ('type', 'const', 'function')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "container_path"],
            },
        ),
        types.Tool(
            name="get_generic_constraints",
            description="Get generic constraints (type bounds, lifetime parameters) for an item",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "item_path": {
                        "type": "string",
                        "description": "Path to the item",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "item_path"],
            },
        ),
        types.Tool(
            name="compare_versions",
            description="Compare two versions of a crate for API changes",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "version_a": {
                        "type": "string",
                        "description": "First version to compare",
                    },
                    "version_b": {
                        "type": "string",
                        "description": "Second version to compare",
                    },
                    "categories": {
                        "type": "string",
                        "description": "Comma-separated categories of changes (breaking,deprecated,added,removed,modified)",
                    },
                    "include_unchanged": {
                        "type": "string",
                        "default": "false",
                        "description": "Include unchanged items in response ('true'/'false')",
                    },
                    "max_results": {
                        "type": "string",
                        "default": "1000",
                        "description": "Maximum number of changes to return (1-5000)",
                    },
                },
                "required": ["crate_name", "version_a", "version_b"],
            },
        ),
        # Phase 7: Workflow Enhancement Tools
        types.Tool(
            name="get_documentation_detail",
            description="Get documentation at specified detail level (summary/detailed/expert)",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "item_path": {
                        "type": "string",
                        "description": "Path to the item",
                    },
                    "detail_level": {
                        "type": "string",
                        "default": "summary",
                        "description": "Level of detail ('summary', 'detailed', 'expert')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "item_path"],
            },
        ),
        types.Tool(
            name="extract_usage_patterns",
            description="Extract common usage patterns from documentation and examples",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                    "limit": {
                        "type": "string",
                        "default": "10",
                        "description": "Maximum patterns to return (1-50)",
                    },
                    "min_frequency": {
                        "type": "string",
                        "default": "2",
                        "description": "Minimum pattern frequency (1-100)",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="generate_learning_path",
            description="Generate learning path for API migration or onboarding",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "from_version": {
                        "type": "string",
                        "default": "",
                        "description": "Starting version (empty for new users)",
                    },
                    "to_version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Target version",
                    },
                    "focus_areas": {
                        "type": "string",
                        "default": "",
                        "description": "Comma-separated focus areas",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        # Phase 6: Cross-References Tools
        types.Tool(
            name="resolve_import",
            description="Resolve import paths and suggest alternatives",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "import_path": {
                        "type": "string",
                        "description": "Import path to resolve (e.g., 'Result', 'io::Error')",
                    },
                    "include_alternatives": {
                        "type": "string",
                        "default": "false",
                        "description": "Include alternative import paths ('true'/'false')",
                    },
                },
                "required": ["crate_name", "import_path"],
            },
        ),
        types.Tool(
            name="get_dependency_graph",
            description="Get dependency graph with version constraints",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "max_depth": {
                        "type": "string",
                        "default": "3",
                        "description": "Maximum depth to traverse (1-10)",
                    },
                    "include_versions": {
                        "type": "string",
                        "default": "true",
                        "description": "Include version information ('true'/'false')",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="suggest_migrations",
            description="Suggest migration paths for breaking changes between versions",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "from_version": {
                        "type": "string",
                        "description": "Starting version (e.g., '1.0.0')",
                    },
                    "to_version": {
                        "type": "string",
                        "description": "Target version (e.g., '2.0.0')",
                    },
                },
                "required": ["crate_name", "from_version", "to_version"],
            },
        ),
        types.Tool(
            name="trace_reexports",
            description="Trace re-exported items to their original source",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "item_path": {
                        "type": "string",
                        "description": "Path of the item to trace (e.g., 'Result', 'io::Error')",
                    },
                },
                "required": ["crate_name", "item_path"],
            },
        ),
        # Phase 5: Code Intelligence Tools
        types.Tool(
            name="get_code_intelligence",
            description="Get comprehensive code intelligence for a specific item including safety info, error types, and feature requirements",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "item_path": {
                        "type": "string",
                        "description": "Full path to the item (e.g., 'tokio::spawn')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name", "item_path"],
            },
        ),
        types.Tool(
            name="get_error_types",
            description="List all error types in a crate or matching a pattern",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "pattern": {
                        "type": "string",
                        "description": "Optional pattern to filter error types",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        types.Tool(
            name="get_unsafe_items",
            description="List all unsafe items in a crate with optional safety documentation",
            inputSchema={
                "type": "object",
                "properties": {
                    "crate_name": {
                        "type": "string",
                        "description": "Name of the Rust crate",
                    },
                    "include_reasons": {
                        "type": "string",
                        "default": "false",
                        "description": "Include detailed unsafe reasons and safety documentation ('true'/'false')",
                    },
                    "version": {
                        "type": "string",
                        "default": "latest",
                        "description": "Specific version or 'latest'",
                    },
                },
                "required": ["crate_name"],
            },
        ),
        # Health Monitoring Tools
        types.Tool(
            name="server_health",
            description="Get comprehensive server health status including subsystems",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_subsystems": {
                        "type": "string",
                        "default": "true",
                        "description": "Include detailed subsystem status",
                    },
                    "include_metrics": {
                        "type": "string",
                        "default": "false",
                        "description": "Include performance metrics",
                    },
                },
            },
        ),
        types.Tool(
            name="get_ingestion_status",
            description="Get detailed pre-ingestion progress and statistics",
            inputSchema={
                "type": "object",
                "properties": {
                    "include_progress": {
                        "type": "string",
                        "default": "true",
                        "description": "Include detailed progress breakdown",
                    },
                },
            },
        ),
    ]


@server.call_tool()
async def handle_call_tool(
    name: str, arguments: dict | None
) -> list[types.TextContent | types.ImageContent | types.EmbeddedResource]:
    """Handle tool calls from clients."""
    try:
        # Handle existing tools
        if name == "get_crate_summary":
            result = await get_crate_summary(**arguments)
        elif name == "search_items":
            result = await search_items(**arguments)
        elif name == "get_item_doc":
            result = await get_item_doc(**arguments)
        elif name == "get_module_tree":
            result = await get_module_tree(**arguments)
        elif name == "search_examples":
            result = await search_examples(**arguments)
        elif name == "list_versions":
            result = await list_versions(**arguments)
        elif name == "start_pre_ingestion":
            result = await start_pre_ingestion(**arguments)
        elif name == "control_pre_ingestion":
            result = await control_pre_ingestion(**arguments)
        elif name == "ingest_cargo_file":
            result = await ingest_cargo_file(**arguments)
        elif name == "get_trait_implementors":
            result = await get_trait_implementors(**arguments)
        elif name == "get_type_traits":
            result = await get_type_traits(**arguments)
        elif name == "resolve_method":
            result = await resolve_method(**arguments)
        elif name == "get_associated_items":
            result = await get_associated_items(**arguments)
        elif name == "get_generic_constraints":
            result = await get_generic_constraints(**arguments)
        elif name == "compare_versions":
            result = await compare_versions(**arguments)
        # Phase 7: Workflow Enhancement Tools
        elif name == "get_documentation_detail":
            result = await get_documentation_detail(**arguments)
        elif name == "extract_usage_patterns":
            result = await extract_usage_patterns(**arguments)
        elif name == "generate_learning_path":
            result = await generate_learning_path(**arguments)
        # Phase 6: Cross-References Tools
        elif name == "resolve_import":
            result = await resolve_import_handler(**arguments)
        elif name == "get_dependency_graph":
            result = await get_dependency_graph_handler(**arguments)
        elif name == "suggest_migrations":
            result = await suggest_migrations_handler(**arguments)
        elif name == "trace_reexports":
            result = await trace_reexports_handler(**arguments)
        # Phase 5: Code Intelligence Tools
        elif name == "get_code_intelligence":
            result = await get_code_intelligence(**arguments)
        elif name == "get_error_types":
            result = await get_error_types(**arguments)
        elif name == "get_unsafe_items":
            result = await get_unsafe_items(**arguments)
        # Health Monitoring Tools
        elif name == "server_health":
            result = await server_health(**arguments)
        elif name == "get_ingestion_status":
            result = await get_ingestion_status(**arguments)
        else:
            raise ValueError(f"Unknown tool: {name}")

        return [types.TextContent(type="text", text=json.dumps(result, indent=2))]
    except Exception as e:
        logger.error(f"Error calling tool {name}: {e}")
        return [types.TextContent(type="text", text=f"Error: {str(e)}")]


@server.list_resources()
async def handle_list_resources() -> list[types.Resource]:
    """List available resources."""
    try:
        # Import config here to avoid circular imports
        from .mcp_tools_config import MCP_RESOURCES_CONFIG
        
        resources = []
        for resource_config in MCP_RESOURCES_CONFIG:
            resources.append(
                types.Resource(
                    uri=resource_config["uri"],
                    name=resource_config["name"],
                    description=resource_config["description"],
                )
            )
        
        logger.info(f"Listed {len(resources)} resources")
        return resources
    except Exception as e:
        logger.error(f"Error listing resources: {e}")
        return []


@server.read_resource()
async def handle_read_resource(uri: AnyUrl) -> list[ReadResourceContents]:
    """Read a specific resource by URI."""
    try:
        uri_str = str(uri)
        logger.info(f"Reading resource: {uri_str}")
        
        if uri_str == "cache://status":
            # Get cache status (not async)
            from .popular_crates import get_popular_crates_status
            
            status = get_popular_crates_status()
            content = json.dumps(status, indent=2)
            
            return [
                ReadResourceContents(
                    content=content,
                    mime_type="application/json",
                )
            ]
            
        elif uri_str == "cache://popular":
            # Get popular crates list
            from .popular_crates import PopularCratesManager
            
            manager = PopularCratesManager()
            crates = await manager.get_popular_crates()
            
            # Convert to dict for JSON serialization
            # Handle both PopularCrate objects and strings
            crates_data = []
            for crate in crates:
                if isinstance(crate, str):
                    # Simple string crate name
                    crates_data.append({
                        "name": crate,
                        "version": "latest",
                        "downloads": 0,
                        "description": "",
                    })
                else:
                    # PopularCrate object
                    crates_data.append({
                        "name": crate.name,
                        "version": crate.version if hasattr(crate, 'version') else "latest",
                        "downloads": crate.downloads if hasattr(crate, 'downloads') else 0,
                        "description": crate.description if hasattr(crate, 'description') else "",
                    })
            
            content = json.dumps({"crates": crates_data, "count": len(crates_data)}, indent=2)
            
            return [
                ReadResourceContents(
                    content=content,
                    mime_type="application/json",
                )
            ]
        
        else:
            logger.warning(f"Unknown resource URI: {uri_str}")
            return [
                ReadResourceContents(
                    content=f"Resource not found: {uri_str}",
                    mime_type="text/plain",
                )
            ]
            
    except Exception as e:
        logger.error(f"Error reading resource {uri}: {e}")
        return [
            ReadResourceContents(
                content=f"Error reading resource: {str(e)}",
                mime_type="text/plain",
            )
        ]


def setup_uvx_environment():
    """Configure environment for uvx execution compatibility."""
    # Essential for real-time STDIO output
    os.environ["PYTHONUNBUFFERED"] = "1"

    # Ensure proper encoding for JSON-RPC messages
    os.environ["PYTHONIOENCODING"] = "utf-8"

    # Force asyncio to use proper event loop policy on Windows
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    # Log environment setup to stderr
    sys.stderr.write("MCP Server: Environment configured for uvx execution\n")
    sys.stderr.flush()


def is_retryable_error(error: Exception) -> bool:
    """Determine if an error is retryable for STDIO connection."""
    # Connection-related errors are retryable
    retryable_types = (
        ConnectionError,
        BrokenPipeError,
        asyncio.TimeoutError,
        OSError,  # Includes stream-related errors
    )

    # Check error type
    if isinstance(error, retryable_types):
        return True

    # Check error message for known retryable patterns
    error_str = str(error).lower()
    retryable_patterns = [
        "broken pipe",
        "connection reset",
        "stream closed",
        "eof",
        "timeout",
    ]

    return any(pattern in error_str for pattern in retryable_patterns)


async def initialize_stdio_server_with_retry(server_instance, options):
    """Initialize STDIO server with retry logic for uvx compatibility."""
    max_retries = 3
    base_delay = 0.1  # Start with 100ms

    for attempt in range(max_retries):
        try:
            # Add small jitter to prevent retry storms
            if attempt > 0:
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.1)
                await asyncio.sleep(delay)
                sys.stderr.write(
                    f"MCP Server: Retry attempt {attempt + 1}/{max_retries} after {delay:.2f}s\n"
                )
                sys.stderr.flush()

            # Attempt STDIO server initialization
            async with stdio_server() as (read_stream, write_stream):
                sys.stderr.write(
                    "MCP Server: STDIO transport established successfully\n"
                )
                sys.stderr.flush()

                # Run the server
                await server_instance.run(read_stream, write_stream, options)
                return  # Success

        except asyncio.CancelledError:
            # Non-retryable - propagate immediately
            raise

        except Exception as e:
            error_msg = f"MCP Server: STDIO initialization failed (attempt {attempt + 1}/{max_retries}): {e}"
            sys.stderr.write(error_msg + "\n")
            sys.stderr.flush()
            logger.error(error_msg)

            # Check if error is retryable
            if not is_retryable_error(e):
                raise

            if attempt == max_retries - 1:
                # Final attempt failed
                raise RuntimeError(
                    f"Failed to initialize STDIO server after {max_retries} attempts"
                ) from e


async def run_sdk_server():
    """Run the MCP SDK server with uvx compatibility."""
    # Setup environment first
    setup_uvx_environment()

    # Initialize with retry logic
    try:
        await initialize_stdio_server_with_retry(
            server,
            InitializationOptions(
                server_name="docsrs-mcp",
                server_version="0.1.0",
                capabilities={
                    "tools": {},
                    "resources": {},
                },
            ),
        )
    except Exception as e:
        sys.stderr.write(f"MCP Server: Fatal error during initialization: {e}\n")
        sys.stderr.flush()
        logger.error(f"Fatal STDIO server initialization error: {e}")
        # Exit with error code for proper client notification
        sys.exit(1)
