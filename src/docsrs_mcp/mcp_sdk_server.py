"""MCP SDK implementation for docsrs-mcp server."""

import logging

from mcp import types
from mcp.server import Server
from mcp.server.models import InitializationOptions

from .models import (
    CompareVersionsResponse,
    GetCrateSummaryResponse,
    GetItemDocResponse,
    GetModuleTreeResponse,
    IngestCargoFileResponse,
    ListVersionsResponse,
    PreIngestionControlResponse,
    SearchExamplesResponse,
    SearchItemsResponse,
    StartPreIngestionResponse,
)
from .services import CrateService, IngestionService

logger = logging.getLogger(__name__)

# Initialize the MCP server
server = Server("docsrs-mcp")

# Initialize service layer
crate_service = CrateService()
ingestion_service = IngestionService()


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


@server.tool()
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
        result = await crate_service.get_crate_summary(
            crate_name, version if version != "latest" else None
        )
        return GetCrateSummaryResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_crate_summary: {e}")
        raise


@server.tool()
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

        results = await crate_service.search_items(
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


@server.tool()
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
        result = await crate_service.get_item_doc(
            crate_name, item_path, version if version != "latest" else None
        )
        return GetItemDocResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_item_doc: {e}")
        raise


@server.tool()
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
        result = await crate_service.get_module_tree(
            crate_name, version if version != "latest" else None
        )
        return GetModuleTreeResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in get_module_tree: {e}")
        raise


@server.tool()
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

        result = await crate_service.search_examples(
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


@server.tool()
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
        result = await crate_service.list_versions(crate_name)
        return ListVersionsResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in list_versions: {e}")
        raise


@server.tool()
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

        result = await ingestion_service.start_pre_ingestion(
            count=count_int, concurrency=concurrency_int, force=force_bool
        )
        return StartPreIngestionResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in start_pre_ingestion: {e}")
        raise


@server.tool()
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

        result = await ingestion_service.control_pre_ingestion(action)
        return PreIngestionControlResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in control_pre_ingestion: {e}")
        raise


@server.tool()
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

        result = await ingestion_service.ingest_cargo_file(
            file_path,
            concurrency=concurrency_int,
            skip_existing=skip_existing_bool,
            resolve_versions=resolve_versions_bool,
        )
        return IngestCargoFileResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in ingest_cargo_file: {e}")
        raise


@server.tool()
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

        # Parse categories if provided
        categories_list = None
        if categories:
            categories_list = [cat.strip() for cat in categories.split(",")]

        result = await crate_service.compare_versions(
            crate_name,
            version_a,
            version_b,
            categories=categories_list,
            include_unchanged=include_unchanged_bool,
            max_results=max_results_int,
        )
        return CompareVersionsResponse(**result).model_dump()
    except Exception as e:
        logger.error(f"Error in compare_versions: {e}")
        raise


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
    ]


async def run_sdk_server():
    """Run the MCP SDK server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            InitializationOptions(
                server_name="docsrs-mcp",
                server_version="0.1.0",
                capabilities=server.get_capabilities(
                    notification_options=types.NotificationOptions(),
                    experimental_capabilities={},
                ),
            ),
        )
