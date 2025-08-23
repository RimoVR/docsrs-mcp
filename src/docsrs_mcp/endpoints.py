"""API endpoint handlers for docsrs-mcp server."""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import aiosqlite
import psutil
from fastapi import APIRouter, HTTPException, Request

from . import config
from .database import (
    CACHE_DIR,
    get_see_also_suggestions,
    search_embeddings,
    search_example_embeddings,
)
from .fuzzy_resolver import get_fuzzy_suggestions_with_fallback, resolve_path_alias
from .ingest import ingest_crate
from .middleware import limiter
from .popular_crates import (
    WorkerState,
    _ingestion_scheduler,
    _popular_crates_manager,
    _pre_ingestion_worker,
    get_popular_crates_status,
)
from .utils import extract_smart_snippet

if TYPE_CHECKING:
    from .models import (
        GetCrateSummaryRequest,
        GetItemDocRequest,
        SearchExamplesRequest,
        SearchItemsRequest,
    )
else:
    from .models import (
        CodeExample,
        CrateModule,
        GetCrateSummaryRequest,
        GetCrateSummaryResponse,
        GetItemDocRequest,
        MCPManifest,
        MCPResource,
        MCPTool,
        SearchExamplesRequest,
        SearchExamplesResponse,
        SearchItemsRequest,
        SearchItemsResponse,
        SearchResult,
    )

logger = logging.getLogger(__name__)

# Health check cache for <10ms response time
_health_cache = None
_health_cache_time = 0.0
HEALTH_CACHE_TTL = 5.0  # 5 second cache TTL

# Create APIRouter instance
router = APIRouter()


# ===== Health Endpoints =====


@router.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    response_description="Service health status with pre-ingestion details",
)
async def health_check():
    """
    Check service health status with enhanced monitoring.

    Returns detailed status including:
    - Service operational status
    - Popular crates cache statistics
    - Pre-ingestion progress if enabled
    - Subsystem health checks
    """
    global _health_cache, _health_cache_time

    # Check cache validity
    current_time = time.time()
    if _health_cache and (current_time - _health_cache_time) < HEALTH_CACHE_TTL:
        return _health_cache

    base_health = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "docsrs-mcp",
        "version": "0.1.0",
    }

    # Add subsystem health checks
    subsystems = {}

    # Database health - check cache directory
    try:
        cache_count = len(list(CACHE_DIR.glob("*/*.db")))
        subsystems["database"] = {
            "status": "healthy",
            "cache_count": cache_count,
            "cache_dir": str(CACHE_DIR),
        }
    except Exception as e:
        subsystems["database"] = {"status": "unhealthy", "error": str(e)}

    # Pre-ingestion health
    popular_crates_status = get_popular_crates_status()
    if popular_crates_status and any(popular_crates_status.values()):
        subsystems["pre_ingestion"] = popular_crates_status

        # Check worker state using proper enum comparison
        if _pre_ingestion_worker:
            worker_state = _pre_ingestion_worker._state
            is_running = worker_state == WorkerState.RUNNING

            # Update status based on actual worker state
            if is_running:
                subsystems["pre_ingestion"]["status"] = "active"
            elif worker_state == WorkerState.PAUSED:
                subsystems["pre_ingestion"]["status"] = "paused"
            elif worker_state == WorkerState.STOPPED:
                subsystems["pre_ingestion"]["status"] = "stopped"
            else:
                subsystems["pre_ingestion"]["status"] = "idle"

            subsystems["pre_ingestion"]["worker_state"] = worker_state.value
        else:
            subsystems["pre_ingestion"]["status"] = "idle"

        # Add ingestion progress if available
        if _pre_ingestion_worker and hasattr(_pre_ingestion_worker, "crate_progress"):
            progress_data = _pre_ingestion_worker.crate_progress
            subsystems["pre_ingestion"]["ingestion_progress"] = {
                "active": len(
                    [
                        p
                        for p in progress_data.values()
                        if p["status"] in ["downloading", "processing"]
                    ]
                ),
                "completed": len(
                    [p for p in progress_data.values() if p["status"] == "completed"]
                ),
                "failed": len(
                    [
                        p
                        for p in progress_data.values()
                        if p["status"] in ["failed", "error"]
                    ]
                ),
                "skipped": len(
                    [p for p in progress_data.values() if p["status"] == "skipped"]
                ),
                "details": progress_data,
            }

    # Check scheduler status
    if _ingestion_scheduler:
        subsystems["scheduler"] = {
            "status": "healthy",
            "enabled": _ingestion_scheduler.enabled,
            "details": _ingestion_scheduler.get_scheduler_status(),
        }

    # Check cache manager
    if _popular_crates_manager:
        manager_stats = _popular_crates_manager.get_cache_stats()
        subsystems["popular_crates_cache"] = {
            "status": "healthy",
            "stats": manager_stats,
        }

    # Check memory usage
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        subsystems["memory"] = {
            "status": "healthy" if memory_mb < 900 else "warning",
            "usage_mb": round(memory_mb, 2),
            "threshold_mb": 900,
            "percent": round(process.memory_percent(), 2),
        }
    except Exception as e:
        subsystems["memory"] = {"status": "unknown", "error": str(e)}

    base_health["subsystems"] = subsystems

    # Overall health based on subsystems
    if any(s.get("status") == "unhealthy" for s in subsystems.values()):
        base_health["status"] = "unhealthy"
    elif any(s.get("status") == "warning" for s in subsystems.values()):
        base_health["status"] = "degraded"

    # Update cache
    _health_cache = base_health
    _health_cache_time = current_time

    return base_health


@router.get(
    "/health/pre-ingestion",
    tags=["health"],
    summary="Pre-ingestion Health",
    response_description="Detailed pre-ingestion progress and health",
)
async def pre_ingestion_health():
    """
    Get detailed pre-ingestion progress and health.

    Returns comprehensive status including:
    - Worker running status
    - Ingestion statistics with ETA
    - Scheduler status if enabled
    - Cache statistics from PopularCratesManager
    """

    if not _pre_ingestion_worker:
        return {
            "status": "available",
            "message": "Pre-ingestion available. Use 'start_pre_ingestion' MCP tool to begin.",
        }

    # Check worker state using proper enum comparison
    worker_state = _pre_ingestion_worker._state
    is_running = worker_state == WorkerState.RUNNING

    response = {
        "status": "healthy",
        "worker": {
            "running": is_running,
            "state": worker_state.value,
            "stats": _pre_ingestion_worker.get_ingestion_stats()
            if _pre_ingestion_worker
            else {},
        },
    }

    # Add detailed ingestion progress if available
    if _pre_ingestion_worker and hasattr(_pre_ingestion_worker, "crate_progress"):
        progress_data = _pre_ingestion_worker.crate_progress
        response["ingestion_progress"] = {
            "active": len(
                [
                    p
                    for p in progress_data.values()
                    if p["status"] in ["downloading", "processing"]
                ]
            ),
            "completed": len(
                [p for p in progress_data.values() if p["status"] == "completed"]
            ),
            "failed": len(
                [
                    p
                    for p in progress_data.values()
                    if p["status"] in ["failed", "error"]
                ]
            ),
            "skipped": len(
                [p for p in progress_data.values() if p["status"] == "skipped"]
            ),
            "total": len(progress_data),
            "details": progress_data,
        }

    # Add scheduler status if available
    if _ingestion_scheduler:
        response["scheduler"] = {
            "enabled": _ingestion_scheduler.enabled,
            "status": _ingestion_scheduler.get_scheduler_status(),
        }
    else:
        response["scheduler"] = {"enabled": False, "status": "not_initialized"}

    # Add cache statistics if available
    if _popular_crates_manager:
        response["cache"] = {
            "stats": _popular_crates_manager.get_cache_stats(),
        }
    else:
        response["cache"] = {"status": "not_initialized"}

    return response


@router.get(
    "/",
    tags=["health"],
    summary="Service Information",
    response_description="Basic service metadata",
)
@limiter.limit("30/second")
async def root(request: Request):
    """
    Get basic service information.

    Returns service name, version, and links to documentation.
    This is the root endpoint that provides an overview of the API.
    """
    return {
        "service": "docsrs-mcp",
        "version": "0.1.0",
        "description": "MCP server for Rust crate documentation",
        "mcp_manifest": "/mcp/manifest",
    }


# ===== MCP Manifest Endpoint =====


@router.get(
    "/mcp/manifest",
    response_model=MCPManifest,
    tags=["mcp"],
    summary="Get MCP Manifest",
    response_description="Complete MCP server manifest",
)
@limiter.limit("30/second")
async def get_mcp_manifest(request: Request):
    """
    Get the Model Context Protocol server manifest.

    Returns a complete manifest describing all available MCP tools and resources,
    including their input schemas and descriptions. This endpoint is used by
    MCP clients to discover server capabilities.
    """
    manifest = MCPManifest(
        tools=[
            MCPTool(
                name="get_crate_summary",
                description="Get summary information about a Rust crate",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the Rust crate",
                        },
                        "version": {
                            "type": "string",
                            "description": "Specific version or 'latest'",
                        },
                    },
                    "required": ["crate_name"],
                },
            ),
            MCPTool(
                name="search_items",
                description="Search for items in crate documentation",
                input_schema={
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
                            "description": "Specific version or 'latest'",
                        },
                        "k": {
                            "type": "string",
                            "description": "Number of results to return (1-20)",
                        },
                        "item_type": {
                            "type": "string",
                            "description": "Filter by item type",
                        },
                        "module_path": {
                            "type": "string",
                            "description": "Filter to specific module path",
                        },
                        "deprecated": {
                            "type": "string",
                            "description": "Filter by deprecation status (true/false)",
                        },
                        "has_examples": {
                            "type": "string",
                            "description": "Filter to items with examples (true/false)",
                        },
                        "visibility": {
                            "type": "string",
                            "description": "Filter by visibility (public/private/crate)",
                        },
                        "min_doc_length": {
                            "type": "string",
                            "description": "Minimum documentation length",
                        },
                        "crate_filter": {
                            "type": "string",
                            "description": "Filter results to specific crate",
                        },
                    },
                    "required": ["crate_name", "query"],
                },
            ),
            MCPTool(
                name="get_item_doc",
                description="Get complete documentation for a specific item",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate containing the item",
                        },
                        "item_path": {
                            "type": "string",
                            "description": "Full path to the item",
                        },
                        "version": {
                            "type": "string",
                            "description": "Specific version or 'latest'",
                        },
                    },
                    "required": ["crate_name", "item_path"],
                },
            ),
            MCPTool(
                name="search_examples",
                description="Search for code examples in crate documentation",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to search within",
                        },
                        "query": {
                            "type": "string",
                            "description": "Search query for finding examples",
                        },
                        "version": {
                            "type": "string",
                            "description": "Specific version to search",
                        },
                        "k": {
                            "type": "string",
                            "description": "Number of examples to return",
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter by programming language",
                        },
                    },
                    "required": ["crate_name", "query"],
                },
            ),
            MCPTool(
                name="get_module_tree",
                description="Get the module hierarchy tree for a Rust crate",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the Rust crate",
                        },
                        "version": {
                            "type": "string",
                            "description": "Specific version or 'latest'",
                        },
                    },
                    "required": ["crate_name"],
                },
            ),
            MCPTool(
                name="start_pre_ingestion",
                description="Start background pre-ingestion of popular Rust crates",
                input_schema={
                    "type": "object",
                    "properties": {
                        "count": {
                            "type": "string",
                            "description": "Number of crates to pre-ingest (10-500)",
                        },
                        "concurrency": {
                            "type": "string",
                            "description": "Number of parallel workers (1-10)",
                        },
                        "force": {
                            "type": "string",
                            "description": "Force restart if already running (true/false)",
                        },
                    },
                    "required": [],
                },
            ),
            MCPTool(
                name="ingest_cargo_file",
                description="Ingest crates from a Cargo.toml or Cargo.lock file",
                input_schema={
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to Cargo.toml or Cargo.lock file",
                        },
                        "concurrency": {
                            "type": "string",
                            "description": "Number of parallel workers (1-10)",
                        },
                        "skip_existing": {
                            "type": "string",
                            "description": "Skip already ingested crates (true/false)",
                        },
                        "resolve_versions": {
                            "type": "string",
                            "description": "Resolve version specifications (true/false)",
                        },
                    },
                    "required": ["file_path"],
                },
            ),
            MCPTool(
                name="control_pre_ingestion",
                description="Control the pre-ingestion worker (pause/resume/stop)",
                input_schema={
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
            MCPTool(
                name="compare_versions",
                description="Compare two versions of a crate for API changes",
                input_schema={
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
                        "include_unchanged": {
                            "type": "string",
                            "description": "Include unchanged items (true/false)",
                        },
                        "max_results": {
                            "type": "string",
                            "description": "Maximum number of changes (1-5000)",
                        },
                        "categories": {
                            "type": "string",
                            "description": "Comma-separated change categories",
                        },
                    },
                    "required": ["crate_name", "version_a", "version_b"],
                },
            ),
        ],
        resources=[
            MCPResource(
                uri="docsrs://versions",
                name="list_versions",
                description="List all locally cached versions of a crate",
            ),
        ],
    )

    return manifest


# ===== MCP Tool Endpoints =====


@router.post(
    "/mcp/tools/get_crate_summary",
    response_model=GetCrateSummaryResponse,
    tags=["tools"],
    summary="Get Crate Summary",
    response_description="Crate metadata and module structure",
    operation_id="getCrateSummary",
)
@limiter.limit("30/second")
async def get_crate_summary(request: Request, params: GetCrateSummaryRequest):
    """
    Get summary information about a Rust crate.

    Fetches crate metadata including name, version, description, repository URL,
    and module structure. If the crate hasn't been ingested yet, it will be
    downloaded and processed automatically.

    **Note**: First-time ingestion may take 1-10 seconds depending on crate size.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        async with aiosqlite.connect(db_path) as db:
            # Get crate metadata
            cursor = await db.execute(
                """
                SELECT name, version, description, repository, documentation
                FROM crate_metadata
                LIMIT 1
                """
            )
            row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Crate not found")

            name, version, description, repository, documentation = row

            # Get top-level modules
            cursor = await db.execute(
                """
                SELECT
                    m.id,
                    m.name,
                    m.path,
                    m.item_count,
                    m.parent_id,
                    m.depth
                FROM modules m
                WHERE m.parent_id IS NULL
                ORDER BY m.name
                """
            )

            modules = [
                CrateModule(
                    name=row[1],
                    path=row[2],
                    item_count=row[3],
                    parent_id=row[4],
                    depth=row[5],
                )
                for row in await cursor.fetchall()
            ]

            # TODO: Add module filtering configuration if needed
            # For now, return all modules

            return GetCrateSummaryResponse(
                name=name,
                version=version,
                description=description or "",
                modules=modules,
                repository=repository,
                documentation=documentation,
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_crate_summary: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/mcp/tools/search_items",
    response_model=SearchItemsResponse,
    tags=["tools"],
    summary="Search Documentation with Smart Snippets",
    response_description="Ranked search results",
    operation_id="searchItems",
)
@limiter.limit("30/second")
async def search_items(request: Request, params: SearchItemsRequest):
    """
    Search for items in a crate's documentation using semantic similarity.

    Performs vector similarity search across all documentation in the specified crate.
    Results are ranked by semantic similarity to the query using BAAI/bge-small-en-v1.5
    embeddings. Snippets use smart extraction (200-400 chars) with intelligent boundary
    detection for improved readability.

    **Performance**: Warm searches typically complete in < 50ms.
    **Rate limit**: 30 requests/second per IP address.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Generate embedding for query
        from .ingest import get_embedding_model

        model = get_embedding_model()
        query_embedding = list(model.embed([params.query]))[0]

        # Search embeddings with filters
        results = await search_embeddings(
            db_path,
            query_embedding,
            k=params.k or 5,
            type_filter=params.item_type,
            crate_filter=params.crate_filter,
            module_path=params.module_path,
            has_examples=params.has_examples,
            min_doc_length=params.min_doc_length,
            visibility=params.visibility,
            deprecated=params.deprecated,
        )

        # Get see-also suggestions if we have results
        suggestions = []
        if results:
            # Get the item paths from the main results to exclude from suggestions
            original_paths = {item_path for _, item_path, _, _ in results}

            # Get suggestions using the same query embedding
            # This finds items similar to what the user is searching for
            suggestions = await get_see_also_suggestions(
                db_path,
                query_embedding,
                original_paths,
                k=10,  # Over-fetch to allow filtering
                similarity_threshold=0.7,
                max_suggestions=5,
            )

        # Convert to response format, adding suggestions only to the first result
        search_results = []
        for i, (score, item_path, header, content) in enumerate(results):
            result = SearchResult(
                score=score,
                item_path=item_path,
                header=header,
                snippet=extract_smart_snippet(content),
            )
            # Add suggestions only to the first result to avoid redundancy
            if i == 0 and suggestions:
                result.suggestions = suggestions
            search_results.append(result)

        return SearchItemsResponse(results=search_results)

    except Exception as e:
        logger.error(f"Error in search_items: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/mcp/tools/get_item_doc",
    tags=["tools"],
    summary="Get Item Documentation",
    response_description="Complete item documentation",
    operation_id="getItemDoc",
)
@limiter.limit("30/second")
async def get_item_doc(request: Request, params: GetItemDocRequest):
    """
    Get complete documentation for a specific item in a crate.

    Retrieves the full documentation for a specific item identified by its path
    (e.g., 'tokio::spawn', 'serde::Deserialize'). Returns markdown-formatted
    documentation including examples and detailed descriptions.

    **Tip**: Use search_items first if you're unsure of the exact item path.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Resolve any path aliases first
        resolved_path = await resolve_path_alias(
            params.crate_name, params.item_path, str(db_path)
        )

        # Search for the specific item
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT content FROM embeddings WHERE item_path = ? LIMIT 1",
                (resolved_path,),
            )
            row = await cursor.fetchone()

            if row:
                # Get cross-references for this item
                from .database import get_cross_references

                cross_refs = await get_cross_references(db_path, resolved_path)

                # Build response with cross-references
                response = {"content": row[0], "format": "markdown"}

                # Add cross-references if any exist
                if cross_refs:
                    response["cross_references"] = cross_refs

                return response
            else:
                # Try fuzzy matching for suggestions
                suggestions = await get_fuzzy_suggestions_with_fallback(
                    query=params.item_path,
                    db_path=str(db_path),
                    crate_name=params.crate_name,
                    version=params.version or "latest",
                )

                # Return error with suggestions
                error_detail = f"No documentation found for '{params.item_path}'"
                if suggestions:
                    error_detail += (
                        f". Did you mean one of these? {', '.join(suggestions[:3])}"
                    )

                # FastAPI's HTTPException expects string detail, so just raise with string
                raise HTTPException(
                    status_code=404,
                    detail=error_detail,
                )

    except HTTPException:
        # Re-raise HTTPExceptions as-is (e.g., 404 with fuzzy suggestions)
        raise
    except Exception as e:
        logger.error(f"Error in get_item_doc: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post(
    "/mcp/tools/search_examples",
    response_model=SearchExamplesResponse,
    tags=["tools"],
    summary="Search Code Examples with Smart Context",
    response_description="Code examples from documentation",
    operation_id="searchExamples",
)
@limiter.limit("30/second")
async def search_examples(request: Request, params: SearchExamplesRequest):
    """
    Search for code examples in a crate's documentation.

    Searches through extracted code examples from documentation and returns
    matching examples with language detection and metadata. Context snippets
    use smart extraction (200-400 chars) with intelligent boundary detection.

    **Features**:
    - Language detection for code blocks
    - Filtering by programming language
    - Semantic search across example content
    - Smart context snippets with boundary detection
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Generate embedding for query
        from .ingest import get_embedding_model

        model = get_embedding_model()
        query_embedding = list(model.embed([params.query]))[0]

        # Get crate version info from the database path
        version = "latest"
        if db_path:
            db_name = Path(db_path).stem  # Gets "1.0.219" from "1.0.219.db"
            if db_name and db_name != "latest":
                version = db_name

        # Try to use dedicated example embeddings first
        try:
            example_results = await search_example_embeddings(
                db_path,
                query_embedding,
                k=params.k,
                crate_filter=params.crate_name,
                language_filter=params.language,
            )

            if example_results:
                # Convert to response format
                examples_list = [
                    CodeExample(
                        code=result["code"],
                        language=result["language"],
                        detected=False,  # We don't track this in new format
                        item_path=result["item_path"],
                        context=result["context"],
                        score=result["score"],
                    )
                    for result in example_results
                ]

                return SearchExamplesResponse(
                    crate_name=params.crate_name,
                    version=version,
                    query=params.query,
                    examples=examples_list,
                    total_count=len(examples_list),
                )
        except Exception as e:
            logger.debug(f"Dedicated example search failed, falling back: {e}")

        # Fallback to searching in the main embeddings table
        import sqlite_vec

        async with aiosqlite.connect(db_path) as db:
            # Load sqlite-vec extension
            await db.enable_load_extension(True)
            await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
            await db.enable_load_extension(False)

            # Search for items containing examples with semantic similarity
            query_sql = """
                SELECT 
                    vec_distance_L2(embedding, ?) as distance,
                    item_path,
                    examples,
                    content
                FROM embeddings
                WHERE examples IS NOT NULL
                ORDER BY distance
                LIMIT ?
            """

            cursor = await db.execute(
                query_sql,
                (
                    bytes(sqlite_vec.serialize_float32(query_embedding)),
                    params.k * 3,
                ),  # Get more results to filter
            )

            examples_list = []
            seen_codes = set()  # For deduplication

            async for row in cursor:
                distance, item_path, examples_json, content = row

                if not examples_json:
                    continue

                try:
                    # Parse the JSON examples - handle both old list format and new dict format
                    examples_data = json.loads(examples_json)

                    # Handle string input - wrap in list to prevent character iteration
                    if isinstance(examples_data, str):
                        # Ensure string is not fragmented into characters
                        examples_data = [
                            {
                                "code": examples_data,
                                "language": "rust",
                                "detected": False,
                            }
                        ]
                    elif not examples_data:
                        logger.warning(
                            f"Empty examples_data for {params.crate_name}/{params.version} at {item_path}"
                        )
                        continue

                    # Additional validation for unexpected data types
                    if not isinstance(examples_data, (list, dict)):
                        logger.warning(
                            f"Unexpected examples_data type {type(examples_data).__name__} for {item_path}"
                        )
                        continue

                    # Handle old format (list of strings)
                    if isinstance(examples_data, list) and all(
                        isinstance(e, str) for e in examples_data
                    ):
                        examples_data = [
                            {"code": e, "language": "rust", "detected": False}
                            for e in examples_data
                        ]

                    for example in examples_data:
                        # Wrap in try-catch for resilience
                        try:
                            # Handle both dict format and potential string format
                            if isinstance(example, str):
                                example = {
                                    "code": example,
                                    "language": "rust",
                                    "detected": False,
                                }

                            # Validate example structure
                            if not isinstance(example, dict):
                                logger.debug(
                                    f"Skipping non-dict example: {type(example).__name__}"
                                )
                                continue

                            # Filter by language if specified
                            if (
                                params.language
                                and example.get("language") != params.language
                            ):
                                continue

                            code = example.get("code", "")

                            # Validate code content
                            if not code or not isinstance(code, str):
                                logger.debug(
                                    f"Skipping example with invalid code for {item_path}"
                                )
                                continue

                            # Simple deduplication by code hash
                            code_hash = hash(code)
                            if code_hash in seen_codes:
                                continue
                            seen_codes.add(code_hash)

                            # Calculate relevance score (inverse of distance)
                            score = 1.0 / (1.0 + distance)

                            examples_list.append(
                                CodeExample(
                                    code=code,
                                    language=example.get("language", "rust"),
                                    detected=example.get("detected", False),
                                    item_path=item_path,
                                    context=extract_smart_snippet(content)
                                    if content
                                    else None,
                                    score=score,
                                )
                            )
                        except Exception as ex:
                            logger.debug(
                                f"Error processing example for {item_path}: {ex}"
                            )
                            continue

                        if len(examples_list) >= params.k:
                            break

                    if len(examples_list) >= params.k:
                        break

                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse examples JSON for {item_path}")
                    continue

            return SearchExamplesResponse(
                crate_name=params.crate_name,
                version=version,
                query=params.query,
                examples=examples_list[: params.k],
                total_count=len(examples_list),
            )

    except Exception as e:
        logger.error(f"Error in search_examples: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
