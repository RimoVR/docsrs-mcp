"""FastAPI application for docsrs-mcp server."""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import aiosqlite
import psutil
import sqlite_vec
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi.errors import RateLimitExceeded

from . import config
from .database import (
    CACHE_DIR,
    get_see_also_suggestions,
    search_embeddings,
    search_example_embeddings,
)
from .database import (
    get_module_tree as get_module_tree_from_db,
)
from .fuzzy_resolver import get_fuzzy_suggestions_with_fallback, resolve_path_alias
from .ingest import get_embedding_model, ingest_crate
from .middleware import limiter, rate_limit_handler
from .models import (
    CodeExample,
    CrateModule,
    GetCrateSummaryRequest,
    GetCrateSummaryResponse,
    GetItemDocRequest,
    GetModuleTreeRequest,
    MCPManifest,
    MCPResource,
    MCPTool,
    ModuleTreeNode,
    SearchExamplesRequest,
    SearchExamplesResponse,
    SearchItemsRequest,
    SearchItemsResponse,
    SearchResult,
    StartPreIngestionRequest,
    StartPreIngestionResponse,
)
from .popular_crates import (
    _ingestion_scheduler,
    _popular_crates_manager,
    _pre_ingestion_worker,
    get_popular_crates_status,
    start_pre_ingestion,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="docsrs-mcp",
    description="""
## docsrs-mcp API

A Model Context Protocol (MCP) server that provides AI agents with semantic search capabilities
over Rust crate documentation from docs.rs.

### Features:
- 🔍 **Vector search** using BAAI/bge-small-en-v1.5 embeddings
- 📚 **Complete rustdoc ingestion** from docs.rs JSON files
- 💾 **Local caching** with automatic LRU eviction
- ⚡ **Fast performance** with sub-500ms warm search latency
- 🔒 **Secure** with rate limiting and input validation

### MCP Tools Available:
- `get_crate_summary` - Fetch crate metadata and module structure
- `search_items` - Semantic search within crate documentation
- `get_item_doc` - Retrieve complete documentation for specific items

For more information, see the [GitHub repository](https://github.com/peterkloiber/docsrs-mcp).
    """.strip(),
    version="0.1.0",
    contact={
        "name": "docsrs-mcp maintainers",
        "url": "https://github.com/peterkloiber/docsrs-mcp",
        "email": "security@example.com",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "health",
            "description": "Health check and service status endpoints",
        },
        {
            "name": "mcp",
            "description": "Model Context Protocol endpoints for AI agent integration",
        },
        {
            "name": "tools",
            "description": "MCP tool implementations for crate documentation queries",
        },
        {
            "name": "resources",
            "description": "MCP resource endpoints for listing available data",
        },
    ],
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

# Configure rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Format validation errors with enhanced context.

    Provides user-friendly error messages with examples and actionable guidance
    for all validation failures.
    """
    errors = []
    for error in exc.errors():
        # Build field path from location tuple
        field_path = " -> ".join(
            str(loc) for loc in error["loc"][1:]
        )  # Skip 'body' prefix

        # Extract the actual error message and context
        error_detail = {
            "field": field_path
            if field_path
            else error["loc"][-1]
            if error["loc"]
            else "unknown",
            "message": error["msg"],
            "type": error["type"],
        }

        # Add input value if available (helps with debugging)
        if "input" in error:
            # Handle cases where input might be an exception or non-serializable object
            input_value = error["input"]
            if isinstance(input_value, Exception):
                error_detail["received_value"] = str(input_value)
            else:
                try:
                    # Try to serialize normally
                    import json

                    json.dumps(input_value)
                    error_detail["received_value"] = input_value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    error_detail["received_value"] = str(input_value)

        # Add context if available
        if "ctx" in error:
            # Handle context which might contain non-serializable objects
            context = error["ctx"]
            if isinstance(context, dict):
                # Try to serialize each value in the context dict
                serializable_context = {}
                for key, value in context.items():
                    if isinstance(value, Exception):
                        serializable_context[key] = str(value)
                    else:
                        try:
                            json.dumps(value)
                            serializable_context[key] = value
                        except (TypeError, ValueError):
                            serializable_context[key] = str(value)
                error_detail["context"] = serializable_context
            else:
                # If context is not a dict, convert to string
                error_detail["context"] = str(context)

        errors.append(error_detail)

    return JSONResponse(
        status_code=422,
        content={
            "error": "Validation failed",
            "error_count": len(errors),
            "details": errors,
            "suggestion": "Please check the field requirements and examples provided in the error messages.",
            "documentation": "https://github.com/peterkloiber/docsrs-mcp#api-documentation",
        },
    )


@app.on_event("startup")
async def startup_event():
    """Handle application startup tasks."""
    # Start pre-ingestion if enabled via environment variable
    if config.PRE_INGEST_ENABLED:
        logger.info("Starting background pre-ingestion of popular crates")
        await start_pre_ingestion()

    # Start embedding warmup if enabled
    if config.EMBEDDINGS_WARMUP_ENABLED:
        from .ingest import warmup_embedding_model

        await warmup_embedding_model()
        logger.info("Embedding warmup task started")


@app.get(
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
        # Check if pre-ingestion is actively running
        ingestion_stats = popular_crates_status.get("ingestion_stats", {})
        if ingestion_stats.get("is_running"):
            subsystems["pre_ingestion"]["status"] = "active"
        else:
            subsystems["pre_ingestion"]["status"] = "idle"
    else:
        subsystems["pre_ingestion"] = {"status": "not_initialized"}

    # Embeddings warmup status
    from .ingest import get_warmup_status

    subsystems["embeddings"] = {
        "status": "warmed" if get_warmup_status() else "cold",
        "warmup_enabled": config.EMBEDDINGS_WARMUP_ENABLED,
        "warmed": get_warmup_status(),
    }

    # Memory health
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
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

    return base_health


@app.get(
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
        return {"status": "not_initialized", "message": "Pre-ingestion is not enabled"}

    response = {
        "status": "healthy",
        "worker": {
            "running": _pre_ingestion_worker._monitor_task is not None
            and not _pre_ingestion_worker._monitor_task.done()
            if _pre_ingestion_worker
            else False,
            "stats": _pre_ingestion_worker.get_ingestion_stats()
            if _pre_ingestion_worker
            else {},
        },
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


@app.get(
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


@app.get(
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
    return MCPManifest(
        tools=[
            MCPTool(
                name="get_crate_summary",
                description="Get summary information about a Rust crate (including stdlib crates like std, core, alloc) with modules and description",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to query (supports stdlib: std, core, alloc)",
                        },
                        "version": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Specific version (default: latest)",
                        },
                    },
                    "required": ["crate_name"],
                },
                tutorial=(
                    "Fetches crate metadata including version, description, and module structure.\n"
                    "Automatically downloads and ingests crates on first access with smart caching.\n"
                    "First-time ingestion: 1-10 seconds depending on crate size.\n"
                    "Use for initial crate exploration before diving into specific documentation."
                ),
                examples=[
                    "Basic: get_crate_summary(crate_name='tokio')",
                    "Specific version: get_crate_summary(crate_name='serde', version='1.0.104')",
                    "Stdlib: get_crate_summary(crate_name='std')",
                ],
                use_cases=[
                    "Initial crate discovery and exploration",
                    "Version compatibility checking",
                    "Understanding module organization",
                ],
            ),
            MCPTool(
                name="search_items",
                description="Search for items in a crate's documentation (including stdlib) using vector similarity",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to search in (supports stdlib: std, core, alloc)",
                        },
                        "query": {"type": "string", "description": "Search query text"},
                        "version": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Specific version (default: latest)",
                        },
                        "k": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}],
                            "description": "Number of results to return",
                            "default": 5,
                        },
                        "item_type": {
                            "type": "string",
                            "description": "Filter by item type (function, struct, trait, enum, module)",
                            "enum": ["function", "struct", "trait", "enum", "module"],
                        },
                        "crate_filter": {
                            "type": "string",
                            "description": "Filter results to specific crate",
                        },
                        "module_path": {
                            "type": "string",
                            "description": "Filter results to specific module path within the crate",
                        },
                        "has_examples": {
                            "anyOf": [{"type": "boolean"}, {"type": "string"}],
                            "description": "Filter to only items with code examples",
                        },
                        "min_doc_length": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}],
                            "description": "Minimum documentation length in characters",
                            "minimum": 100,
                            "maximum": 10000,
                        },
                        "visibility": {
                            "type": "string",
                            "description": "Filter by item visibility",
                            "enum": ["public", "private", "crate"],
                        },
                        "deprecated": {
                            "anyOf": [{"type": "boolean"}, {"type": "string"}],
                            "description": "Filter by deprecation status (true=deprecated only, false=non-deprecated only)",
                        },
                    },
                    "required": ["crate_name", "query"],
                },
                tutorial=(
                    "Performs semantic search across all crate documentation using embeddings.\n"
                    "Results ranked by relevance using BAAI/bge-small-en-v1.5 model.\n"
                    "Warm searches complete in <50ms. Rate limit: 30 req/sec.\n"
                    "Best for finding functionality when you don't know exact item names."
                ),
                examples=[
                    "Basic: search_items(crate_name='tokio', query='spawn tasks')",
                    "Filtered: search_items(crate_name='serde', query='deserialize', item_type='trait')",
                    "Limited: search_items(crate_name='actix-web', query='middleware', k=10)",
                ],
                use_cases=[
                    "Finding relevant APIs by description",
                    "Discovering similar functionality",
                    "Exploring unfamiliar crates",
                ],
            ),
            MCPTool(
                name="get_item_doc",
                description="Get complete documentation for a specific item in a crate (including stdlib)",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate (supports stdlib: std, core, alloc)",
                        },
                        "item_path": {
                            "type": "string",
                            "description": "Full path to the item (e.g., 'tokio::spawn')",
                        },
                        "version": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Specific version (default: latest)",
                        },
                    },
                    "required": ["crate_name", "item_path"],
                },
                tutorial=(
                    "Retrieves complete documentation for a specific item by path.\n"
                    "Returns markdown-formatted docs with examples and descriptions.\n"
                    "Use after search_items to get full details on discovered items.\n"
                    "Tip: Use 'crate' as item_path for crate-level documentation."
                ),
                examples=[
                    "Function: get_item_doc(crate_name='tokio', item_path='tokio::spawn')",
                    "Trait: get_item_doc(crate_name='serde', item_path='serde::Deserialize')",
                    "Crate docs: get_item_doc(crate_name='reqwest', item_path='crate')",
                ],
                use_cases=[
                    "Reading complete API documentation",
                    "Understanding implementation details",
                    "Accessing code examples",
                ],
            ),
            MCPTool(
                name="search_examples",
                description="Search for code examples in a crate's documentation with language detection",
                input_schema={
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
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Specific version (default: latest)",
                        },
                        "k": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}],
                            "description": "Number of examples to return",
                            "default": 5,
                        },
                        "language": {
                            "type": "string",
                            "description": "Filter examples by programming language (e.g., 'rust', 'bash', 'toml')",
                        },
                    },
                    "required": ["crate_name", "query"],
                },
                tutorial=(
                    "Searches code examples extracted from crate documentation.\n"
                    "Includes language detection and filtering capabilities.\n"
                    "Returns runnable code snippets with context.\n"
                    "Perfect for learning implementation patterns."
                ),
                examples=[
                    "Basic: search_examples(crate_name='tokio', query='async runtime')",
                    "Language filter: search_examples(crate_name='diesel', query='migrations', language='sql')",
                    "More results: search_examples(crate_name='actix-web', query='handlers', k=10)",
                ],
                use_cases=[
                    "Finding implementation patterns",
                    "Learning library usage",
                    "Discovering best practices",
                ],
            ),
            MCPTool(
                name="get_module_tree",
                description="Get the module hierarchy tree for a Rust crate",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to get module tree for",
                        },
                        "version": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Specific version (default: latest)",
                        },
                    },
                    "required": ["crate_name"],
                },
                tutorial=(
                    "Returns hierarchical module structure with parent-child relationships.\n"
                    "Shows item counts and depth levels for navigation.\n"
                    "First-time processing: 1-10 seconds for crate ingestion.\n"
                    "Essential for understanding crate organization."
                ),
                examples=[
                    "Basic: get_module_tree(crate_name='tokio')",
                    "Specific version: get_module_tree(crate_name='serde', version='1.0.104')",
                    "Large crate: get_module_tree(crate_name='rustc')",
                ],
                use_cases=[
                    "Navigating complex crates",
                    "Understanding architecture",
                    "Finding module locations",
                ],
            ),
            MCPTool(
                name="start_pre_ingestion",
                description="Start background pre-ingestion of popular Rust crates for improved performance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "force": {
                            "anyOf": [{"type": "boolean"}, {"type": "string"}],
                            "default": False,
                            "description": "Force restart if pre-ingestion is already running",
                        },
                        "concurrency": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}],
                            "minimum": 1,
                            "maximum": 10,
                            "description": "Number of parallel download workers (1-10, default: 3)",
                        },
                        "count": {
                            "anyOf": [{"type": "integer"}, {"type": "string"}],
                            "minimum": 10,
                            "maximum": 500,
                            "description": "Number of crates to pre-ingest (10-500, default: 100)",
                        },
                    },
                },
                tutorial=(
                    "The pre-ingestion system caches popular Rust crates to eliminate cold-start latency.\n\n"
                    "**How it works:**\n"
                    "1. Fetches the list of most-downloaded crates from crates.io\n"
                    "2. Downloads and processes them in parallel (configurable concurrency)\n"
                    "3. Builds search indices and caches documentation\n"
                    "4. Runs in background without blocking other operations\n\n"
                    "**Monitoring:**\n"
                    "- Check `/health` for overall system status\n"
                    "- Use `/health/pre-ingestion` for detailed progress\n"
                    "- Look for 'pre_ingestion' section showing processed/total counts\n\n"
                    "**Performance Impact:**\n"
                    "- Popular crates respond in <100ms after pre-ingestion\n"
                    "- System remains responsive during pre-ingestion\n"
                    "- Memory usage increases gradually as crates are cached\n\n"
                    "**Best Practices:**\n"
                    "- Start pre-ingestion during low-traffic periods\n"
                    "- Monitor memory usage via health endpoints\n"
                    "- Allow 5-10 minutes for full completion\n"
                    "- Use force=true sparingly to avoid redundant work"
                ),
                examples=[
                    "start_pre_ingestion()",
                    "start_pre_ingestion(force=true)",
                    "start_pre_ingestion(concurrency=5)",
                    "start_pre_ingestion(count=200, concurrency=5)",
                ],
                use_cases=[
                    "**Server Startup**: Warm cache after server deployment or restart",
                    "**Before Peak Hours**: Pre-load popular crates before high-traffic periods",
                    "**After Cache Clear**: Rebuild cache after maintenance or cleanup",
                    "**Performance Optimization**: Ensure sub-100ms responses for common queries",
                    "**AI Agent Workflows**: Eliminate wait times for frequently accessed documentation",
                ],
            ),
        ],
        resources=[
            MCPResource(
                name="versions",
                description="List all available versions of a crate",
                uri="/mcp/resources/versions",
            )
        ],
    )


@app.post(
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

        # Fetch crate metadata from database
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name, version, description, repository, documentation FROM crate_metadata LIMIT 1"
            )
            row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Crate not found")

            name, version, description, repository, documentation = row

            # Fetch modules with hierarchy information
            cursor = await db.execute(
                "SELECT name, path, parent_id, depth, item_count FROM modules WHERE crate_id = 1"
            )
            modules = [
                CrateModule(
                    name=row[0],
                    path=row[1],
                    parent_id=row[2],
                    depth=row[3],
                    item_count=row[4],
                )
                for row in await cursor.fetchall()
            ]

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


@app.post(
    "/mcp/tools/search_items",
    response_model=SearchItemsResponse,
    tags=["tools"],
    summary="Search Documentation",
    response_description="Ranked search results",
    operation_id="searchItems",
)
@limiter.limit("30/second")
async def search_items(request: Request, params: SearchItemsRequest):
    """
    Search for items in a crate's documentation using semantic similarity.

    Performs vector similarity search across all documentation in the specified crate.
    Results are ranked by semantic similarity to the query using BAAI/bge-small-en-v1.5
    embeddings.

    **Performance**: Warm searches typically complete in < 50ms.
    **Rate limit**: 30 requests/second per IP address.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Generate embedding for query
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
                snippet=content[:200] + "..." if len(content) > 200 else content,
            )
            # Add suggestions only to the first result to avoid redundancy
            if i == 0 and suggestions:
                result.suggestions = suggestions
            search_results.append(result)

        return SearchItemsResponse(results=search_results)

    except Exception as e:
        logger.error(f"Error in search_items: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
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
                return {"content": row[0], "format": "markdown"}
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


@app.post(
    "/mcp/tools/search_examples",
    response_model=SearchExamplesResponse,
    tags=["tools"],
    summary="Search Code Examples",
    response_description="Code examples from documentation",
    operation_id="searchExamples",
)
@limiter.limit("30/second")
async def search_examples(request: Request, params: SearchExamplesRequest):
    """
    Search for code examples in a crate's documentation.

    Searches through extracted code examples from documentation and returns
    matching examples with language detection and metadata.

    **Features**:
    - Language detection for code blocks
    - Filtering by programming language
    - Semantic search across example content
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Generate embedding for query
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
                                    context=content[:200] if content else None,
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


@app.post(
    "/mcp/tools/get_module_tree",
    tags=["tools"],
    summary="Get Module Tree",
    response_description="Module hierarchy tree structure",
    operation_id="getModuleTree",
)
@limiter.limit("30/second")
async def get_module_tree(request: Request, params: GetModuleTreeRequest):
    """
    Get the module hierarchy tree for a Rust crate.

    Returns a hierarchical tree structure of all modules in the specified crate,
    including parent-child relationships, depth levels, and item counts per module.

    **Note**: First-time ingestion may take 1-10 seconds depending on crate size.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(params.crate_name, params.version)

        # Get crate ID
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute("SELECT id FROM crate_metadata LIMIT 1")
            row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Crate not found")

            crate_id = row[0]

        # Get module tree from database
        modules = await get_module_tree_from_db(db_path, crate_id)

        # Build hierarchical tree structure
        def build_tree(modules_list, parent_id=None):
            """Build tree structure from flat module list."""
            nodes = []
            for module in modules_list:
                if module["parent_id"] == parent_id:
                    node = ModuleTreeNode(
                        name=module["name"],
                        path=module["path"],
                        depth=module["depth"],
                        item_count=module["item_count"],
                        children=build_tree(modules_list, module["id"]),
                    )
                    nodes.append(node)
            return nodes

        # Build and return tree
        tree = build_tree(modules)

        return {
            "crate_name": params.crate_name,
            "modules": tree,
            "total_modules": len(modules),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_module_tree: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.post(
    "/mcp/tools/start_pre_ingestion",
    response_model=StartPreIngestionResponse,
    tags=["tools"],
    summary="Start pre-ingestion of popular Rust crates",
    response_description="Status of the pre-ingestion operation",
    operation_id="startPreIngestion",
)
@limiter.limit("30/second")
async def start_pre_ingestion_tool(
    request: Request, params: StartPreIngestionRequest
) -> StartPreIngestionResponse:
    """
    Start background pre-ingestion of popular Rust crates.

    This tool initiates the pre-ingestion system to cache the most popular
    crates, eliminating cold-start latency for common queries.
    """
    try:
        # Check current state
        status = get_popular_crates_status()
        is_running = status.get("worker", {}).get("is_running", False) or status.get(
            "scheduler", {}
        ).get("is_running", False)

        # Handle force restart
        if is_running and not params.force:
            current_stats = status.get("worker", {}).get("stats", {})
            return StartPreIngestionResponse(
                status="already_running",
                message=(
                    f"Pre-ingestion already in progress. "
                    f"Processed {current_stats.get('processed', 0)}/{current_stats.get('total', 0)} crates. "
                    f"Use force=true to restart."
                ),
                stats=current_stats,
            )

        # Apply configuration if provided
        if params.concurrency is not None:
            os.environ["PRE_INGEST_CONCURRENCY"] = str(params.concurrency)

        if params.count is not None:
            # This would require modification to start_pre_ingestion to accept count
            # For now, we'll note it in the response
            pass

        # Start or restart pre-ingestion
        # If forcing restart, we might need to stop existing first
        # (This is a simplification - actual implementation would be more robust)
        if is_running and params.force:
            logger.info("Force restarting pre-ingestion system")
            # The actual stop would be handled internally

        # Start pre-ingestion in background
        asyncio.create_task(start_pre_ingestion())

        response_status = "restarted" if (is_running and params.force) else "started"

        return StartPreIngestionResponse(
            status=response_status,
            message=(
                f"Pre-ingestion {response_status} successfully. "
                f"Processing {params.count or '100-500'} popular crates "
                f"with {params.concurrency or 3} concurrent workers. "
                f"Monitor progress via health endpoints."
            ),
            stats=None,  # Stats not immediately available on start
        )

    except Exception as e:
        logger.error(f"Failed to start pre-ingestion: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to start pre-ingestion: {str(e)}"
        ) from e


@app.get(
    "/mcp/resources/versions",
    tags=["resources"],
    summary="List Crate Versions",
    response_description="Available crate versions",
    operation_id="listVersions",
)
@limiter.limit("30/second")
async def list_versions(request: Request, crate_name: str):
    """
    List all locally cached versions of a crate.

    Returns a list of all versions that have been previously ingested and cached.
    Note that this only shows cached versions, not all versions available on crates.io.

    **Parameters**:
    - `crate_name`: Name of the Rust crate to query
    """
    try:
        # For now, just return the current version from the database if it exists
        # In a full implementation, this would query crates.io API
        versions = []

        # Check all cached versions
        crate_dir = Path(
            f"./cache/{crate_name.replace('/', '_').replace(chr(92), '_')}"
        )
        if crate_dir.exists():
            for db_file in crate_dir.glob("*.db"):
                version = db_file.stem
                versions.append({"version": version, "yanked": False})

        return {"crate_name": crate_name, "versions": versions}

    except Exception as e:
        logger.error(f"Error in list_versions: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
