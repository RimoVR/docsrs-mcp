"""API endpoint handlers for docsrs-mcp server."""

import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING

import aiosqlite
import psutil
from fastapi import APIRouter, HTTPException, Request

from .database import CACHE_DIR
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

if TYPE_CHECKING:
    from .models import (
        GetCrateSummaryRequest,
        GetItemDocRequest,
    )
else:
    from .models import (
        CrateModule,
        GetCrateSummaryRequest,
        GetCrateSummaryResponse,
        GetItemDocRequest,
        MCPManifest,
        MCPResource,
        MCPTool,
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
    from .mcp_tools_config import MCP_RESOURCES_CONFIG, MCP_TOOLS_CONFIG

    # Create MCPTool objects from configuration
    tools = [
        MCPTool(
            name=tool_config["name"],
            description=tool_config["description"],
            input_schema=tool_config["input_schema"],
        )
        for tool_config in MCP_TOOLS_CONFIG
    ]

    # Create MCPResource objects from configuration
    resources = [
        MCPResource(
            uri=resource_config["uri"],
            name=resource_config["name"],
            description=resource_config["description"],
        )
        for resource_config in MCP_RESOURCES_CONFIG
    ]

    manifest = MCPManifest(tools=tools, resources=resources)
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
