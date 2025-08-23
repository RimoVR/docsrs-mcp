"""Additional MCP tool endpoints for docsrs-mcp server."""

import asyncio
import json
import logging
import os
from pathlib import Path

import aiosqlite
from fastapi import APIRouter, HTTPException, Request

from .database import (
    CACHE_DIR,
    get_see_also_suggestions,
    search_embeddings,
    search_example_embeddings,
)
from .database import get_module_tree as get_module_tree_from_db
from .ingest import ingest_crate
from .middleware import limiter
from .models import (
    CodeExample,
    CompareVersionsRequest,
    GetModuleTreeRequest,
    IngestCargoFileRequest,
    IngestCargoFileResponse,
    ModuleTreeNode,
    PreIngestionControlRequest,
    PreIngestionControlResponse,
    SearchExamplesRequest,
    SearchExamplesResponse,
    SearchItemsRequest,
    SearchItemsResponse,
    SearchResult,
    StartPreIngestionRequest,
    StartPreIngestionResponse,
    VersionDiffResponse,
)
from .popular_crates import get_popular_crates_status, start_pre_ingestion
from .utils import extract_smart_snippet

logger = logging.getLogger(__name__)

# Create APIRouter instance for additional tool endpoints
router = APIRouter()


@router.post(
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


@router.post(
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

        # Start pre-ingestion in background with force_start for MCP control
        asyncio.create_task(start_pre_ingestion(force_start=True))

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


@router.post(
    "/mcp/tools/ingest_cargo_file",
    response_model=IngestCargoFileResponse,
    tags=["tools"],
    summary="Ingest crates from Cargo.toml or Cargo.lock file",
    response_description="Status of the cargo file ingestion",
    operation_id="ingestCargoFile",
)
@limiter.limit("30/second")
async def ingest_cargo_file(
    request: Request, params: IngestCargoFileRequest
) -> IngestCargoFileResponse:
    """Ingest crates from a Cargo.toml or Cargo.lock file.

    Parses the specified Cargo file and queues all dependencies for ingestion.
    Skips crates that are already cached to avoid redundant downloads.
    """
    import aiohttp

    from .cargo import extract_crates_from_cargo, resolve_cargo_versions
    from .popular_crates import check_crate_exists, queue_for_ingestion

    try:
        # Parse the Cargo file
        file_path = Path(params.file_path)
        crates = extract_crates_from_cargo(file_path)

        if not crates:
            return IngestCargoFileResponse(
                status="completed",
                message=f"No dependencies found in {file_path.name}",
                crates_found=0,
                crates_queued=0,
                crates_skipped=0,
            )

        # Resolve version specifications if requested
        if params.resolve_versions:
            async with aiohttp.ClientSession() as session:
                crates = await resolve_cargo_versions(crates, session, resolve=True)
                logger.info(f"Resolved {len(crates)} crate versions")

        # Check which crates already exist
        crates_to_ingest = []
        crates_skipped = 0

        if params.skip_existing:
            for crate_spec in crates:
                if not await check_crate_exists(crate_spec):
                    crates_to_ingest.append(crate_spec)
                else:
                    crates_skipped += 1
        else:
            crates_to_ingest = crates

        # Queue for ingestion
        if crates_to_ingest:
            await queue_for_ingestion(
                crates_to_ingest, concurrency=params.concurrency or 3
            )
            estimated_time = (
                len(crates_to_ingest) * 2.0 / (params.concurrency or 3)
            )  # Rough estimate
        else:
            estimated_time = 0

        return IngestCargoFileResponse(
            status="started" if crates_to_ingest else "completed",
            message=f"Queued {len(crates_to_ingest)} crates from {file_path.name}",
            crates_found=len(crates),
            crates_queued=len(crates_to_ingest),
            crates_skipped=crates_skipped,
            estimated_time_seconds=estimated_time,
        )

    except Exception as e:
        logger.error(f"Failed to ingest Cargo file: {e}")
        return IngestCargoFileResponse(
            status="failed",
            message=str(e),
            crates_found=0,
            crates_queued=0,
            crates_skipped=0,
        )


@router.post(
    "/mcp/tools/control_pre_ingestion",
    response_model=PreIngestionControlResponse,
    tags=["tools"],
    summary="Control pre-ingestion worker",
    response_description="Result of the control operation",
    operation_id="controlPreIngestion",
)
@limiter.limit("30/second")
async def control_pre_ingestion(
    request: Request, params: PreIngestionControlRequest
) -> PreIngestionControlResponse:
    """Control the pre-ingestion worker (pause/resume/stop).

    Allows runtime control of the background pre-ingestion process without
    requiring server restart.
    """
    from .popular_crates import _ingestion_scheduler, _pre_ingestion_worker

    if not _pre_ingestion_worker:
        return PreIngestionControlResponse(
            status="failed",
            message="Pre-ingestion worker not initialized. Start pre-ingestion first.",
            worker_state=None,
        )

    success = False
    if params.action == "pause":
        success = await _pre_ingestion_worker.pause()
    elif params.action == "resume":
        success = await _pre_ingestion_worker.resume()
    elif params.action == "stop":
        success = await _pre_ingestion_worker.stop()

    # Get current stats
    current_stats = None
    if _ingestion_scheduler:
        current_stats = await _ingestion_scheduler.get_ingestion_stats()
    elif _pre_ingestion_worker:
        current_stats = _pre_ingestion_worker.get_ingestion_stats()

    return PreIngestionControlResponse(
        status="success" if success else "no_change",
        message=f"Worker {params.action} {'successful' if success else 'had no effect'}",
        worker_state=str(_pre_ingestion_worker._state.value)
        if _pre_ingestion_worker
        else None,
        current_stats=current_stats,
    )


@router.post(
    "/admin/recover_ingestions",
    tags=["admin"],
    summary="Recover Incomplete Ingestions",
    response_description="Recovery status for incomplete ingestions",
)
@limiter.limit("1/minute")  # Rate limit recovery operations
async def recover_ingestions(request: Request) -> dict:
    """
    Trigger recovery for all stalled/incomplete ingestions.

    This endpoint finds all incomplete ingestions and attempts to recover them
    by deleting the incomplete database and re-ingesting the crate.
    """
    from .database import find_incomplete_ingestions, recover_incomplete_ingestion

    # Find all incomplete ingestions
    incomplete = await find_incomplete_ingestions(
        CACHE_DIR, stale_threshold_seconds=1800
    )

    recovery_results = {
        "total_incomplete": len(incomplete),
        "recovery_attempted": 0,
        "recovery_succeeded": 0,
        "recovery_failed": 0,
        "details": [],
    }

    # Attempt recovery for each incomplete ingestion
    for ingestion in incomplete:
        if ingestion.get("is_stalled") or ingestion.get("status") == "failed":
            crate_name = ingestion["crate_name"]
            version = ingestion["version"]
            db_path = Path(ingestion["db_path"])

            recovery_results["recovery_attempted"] += 1

            try:
                # Attempt recovery
                recovered_path = await recover_incomplete_ingestion(
                    crate_name, version, db_path
                )

                recovery_results["recovery_succeeded"] += 1
                recovery_results["details"].append(
                    {
                        "crate": f"{crate_name}@{version}",
                        "status": "recovered",
                        "db_path": str(recovered_path),
                    }
                )

            except Exception as e:
                recovery_results["recovery_failed"] += 1
                recovery_results["details"].append(
                    {
                        "crate": f"{crate_name}@{version}",
                        "status": "failed",
                        "error": str(e),
                    }
                )

    return recovery_results


@router.post(
    "/mcp/tools/compare_versions",
    tags=["tools"],
    summary="Compare Versions",
    response_description="Version diff with breaking changes and migration hints",
    operation_id="compareVersions",
)
@limiter.limit("30/second")
async def compare_versions(
    request: Request, params: CompareVersionsRequest
) -> VersionDiffResponse:
    """
    Compare two versions of a crate for API changes.

    Performs semantic diff between two crate versions, identifying breaking changes,
    deprecations, and providing migration hints. Optimized for Rust coding agents
    to understand API evolution and assist with code migration.

    **Features**:
    - Detects breaking changes according to Rust semver rules
    - Provides migration hints for breaking changes
    - Categorizes changes (added, removed, modified, deprecated)
    - Caches results for improved performance
    """
    try:
        # Import here to avoid circular dependency
        from .version_diff import get_diff_engine

        # Get or create the diff engine
        engine = get_diff_engine()

        # Perform comparison
        result = await engine.compare_versions(params)

        return result

    except FileNotFoundError as e:
        # One or both versions don't exist
        logger.error(f"Version not found for comparison: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Version not found: {str(e)}. Ensure both versions exist.",
        ) from e
    except Exception as e:
        logger.error(f"Error comparing versions: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to compare versions: {str(e)}",
        ) from e


@router.get(
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
    If the crate hasn't been ingested yet, it will be downloaded and processed automatically
    using the latest version.

    **Parameters**:
    - `crate_name`: Name of the Rust crate to query

    **Note**: First-time ingestion may take 1-10 seconds depending on crate size.
    """
    try:
        # Ensure crate is ingested (will wait for completion)
        await ingest_crate(crate_name, "latest")

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
        # Handle cross-crate search
        if params.crates and len(params.crates) > 1:
            # Cross-crate search
            from .database.search import cross_crate_search
            from .ingest import get_embedding_model
            
            crate_paths = []
            for crate in params.crates[:5]:  # Limit to 5 crates
                db_path = await ingest_crate(crate, "latest")
                crate_paths.append(db_path)
            
            model = get_embedding_model()
            query_embedding = list(model.embed([params.query]))[0]
            
            results = await cross_crate_search(
                crate_paths,
                query_embedding,
                k=params.k or 5,
                type_filter=params.item_type,
                has_examples=params.has_examples,
                min_doc_length=params.min_doc_length,
                visibility=params.visibility,
                deprecated=params.deprecated,
                stability_filter=params.stability_filter,
            )
        else:
            # Single crate search
            db_path = await ingest_crate(params.crate_name, params.version)
            
            # Route based on search mode
            if params.search_mode == "regex" and params.regex_pattern:
                # Regex search
                from .database.search import regex_search
                
                results = await regex_search(
                    db_path,
                    params.regex_pattern,
                    k=params.k or 10,
                    type_filter=params.item_type,
                    crate_filter=params.crate_filter,
                    module_path=params.module_path,
                    has_examples=params.has_examples,
                    min_doc_length=params.min_doc_length,
                    visibility=params.visibility,
                    deprecated=params.deprecated,
                    stability_filter=params.stability_filter,
                )
            elif params.search_mode == "fuzzy":
                # Enhanced fuzzy search
                from .fuzzy_resolver import get_fuzzy_suggestions
                
                # Get fuzzy suggestions first
                fuzzy_paths = await get_fuzzy_suggestions(
                    params.query,
                    db_path,
                    params.crate_name,
                    params.version or "latest",
                    limit=params.k or 5,
                    threshold=params.fuzzy_tolerance or 0.7,
                )
                
                # Then retrieve full docs for those paths
                results = []
                if fuzzy_paths:
                    import aiosqlite
                    async with aiosqlite.connect(db_path) as db:
                        for path in fuzzy_paths:
                            cursor = await db.execute(
                                "SELECT item_path, header, content FROM embeddings WHERE item_path = ?",
                                (path,)
                            )
                            row = await cursor.fetchone()
                            if row:
                                results.append((1.0, row[0], row[1], row[2]))
            else:
                # Default vector search (including hybrid mode)
                from .ingest import get_embedding_model
                
                model = get_embedding_model()
                query_embedding = list(model.embed([params.query]))[0]
                
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
                    stability_filter=params.stability_filter,
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
