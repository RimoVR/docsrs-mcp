"""FastAPI application for docsrs-mcp server."""

import logging
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, HTTPException, Request
from slowapi.errors import RateLimitExceeded

from .database import get_module_tree as get_module_tree_from_db
from .database import search_embeddings
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


@app.get(
    "/health",
    tags=["health"],
    summary="Health Check",
    response_description="Service health status",
)
async def health_check():
    """
    Check service health status.

    Returns a simple status object indicating the service is operational.
    Useful for monitoring and load balancer health checks.
    """
    return {"status": "ok", "service": "docsrs-mcp"}


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
                            "type": "string",
                            "description": "Specific version (default: latest)",
                        },
                    },
                    "required": ["crate_name"],
                },
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
                            "type": "string",
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
                            "type": "string",
                            "description": "Specific version (default: latest)",
                        },
                    },
                    "required": ["crate_name", "item_path"],
                },
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
                            "type": "string",
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

        # Convert to response format
        search_results = [
            SearchResult(
                score=score,
                item_path=item_path,
                header=header,
                snippet=content[:200] + "..." if len(content) > 200 else content,
            )
            for score, item_path, header, content in results
        ]

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

        # Search for the specific item
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT content FROM embeddings WHERE item_path = ? LIMIT 1",
                (params.item_path,),
            )
            row = await cursor.fetchone()

            if row:
                return {"content": row[0], "format": "markdown"}
            else:
                # If not found in embeddings, return a helpful message
                return {
                    "content": f"Documentation for `{params.item_path}` not found in the current index. Try searching for it using the search_items tool.",
                    "format": "markdown",
                }

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
        import json

        from .database import search_example_embeddings

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
            import sqlite_vec

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

                    # Handle old format (list of strings)
                    if isinstance(examples_data, list) and all(
                        isinstance(e, str) for e in examples_data
                    ):
                        examples_data = [
                            {"code": e, "language": "rust", "detected": False}
                            for e in examples_data
                        ]

                    for example in examples_data:
                        # Handle both dict format and potential string format
                        if isinstance(example, str):
                            example = {
                                "code": example,
                                "language": "rust",
                                "detected": False,
                            }

                        # Filter by language if specified
                        if (
                            params.language
                            and example.get("language") != params.language
                        ):
                            continue

                        code = example.get("code", "")
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
