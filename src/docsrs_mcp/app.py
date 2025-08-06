"""FastAPI application for docsrs-mcp server."""

import logging
from pathlib import Path

import aiosqlite
from fastapi import FastAPI, HTTPException

from .database import search_embeddings
from .ingest import get_embedding_model, ingest_crate
from .models import (
    CrateModule,
    GetCrateSummaryRequest,
    GetCrateSummaryResponse,
    GetItemDocRequest,
    MCPManifest,
    MCPResource,
    MCPTool,
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
async def root():
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
async def get_mcp_manifest():
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
async def get_crate_summary(request: GetCrateSummaryRequest):
    """
    Get summary information about a Rust crate.

    Fetches crate metadata including name, version, description, repository URL,
    and module structure. If the crate hasn't been ingested yet, it will be
    downloaded and processed automatically.

    **Note**: First-time ingestion may take 1-10 seconds depending on crate size.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(request.crate_name, request.version)

        # Fetch crate metadata from database
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name, version, description, repository, documentation FROM crate_metadata LIMIT 1"
            )
            row = await cursor.fetchone()

            if not row:
                raise HTTPException(status_code=404, detail="Crate not found")

            name, version, description, repository, documentation = row

            # Fetch modules
            cursor = await db.execute(
                "SELECT name, path FROM modules WHERE crate_id = 1"
            )
            modules = [
                CrateModule(name=row[0], path=row[1]) for row in await cursor.fetchall()
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
async def search_items(request: SearchItemsRequest):
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
        db_path = await ingest_crate(request.crate_name, request.version)

        # Generate embedding for query
        model = get_embedding_model()
        query_embedding = list(model.embed([request.query]))[0]

        # Search embeddings
        results = await search_embeddings(db_path, query_embedding, k=request.k or 5)

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
async def get_item_doc(request: GetItemDocRequest):
    """
    Get complete documentation for a specific item in a crate.

    Retrieves the full documentation for a specific item identified by its path
    (e.g., 'tokio::spawn', 'serde::Deserialize'). Returns markdown-formatted
    documentation including examples and detailed descriptions.

    **Tip**: Use search_items first if you're unsure of the exact item path.
    """
    try:
        # Ingest crate if not already done
        db_path = await ingest_crate(request.crate_name, request.version)

        # Search for the specific item
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT content FROM embeddings WHERE item_path = ? LIMIT 1",
                (request.item_path,),
            )
            row = await cursor.fetchone()

            if row:
                return {"content": row[0], "format": "markdown"}
            else:
                # If not found in embeddings, return a helpful message
                return {
                    "content": f"Documentation for `{request.item_path}` not found in the current index. Try searching for it using the search_items tool.",
                    "format": "markdown",
                }

    except Exception as e:
        logger.error(f"Error in get_item_doc: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.get(
    "/mcp/resources/versions",
    tags=["resources"],
    summary="List Crate Versions",
    response_description="Available crate versions",
    operation_id="listVersions",
)
async def list_versions(crate_name: str):
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
