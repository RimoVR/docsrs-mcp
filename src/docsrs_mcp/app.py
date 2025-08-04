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
    description="MCP server for querying Rust crate documentation with vector search",
    version="0.1.0",
)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "service": "docsrs-mcp"}


@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "service": "docsrs-mcp",
        "version": "0.1.0",
        "description": "MCP server for Rust crate documentation",
        "mcp_manifest": "/mcp/manifest",
    }


@app.get("/mcp/manifest", response_model=MCPManifest)
async def get_mcp_manifest():
    """Return MCP server manifest with available tools and resources."""
    return MCPManifest(
        tools=[
            MCPTool(
                name="get_crate_summary",
                description="Get summary information about a Rust crate including modules and description",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to query",
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
                description="Search for items in a crate's documentation using vector similarity",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate to search in",
                        },
                        "query": {"type": "string", "description": "Search query text"},
                        "version": {
                            "type": "string",
                            "description": "Specific version (default: latest)",
                        },
                        "k": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 5,
                            "minimum": 1,
                            "maximum": 20,
                        },
                    },
                    "required": ["crate_name", "query"],
                },
            ),
            MCPTool(
                name="get_item_doc",
                description="Get complete documentation for a specific item in a crate",
                input_schema={
                    "type": "object",
                    "properties": {
                        "crate_name": {
                            "type": "string",
                            "description": "Name of the crate",
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


@app.post("/mcp/tools/get_crate_summary", response_model=GetCrateSummaryResponse)
async def get_crate_summary(request: GetCrateSummaryRequest):
    """Get summary information about a Rust crate."""
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


@app.post("/mcp/tools/search_items", response_model=SearchItemsResponse)
async def search_items(request: SearchItemsRequest):
    """Search for items in a crate's documentation."""
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


@app.post("/mcp/tools/get_item_doc")
async def get_item_doc(request: GetItemDocRequest):
    """Get complete documentation for a specific item."""
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


@app.get("/mcp/resources/versions")
async def list_versions(crate_name: str):
    """List all available versions of a crate."""
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
