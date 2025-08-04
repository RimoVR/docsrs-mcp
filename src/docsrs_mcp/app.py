"""FastAPI application for docsrs-mcp server."""

from fastapi import FastAPI

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
