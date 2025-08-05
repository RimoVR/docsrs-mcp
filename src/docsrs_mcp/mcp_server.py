"""MCP server implementation using FastMCP."""

import logging
import sys

from fastmcp import FastMCP

from .app import app

# Configure logging to stderr to avoid STDIO corruption
# This is critical for STDIO transport to work properly
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Create MCP server from existing FastAPI app
# FastMCP.from_fastapi() automatically discovers and converts endpoints
mcp = FastMCP.from_fastapi(app)


def run_mcp_server():
    """Run the MCP server with STDIO transport."""
    try:
        logger.info("Starting docsrs-mcp server in MCP mode with STDIO transport")
        # Run with default STDIO transport for Claude Desktop
        mcp.run()
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_mcp_server()
