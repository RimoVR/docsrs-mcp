"""MCP server implementation using FastMCP."""

import asyncio
import logging
import sys
import threading

from fastmcp import FastMCP

from . import config
from .app import app
from .popular_crates import start_pre_ingestion

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

        # Start pre-ingestion in background if enabled (matching app.py pattern)
        if config.PRE_INGEST_ENABLED:
            logger.info("Starting background pre-ingestion of popular crates")

            # Create separate event loop for pre-ingestion (maintaining existing pattern)
            def run_pre_ingestion():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(start_pre_ingestion())
                finally:
                    loop.close()

            # Start in background thread (non-blocking)
            thread = threading.Thread(target=run_pre_ingestion, daemon=True)
            thread.start()

        # Start embedding warmup if enabled (matching app.py pattern)
        if config.EMBEDDINGS_WARMUP_ENABLED:
            logger.info("Starting embedding warmup in background")

            # Create separate event loop for warmup
            def run_warmup():
                from .ingest import warmup_embedding_model

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(warmup_embedding_model())
                finally:
                    loop.close()

            # Start in background thread (non-blocking)
            warmup_thread = threading.Thread(target=run_warmup, daemon=True)
            warmup_thread.start()

        # Run with default STDIO transport for Claude Desktop
        mcp.run()
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_mcp_server()
