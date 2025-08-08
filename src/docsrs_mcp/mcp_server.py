"""MCP server implementation using FastMCP."""

import asyncio
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


async def start_pre_ingestion_if_enabled(args):
    """Start pre-ingestion if enabled via CLI flag."""
    if hasattr(args, "pre_ingest") and args.pre_ingest:
        from .popular_crates import PopularCratesManager, PreIngestionWorker

        logger.info("Starting background pre-ingestion of popular crates")
        manager = PopularCratesManager()
        worker = PreIngestionWorker(manager)
        await worker.start()


def run_mcp_server(args: object | None = None):
    """Run the MCP server with STDIO transport."""
    try:
        logger.info("Starting docsrs-mcp server in MCP mode with STDIO transport")

        # Start pre-ingestion in background if enabled
        if args and hasattr(args, "pre_ingest") and args.pre_ingest:
            # Create a new event loop for the pre-ingestion task
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # Start the pre-ingestion task
            loop.run_until_complete(start_pre_ingestion_if_enabled(args))

            # Keep the loop running in a separate thread while MCP runs
            import threading

            def run_loop():
                loop.run_forever()

            thread = threading.Thread(target=run_loop, daemon=True)
            thread.start()

        # Run with default STDIO transport for Claude Desktop
        mcp.run()
    except Exception as e:
        logger.error(f"Error running MCP server: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_mcp_server()
