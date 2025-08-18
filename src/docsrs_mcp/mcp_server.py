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


async def override_fastmcp_schemas():
    """Override FastMCP auto-generated schemas for Claude Code compatibility.

    CRITICAL CONTEXT:
    ----------------
    Claude Code (our PRIMARY target MCP client) has a critical bug where it:
    1. Sends parameters as native JSON types (integers, booleans)
    2. Does NOT support anyOf/oneOf/allOf JSON schema patterns

    THE PROBLEM:
    ------------
    FastMCP.from_fastapi() auto-generates schemas from Pydantic type hints.
    These schemas declare parameters with their actual types (integer, boolean).
    When Claude Code sends native types, FastMCP's JSON Schema validation layer
    REJECTS them before our Pydantic field validators can perform type coercion.

    OUR SOLUTION:
    -------------
    We intercept the auto-generated schemas AFTER FastMCP creates them but
    BEFORE they're used for validation. We REPLACE the entire parameter schema
    with a simple string type declaration. This creates a deliberate mismatch:
    - Schema says: "I expect a string"
    - Claude Code sends: native integer/boolean
    - JSON Schema validation: Passes (because it's lenient about type checking)
    - Pydantic validators: Convert the native types to correct internal types

    WHY THIS WORKS:
    ---------------
    1. JSON Schema validation in FastMCP is more lenient than strict
    2. Our Pydantic field validators with mode='before' handle actual conversion
    3. This approach avoids the anyOf pattern that Claude Code can't handle
    4. REST mode is unaffected since it uses different validation paths

    IMPORTANT: This is a workaround for Claude Code's bugs. When Claude Code
    is fixed, this entire function can be removed.
    """
    try:
        # Access the tool manager which stores all tools
        # The _tool_manager attribute is FastMCP's internal registry
        if hasattr(mcp, "_tool_manager"):
            tool_manager = mcp._tool_manager

            # Get all tools from the manager
            # This returns Tool objects that have been auto-generated from our FastAPI endpoints
            tools_dict = await tool_manager.list_tools()

            # Tools that need schema modifications for Claude Code compatibility
            # Each tool maps to parameters that need type override and their original types
            # We track original types for logging/debugging purposes
            tools_to_fix = {
                "search_items": {
                    "k": "integer",  # Number of results
                    "min_doc_length": "integer",  # Min doc length filter
                    "has_examples": "boolean",  # Filter for examples
                    "deprecated": "boolean",  # Deprecation filter
                },
                "start_pre_ingestion": {
                    "count": "integer",  # Number of crates to ingest
                    "concurrency": "integer",  # Parallel workers
                    "force": "boolean",  # Force restart flag
                },
                "ingest_cargo_file": {
                    "concurrency": "integer",  # Parallel workers
                    "skip_existing": "boolean",  # Skip already ingested
                    "resolve_versions": "boolean",  # Resolve version specs
                },
                "compare_versions": {
                    "include_unchanged": "boolean",  # Include unchanged items
                    "max_results": "integer",  # Max changes to return
                },
                "get_crate_summary": {},  # Only has string parameters
                "search_examples": {
                    "k": "integer"  # Number of examples
                },
                # These tools have no problematic parameters but are listed for completeness
                "list_versions": {},  # Only has string parameters
                "control_pre_ingestion": {},  # Only has enum parameters
            }

            # Iterate through tools and modify their schemas
            for tool in tools_dict:
                if tool.name in tools_to_fix:
                    params_to_fix = tools_to_fix[tool.name]

                    # Access the tool's parameters (which become inputSchema in MCP protocol)
                    # The 'parameters' attribute contains the JSON Schema for validation
                    if hasattr(tool, "parameters") and tool.parameters:
                        if "properties" in tool.parameters:
                            for param_name, original_type in params_to_fix.items():
                                if param_name in tool.parameters["properties"]:
                                    # CRITICAL FIX: We must REPLACE the entire schema, not just add a type
                                    # The original has anyOf patterns that Claude Code can't handle
                                    # We create a simple string schema that allows any string value

                                    # Preserve useful metadata but replace the type structure
                                    original_schema = tool.parameters["properties"][
                                        param_name
                                    ]
                                    description = original_schema.get("description", "")
                                    title = original_schema.get("title", param_name)

                                    # REPLACE the entire schema with a simple string type
                                    # This removes anyOf patterns and makes it Claude Code compatible
                                    tool.parameters["properties"][param_name] = {
                                        "type": "string",
                                        "description": description,
                                        "title": title,
                                    }

                                    # Log the modification for debugging and monitoring
                                    logger.debug(
                                        f"Replaced {tool.name}.{param_name} schema from {original_type} to string for Claude Code"
                                    )

            logger.info(
                "Successfully overrode FastMCP schemas for Claude Code compatibility"
            )

    except Exception as e:
        # Log the error but don't fail server startup
        # The server can still function, just without the compatibility fix
        logger.error(f"Failed to override FastMCP schemas: {e}", exc_info=True)
        # Don't fail the server startup if schema override fails
        # Better to have a partially working server than no server


def run_mcp_server():
    """Run the MCP server with STDIO transport."""
    try:
        logger.info("Starting docsrs-mcp server in MCP mode with STDIO transport")

        # Override FastMCP schemas for Claude Code compatibility
        # This must happen before starting the server
        asyncio.run(override_fastmcp_schemas())

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
