"""CLI entry point for docsrs-mcp server."""

import argparse
import logging
import os

import uvicorn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main() -> None:
    """Main entry point for the docsrs-mcp server."""
    parser = argparse.ArgumentParser(
        description="docsrs-mcp server - MCP server for Rust crate documentation"
    )
    parser.add_argument(
        "--mode",
        choices=["rest", "mcp"],
        default="mcp",
        help="Server mode: 'rest' for HTTP API, 'mcp' for MCP protocol via STDIO (default: mcp)",
    )
    parser.add_argument(
        "--mcp-implementation",
        choices=["fastmcp", "sdk", "both"],
        default="fastmcp",
        help="MCP implementation to use: 'fastmcp' (current), 'sdk' (new), 'both' (parallel validation) (default: fastmcp)",
    )
    parser.add_argument(
        "--pre-ingest",
        action="store_true",
        default=False,
        help="Enable background pre-ingestion of popular crates (default: disabled)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port for REST server (default: 8000, ignored in MCP mode)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=None,
        help="Number of concurrent pre-ingestion workers (default: 3, range: 1-10)",
    )
    args = parser.parse_args()

    # Convert CLI arguments to environment variables
    if args.port is not None:
        if 1024 <= args.port <= 65535:
            os.environ["DOCSRS_PORT"] = str(args.port)
        else:
            print(f"Warning: Port {args.port} out of range (1024-65535), using default")

    if args.concurrency is not None:
        if 1 <= args.concurrency <= 10:
            os.environ["DOCSRS_CONCURRENCY"] = str(args.concurrency)
        else:
            print(
                f"Warning: Concurrency {args.concurrency} out of range (1-10), using default"
            )

    # Set pre-ingestion environment variable if needed
    if args.pre_ingest:
        os.environ["DOCSRS_PRE_INGEST_ENABLED"] = "true"

    # Import modules after setting environment variables to ensure they see the correct config
    from . import config  # noqa: PLC0415

    # Log startup configuration
    logger.info("Starting docsrs-mcp server with configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Port: {config.PORT if args.mode == 'rest' else 'N/A (MCP mode)'}")
    logger.info(
        f"  Pre-ingestion: {'enabled' if config.PRE_INGEST_ENABLED else 'disabled'}"
    )
    if config.PRE_INGEST_ENABLED:
        logger.info(f"  Concurrency: {config.CONCURRENCY} workers")

    if args.mode == "mcp":
        logger.info(f"  MCP Implementation: {args.mcp_implementation}")

    if args.mode == "mcp":
        # Warn if port was specified in MCP mode
        if args.port is not None:
            logger.warning("--port flag is ignored in MCP mode")

        # Run the appropriate MCP implementation
        if args.mcp_implementation == "sdk":
            # Run new SDK implementation
            import asyncio  # noqa: PLC0415

            from .mcp_sdk_server import run_sdk_server  # noqa: PLC0415

            asyncio.run(run_sdk_server())
        elif args.mcp_implementation == "both":
            # Run parallel validation mode
            from .parallel_validation import run_parallel_validation  # noqa: PLC0415

            run_parallel_validation()
        else:
            # Run current FastMCP implementation (default)
            from .mcp_server import run_mcp_server  # noqa: PLC0415

            run_mcp_server()
    else:
        # Run REST API server
        host = os.getenv("HOST", "0.0.0.0")

        print(f"Starting docsrs-mcp server in REST mode on {host}:{config.PORT}")
        if args.pre_ingest:
            print("Pre-ingestion of popular crates enabled")

        uvicorn.run(
            "docsrs_mcp.app:app",
            host=host,
            port=config.PORT,  # Use config.PORT instead of local variable
            loop="uvloop",
            workers=1,
            log_level="info",
        )


if __name__ == "__main__":
    main()
