"""CLI entry point for docsrs-mcp server."""

import argparse
import os

import uvicorn

from .mcp_server import run_mcp_server


def main() -> None:
    """Main entry point for the docsrs-mcp server."""
    parser = argparse.ArgumentParser(
        description="docsrs-mcp server - MCP server for Rust crate documentation"
    )
    parser.add_argument(
        "--mode",
        choices=["rest", "mcp"],
        default="rest",
        help="Server mode: 'rest' for HTTP API, 'mcp' for MCP protocol via STDIO (default: rest)",
    )
    args = parser.parse_args()

    if args.mode == "mcp":
        # Run MCP server with STDIO transport
        run_mcp_server()
    else:
        # Run REST API server (default behavior)
        port = int(os.getenv("PORT", "8000"))
        host = os.getenv("HOST", "0.0.0.0")

        print(f"Starting docsrs-mcp server in REST mode on {host}:{port}")

        uvicorn.run(
            "docsrs_mcp.app:app",
            host=host,
            port=port,
            loop="uvloop",
            workers=1,
            log_level="info",
        )


if __name__ == "__main__":
    main()
