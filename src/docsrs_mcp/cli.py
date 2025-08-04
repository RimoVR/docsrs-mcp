"""CLI entry point for docsrs-mcp server."""

import os

import uvicorn


def main() -> None:
    """Main entry point for the docsrs-mcp server."""
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")

    print(f"Starting docsrs-mcp server on {host}:{port}")

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
