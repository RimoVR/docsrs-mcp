"""FastAPI application initialization and configuration for docsrs-mcp server."""

import logging

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from slowapi.errors import RateLimitExceeded

from .middleware import (
    limiter,
    rate_limit_handler,
    startup_event,
    validation_exception_handler,
)

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
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
        },
        license_info={
            "name": "Apache 2.0",
            "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
        },
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        openapi_tags=[
            {
                "name": "health",
                "description": "Health check endpoints",
            },
            {
                "name": "mcp",
                "description": "MCP manifest endpoint",
            },
            {
                "name": "tools",
                "description": "MCP tool endpoints for interacting with crate documentation",
            },
            {
                "name": "resources",
                "description": "MCP resource endpoints",
            },
            {
                "name": "admin",
                "description": "Administrative endpoints",
            },
        ],
    )

    # Configure rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_handler)

    # Configure validation error handler
    app.add_exception_handler(RequestValidationError, validation_exception_handler)

    # Configure startup event
    app.add_event_handler("startup", startup_event)

    # Register routers
    from .endpoints import router as main_router
    from .endpoints_tools import router as tools_router

    app.include_router(main_router)
    app.include_router(tools_router)

    return app


# Create the app instance
app = create_app()
