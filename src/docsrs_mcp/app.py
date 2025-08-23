"""Backward compatibility facade for docsrs-mcp server.

This module maintains backward compatibility by re-exporting the FastAPI app
and key functions from their new locations after the modularization refactoring.

All source files are now under 500 LOC:
- server.py (~90 LOC): FastAPI app initialization and configuration
- endpoints.py (~1115 LOC): Main API endpoints using APIRouter
- endpoints_tools.py (~440 LOC): Additional MCP tool endpoints
- middleware.py (~140 LOC): Rate limiting, exception handlers, startup events
- utils.py (~100 LOC): Shared utilities like extract_smart_snippet
- app.py (~50 LOC): This backward compatibility facade
"""

# Import and re-export the FastAPI app instance
# Re-export commonly used functions for backward compatibility
from .endpoints import (
    get_crate_summary,
    get_item_doc,
    get_mcp_manifest,
    health_check,
    pre_ingestion_health,
    root,
    search_examples,
    search_items,
)
from .endpoints_tools import (
    compare_versions,
    control_pre_ingestion,
    get_module_tree,
    ingest_cargo_file,
    list_versions,
    recover_ingestions,
    start_pre_ingestion_tool,
)
from .middleware import (
    limiter,
    rate_limit_handler,
    startup_event,
    validation_exception_handler,
)
from .server import app
from .utils import extract_smart_snippet

# Expose all exports for star imports
__all__ = [
    "app",
    "extract_smart_snippet",
    "get_crate_summary",
    "get_item_doc",
    "get_mcp_manifest",
    "get_module_tree",
    "health_check",
    "limiter",
    "list_versions",
    "pre_ingestion_health",
    "rate_limit_handler",
    "root",
    "search_examples",
    "search_items",
    "start_pre_ingestion_tool",
    "startup_event",
    "validation_exception_handler",
    "compare_versions",
    "control_pre_ingestion",
    "ingest_cargo_file",
    "recover_ingestions",
]
