"""Middleware, exception handlers, and startup events for the docsrs MCP server."""

import json
import logging

from fastapi import Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from . import config

logger = logging.getLogger(__name__)

# Create limiter with 30 req/s per IP
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=["30/second"],
    enabled=True,  # Can be made configurable via environment variable
)


async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """Return consistent JSON error for rate limit violations.

    Returns HTTP 429 with the error format specified in PRD:
    {"error": "too_many_requests"}
    """
    response = JSONResponse(status_code=429, content={"error": "too_many_requests"})
    # Add standard rate limit headers for client retry guidance
    response.headers["Retry-After"] = (
        str(exc.retry_after) if hasattr(exc, "retry_after") else "1"
    )
    return response


async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Format validation errors with enhanced context.

    Provides user-friendly error messages with examples and actionable guidance
    for all validation failures.
    """
    errors = []
    for error in exc.errors():
        # Build field path from location tuple
        field_path = " -> ".join(
            str(loc) for loc in error["loc"][1:]
        )  # Skip 'body' prefix

        # Extract the actual error message and context
        error_detail = {
            "field": field_path
            if field_path
            else error["loc"][-1]
            if error["loc"]
            else "unknown",
            "message": error["msg"],
            "type": error["type"],
        }

        # Add input value if available (helps with debugging)
        if "input" in error:
            # Handle cases where input might be an exception or non-serializable object
            input_value = error["input"]
            if isinstance(input_value, Exception):
                error_detail["received_value"] = str(input_value)
            else:
                try:
                    # Try to serialize normally
                    json.dumps(input_value)
                    error_detail["received_value"] = input_value
                except (TypeError, ValueError):
                    # If not serializable, convert to string
                    error_detail["received_value"] = str(input_value)

        # Add context if available
        if "ctx" in error:
            # Handle context which might contain non-serializable objects
            context = error["ctx"]
            if isinstance(context, dict):
                # Try to serialize each value in the context dict
                serializable_context = {}
                for key, value in context.items():
                    if isinstance(value, Exception):
                        serializable_context[key] = str(value)
                    else:
                        try:
                            json.dumps(value)
                            serializable_context[key] = value
                        except (TypeError, ValueError):
                            serializable_context[key] = str(value)
                error_detail["context"] = serializable_context
            else:
                # If context is not a dict, convert to string
                error_detail["context"] = str(context)

        errors.append(error_detail)

    # Enhanced error response with examples
    error_response = {
        "error": "validation_error",
        "message": "Request validation failed",
        "details": errors,
        "examples": {
            "get_crate_summary": {
                "crate_name": "serde",
                "version": "1.0.200",  # Optional, defaults to latest
            },
            "search_items": {
                "crate_name": "tokio",
                "query": "spawn async task",
                "k": "10",  # String accepted for numeric parameters
            },
        },
        "documentation": "https://github.com/peterkloiber/docsrs-mcp#api-documentation",
    }

    return JSONResponse(status_code=422, content=error_response)


async def startup_event():
    """Handle application startup tasks."""
    # Start pre-ingestion if enabled via environment variable
    if config.PRE_INGEST_ENABLED:
        logger.info("Starting background pre-ingestion of popular crates")
        from .popular_crates import start_pre_ingestion

        await start_pre_ingestion()

    # Start embedding warmup if enabled
    if config.EMBEDDINGS_WARMUP_ENABLED:
        from .ingest import warmup_embedding_model

        await warmup_embedding_model()
        logger.info("Embedding warmup task started")
