"""Rate limiting middleware for the docsrs MCP server."""

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from slowapi import Limiter
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
