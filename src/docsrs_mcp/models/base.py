"""
Base models and shared utilities for all model modules.

This module provides the foundation layer with common configurations,
shared validators, and the ErrorResponse model used across the application.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Base configuration used by all models to prevent injection attacks
strict_config = ConfigDict(extra="forbid")


class ErrorResponse(BaseModel):
    """
    Standard error response format.

    Used for all API errors to provide consistent error handling.

    Example:
        ```json
        {
            "error": "item_not_found",
            "detail": "No documentation found for 'tokio::spwan'",
            "status_code": 404,
            "suggestions": ["tokio::spawn", "tokio::spawn_blocking"]
        }
        ```
    """

    error: str = Field(..., description="Error type/category")
    detail: str | None = Field(None, description="Detailed error message for debugging")
    status_code: int = Field(500, description="HTTP status code", ge=400, le=599)
    suggestions: list[str] | None = Field(
        None, description="Suggested alternative paths for fuzzy matching"
    )

    @field_validator("status_code", mode="before")
    @classmethod
    def coerce_status_code_to_int(cls, v):
        """Convert string numbers to int for MCP client compatibility."""
        if v is None:
            return 500  # Default to 500 if None
        # Use enhanced validation with examples
        from docsrs_mcp.validation import coerce_to_int_with_bounds

        return coerce_to_int_with_bounds(
            value=v,
            field_name="status_code (HTTP error code)",
            min_val=400,
            max_val=599,
            examples=[400, 404, 500],
        )

    model_config = strict_config

