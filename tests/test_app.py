"""Tests for the FastAPI application."""

import pytest
from httpx import ASGITransport, AsyncClient

from docsrs_mcp.app import app


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok", "service": "docsrs-mcp"}


@pytest.mark.asyncio
async def test_root():
    """Test the root endpoint."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "docsrs-mcp"
        assert data["version"] == "0.1.0"
        assert "mcp_manifest" in data
