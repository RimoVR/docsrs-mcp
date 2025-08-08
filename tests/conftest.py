"""Pytest configuration and fixtures for integration tests."""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from docsrs_mcp.popular_crates import PopularCratesManager, PreIngestionWorker


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the entire test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest_asyncio.fixture
async def mock_crates_api():
    """Mock crates.io API responses."""
    with patch("httpx.AsyncClient") as mock_client:
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "crates": [
                {
                    "id": "tokio",
                    "downloads": 1000000,
                    "description": "An asynchronous runtime for Rust",
                    "max_stable_version": {"num": "1.35.1"},
                },
                {
                    "id": "serde",
                    "downloads": 900000,
                    "description": "Serialization framework",
                    "max_stable_version": {"num": "1.0.195"},
                },
                {
                    "id": "clap",
                    "downloads": 800000,
                    "description": "Command-line argument parser",
                    "max_stable_version": {"num": "4.5.0"},
                },
            ]
        }

        # Mock both get and post methods
        mock_instance = mock_client.return_value.__aenter__.return_value
        mock_instance.get = AsyncMock(return_value=mock_response)
        mock_instance.post = AsyncMock(return_value=mock_response)

        yield mock_client


@pytest_asyncio.fixture
async def mock_aiohttp_session():
    """Mock aiohttp.ClientSession for crates.io API calls."""
    with patch("aiohttp.ClientSession") as mock_session_class:
        mock_session = MagicMock()
        mock_session_class.return_value.__aenter__.return_value = mock_session

        # Mock response for popular crates
        mock_resp = AsyncMock()
        mock_resp.status = 200
        mock_resp.json = AsyncMock(
            return_value={
                "crates": [
                    {
                        "id": "tokio",
                        "downloads": 1000000,
                        "description": "An asynchronous runtime",
                        "max_stable_version": {"num": "1.35.1"},
                    },
                    {
                        "id": "serde",
                        "downloads": 900000,
                        "description": "Serialization framework",
                        "max_stable_version": {"num": "1.0.195"},
                    },
                ]
            }
        )

        mock_session.get.return_value.__aenter__.return_value = mock_resp

        yield mock_session


@pytest_asyncio.fixture
async def pre_ingestion_system(tmp_path):
    """Complete pre-ingestion system for testing."""
    # Set up test environment with temp directory
    os.environ["DOCSRS_CACHE_DIR"] = str(tmp_path)
    os.environ["DOCSRS_PRE_INGEST_ENABLED"] = "true"
    os.environ["DOCSRS_PRE_INGEST_CONCURRENCY"] = "2"

    # Create manager and worker
    manager = PopularCratesManager()
    worker = PreIngestionWorker(manager)

    yield manager, worker

    # Cleanup
    if worker._monitor_task and not worker._monitor_task.done():
        worker._monitor_task.cancel()
        try:
            await worker._monitor_task
        except asyncio.CancelledError:
            pass

    if worker._memory_monitor_task and not worker._memory_monitor_task.done():
        worker._memory_monitor_task.cancel()
        try:
            await worker._memory_monitor_task
        except asyncio.CancelledError:
            pass

    # Close manager session if exists
    await manager.close()


@pytest_asyncio.fixture
async def test_client():
    """Create a test client for the FastAPI app."""
    from docsrs_mcp.app import app

    with TestClient(app) as client:
        yield client


@pytest_asyncio.fixture
async def mock_ingest_crate():
    """Mock the ingest_crate function for testing."""
    with patch("docsrs_mcp.popular_crates.ingest_crate") as mock:
        # Mock returns a Path object representing successful ingestion
        mock.return_value = Path("/tmp/test_cache/crate/version.db")
        yield mock


@pytest_asyncio.fixture
async def mock_psutil():
    """Mock psutil for memory monitoring tests."""
    with patch("psutil.Process") as mock_process_class:
        mock_process = MagicMock()
        mock_process_class.return_value = mock_process

        # Default memory values (under threshold)
        mock_memory_info = MagicMock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500 MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_process.memory_percent.return_value = 25.0

        yield mock_process


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    with patch("docsrs_mcp.config") as mock_cfg:
        mock_cfg.PRE_INGEST_ENABLED = True
        mock_cfg.PRE_INGEST_CONCURRENCY = 3
        mock_cfg.POPULAR_CRATES_COUNT = 10
        mock_cfg.SCHEDULER_ENABLED = False
        mock_cfg.VERSION = "0.1.0-test"
        mock_cfg.POPULAR_CRATES_URL = (
            "https://crates.io/api/v1/crates?page=1&per_page={}"
        )
        mock_cfg.POPULAR_CRATES_REFRESH_HOURS = 24
        mock_cfg.FALLBACK_POPULAR_CRATES = [
            "tokio",
            "serde",
            "clap",
            "regex",
            "anyhow",
            "thiserror",
            "log",
            "tracing",
            "async-trait",
            "futures",
        ]
        yield mock_cfg


@pytest.fixture
def temp_cache_dir(tmp_path):
    """Create a temporary cache directory for tests."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    os.environ["DOCSRS_CACHE_DIR"] = str(cache_dir)
    yield cache_dir

    # Cleanup
    if "DOCSRS_CACHE_DIR" in os.environ:
        del os.environ["DOCSRS_CACHE_DIR"]
