"""Unit tests for CrossReferenceService."""

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio

from docsrs_mcp.services.cross_reference_service import (
    CircuitOpenError,
    CrossReferenceService,
    SimpleCircuitBreaker,
)


@pytest.fixture
def temp_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        db_path = tmp.name

    # Create minimal schema for testing
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create tables
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS crate_metadata (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            version TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS items (
            id INTEGER PRIMARY KEY,
            crate_id INTEGER,
            name TEXT NOT NULL,
            fingerprint TEXT,
            FOREIGN KEY (crate_id) REFERENCES crate_metadata(id)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS reexports (
            id INTEGER PRIMARY KEY,
            source_item_id INTEGER,
            target_item_id INTEGER,
            link_type TEXT,
            confidence_score REAL,
            FOREIGN KEY (source_item_id) REFERENCES items(id),
            FOREIGN KEY (target_item_id) REFERENCES items(id)
        )
    """)

    # Insert test data
    cursor.execute(
        "INSERT INTO crate_metadata (id, name, version) VALUES (1, 'test_crate', '1.0.0')"
    )
    cursor.execute(
        "INSERT INTO items (id, crate_id, name, fingerprint) VALUES (1, 1, 'TestStruct', 'abc123')"
    )
    cursor.execute(
        "INSERT INTO items (id, crate_id, name, fingerprint) VALUES (2, 1, 'test_mod::TestStruct', 'abc123')"
    )
    cursor.execute(
        "INSERT INTO items (id, crate_id, name, fingerprint) VALUES (3, 1, 'OtherStruct', 'def456')"
    )

    # Add some re-exports
    cursor.execute("""
        INSERT INTO reexports (source_item_id, target_item_id, link_type, confidence_score)
        VALUES (1, 2, 'reexport', 0.9)
    """)
    cursor.execute("""
        INSERT INTO reexports (source_item_id, target_item_id, link_type, confidence_score)
        VALUES (1, 3, 'crossref', 0.7)
    """)

    conn.commit()
    conn.close()

    yield db_path

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest_asyncio.fixture
async def service(temp_db):
    """Create a CrossReferenceService instance with test database."""
    return CrossReferenceService(temp_db)


class TestCrossReferenceService:
    """Test CrossReferenceService methods."""

    @pytest.mark.asyncio
    async def test_init(self, temp_db):
        """Test service initialization."""
        service = CrossReferenceService(temp_db)
        assert service.db_path == temp_db
        assert service._cache_ttl == 300
        assert service.circuit_breaker is not None

    @pytest.mark.asyncio
    async def test_cache_validity(self, service):
        """Test cache validity checking."""
        # Initially not valid
        assert not service._is_cache_valid("test_key")

        # Add to cache
        service._cache_timestamps["test_key"] = service._cache_ttl * 2
        assert not service._is_cache_valid("test_key")

        # Add recent timestamp
        service._cache_timestamps["test_key"] = time.time()
        assert service._is_cache_valid("test_key")

        # Simulate expired cache
        service._cache_timestamps["test_key"] = time.time() - 301
        assert not service._is_cache_valid("test_key")

    @pytest.mark.asyncio
    async def test_resolve_import_cached(self, service):
        """Test that resolve_import uses caching."""
        # First call
        result1 = await service.resolve_import("test_crate", "TestStruct")
        assert "resolved_path" in result1

        # Check cache was populated
        cache_key = "import:test_crate:TestStruct"
        assert cache_key in service._graph_cache

        # Second call should use cache
        result2 = await service.resolve_import("test_crate", "TestStruct")

        assert result1 == result2

    @pytest.mark.asyncio
    async def test_resolve_import_with_alternatives(self, service):
        """Test resolve_import with alternatives."""
        result = await service.resolve_import(
            "test_crate", "NonExistent", include_alternatives=True
        )

        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    @pytest.mark.asyncio
    async def test_get_dependency_graph_basic(self, service):
        """Test basic dependency graph generation."""
        result = await service.get_dependency_graph("test_crate", max_depth=2)

        assert "root" in result
        assert "total_nodes" in result
        assert "max_depth" in result
        assert "has_cycles" in result
        assert result["max_depth"] == 2

    @pytest.mark.asyncio
    async def test_get_dependency_graph_cached(self, service):
        """Test that dependency graphs are cached."""
        # First call
        result1 = await service.get_dependency_graph("test_crate", max_depth=2)

        # Check cache was populated
        cache_key = "graph:test_crate:2"
        assert cache_key in service._graph_cache

        # Second call should use cache
        result2 = await service.get_dependency_graph("test_crate", max_depth=2)
        assert result1 == result2

    def test_detect_cycles_no_cycle(self, service):
        """Test cycle detection with acyclic graph."""
        graph = {"A": [{"name": "B"}], "B": [{"name": "C"}], "C": []}
        assert not service._detect_cycles(graph)

    def test_detect_cycles_with_cycle(self, service):
        """Test cycle detection with cyclic graph."""
        graph = {"A": [{"name": "B"}], "B": [{"name": "C"}], "C": [{"name": "A"}]}
        assert service._detect_cycles(graph)

    def test_detect_cycles_self_loop(self, service):
        """Test cycle detection with self-loop."""
        graph = {
            "A": [{"name": "A"}]  # Self-loop
        }
        assert service._detect_cycles(graph)

    def test_build_hierarchy(self, service):
        """Test hierarchy building."""
        graph = {
            "root": [{"name": "child1"}, {"name": "child2"}],
            "child1": [{"name": "grandchild"}],
            "child2": [],
            "grandchild": [],
        }

        result = service._build_hierarchy("root", graph, max_depth=3)

        assert result["name"] == "root"
        assert len(result["dependencies"]) == 2
        assert result["dependencies"][0]["name"] == "child1"
        assert result["dependencies"][1]["name"] == "child2"
        assert len(result["dependencies"][0]["dependencies"]) == 1

    def test_build_hierarchy_max_depth(self, service):
        """Test hierarchy building respects max depth."""
        graph = {
            "root": [{"name": "level1"}],
            "level1": [{"name": "level2"}],
            "level2": [{"name": "level3"}],
            "level3": [{"name": "level4"}],
        }

        result = service._build_hierarchy("root", graph, max_depth=2)

        # Should only go 2 levels deep
        assert result["name"] == "root"
        assert len(result["dependencies"]) == 1
        assert result["dependencies"][0]["name"] == "level1"
        assert len(result["dependencies"][0]["dependencies"]) == 1
        assert result["dependencies"][0]["dependencies"][0]["name"] == "level2"
        # level3 should not be included due to max_depth
        assert len(result["dependencies"][0]["dependencies"][0]["dependencies"]) == 0

    @pytest.mark.asyncio
    async def test_suggest_migrations_fallback(self, service):
        """Test migration suggestions with fallback to patterns."""
        # Mock the database query to fail
        with patch("aiosqlite.connect") as mock_connect:
            mock_conn = AsyncMock()
            mock_cursor = AsyncMock()
            mock_cursor.fetchall = AsyncMock(side_effect=Exception("DB Error"))
            mock_conn.execute = AsyncMock(return_value=mock_cursor)
            mock_conn.__aenter__ = AsyncMock(return_value=mock_conn)
            mock_conn.__aexit__ = AsyncMock()
            mock_connect.return_value = mock_conn

            # Should fall back to pattern-based suggestions
            result = await service.suggest_migrations("test_crate", "1.0.0", "2.0.0")
            assert isinstance(result, list)

    def test_pattern_based_migrations(self, service):
        """Test pattern-based migration generation."""
        suggestions = service._pattern_based_migrations("test_crate", "1.0.0", "2.0.0")

        assert isinstance(suggestions, list)
        # Major version change should generate some suggestions
        assert len(suggestions) > 0

        for suggestion in suggestions:
            assert "old_path" in suggestion
            assert "new_path" in suggestion
            assert "change_type" in suggestion
            assert "confidence" in suggestion

    @pytest.mark.asyncio
    async def test_trace_reexports_basic(self, service):
        """Test basic re-export tracing."""
        result = await service.trace_reexports("test_crate", "TestStruct")

        assert "chain" in result
        assert "original_source" in result
        assert "original_crate" in result
        assert isinstance(result["chain"], list)

    @pytest.mark.asyncio
    async def test_trace_reexports_no_reexports(self, service):
        """Test tracing when no re-exports exist."""
        result = await service.trace_reexports("test_crate", "NoReexport")

        assert result["chain"] == []
        assert result["original_source"] == "NoReexport"
        assert result["original_crate"] == "test_crate"


class TestSimpleCircuitBreaker:
    """Test SimpleCircuitBreaker class."""

    def test_init(self):
        """Test circuit breaker initialization."""
        breaker = SimpleCircuitBreaker(failure_threshold=3, timeout=30)
        assert breaker.failures == 0
        assert breaker.threshold == 3
        assert breaker.timeout == 30
        assert not breaker.is_open_flag

    def test_is_open_initially_closed(self):
        """Test circuit is initially closed."""
        breaker = SimpleCircuitBreaker()
        assert not breaker.is_open()

    def test_record_failure(self):
        """Test failure recording."""
        breaker = SimpleCircuitBreaker(failure_threshold=3)

        breaker.record_failure()
        assert breaker.failures == 1
        assert not breaker.is_open()

        breaker.record_failure()
        assert breaker.failures == 2
        assert not breaker.is_open()

        breaker.record_failure()
        assert breaker.failures == 3
        assert breaker.is_open()

    def test_reset(self):
        """Test circuit breaker reset."""
        breaker = SimpleCircuitBreaker(failure_threshold=2)

        breaker.record_failure()
        breaker.record_failure()
        assert breaker.is_open()

        breaker.reset()
        assert breaker.failures == 0
        assert not breaker.is_open()

    def test_timeout_reset(self):
        """Test automatic reset after timeout."""
        breaker = SimpleCircuitBreaker(failure_threshold=1, timeout=0.1)

        breaker.record_failure()
        assert breaker.is_open()

        # Wait for timeout
        time.sleep(0.15)
        assert not breaker.is_open()

    @pytest.mark.asyncio
    async def test_call_success(self):
        """Test successful function call through circuit breaker."""
        breaker = SimpleCircuitBreaker()

        async def test_func():
            return "success"

        result = await breaker.call(test_func)
        assert result == "success"
        assert breaker.failures == 0

    @pytest.mark.asyncio
    async def test_call_failure(self):
        """Test failed function call through circuit breaker."""
        breaker = SimpleCircuitBreaker(failure_threshold=2)

        async def failing_func():
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        assert breaker.failures == 1
        assert not breaker.is_open()

        with pytest.raises(ValueError):
            await breaker.call(failing_func)

        assert breaker.failures == 2
        assert breaker.is_open()

    @pytest.mark.asyncio
    async def test_call_when_open(self):
        """Test calling when circuit is open."""
        breaker = SimpleCircuitBreaker(failure_threshold=1)

        async def test_func():
            return "success"

        # Open the circuit
        breaker.record_failure()
        assert breaker.is_open()

        # Should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await breaker.call(test_func)


# Removed TestIntegrationWithFuzzyResolver as fuzzy_resolver is now a module function, not a class
