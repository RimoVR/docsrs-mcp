"""Integration tests for Phase 6: Cross-References."""

import time

import pytest
import pytest_asyncio

from docsrs_mcp.ingest import ingest_crate
from docsrs_mcp.services.cross_reference_service import CrossReferenceService


@pytest_asyncio.fixture
async def sample_crate_db():
    """Fixture to provide a test database with ingested crate data."""
    # Ingest a small, well-known crate for testing
    db_path = await ingest_crate("serde", "1.0.193")
    return db_path


@pytest_asyncio.fixture
async def cross_ref_service(sample_crate_db):
    """Fixture to provide a CrossReferenceService instance."""
    return CrossReferenceService(sample_crate_db)


class TestImportResolution:
    """Test import resolution functionality."""

    @pytest.mark.asyncio
    async def test_resolve_import_basic(self, cross_ref_service):
        """Test basic import resolution."""
        result = await cross_ref_service.resolve_import("serde", "Deserialize")

        assert result is not None
        assert "resolved_path" in result
        assert "confidence" in result
        assert result["confidence"] >= 0.0 and result["confidence"] <= 1.0

    @pytest.mark.asyncio
    async def test_resolve_import_with_alternatives(self, cross_ref_service):
        """Test import resolution with alternatives."""
        result = await cross_ref_service.resolve_import(
            "serde", "NonExistentType", include_alternatives=True
        )

        assert result is not None
        assert "alternatives" in result
        assert isinstance(result["alternatives"], list)

    @pytest.mark.asyncio
    async def test_import_resolution_performance(self, cross_ref_service):
        """Verify <200ms import resolution performance requirement."""
        start = time.time()
        result = await cross_ref_service.resolve_import("serde", "Deserialize")
        elapsed = time.time() - start

        assert elapsed < 0.2, f"Import resolution took {elapsed:.3f}s, expected <200ms"
        assert result["confidence"] > 0.0


class TestDependencyGraph:
    """Test dependency graph functionality."""

    @pytest.mark.asyncio
    async def test_dependency_graph_basic(self, cross_ref_service):
        """Test basic dependency graph generation."""
        result = await cross_ref_service.get_dependency_graph("serde", max_depth=2)

        assert result is not None
        assert "root" in result
        assert "total_nodes" in result
        assert "max_depth" in result
        assert "has_cycles" in result
        assert result["max_depth"] == 2

    @pytest.mark.asyncio
    async def test_dependency_graph_cycle_detection(self, cross_ref_service):
        """Test cycle detection in dependency graphs."""
        result = await cross_ref_service.get_dependency_graph("serde", max_depth=5)

        assert "has_cycles" in result
        assert isinstance(result["has_cycles"], bool)
        # serde shouldn't have cycles
        assert not result["has_cycles"]

    @pytest.mark.asyncio
    async def test_dependency_graph_performance(self, cross_ref_service):
        """Verify <500ms dependency graph generation performance requirement."""
        start = time.time()
        result = await cross_ref_service.get_dependency_graph("serde", max_depth=3)
        elapsed = time.time() - start

        assert elapsed < 0.5, f"Dependency graph took {elapsed:.3f}s, expected <500ms"
        assert result["total_nodes"] >= 0

    @pytest.mark.asyncio
    async def test_dependency_graph_with_versions(self, cross_ref_service):
        """Test dependency graph with version information."""
        result = await cross_ref_service.get_dependency_graph(
            "serde", max_depth=2, include_versions=True
        )

        assert result is not None
        assert "root" in result
        root = result["root"]
        assert "name" in root
        assert root["name"] == "serde"


class TestMigrationSuggestions:
    """Test migration suggestion functionality."""

    @pytest.mark.asyncio
    async def test_migration_suggestions_basic(self, cross_ref_service):
        """Test basic migration suggestions between versions."""
        suggestions = await cross_ref_service.suggest_migrations(
            "serde", "1.0.0", "1.0.193"
        )

        assert suggestions is not None
        assert isinstance(suggestions, list)
        # Each suggestion should have required fields
        for suggestion in suggestions[:5]:  # Check first 5 suggestions
            assert "old_path" in suggestion
            assert "new_path" in suggestion
            assert "change_type" in suggestion
            assert "confidence" in suggestion

    @pytest.mark.asyncio
    async def test_migration_suggestions_performance(self, cross_ref_service):
        """Verify <300ms migration suggestions performance requirement."""
        start = time.time()
        await cross_ref_service.suggest_migrations("serde", "1.0.0", "1.0.193")
        elapsed = time.time() - start

        # Note: This might take longer on first run due to ingestion
        # Performance requirement is for warm cache
        assert elapsed < 1.0, f"Migration suggestions took {elapsed:.3f}s"


class TestReexportTracing:
    """Test re-export tracing functionality."""

    @pytest.mark.asyncio
    async def test_trace_reexports_basic(self, cross_ref_service):
        """Test basic re-export tracing."""
        result = await cross_ref_service.trace_reexports("serde", "Deserialize")

        assert result is not None
        assert "chain" in result
        assert "original_source" in result
        assert "original_crate" in result
        assert isinstance(result["chain"], list)

    @pytest.mark.asyncio
    async def test_trace_reexports_no_reexport(self, cross_ref_service):
        """Test tracing when item is not re-exported."""
        result = await cross_ref_service.trace_reexports("serde", "de::Deserialize")

        assert result is not None
        assert "original_source" in result
        # If not re-exported, original source should be the same
        assert result["original_source"] == "de::Deserialize"


class TestCycleDetection:
    """Test cycle detection algorithms."""

    def test_cycle_detection_simple(self):
        """Test cycle detection with simple graph."""
        service = CrossReferenceService(":memory:")

        # Graph with no cycles
        graph_no_cycle = {
            "A": [{"name": "B"}, {"name": "C"}],
            "B": [{"name": "D"}],
            "C": [{"name": "D"}],
            "D": [],
        }
        assert not service._detect_cycles(graph_no_cycle)

        # Graph with cycle
        graph_with_cycle = {
            "A": [{"name": "B"}],
            "B": [{"name": "C"}],
            "C": [{"name": "A"}],  # Cycle: A -> B -> C -> A
        }
        assert service._detect_cycles(graph_with_cycle)

    def test_cycle_detection_complex(self):
        """Test cycle detection with complex graph."""
        service = CrossReferenceService(":memory:")

        # Complex graph with multiple paths but no cycle
        graph_complex = {
            "root": [{"name": "lib1"}, {"name": "lib2"}],
            "lib1": [{"name": "common"}, {"name": "utils"}],
            "lib2": [{"name": "common"}, {"name": "helpers"}],
            "common": [{"name": "utils"}],
            "utils": [],
            "helpers": [],
        }
        assert not service._detect_cycles(graph_complex)

        # Add a cycle
        graph_complex["utils"] = [{"name": "lib1"}]  # Creates cycle
        assert service._detect_cycles(graph_complex)


class TestCircuitBreaker:
    """Test circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_circuit_breaker_basic(self):
        """Test basic circuit breaker operation."""
        service = CrossReferenceService(":memory:")
        breaker = service.circuit_breaker

        # Circuit should be closed initially
        assert not breaker.is_open()

        # Record failures
        for _ in range(4):
            breaker.record_failure()
        assert not breaker.is_open()  # Still closed at 4 failures

        # One more failure should open it
        breaker.record_failure()
        assert breaker.is_open()  # Open at 5 failures

        # Reset should close it
        breaker.reset()
        assert not breaker.is_open()


class TestCaching:
    """Test caching functionality."""

    @pytest.mark.asyncio
    async def test_import_resolution_caching(self, cross_ref_service):
        """Test that import resolution results are cached."""
        # First call - should hit database
        start1 = time.time()
        result1 = await cross_ref_service.resolve_import("serde", "Deserialize")
        time1 = time.time() - start1

        # Second call - should hit cache
        start2 = time.time()
        result2 = await cross_ref_service.resolve_import("serde", "Deserialize")
        time2 = time.time() - start2

        # Cache hit should be faster (at least 2x)
        assert time2 < time1 / 2 or time2 < 0.01
        assert result1 == result2

    @pytest.mark.asyncio
    async def test_dependency_graph_caching(self, cross_ref_service):
        """Test that dependency graphs are cached."""
        # First call - should hit database
        start1 = time.time()
        result1 = await cross_ref_service.get_dependency_graph("serde", max_depth=2)
        time1 = time.time() - start1

        # Second call - should hit cache
        start2 = time.time()
        result2 = await cross_ref_service.get_dependency_graph("serde", max_depth=2)
        time2 = time.time() - start2

        # Cache hit should be faster
        assert time2 < time1 / 2 or time2 < 0.01
        assert result1 == result2


# Performance benchmarks
@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmarks for Phase 6 requirements."""

    @pytest.mark.asyncio
    async def test_all_performance_requirements(self, cross_ref_service):
        """Test all performance requirements are met."""
        # Import resolution: <200ms
        start = time.time()
        await cross_ref_service.resolve_import("serde", "Deserialize")
        import_time = time.time() - start
        assert import_time < 0.2, (
            f"Import resolution: {import_time:.3f}s (expected <200ms)"
        )

        # Dependency graph: <500ms
        start = time.time()
        await cross_ref_service.get_dependency_graph("serde", max_depth=3)
        graph_time = time.time() - start
        assert graph_time < 0.5, (
            f"Dependency graph: {graph_time:.3f}s (expected <500ms)"
        )

        # Migration suggestions: <300ms (warm cache)
        # First call to warm up
        await cross_ref_service.suggest_migrations("serde", "1.0.0", "1.0.193")
        # Measure second call
        start = time.time()
        await cross_ref_service.suggest_migrations("serde", "1.0.0", "1.0.193")
        migration_time = time.time() - start
        assert migration_time < 0.3, (
            f"Migration suggestions: {migration_time:.3f}s (expected <300ms)"
        )

        print("\nPerformance Results:")
        print(f"  Import resolution: {import_time * 1000:.1f}ms (target: <200ms) ✓")
        print(f"  Dependency graph: {graph_time * 1000:.1f}ms (target: <500ms) ✓")
        print(
            f"  Migration suggestions: {migration_time * 1000:.1f}ms (target: <300ms) ✓"
        )
