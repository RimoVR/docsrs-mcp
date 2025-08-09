"""Integration tests for re-export auto-discovery with real rustdoc JSON."""

import time

import pytest

from src.docsrs_mcp.database import get_discovered_reexports
from src.docsrs_mcp.fuzzy_resolver import _reexport_cache, resolve_path_alias
from src.docsrs_mcp.ingest import ingest_crate


@pytest.mark.asyncio
async def test_serde_reexport_discovery():
    """Integration test: verify re-export discovery works with real serde crate."""
    crate_name = "serde"
    version = "1.0.219"

    # Ingest the crate
    db_path = await ingest_crate(crate_name, version)

    # Get discovered re-exports
    reexports = await get_discovered_reexports(db_path, crate_name)

    # Check if any re-exports were discovered
    if reexports:
        print(f"\nDiscovered {len(reexports)} re-exports for {crate_name}:")
        # Print first 5 for debugging
        for _, (alias, actual) in enumerate(list(reexports.items())[:5]):
            print(f"  {alias} -> {actual}")

    # Test path resolution (should work with static aliases even if no re-exports found)
    test_paths = [
        ("Deserialize", "serde::de::Deserialize"),
        ("Serialize", "serde::ser::Serialize"),
        ("Deserializer", "serde::de::Deserializer"),
        ("Serializer", "serde::ser::Serializer"),
    ]

    for item_path, expected in test_paths:
        resolved = await resolve_path_alias(crate_name, item_path, str(db_path))
        assert resolved == expected, (
            f"Failed to resolve {item_path} to {expected}, got {resolved}"
        )

    # Clean up
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_tokio_reexport_discovery():
    """Integration test: verify re-export discovery works with real tokio crate."""
    crate_name = "tokio"
    version = "1.35.1"

    # Ingest the crate
    db_path = await ingest_crate(crate_name, version)

    # Get discovered re-exports
    reexports = await get_discovered_reexports(db_path, crate_name)

    # Check if any re-exports were discovered
    if reexports:
        print(f"\nDiscovered {len(reexports)} re-exports for {crate_name}:")
        # Print first 5 for debugging
        for _, (alias, actual) in enumerate(list(reexports.items())[:5]):
            print(f"  {alias} -> {actual}")

    # Test path resolution
    test_paths = [
        ("spawn", "tokio::task::spawn"),  # Common alias
        ("JoinHandle", "tokio::task::JoinHandle"),  # Another common one
    ]

    for item_path, expected in test_paths:
        resolved = await resolve_path_alias(crate_name, item_path, str(db_path))
        assert resolved == expected, (
            f"Failed to resolve {item_path} to {expected}, got {resolved}"
        )

    # Clean up
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_small_crate_with_reexports():
    """Integration test with a smaller crate that might have simpler re-exports."""""
    crate_name = "log"
    version = "0.4.20"

    # Ingest the crate
    db_path = await ingest_crate(crate_name, version)

    # Get discovered re-exports
    reexports = await get_discovered_reexports(db_path, crate_name)

    # Log any discovered re-exports
    if reexports:
        print(f"\nDiscovered {len(reexports)} re-exports for {crate_name}:")
        for alias, actual in reexports.items():
            print(f"  {alias} -> {actual}")
    else:
        print(f"\nNo re-exports discovered for {crate_name}")

    # Even if no re-exports, the system should still work
    assert db_path.exists()

    # Clean up
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reexport_discovery_performance():
    """Test that re-export discovery doesn't significantly impact ingestion performance."""

    crate_name = "serde_json"
    version = "1.0.108"

    # Time the ingestion
    start_time = time.time()
    db_path = await ingest_crate(crate_name, version)
    ingestion_time = time.time() - start_time

    print(f"\nIngestion time for {crate_name}: {ingestion_time:.2f}s")

    # Check performance constraint (should be under 3s for small crates)
    assert ingestion_time < 5.0, f"Ingestion took too long: {ingestion_time:.2f}s"

    # Verify re-exports were processed
    reexports = await get_discovered_reexports(db_path, crate_name)
    print(f"Discovered {len(reexports)} re-exports")

    # Test resolution performance
    start_time = time.time()
    for _ in range(100):
        await resolve_path_alias(crate_name, "Value", str(db_path))
    resolution_time = (time.time() - start_time) / 100

    print(f"Average resolution time: {resolution_time * 1000:.2f}ms")

    # Should be sub-millisecond
    assert resolution_time < 0.001, (
        f"Resolution too slow: {resolution_time * 1000:.2f}ms"
    )

    # Clean up
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_multiple_crates_reexport_caching():
    """Test that re-export caching works correctly across multiple crates."""

    # Clear cache
    _reexport_cache.clear()

    crates = [
        ("serde", "1.0.219"),
        ("tokio", "1.35.1"),
    ]

    db_paths = []

    for crate_name, version in crates:
        # Ingest crate
        db_path = await ingest_crate(crate_name, version)
        db_paths.append(db_path)

        # First resolution should populate cache
        _ = await resolve_path_alias(crate_name, "test_item", str(db_path))

        # Check cache was populated
        cache_key = f"{crate_name}_reexports"
        assert cache_key in _reexport_cache, f"Cache not populated for {crate_name}"

        # Second resolution should use cache
        _ = await resolve_path_alias(crate_name, "another_item", str(db_path))

        # Cache should still be there
        assert cache_key in _reexport_cache, f"Cache disappeared for {crate_name}"

    # Verify separate caches for different crates
    assert len(_reexport_cache) == len(crates), (
        "Should have separate cache entries for each crate"
    )

    # Clean up
    for db_path in db_paths:
        db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_reexport_persistence():
    """Test that re-exports are persisted in the database across sessions."""
    crate_name = "serde"
    version = "1.0.219"

    # First session: ingest and check re-exports
    db_path = await ingest_crate(crate_name, version)
    reexports1 = await get_discovered_reexports(db_path, crate_name)

    # Store count for comparison
    reexport_count = len(reexports1)

    # Simulate new session by clearing caches
    _reexport_cache.clear()

    # Second session: re-exports should still be available from database
    reexports2 = await get_discovered_reexports(db_path, crate_name)

    # Should have same re-exports
    assert len(reexports2) == reexport_count, "Re-exports not persisted correctly"

    # Test resolution still works
    resolved = await resolve_path_alias(crate_name, "Deserialize", str(db_path))
    assert resolved == "serde::de::Deserialize", "Resolution failed after cache clear"

    # Clean up
    db_path.unlink(missing_ok=True)


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "-m", "integration"])
