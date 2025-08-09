"""Unit tests for re-export auto-discovery functionality."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio

from src.docsrs_mcp.database import (
    get_discovered_reexports,
    init_database,
    store_crate_metadata,
    store_reexports,
)
from src.docsrs_mcp.fuzzy_resolver import resolve_path_alias
from src.docsrs_mcp.ingest import parse_rustdoc_items_streaming


@pytest_asyncio.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    # Initialize database
    await init_database(db_path)

    # Store test crate metadata
    crate_id = await store_crate_metadata(
        db_path,
        "test_crate",
        "1.0.0",
        "Test crate for unit tests",
        None,
        None,
    )

    yield db_path, crate_id

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_store_and_retrieve_reexports(temp_db):
    """Test storing and retrieving re-export mappings."""
    db_path, crate_id = temp_db

    # Test data
    reexports = [
        {
            "alias_path": "test_crate::Serialize",
            "actual_path": "test_crate::ser::Serialize",
            "is_glob": False,
        },
        {
            "alias_path": "test_crate::Deserialize",
            "actual_path": "test_crate::de::Deserialize",
            "is_glob": False,
        },
        {
            "alias_path": "test_crate::prelude::*",
            "actual_path": "test_crate::internal::prelude",
            "is_glob": True,
        },
    ]

    # Store re-exports
    await store_reexports(db_path, crate_id, reexports)

    # Retrieve re-exports
    retrieved = await get_discovered_reexports(db_path, "test_crate")

    # Verify all mappings are retrieved
    assert len(retrieved) >= 2  # At least the non-glob mappings
    assert retrieved["test_crate::Serialize"] == "test_crate::ser::Serialize"
    assert retrieved["test_crate::Deserialize"] == "test_crate::de::Deserialize"

    # Verify short aliases are also created
    assert "Serialize" in retrieved
    assert retrieved["Serialize"] == "test_crate::ser::Serialize"


@pytest.mark.asyncio
async def test_parse_rustdoc_reexports():
    """Test parsing re-exports from rustdoc JSON."""
    # Create mock rustdoc JSON with re-export items
    rustdoc_json = {
        "paths": {
            "1:2:3": {"path": ["test_crate"], "kind": "module"},
            "1:2:4": {"path": ["test_crate", "ser"], "kind": "module"},
            "1:2:5": {"path": ["test_crate", "de"], "kind": "module"},
        },
        "index": {
            # Regular item
            "1:2:3": {
                "name": "test_function",
                "docs": "A test function",
                "inner": {"function": {}},
            },
            # Re-export item (Use variant)
            "1:2:6": {
                "name": "Serialize",
                "docs": "",
                "inner": {
                    "use": {
                        "source": "test_crate::ser::Serialize",
                        "is_glob": False,
                    }
                },
            },
            # Another re-export
            "1:2:7": {
                "name": "Deserialize",
                "docs": "",
                "inner": {
                    "Use": {  # Capital U variant
                        "source": "test_crate::de::Deserialize",
                        "is_glob": False,
                    }
                },
            },
            # Glob re-export
            "1:2:8": {
                "name": "*",
                "docs": "",
                "inner": {
                    "use": {
                        "source": "test_crate::prelude",
                        "is_glob": True,
                    }
                },
            },
        },
    }

    json_content = json.dumps(rustdoc_json)

    # Parse the JSON
    items = []
    reexports = []
    modules = {}

    async for item in parse_rustdoc_items_streaming(json_content):
        if "_modules" in item:
            _ = item["_modules"]
        elif "_reexport" in item:
            reexports.append(item["_reexport"])
        else:
            items.append(item)

    # Verify re-exports were extracted
    assert len(reexports) == 3

    # Verify re-export mappings (alias paths may not include crate prefix)
    reexport_dict = {r["alias_path"]: r["actual_path"] for r in reexports}
    # Check if either with or without crate prefix
    assert "Serialize" in reexport_dict or "test_crate::Serialize" in reexport_dict
    if "Serialize" in reexport_dict:
        assert reexport_dict["Serialize"] == "test_crate::ser::Serialize"
    else:
        assert reexport_dict["test_crate::Serialize"] == "test_crate::ser::Serialize"

    assert "Deserialize" in reexport_dict or "test_crate::Deserialize" in reexport_dict
    if "Deserialize" in reexport_dict:
        assert reexport_dict["Deserialize"] == "test_crate::de::Deserialize"
    else:
        assert reexport_dict["test_crate::Deserialize"] == "test_crate::de::Deserialize"

    # Verify glob re-export
    glob_reexports = [r for r in reexports if r["is_glob"]]
    assert len(glob_reexports) == 1
    assert glob_reexports[0]["actual_path"] == "test_crate::prelude"


@pytest.mark.asyncio
async def test_resolve_path_alias_with_discovered_reexports(temp_db):
    """Test path alias resolution with discovered re-exports."""
    db_path, crate_id = temp_db

    # Store some re-exports
    reexports = [
        {
            "alias_path": "test_crate::MyTrait",
            "actual_path": "test_crate::traits::MyTrait",
            "is_glob": False,
        },
        {
            "alias_path": "test_crate::MyStruct",
            "actual_path": "test_crate::types::MyStruct",
            "is_glob": False,
        },
    ]
    await store_reexports(db_path, crate_id, reexports)

    # Test resolution with discovered re-exports
    resolved = await resolve_path_alias("test_crate", "MyTrait", str(db_path))
    assert resolved == "test_crate::traits::MyTrait"

    resolved = await resolve_path_alias("test_crate", "MyStruct", str(db_path))
    assert resolved == "test_crate::types::MyStruct"

    # Test with crate prefix
    resolved = await resolve_path_alias(
        "test_crate", "test_crate::MyTrait", str(db_path)
    )
    assert resolved == "test_crate::traits::MyTrait"

    # Test fallback to static aliases (should still work)
    resolved = await resolve_path_alias("serde", "Deserialize", None)
    assert resolved == "serde::de::Deserialize"

    # Test non-existent path (should return original)
    resolved = await resolve_path_alias("test_crate", "NonExistent", str(db_path))
    assert resolved == "NonExistent"


@pytest.mark.asyncio
async def test_reexport_cache_behavior():
    """Test caching behavior for discovered re-exports."""

    from src.docsrs_mcp.fuzzy_resolver import REEXPORT_CACHE_TTL, _reexport_cache

    # Clear cache
    _reexport_cache.clear()

    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    await init_database(db_path)
    crate_id = await store_crate_metadata(
        db_path, "cache_test", "1.0.0", "Cache test", None, None
    )

    # Store re-exports
    reexports = [
        {
            "alias_path": "cache_test::Item",
            "actual_path": "cache_test::module::Item",
            "is_glob": False,
        }
    ]
    await store_reexports(db_path, crate_id, reexports)

    # First call should query database
    resolved1 = await resolve_path_alias("cache_test", "Item", str(db_path))
    assert resolved1 == "cache_test::module::Item"

    # Verify cache was populated
    assert "cache_test_reexports" in _reexport_cache

    # Second call should use cache (mock database to verify)
    with patch("src.docsrs_mcp.database.get_discovered_reexports") as mock_get:
        mock_get.return_value = {}  # Return empty to verify cache is used
        resolved2 = await resolve_path_alias("cache_test", "Item", str(db_path))
        assert resolved2 == "cache_test::module::Item"  # Should still work from cache
        mock_get.assert_not_called()  # Should not call database

    # Simulate cache expiration
    cache_key = "cache_test_reexports"
    timestamp, cached_data = _reexport_cache[cache_key]
    _reexport_cache[cache_key] = (timestamp - REEXPORT_CACHE_TTL - 1, cached_data)

    # Next call should query database again
    resolved3 = await resolve_path_alias("cache_test", "Item", str(db_path))
    assert resolved3 == "cache_test::module::Item"

    # Cleanup
    db_path.unlink(missing_ok=True)


@pytest.mark.asyncio
async def test_empty_reexports_handling(temp_db):
    """Test handling of empty re-exports."""
    db_path, crate_id = temp_db

    # Store empty re-exports list
    await store_reexports(db_path, crate_id, [])

    # Should return empty dict
    reexports = await get_discovered_reexports(db_path, "test_crate")
    assert reexports == {}

    # Resolution should fall back to static aliases
    resolved = await resolve_path_alias("test_crate", "SomeItem", str(db_path))
    assert resolved == "SomeItem"  # Returns original when not found


@pytest.mark.asyncio
async def test_duplicate_reexports(temp_db):
    """Test handling of duplicate re-export mappings."""
    db_path, crate_id = temp_db

    # Try to store duplicate re-exports
    reexports = [
        {
            "alias_path": "test_crate::Item",
            "actual_path": "test_crate::first::Item",
            "is_glob": False,
        },
        {
            "alias_path": "test_crate::Item",  # Duplicate alias
            "actual_path": "test_crate::second::Item",
            "is_glob": False,
        },
    ]

    # Should handle duplicates gracefully (UNIQUE constraint with IGNORE)
    await store_reexports(db_path, crate_id, reexports)

    # Should only have first mapping
    retrieved = await get_discovered_reexports(db_path, "test_crate")
    assert retrieved["test_crate::Item"] == "test_crate::first::Item"


@pytest.mark.asyncio
async def test_database_error_handling():
    """Test error handling when database operations fail."""
    # Use non-existent database path
    bad_db_path = Path("/nonexistent/path/test.db")

    # Should handle gracefully and return empty dict
    reexports = await get_discovered_reexports(bad_db_path, "test_crate")
    assert reexports == {}

    # Resolution should fall back to static aliases
    resolved = await resolve_path_alias("serde", "Serialize", str(bad_db_path))
    assert resolved == "serde::ser::Serialize"  # Static alias still works


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])
