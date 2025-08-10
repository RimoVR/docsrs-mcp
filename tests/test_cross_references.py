"""Unit tests for cross-reference extraction and resolution."""

import json
import tempfile
from pathlib import Path

import pytest

from docsrs_mcp.database import (
    get_cross_references,
    get_discovered_reexports,
    init_database,
    migrate_reexports_for_crossrefs,
    store_reexports,
)
from docsrs_mcp.ingest import parse_rustdoc_items_streaming


@pytest.fixture
async def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    await init_database(db_path)
    await migrate_reexports_for_crossrefs(db_path)

    yield db_path

    # Cleanup
    try:
        db_path.unlink()
    except:
        pass


def create_test_rustdoc_json(with_links=True):
    """Create a test rustdoc JSON structure."""
    rustdoc = {
        "root": "0:0",
        "paths": {
            "0:0": {"path": ["test_crate"], "kind": "module"},
            "0:1": {"path": ["test_crate", "foo"], "kind": "function"},
            "0:2": {"path": ["test_crate", "Bar"], "kind": "struct"},
            "0:3": {"path": ["test_crate", "Baz"], "kind": "trait"},
        },
        "index": {
            "0:0": {
                "name": "test_crate",
                "docs": "Test crate documentation",
                "inner": {"Module": {}},
            },
            "0:1": {
                "name": "foo",
                "docs": "Function that uses [`Bar`] and implements [`Baz`]",
                "inner": {"Function": {}},
                "links": {"Bar": "0:2", "Baz": "0:3"} if with_links else {},
            },
            "0:2": {
                "name": "Bar",
                "docs": "A struct that implements [`Baz`]",
                "inner": {"Struct": {}},
                "links": {"Baz": "0:3"} if with_links else {},
            },
            "0:3": {
                "name": "Baz",
                "docs": "A trait",
                "inner": {"Trait": {}},
                "links": {} if with_links else {},
            },
        },
    }
    return json.dumps(rustdoc)


@pytest.mark.asyncio
async def test_parse_rustdoc_with_links():
    """Test that links field is extracted during parsing."""
    json_content = create_test_rustdoc_json(with_links=True)

    items = []
    crossrefs = []

    async for item in parse_rustdoc_items_streaming(json_content):
        if "_crossref" in item:
            crossrefs.append(item["_crossref"])
        elif "_modules" not in item and "_reexport" not in item:
            items.append(item)

    # Should have extracted cross-references
    assert len(crossrefs) > 0

    # Check specific cross-references
    foo_to_bar = any(
        cr["source_path"] == "test_crate::foo"
        and cr["target_path"] == "test_crate::Bar"
        and cr["link_text"] == "Bar"
        for cr in crossrefs
    )
    assert foo_to_bar, "Should have cross-reference from foo to Bar"

    foo_to_baz = any(
        cr["source_path"] == "test_crate::foo"
        and cr["target_path"] == "test_crate::Baz"
        and cr["link_text"] == "Baz"
        for cr in crossrefs
    )
    assert foo_to_baz, "Should have cross-reference from foo to Baz"

    bar_to_baz = any(
        cr["source_path"] == "test_crate::Bar"
        and cr["target_path"] == "test_crate::Baz"
        and cr["link_text"] == "Baz"
        for cr in crossrefs
    )
    assert bar_to_baz, "Should have cross-reference from Bar to Baz"


@pytest.mark.asyncio
async def test_parse_rustdoc_without_links():
    """Test parsing works when links field is missing."""
    json_content = create_test_rustdoc_json(with_links=False)

    items = []
    crossrefs = []

    async for item in parse_rustdoc_items_streaming(json_content):
        if "_crossref" in item:
            crossrefs.append(item["_crossref"])
        elif "_modules" not in item and "_reexport" not in item:
            items.append(item)

    # Should have no cross-references
    assert len(crossrefs) == 0

    # But should still have items
    assert len(items) > 0


@pytest.mark.asyncio
async def test_store_and_retrieve_crossrefs(temp_db):
    """Test storing and retrieving cross-references."""
    crate_id = 1

    # Store some cross-references
    crossrefs = [
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Bar",
            "is_glob": False,
            "link_text": "Bar",
            "link_type": "crossref",
            "target_item_id": "0:2",
            "confidence_score": 1.0,
        },
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Baz",
            "is_glob": False,
            "link_text": "Baz",
            "link_type": "crossref",
            "target_item_id": "0:3",
            "confidence_score": 1.0,
        },
    ]

    await store_reexports(temp_db, crate_id, crossrefs)

    # Retrieve cross-references for foo
    cross_refs = await get_cross_references(
        temp_db, "test_crate::foo", direction="from"
    )

    assert "from" in cross_refs
    assert len(cross_refs["from"]) == 2

    # Check specific references
    bar_ref = next(
        (r for r in cross_refs["from"] if r["target_path"] == "test_crate::Bar"), None
    )
    assert bar_ref is not None
    assert bar_ref["link_text"] == "Bar"
    assert bar_ref["confidence"] == 1.0


@pytest.mark.asyncio
async def test_bidirectional_crossrefs(temp_db):
    """Test bidirectional cross-reference retrieval."""
    crate_id = 1

    # Store cross-references
    crossrefs = [
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Bar",
            "is_glob": False,
            "link_text": "Bar",
            "link_type": "crossref",
        },
        {
            "alias_path": "test_crate::Baz",
            "actual_path": "test_crate::Bar",
            "is_glob": False,
            "link_text": "Bar",
            "link_type": "crossref",
        },
    ]

    await store_reexports(temp_db, crate_id, crossrefs)

    # Get incoming references to Bar
    cross_refs = await get_cross_references(temp_db, "test_crate::Bar", direction="to")

    assert "to" in cross_refs
    assert len(cross_refs["to"]) == 2

    # Check sources
    sources = {r["source_path"] for r in cross_refs["to"]}
    assert "test_crate::foo" in sources
    assert "test_crate::Baz" in sources


@pytest.mark.asyncio
async def test_mixed_reexports_and_crossrefs(temp_db):
    """Test that regular reexports and cross-references can coexist."""
    # First, create a crate metadata entry
    import aiosqlite

    async with aiosqlite.connect(temp_db) as db:
        await db.execute(
            """INSERT INTO crate_metadata (name, version, description) 
               VALUES (?, ?, ?)""",
            ("test_crate", "1.0.0", "Test crate"),
        )
        await db.commit()
        crate_id = 1

    # Store mixed entries
    entries = [
        {
            "alias_path": "test_crate::prelude::Vec",
            "actual_path": "alloc::vec::Vec",
            "is_glob": False,
            "link_type": "reexport",  # Regular re-export
        },
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Bar",
            "is_glob": False,
            "link_text": "Bar",
            "link_type": "crossref",  # Cross-reference
        },
    ]

    await store_reexports(temp_db, crate_id, entries)

    # Get reexports only (no cross-refs)
    reexports = await get_discovered_reexports(
        temp_db, "test_crate", include_crossrefs=False
    )

    assert "test_crate::prelude::Vec" in reexports
    assert "test_crate::foo" not in reexports  # Cross-ref should be excluded

    # Get all mappings including cross-refs
    all_mappings = await get_discovered_reexports(
        temp_db, "test_crate", include_crossrefs=True
    )

    assert "test_crate::prelude::Vec" in all_mappings
    assert "test_crate::foo" in all_mappings  # Cross-ref should be included


@pytest.mark.asyncio
async def test_crossref_confidence_scores(temp_db):
    """Test that confidence scores are properly stored and retrieved."""
    crate_id = 1

    # Store cross-references with varying confidence
    crossrefs = [
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Bar",
            "is_glob": False,
            "link_text": "Bar",
            "link_type": "crossref",
            "confidence_score": 0.9,
        },
        {
            "alias_path": "test_crate::foo",
            "actual_path": "test_crate::Baz",
            "is_glob": False,
            "link_text": "Baz",
            "link_type": "crossref",
            "confidence_score": 0.7,
        },
    ]

    await store_reexports(temp_db, crate_id, crossrefs)

    # Retrieve and check confidence scores
    cross_refs = await get_cross_references(
        temp_db, "test_crate::foo", direction="from"
    )

    assert "from" in cross_refs

    bar_ref = next(
        (r for r in cross_refs["from"] if r["target_path"] == "test_crate::Bar"), None
    )
    assert bar_ref["confidence"] == 0.9

    baz_ref = next(
        (r for r in cross_refs["from"] if r["target_path"] == "test_crate::Baz"), None
    )
    assert baz_ref["confidence"] == 0.7
