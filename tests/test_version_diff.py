"""Tests for version diff functionality."""

from unittest.mock import MagicMock, patch

import pytest

from docsrs_mcp.change_analyzer import RustBreakingChangeDetector
from docsrs_mcp.models import (
    ChangeCategory,
    ChangeType,
    CompareVersionsRequest,
    ItemChange,
    ItemKind,
    Severity,
)
from docsrs_mcp.version_diff import VersionDiffEngine


@pytest.fixture
def diff_engine():
    """Create a diff engine instance for testing."""
    return VersionDiffEngine()


@pytest.fixture
def breaking_detector():
    """Create a breaking change detector for testing."""
    return RustBreakingChangeDetector()


@pytest.fixture
def sample_items_v1():
    """Sample items for version 1."""
    return {
        "tokio::spawn": {
            "item_path": "tokio::spawn",
            "item_type": "function",
            "signature": "pub fn spawn<F>(future: F) -> JoinHandle<F::Output>",
            "visibility": "public",
            "deprecated": False,
            "generic_params": "<F>",
            "trait_bounds": "F: Future + Send + 'static",
        },
        "tokio::sync::Mutex": {
            "item_path": "tokio::sync::Mutex",
            "item_type": "struct",
            "signature": "pub struct Mutex<T>",
            "visibility": "public",
            "deprecated": False,
            "generic_params": "<T>",
            "trait_bounds": None,
        },
        "tokio::old_function": {
            "item_path": "tokio::old_function",
            "item_type": "function",
            "signature": "pub fn old_function()",
            "visibility": "public",
            "deprecated": False,
            "generic_params": None,
            "trait_bounds": None,
        },
    }


@pytest.fixture
def sample_items_v2():
    """Sample items for version 2."""
    return {
        "tokio::spawn": {
            "item_path": "tokio::spawn",
            "item_type": "function",
            "signature": "pub fn spawn<F>(future: F) -> JoinHandle<F::Output>",
            "visibility": "public",
            "deprecated": False,
            "generic_params": "<F>",
            "trait_bounds": "F: Future + Send + 'static + Unpin",  # Changed
        },
        "tokio::sync::Mutex": {
            "item_path": "tokio::sync::Mutex",
            "item_type": "struct",
            "signature": "pub struct Mutex<T>",
            "visibility": "public",
            "deprecated": True,  # Now deprecated
            "generic_params": "<T>",
            "trait_bounds": None,
        },
        "tokio::new_function": {
            "item_path": "tokio::new_function",
            "item_type": "function",
            "signature": "pub async fn new_function()",
            "visibility": "public",
            "deprecated": False,
            "generic_params": None,
            "trait_bounds": None,
        },
    }


@pytest.mark.asyncio
async def test_diff_engine_initialization(diff_engine):
    """Test that diff engine initializes correctly."""
    assert diff_engine is not None
    assert diff_engine.cache is not None
    assert diff_engine.cache.max_size == 100


@pytest.mark.asyncio
async def test_diff_cache_operations(diff_engine):
    """Test cache operations."""
    # Test cache miss
    result = await diff_engine.cache.get("test_crate", "1.0.0", "1.0.1")
    assert result is None

    # Test cache key generation
    key1 = diff_engine.cache._make_key("test_crate", "1.0.0", "1.0.1")
    key2 = diff_engine.cache._make_key("test_crate", "1.0.1", "1.0.0")
    assert key1 == key2  # Should be same regardless of version order


@pytest.mark.asyncio
@patch("docsrs_mcp.version_diff.ingest_crate")
@patch("docsrs_mcp.version_diff.get_all_items_for_version")
@patch("docsrs_mcp.version_diff.compute_item_hash")
async def test_compare_versions_basic(
    mock_compute_hash,
    mock_get_items,
    mock_ingest,
    diff_engine,
    sample_items_v1,
    sample_items_v2,
):
    """Test basic version comparison."""
    # Setup mocks
    mock_ingest.side_effect = ["/path/to/v1.db", "/path/to/v2.db"]
    mock_get_items.side_effect = [sample_items_v1, sample_items_v2]

    # Mock hash computation to detect changes
    def compute_hash_side_effect(item):
        # Make spawn and Mutex have different hashes between versions
        path = item.get("item_path", "")
        if path == "tokio::spawn":
            return (
                "hash_spawn_v2"
                if item.get("trait_bounds", "").endswith("Unpin")
                else "hash_spawn_v1"
            )
        elif path == "tokio::sync::Mutex":
            return "hash_mutex_v2" if item.get("deprecated") else "hash_mutex_v1"
        else:
            return f"hash_{path}"

    mock_compute_hash.side_effect = compute_hash_side_effect

    # Create request
    request = CompareVersionsRequest(
        crate_name="tokio",
        version_a="1.0.0",
        version_b="2.0.0",
        categories=[
            ChangeCategory.ADDED,
            ChangeCategory.REMOVED,
            ChangeCategory.MODIFIED,
        ],
    )

    # Run comparison
    result = await diff_engine.compare_versions(request)

    # Verify results
    assert result.crate_name == "tokio"
    assert result.version_a == "1.0.0"
    assert result.version_b == "2.0.0"
    assert result.summary.total_changes > 0
    assert result.summary.added_items == 1  # new_function
    assert result.summary.removed_items == 1  # old_function
    assert result.summary.modified_items == 2  # spawn and Mutex


def test_breaking_change_detector_removed_item(breaking_detector):
    """Test detection of removed public items as breaking."""
    change = ItemChange(
        path="test::removed_fn",
        kind=ItemKind.FUNCTION,
        change_type=ChangeType.REMOVED,
        severity=Severity.BREAKING,
        details=MagicMock(
            before=MagicMock(visibility="public"),
            after=None,
            semantic_changes=["Item removed from API"],
        ),
    )

    is_breaking, reasons = breaking_detector.analyze_change(change)
    assert is_breaking
    assert "Public item removed from API" in reasons[0]


def test_breaking_change_detector_signature_change(breaking_detector):
    """Test detection of signature changes as breaking."""
    change = ItemChange(
        path="test::changed_fn",
        kind=ItemKind.FUNCTION,
        change_type=ChangeType.MODIFIED,
        severity=Severity.BREAKING,
        details=MagicMock(
            before=MagicMock(
                raw_signature="fn foo(x: i32)",
                visibility="public",
            ),
            after=MagicMock(
                raw_signature="fn foo(x: i32, y: i32)",
                visibility="public",
            ),
            semantic_changes=["Function signature changed"],
        ),
    )

    is_breaking, reasons = breaking_detector.analyze_change(change)
    assert is_breaking
    assert any("signature" in reason.lower() for reason in reasons)


def test_breaking_change_detector_visibility_change(breaking_detector):
    """Test detection of visibility reduction as breaking."""
    change = ItemChange(
        path="test::now_private",
        kind=ItemKind.STRUCT,
        change_type=ChangeType.MODIFIED,
        severity=Severity.BREAKING,
        details=MagicMock(
            before=MagicMock(visibility="public"),
            after=MagicMock(visibility="private"),
            semantic_changes=["Changed from public to private"],
        ),
    )

    is_breaking, reasons = breaking_detector.analyze_change(change)
    assert is_breaking
    assert any("visibility" in reason.lower() for reason in reasons)


def test_extract_parameters():
    """Test parameter extraction from function signatures."""
    detector = RustBreakingChangeDetector()

    # Test simple function
    params = detector._extract_parameters("fn foo(x: i32, y: String)")
    assert params == ["x: i32", "y: String"]

    # Test generic function
    params = detector._extract_parameters("fn bar<T>(items: Vec<T>)")
    assert params == ["items: Vec<T>"]

    # Test no parameters
    params = detector._extract_parameters("fn baz()")
    assert params == []

    # Test complex nested types
    params = detector._extract_parameters(
        "fn complex(map: HashMap<String, Vec<(i32, i32)>>)"
    )
    assert params == ["map: HashMap<String, Vec<(i32, i32)>>"]


def test_extract_return_type():
    """Test return type extraction from function signatures."""
    detector = RustBreakingChangeDetector()

    # Test simple return
    ret = detector._extract_return_type("fn foo() -> i32")
    assert ret == "i32"

    # Test generic return
    ret = detector._extract_return_type("fn bar<T>() -> Vec<T>")
    assert ret == "Vec<T>"

    # Test no return (unit type)
    ret = detector._extract_return_type("fn baz()")
    assert ret is None

    # Test with where clause
    ret = detector._extract_return_type(
        "fn complex() -> Result<String, Error> where T: Clone"
    )
    assert ret == "Result<String, Error>"
