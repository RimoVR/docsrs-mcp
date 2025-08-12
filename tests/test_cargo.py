"""Tests for cargo file parsing and version resolution."""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from docsrs_mcp.cargo import (
    fetch_crate_info_cached,
    find_best_version_match,
    parse_cargo_toml,
    parse_cargo_version_spec,
    resolve_cargo_versions,
)


class TestVersionParsing:
    """Test version specification parsing."""

    def test_parse_caret_version(self):
        """Test parsing caret version specifications."""
        assert parse_cargo_version_spec("^1.2.3") == ("caret", "1.2.3")
        assert parse_cargo_version_spec("^0.5.0") == ("caret", "0.5.0")
        assert parse_cargo_version_spec("^ 1.0.0") == ("caret", "1.0.0")

    def test_parse_tilde_version(self):
        """Test parsing tilde version specifications."""
        assert parse_cargo_version_spec("~1.2.3") == ("tilde", "1.2.3")
        assert parse_cargo_version_spec("~0.5") == ("tilde", "0.5")
        assert parse_cargo_version_spec("~ 2.0.0") == ("tilde", "2.0.0")

    def test_parse_wildcard_version(self):
        """Test parsing wildcard version specifications."""
        assert parse_cargo_version_spec("1.*") == ("wildcard", "1.*")
        assert parse_cargo_version_spec("1.2.*") == ("wildcard", "1.2.*")
        assert parse_cargo_version_spec("*") == ("wildcard", "*")

    def test_parse_exact_version(self):
        """Test parsing exact version specifications."""
        assert parse_cargo_version_spec("1.2.3") == ("exact", "1.2.3")
        assert parse_cargo_version_spec("0.1.0") == ("exact", "0.1.0")
        # Test incomplete versions get padded
        assert parse_cargo_version_spec("1.2") == ("exact", "1.2.0")
        assert parse_cargo_version_spec("1") == ("exact", "1.0.0")

    def test_parse_complex_version(self):
        """Test parsing complex version specifications."""
        assert parse_cargo_version_spec(">=1.0, <2.0") == ("complex", ">=1.0, <2.0")
        assert parse_cargo_version_spec(">1.0.0") == ("complex", ">1.0.0")
        assert parse_cargo_version_spec("<=2.5.0") == ("complex", "<=2.5.0")

    def test_parse_unknown_version(self):
        """Test parsing unknown/invalid version specifications."""
        assert parse_cargo_version_spec("invalid-version") == (
            "unknown",
            "invalid-version",
        )
        assert parse_cargo_version_spec("git:something") == ("unknown", "git:something")


class TestVersionMatching:
    """Test version matching logic."""

    def test_find_exact_match(self):
        """Test finding exact version matches."""
        versions = ["1.0.0", "1.2.3", "2.0.0"]
        assert find_best_version_match("exact", "1.2.3", versions) == "1.2.3"
        assert find_best_version_match("exact", "1.0.0", versions) == "1.0.0"
        assert find_best_version_match("exact", "1.5.0", versions) is None

    def test_find_caret_match(self):
        """Test finding caret version matches."""
        versions = ["1.0.0", "1.2.3", "1.5.0", "2.0.0", "2.1.0"]
        # ^1.2.3 should match highest 1.x.x version >= 1.2.3
        assert find_best_version_match("caret", "1.2.3", versions) == "1.5.0"
        # ^1.0.0 should match highest 1.x.x version
        assert find_best_version_match("caret", "1.0.0", versions) == "1.5.0"
        # ^2.0.0 should match highest 2.x.x version
        assert find_best_version_match("caret", "2.0.0", versions) == "2.1.0"

    def test_find_tilde_match(self):
        """Test finding tilde version matches."""
        versions = ["1.2.0", "1.2.3", "1.2.5", "1.3.0", "2.0.0"]
        # ~1.2.3 should match highest 1.2.x version >= 1.2.3
        assert find_best_version_match("tilde", "1.2.3", versions) == "1.2.5"
        # ~1.2.0 should match highest 1.2.x version
        assert find_best_version_match("tilde", "1.2.0", versions) == "1.2.5"

    def test_find_wildcard_match(self):
        """Test finding wildcard version matches."""
        versions = ["1.0.0", "1.2.3", "1.5.0", "2.0.0", "2.1.0"]
        # 1.* should match highest 1.x.x version
        assert find_best_version_match("wildcard", "1.*", versions) == "1.5.0"
        # 2.* should match highest 2.x.x version
        assert find_best_version_match("wildcard", "2.*", versions) == "2.1.0"

    def test_filter_prerelease_versions(self):
        """Test that pre-release versions are filtered out by default."""
        versions = ["1.0.0", "1.1.0-beta", "1.2.0", "1.3.0-alpha", "2.0.0-rc1"]
        # Should only consider stable versions
        assert find_best_version_match("caret", "1.0.0", versions) == "1.2.0"

    def test_fallback_to_latest_stable(self):
        """Test fallback to latest stable version for unknown constraints."""
        versions = ["1.0.0", "1.2.3", "2.0.0", "2.1.0-beta"]
        assert find_best_version_match("unknown", "something", versions) == "2.0.0"
        assert find_best_version_match("complex", ">=1.0", versions) == "2.0.0"


class TestCargoFileParsing:
    """Test Cargo.toml and Cargo.lock parsing."""

    def test_parse_cargo_toml_simple(self):
        """Test parsing simple Cargo.toml."""
        toml_content = """
[dependencies]
serde = "1.0"
tokio = { version = "1.35", features = ["full"] }
rand = "0.8"

[dev-dependencies]
criterion = "0.5"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="Cargo.toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        result = parse_cargo_toml(path)
        crates = result["crates"]

        assert "serde@1.0" in crates
        assert "tokio@1.35" in crates
        assert "rand@0.8" in crates
        assert "criterion@0.5" in crates

        path.unlink()

    def test_parse_cargo_toml_workspace(self):
        """Test parsing Cargo.toml with workspace dependencies."""
        toml_content = """
[workspace.dependencies]
serde = "1.0"
tokio = { version = "1.35" }

[dependencies]
rand = "0.8"
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="Cargo.toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        result = parse_cargo_toml(path)
        crates = result["crates"]

        assert "serde@1.0" in crates
        assert "tokio@1.35" in crates
        assert "rand@0.8" in crates

        path.unlink()

    def test_parse_cargo_toml_no_version(self):
        """Test parsing Cargo.toml with dependencies without versions."""
        toml_content = """
[dependencies]
my-crate = { path = "../my-crate" }
another-crate = { git = "https://github.com/example/repo" }
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix="Cargo.toml", delete=False
        ) as f:
            f.write(toml_content)
            f.flush()
            path = Path(f.name)

        result = parse_cargo_toml(path)
        crates = result["crates"]

        assert "my-crate@latest" in crates
        assert "another-crate@latest" in crates

        path.unlink()


@pytest.mark.asyncio
class TestVersionResolution:
    """Test async version resolution."""

    async def test_resolve_versions_disabled(self):
        """Test that resolution is skipped when disabled."""
        crates = ["serde@^1.0", "tokio@~1.35"]
        session = AsyncMock()

        result = await resolve_cargo_versions(crates, session, resolve=False)
        assert result == crates
        session.get.assert_not_called()

    async def test_resolve_versions_with_cache(self):
        """Test version resolution with caching."""
        crates = ["serde@^1.0", "tokio@1.35.0"]
        session = AsyncMock()

        # Mock crate info responses (simplified resolution uses max_stable_version)
        mock_crate_info = {
            "serde": {
                "max_stable_version": "1.0.197"
            },
            "tokio": {"max_stable_version": "1.35.1"},
        }

        with patch(
            "docsrs_mcp.cargo.fetch_crate_info_cached",
            side_effect=lambda name, _: mock_crate_info.get(name),
        ):
            result = await resolve_cargo_versions(crates, session, resolve=True)

            # Caret version should resolve to max_stable_version
            assert "serde@1.0.197" in result
            # Exact version should also resolve to max_stable_version in simplified approach
            assert "tokio@1.35.1" in result

    async def test_resolve_versions_fallback(self):
        """Test fallback when resolution fails."""
        crates = ["unknown-crate@^1.0"]
        session = AsyncMock()

        with patch("docsrs_mcp.cargo.fetch_crate_info_cached", return_value=None):
            result = await resolve_cargo_versions(crates, session, resolve=True)
            # Should fall back to original spec
            assert result == ["unknown-crate@^1.0"]

    async def test_resolve_versions_latest(self):
        """Test that 'latest' version is preserved."""
        crates = ["serde@latest", "tokio@latest"]
        session = AsyncMock()

        result = await resolve_cargo_versions(crates, session, resolve=True)
        assert result == crates

    async def test_resolve_versions_no_at_sign(self):
        """Test handling crates without version specs."""
        crates = ["serde", "tokio"]
        session = AsyncMock()

        result = await resolve_cargo_versions(crates, session, resolve=True)
        assert result == ["serde@latest", "tokio@latest"]


@pytest.mark.asyncio
async def test_fetch_crate_info_cached():
    """Test crate info caching."""
    session = AsyncMock()
    mock_info = {"name": "serde", "versions": [{"num": "1.0.197"}]}

    with patch("docsrs_mcp.ingest.fetch_crate_info", return_value=mock_info):
        # First call should fetch from API
        result1 = await fetch_crate_info_cached("serde", session)
        assert result1 == mock_info

        # Second call should use cache
        with patch("docsrs_mcp.ingest.fetch_crate_info") as mock_fetch:
            result2 = await fetch_crate_info_cached("serde", session)
            assert result2 == mock_info
            mock_fetch.assert_not_called()
