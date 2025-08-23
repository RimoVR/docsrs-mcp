"""
Rustup toolchain detector for docsrs-mcp.

This module provides cross-platform detection of rustup installations
and available Rust toolchains, with caching and error handling.
"""

import logging
import os
import platform
import subprocess
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)


class RustupDetector:
    """Detect and manage rustup toolchain information."""

    def __init__(self):
        """Initialize the rustup detector."""
        self._rustup_home: Path | None = None
        self._toolchains: list[str] = []
        self._detected = False

    @lru_cache(maxsize=1)
    def detect_rustup_home(self) -> Path | None:
        """
        Detect rustup installation with cross-platform support.

        Returns:
            Path to rustup home directory or None if not found
        """
        if self._rustup_home:
            return self._rustup_home

        # Check environment variables first (highest priority)
        if rustup_home := os.environ.get("RUSTUP_HOME"):
            path = Path(rustup_home)
            if self._validate_rustup_home(path):
                logger.info(f"Found rustup via RUSTUP_HOME environment: {path}")
                self._rustup_home = path
                return path

        # Platform-specific detection
        home = Path.home()

        # Common installation paths
        candidates = [
            home / ".rustup",  # Unix/Linux/macOS standard
            home / ".cargo" / ".." / ".rustup",  # Alternative via cargo
        ]

        # Windows-specific paths
        if platform.system() == "Windows":
            candidates.extend(
                [
                    Path(os.environ.get("USERPROFILE", "")) / ".rustup",
                    Path(os.environ.get("LOCALAPPDATA", "")) / "rustup",
                ]
            )

        # WSL detection
        if "microsoft" in platform.uname().release.lower():
            # Running in WSL, check Windows paths via /mnt/c
            windows_home = Path("/mnt/c/Users") / os.environ.get("USER", "")
            if windows_home.exists():
                candidates.append(windows_home / ".rustup")

        # Check each candidate path
        for path in candidates:
            if self._validate_rustup_home(path):
                logger.info(f"Found rustup at: {path}")
                self._rustup_home = path
                return path

        logger.warning("Rustup installation not detected")
        return None

    def _validate_rustup_home(self, path: Path) -> bool:
        """
        Validate that a path is a valid rustup home directory.

        Args:
            path: Path to validate

        Returns:
            True if valid rustup home, False otherwise
        """
        try:
            if not path.exists():
                return False

            # Check for expected rustup structure
            toolchains_dir = path / "toolchains"
            settings_file = path / "settings.toml"

            # At minimum, toolchains directory should exist
            if toolchains_dir.exists() and toolchains_dir.is_dir():
                return True

            # Also accept if settings.toml exists (even if toolchains empty)
            if settings_file.exists():
                return True

        except (PermissionError, OSError) as e:
            logger.debug(f"Cannot access {path}: {e}")

        return False

    def get_available_toolchains(self) -> list[str]:
        """
        Get list of available Rust toolchains.

        Returns:
            List of toolchain names (e.g., ["stable", "nightly", "1.75.0"])
        """
        if self._toolchains:
            return self._toolchains

        # Try using rustup command first
        try:
            result = subprocess.run(
                ["rustup", "toolchain", "list"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                # Parse toolchain list (format: "stable-x86_64-unknown-linux-gnu (default)")
                toolchains = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        # Extract just the channel name (stable, nightly, etc.)
                        toolchain = line.split("-")[0].split()[0]
                        if toolchain and toolchain not in toolchains:
                            toolchains.append(toolchain)

                self._toolchains = toolchains
                logger.info(f"Found toolchains via rustup command: {toolchains}")
                return toolchains

        except (subprocess.SubprocessError, FileNotFoundError) as e:
            logger.debug(f"Cannot run rustup command: {e}")

        # Fallback to filesystem detection
        rustup_home = self.detect_rustup_home()
        if not rustup_home:
            return []

        toolchains_dir = rustup_home / "toolchains"
        if not toolchains_dir.exists():
            return []

        toolchains = []
        try:
            for item in toolchains_dir.iterdir():
                if item.is_dir():
                    # Extract channel from directory name
                    # Format: stable-x86_64-unknown-linux-gnu
                    channel = item.name.split("-")[0]
                    if channel and channel not in toolchains:
                        toolchains.append(channel)
        except (PermissionError, OSError) as e:
            logger.warning(f"Cannot list toolchains: {e}")

        self._toolchains = toolchains
        logger.info(f"Found toolchains via filesystem: {toolchains}")
        return toolchains

    def has_rust_docs_json(self, toolchain: str = "nightly") -> bool:
        """
        Check if rust-docs-json component is installed for a toolchain.

        Args:
            toolchain: Toolchain to check (default: nightly)

        Returns:
            True if rust-docs-json is available
        """
        # Try rustup command
        try:
            result = subprocess.run(
                ["rustup", "component", "list", "--toolchain", toolchain],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return "rust-docs-json" in result.stdout
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

        # Fallback to filesystem check
        rustup_home = self.detect_rustup_home()
        if not rustup_home:
            return False

        # Look for rust-docs-json in toolchain
        toolchains_dir = rustup_home / "toolchains"
        for item in toolchains_dir.iterdir():
            if item.is_dir() and toolchain in item.name:
                docs_json = item / "share" / "doc" / "rust" / "json"
                if docs_json.exists():
                    return True

        return False

    def get_stdlib_json_path(
        self, crate_name: str, toolchain: str = "nightly"
    ) -> Path | None:
        """
        Get path to standard library JSON documentation if available.

        Args:
            crate_name: Name of stdlib crate (std, core, alloc, etc.)
            toolchain: Toolchain to use (default: nightly)

        Returns:
            Path to JSON file or None if not found
        """
        rustup_home = self.detect_rustup_home()
        if not rustup_home:
            return None

        # Find the toolchain directory
        toolchains_dir = rustup_home / "toolchains"
        for toolchain_dir in toolchains_dir.iterdir():
            if toolchain_dir.is_dir() and toolchain in toolchain_dir.name:
                # Standard library JSON location
                json_path = (
                    toolchain_dir
                    / "share"
                    / "doc"
                    / "rust"
                    / "json"
                    / f"{crate_name}.json"
                )
                if json_path.exists():
                    logger.info(f"Found stdlib JSON: {json_path}")
                    return json_path

        return None

    def get_installation_instructions(self) -> str:
        """
        Get platform-specific rustup installation instructions.

        Returns:
            Installation command for the current platform
        """
        if platform.system() == "Windows":
            return (
                "Download and run rustup-init.exe from https://rustup.rs/\n"
                "Or use: winget install Rustlang.Rustup"
            )
        else:
            return (
                "Install rustup with:\n"
                "curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh"
            )

    def get_rust_docs_json_instructions(self) -> str:
        """
        Get instructions for installing rust-docs-json component.

        Returns:
            Installation command for rust-docs-json
        """
        return (
            "Install rust-docs-json component with:\n"
            "rustup component add rust-docs-json --toolchain nightly\n\n"
            "Note: rust-docs-json is only available on the nightly toolchain."
        )


# Global detector instance
_detector = RustupDetector()


def detect_rustup() -> Path | None:
    """
    Detect rustup installation (convenience function).

    Returns:
        Path to rustup home or None
    """
    return _detector.detect_rustup_home()


def get_available_toolchains() -> list[str]:
    """
    Get available Rust toolchains (convenience function).

    Returns:
        List of toolchain names
    """
    return _detector.get_available_toolchains()


def has_rust_docs_json(toolchain: str = "nightly") -> bool:
    """
    Check if rust-docs-json is available (convenience function).

    Args:
        toolchain: Toolchain to check

    Returns:
        True if available
    """
    return _detector.has_rust_docs_json(toolchain)


def get_stdlib_json_path(crate_name: str, toolchain: str = "nightly") -> Path | None:
    """
    Get path to stdlib JSON documentation (convenience function).

    Args:
        crate_name: Stdlib crate name
        toolchain: Toolchain to use

    Returns:
        Path to JSON file or None
    """
    return _detector.get_stdlib_json_path(crate_name, toolchain)
