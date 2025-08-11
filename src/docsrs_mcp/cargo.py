"""Cargo.toml and Cargo.lock parsing utilities."""

import logging
import sys
from pathlib import Path

# Use tomllib for Python 3.11+, fall back to toml
if sys.version_info >= (3, 11):
    import tomllib
else:
    import toml as tomllib

logger = logging.getLogger(__name__)


def parse_cargo_toml(file_path: Path) -> dict[str, list[str]]:
    """Parse Cargo.toml and extract dependencies.

    Args:
        file_path: Path to Cargo.toml file

    Returns:
        Dict with 'crates' key containing list of crate@version strings

    Raises:
        ValueError: If the file is invalid or cannot be parsed
    """
    try:
        # Open in binary mode for tomllib compatibility
        with open(file_path, "rb" if sys.version_info >= (3, 11) else "r") as f:
            data = tomllib.load(f)

        crates = set()

        # Extract from [dependencies], [dev-dependencies], [build-dependencies]
        for dep_section in ["dependencies", "dev-dependencies", "build-dependencies"]:
            if dep_section in data:
                for crate_name, spec in data[dep_section].items():
                    # Handle various dependency formats
                    if isinstance(spec, str):
                        # Simple version string
                        crates.add(f"{crate_name}@{spec}")
                    elif isinstance(spec, dict) and "version" in spec:
                        # Complex dependency with version field
                        crates.add(f"{crate_name}@{spec['version']}")
                    else:
                        # Default to latest if no version specified
                        crates.add(f"{crate_name}@latest")

        # Handle workspace dependencies
        if "workspace" in data and "dependencies" in data["workspace"]:
            for crate_name, spec in data["workspace"]["dependencies"].items():
                if isinstance(spec, str):
                    crates.add(f"{crate_name}@{spec}")
                elif isinstance(spec, dict) and "version" in spec:
                    crates.add(f"{crate_name}@{spec['version']}")
                else:
                    crates.add(f"{crate_name}@latest")

        # Handle target-specific dependencies
        if "target" in data:
            for target_spec in data["target"].values():
                if "dependencies" in target_spec:
                    for crate_name, spec in target_spec["dependencies"].items():
                        if isinstance(spec, str):
                            crates.add(f"{crate_name}@{spec}")
                        elif isinstance(spec, dict) and "version" in spec:
                            crates.add(f"{crate_name}@{spec['version']}")
                        else:
                            crates.add(f"{crate_name}@latest")

        return {"crates": sorted(list(crates))}
    except Exception as e:
        logger.error(f"Failed to parse Cargo.toml: {e}")
        raise ValueError(f"Invalid Cargo.toml file: {e}")


def parse_cargo_lock(file_path: Path) -> dict[str, list[str]]:
    """Parse Cargo.lock and extract exact versions.

    Args:
        file_path: Path to Cargo.lock file

    Returns:
        Dict with 'crates' key containing list of crate@version strings

    Raises:
        ValueError: If the file is invalid or cannot be parsed
    """
    try:
        with open(file_path, "rb" if sys.version_info >= (3, 11) else "r") as f:
            data = tomllib.load(f)

        crates = set()

        # Extract from [[package]] entries
        if "package" in data:
            for package in data["package"]:
                if "name" in package and "version" in package:
                    # Use exact version from lock file
                    crates.add(f"{package['name']}@{package['version']}")

        return {"crates": sorted(list(crates))}
    except Exception as e:
        logger.error(f"Failed to parse Cargo.lock: {e}")
        raise ValueError(f"Invalid Cargo.lock file: {e}")


def extract_crates_from_cargo(file_path: Path) -> list[str]:
    """Extract crate list from either Cargo.toml or Cargo.lock.

    Args:
        file_path: Path to Cargo.toml or Cargo.lock file

    Returns:
        List of crate@version strings

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file is not a supported Cargo file
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if file_path.name == "Cargo.toml":
        return parse_cargo_toml(file_path)["crates"]
    elif file_path.name == "Cargo.lock":
        return parse_cargo_lock(file_path)["crates"]
    else:
        raise ValueError(
            f"Unsupported file: {file_path.name}. Expected Cargo.toml or Cargo.lock"
        )
