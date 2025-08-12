"""Cargo.toml and Cargo.lock parsing utilities."""

import logging
import sys
from pathlib import Path
from typing import Any

import aiohttp
import semver

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


def parse_cargo_version_spec(spec: str) -> tuple[str, str | None]:
    """Parse cargo version specification into a constraint type and version.

    Args:
        spec: Version specification (e.g., "^1.2.3", "~1.2", "1.*", ">=1.0, <2.0")

    Returns:
        Tuple of (constraint_type, version) where constraint_type is 'caret', 'tilde',
        'wildcard', 'exact', or 'complex'
    """
    spec = spec.strip()

    # Handle caret requirements (default in Cargo)
    if spec.startswith("^"):
        return ("caret", spec[1:].strip())

    # Handle tilde requirements
    elif spec.startswith("~"):
        return ("tilde", spec[1:].strip())

    # Handle wildcard requirements
    elif "*" in spec:
        return ("wildcard", spec)

    # Handle complex requirements (with operators)
    elif any(op in spec for op in [">=", "<=", ">", "<", ","]):
        return ("complex", spec)

    # Try to parse as exact version
    else:
        # Check if it's a valid semver
        try:
            # Add missing components for incomplete versions
            parts = spec.split(".")
            normalized = spec
            if len(parts) == 1 and parts[0].isdigit():
                normalized = f"{spec}.0.0"
            elif len(parts) == 2 and all(p.isdigit() for p in parts):
                normalized = f"{spec}.0"

            semver.Version.parse(normalized)
            return ("exact", normalized)
        except ValueError:
            # Not a valid version, return as-is
            return ("unknown", spec)


def find_best_version_match(
    constraint_type: str, version_spec: str | None, available_versions: list[str]
) -> str | None:
    """Find the best matching version from available versions based on constraint.

    Args:
        constraint_type: Type of constraint ('caret', 'tilde', 'wildcard', 'exact', etc.)
        version_spec: The version specification
        available_versions: List of available versions from crate info

    Returns:
        Best matching version or None if no match found
    """
    if not version_spec or not available_versions:
        return None

    # Filter out pre-release versions unless explicitly requested
    stable_versions = []
    for v in available_versions:
        try:
            parsed = semver.Version.parse(v)
            if not parsed.prerelease:
                stable_versions.append(v)
        except ValueError:
            continue

    # Use stable versions if available, otherwise fall back to all versions
    versions_to_check = stable_versions if stable_versions else available_versions

    if constraint_type == "exact":
        # Return exact match if it exists
        if version_spec in versions_to_check:
            return version_spec
        # Try adding missing components
        for v in versions_to_check:
            if v.startswith(version_spec):
                return v
        return None

    elif constraint_type == "caret":
        # Caret: ^1.2.3 means >=1.2.3, <2.0.0 (compatible updates)
        try:
            base_version = semver.Version.parse(version_spec)
            max_version = base_version.bump_major()

            matching = []
            for v in versions_to_check:
                try:
                    parsed = semver.Version.parse(v)
                    if parsed >= base_version and parsed < max_version:
                        matching.append(v)
                except ValueError:
                    continue

            # Return highest matching version
            if matching:
                return max(matching, key=lambda v: semver.Version.parse(v))
        except ValueError:
            pass

    elif constraint_type == "tilde":
        # Tilde: ~1.2.3 means >=1.2.3, <1.3.0 (bug fixes only)
        try:
            base_version = semver.Version.parse(version_spec)
            max_version = base_version.bump_minor()

            matching = []
            for v in versions_to_check:
                try:
                    parsed = semver.Version.parse(v)
                    if parsed >= base_version and parsed < max_version:
                        matching.append(v)
                except ValueError:
                    continue

            # Return highest matching version
            if matching:
                return max(matching, key=lambda v: semver.Version.parse(v))
        except ValueError:
            pass

    elif constraint_type == "wildcard":
        # Handle patterns like "1.*" or "1.2.*"
        prefix = version_spec.replace("*", "")
        matching = [v for v in versions_to_check if v.startswith(prefix)]
        if matching:
            # Return highest matching version
            try:
                return max(matching, key=lambda v: semver.Version.parse(v))
            except ValueError:
                return matching[-1]  # Fall back to last in list

    # For complex constraints or unknown types, just return the latest stable version
    if stable_versions:
        try:
            return max(stable_versions, key=lambda v: semver.Version.parse(v))
        except ValueError:
            return stable_versions[-1]

    return None


async def fetch_crate_info_cached(
    crate_name: str, session: aiohttp.ClientSession
) -> dict[str, Any] | None:
    """Fetch crate info with caching.

    Args:
        crate_name: Name of the crate
        session: aiohttp session for making requests

    Returns:
        Crate info dict or None if fetch fails
    """
    from .cache import get_crate_info_cache
    from .ingest import fetch_crate_info

    cache = get_crate_info_cache()

    # Check cache first
    cached = cache.get(crate_name)
    if cached is not None:
        return cached

    # Fetch from API
    try:
        crate_info = await fetch_crate_info(session, crate_name)
        if crate_info:
            # Cache the result
            cache.set(crate_name, crate_info)
        return crate_info
    except Exception as e:
        logger.warning(f"Failed to fetch crate info for {crate_name}: {e}")
        return None


async def resolve_cargo_versions(
    crates: list[str], session: aiohttp.ClientSession, resolve: bool = False
) -> list[str]:
    """Resolve version specifications to concrete versions.

    Args:
        crates: List of crate@version_spec strings
        session: aiohttp session for API requests
        resolve: Whether to resolve version specs to concrete versions

    Returns:
        List of crate@version strings (resolved or original)
    """
    if not resolve:
        return crates

    resolved = []

    for crate_spec in crates:
        if "@" not in crate_spec:
            # No version specified, use latest
            resolved.append(f"{crate_spec}@latest")
            continue

        name, version_spec = crate_spec.split("@", 1)

        # Special case for 'latest'
        if version_spec == "latest":
            resolved.append(crate_spec)
            continue

        # Try to resolve the version
        try:
            # Get crate info
            crate_info = await fetch_crate_info_cached(name, session)

            if crate_info:
                # Parse the version specification
                constraint_type, parsed_spec = parse_cargo_version_spec(version_spec)

                # For simple resolution, use the max_stable_version
                # For more complex resolution, we would need to fetch all versions
                # from /api/v1/crates/{crate}/versions endpoint

                if constraint_type in ["caret", "tilde", "wildcard", "exact"]:
                    # Use the max_stable_version as a simple resolution
                    # This is a simplification - a full implementation would fetch all versions
                    max_stable = crate_info.get("max_stable_version")
                    if max_stable:
                        resolved.append(f"{name}@{max_stable}")
                        logger.info(f"Resolved {crate_spec} to {name}@{max_stable}")
                    else:
                        # Fall back to max_version or newest_version
                        version = crate_info.get("max_version") or crate_info.get(
                            "newest_version"
                        )
                        if version:
                            resolved.append(f"{name}@{version}")
                            logger.info(f"Resolved {crate_spec} to {name}@{version}")
                        else:
                            resolved.append(crate_spec)
                            logger.warning(
                                f"No version found for {crate_spec}, using original"
                            )
                else:
                    # For complex or unknown constraints, use original
                    resolved.append(crate_spec)
                    logger.warning(
                        f"Complex constraint for {crate_spec}, using original"
                    )
            else:
                # Failed to get crate info, use original
                resolved.append(crate_spec)
                logger.warning(
                    f"No crate info for {name}, using original spec: {crate_spec}"
                )

        except Exception as e:
            logger.error(f"Error resolving {crate_spec}: {e}")
            resolved.append(crate_spec)  # Fallback to original

    return resolved


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
