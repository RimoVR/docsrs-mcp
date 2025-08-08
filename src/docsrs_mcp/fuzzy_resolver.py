"""Fuzzy path resolution for improved UX when exact paths aren't found."""

import logging
import time

import aiosqlite
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

logger = logging.getLogger(__name__)

# Path aliases for common Rust documentation patterns
PATH_ALIASES = {
    # serde aliases
    "serde::Serialize": "serde::ser::Serialize",
    "serde::Deserialize": "serde::de::Deserialize",
    "serde::Serializer": "serde::ser::Serializer",
    "serde::Deserializer": "serde::de::Deserializer",
    # tokio aliases
    "tokio::spawn": "tokio::task::spawn",
    "tokio::JoinHandle": "tokio::task::JoinHandle",
    "tokio::select": "tokio::macros::select",
    # std aliases
    "std::HashMap": "std::collections::HashMap",
    "std::HashSet": "std::collections::HashSet",
    "std::BTreeMap": "std::collections::BTreeMap",
    "std::BTreeSet": "std::collections::BTreeSet",
    "std::VecDeque": "std::collections::VecDeque",
    "std::Vec": "std::vec::Vec",
    "std::Result": "std::result::Result",
    "std::Option": "std::option::Option",
}

# Simple path cache with TTL
_path_cache: dict[str, tuple[float, list[str]]] = {}
PATH_CACHE_TTL = 300  # 5 minutes


def resolve_path_alias(crate_name: str, item_path: str) -> str:
    """
    Resolve common path aliases to their actual rustdoc paths.

    Returns the resolved path or original if no alias exists.
    O(1) operation with no exceptions.

    Args:
        crate_name: Name of the crate being searched
        item_path: The path to resolve

    Returns:
        The resolved path or original if no alias found
    """
    # Handle crate-level paths
    if item_path == "crate":
        return item_path

    # Check for direct alias with crate prefix
    crate_qualified = f"{crate_name}::{item_path}"
    if crate_qualified in PATH_ALIASES:
        resolved = PATH_ALIASES[crate_qualified]
        logger.debug(f"Resolved crate-specific alias {item_path} -> {resolved}")
        return resolved

    # Try without crate prefix for common patterns
    if item_path in PATH_ALIASES:
        resolved = PATH_ALIASES[item_path]
        logger.debug(f"Resolved alias {item_path} -> {resolved}")
        return resolved

    # No alias found, return original
    return item_path


async def get_fuzzy_suggestions(
    query: str,
    db_path: str,
    crate_name: str,
    version: str,
    limit: int = 3,
    threshold: float = 0.6,
) -> list[str]:
    """
    Get fuzzy path suggestions when exact match fails.

    Args:
        query: The user's query path that wasn't found
        db_path: Path to the SQLite database
        crate_name: Name of the crate being searched
        version: Version of the crate
        limit: Maximum number of suggestions to return (default: 3)
        threshold: Minimum similarity score threshold (default: 0.6)

    Returns:
        List of suggested paths that are similar to the query
    """
    # Create cache key for this crate's paths
    cache_key = f"{crate_name}_{version}_paths"

    # Try to get paths from cache
    cached_paths = None
    if cache_key in _path_cache:
        timestamp, paths = _path_cache[cache_key]
        if time.time() - timestamp < PATH_CACHE_TTL:
            cached_paths = paths
        else:
            # Expired, remove from cache
            del _path_cache[cache_key]

    if cached_paths is None:
        # Fetch all item paths from database
        try:
            async with aiosqlite.connect(db_path) as db:
                cursor = await db.execute(
                    "SELECT DISTINCT item_path FROM embeddings WHERE item_path IS NOT NULL"
                )
                paths = [row[0] for row in await cursor.fetchall()]

                # Cache the paths for future use
                if paths:
                    _path_cache[cache_key] = (time.time(), paths)
                cached_paths = paths
        except Exception as e:
            logger.error(f"Error fetching paths for fuzzy matching: {e}")
            return []

    if not cached_paths:
        return []

    # Use RapidFuzz to find similar paths
    # Use explicit processor for consistent behavior with v3.x
    matches = process.extract(
        query,
        cached_paths,
        scorer=fuzz.ratio,
        processor=default_process,
        limit=limit * 2,  # Get more candidates for filtering
    )

    # Filter by threshold and limit results
    suggestions = []
    for match_text, score, _ in matches:
        # Convert score from 0-100 to 0-1 range and check threshold
        normalized_score = score / 100.0
        if normalized_score >= threshold:
            suggestions.append(match_text)
            if len(suggestions) >= limit:
                break

    logger.info(f"Found {len(suggestions)} fuzzy suggestions for '{query}'")
    return suggestions


async def get_fuzzy_suggestions_with_fallback(
    query: str,
    db_path: str,
    crate_name: str,
    version: str,
) -> list[str]:
    """
    Get fuzzy suggestions with graceful error handling.

    This is a convenience wrapper that ensures we always return a list,
    even if fuzzy matching fails for any reason.
    """
    try:
        return await get_fuzzy_suggestions(
            query=query,
            db_path=db_path,
            crate_name=crate_name,
            version=version,
        )
    except Exception as e:
        logger.warning(f"Fuzzy matching failed, returning empty list: {e}")
        return []
