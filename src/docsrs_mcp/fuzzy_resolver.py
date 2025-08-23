"""Fuzzy path resolution for improved UX when exact paths aren't found.

This module provides enhanced fuzzy matching for Rust documentation paths with:
- Composite scoring using multiple RapidFuzz algorithms (token_set, token_sort, partial)
- Path component bonus system for exact and partial matches
- Adaptive thresholds based on query length
- Unicode normalization for consistent matching
- Configurable weights via FUZZY_WEIGHTS in config.py

The fuzzy matching system uses a three-tier approach:
1. Re-export discovery from database
2. Static PATH_ALIASES dictionary
3. Enhanced fuzzy matching with composite scoring
"""

import logging
import time
import unicodedata

import aiosqlite
from rapidfuzz import fuzz
from rapidfuzz.utils import default_process

from .config import (
    FUZZY_MAX_EXPANSIONS,
    FUZZY_PREFIX_LENGTH,
    FUZZY_SCORE_CUTOFF,
    FUZZY_WEIGHTS,
)
from .database import get_discovered_reexports

logger = logging.getLogger(__name__)

# Path aliases for common Rust documentation patterns
PATH_ALIASES = {
    # serde aliases
    "serde::Serialize": "serde::ser::Serialize",
    "serde::Deserialize": "serde::de::Deserialize",
    "serde::Serializer": "serde::ser::Serializer",
    "serde::Deserializer": "serde::de::Deserializer",
    "serde::json": "serde_json",
    "serde::Error": "serde::de::Error",
    "serde::Visitor": "serde::de::Visitor",
    # tokio aliases
    "tokio::spawn": "tokio::task::spawn",
    "tokio::JoinHandle": "tokio::task::JoinHandle",
    "tokio::select": "tokio::macros::select",
    "tokio::sleep": "tokio::time::sleep",
    "tokio::timeout": "tokio::time::timeout",
    "tokio::interval": "tokio::time::interval",
    "tokio::Duration": "tokio::time::Duration",
    "tokio::Instant": "tokio::time::Instant",
    "tokio::TcpListener": "tokio::net::TcpListener",
    "tokio::TcpStream": "tokio::net::TcpStream",
    "tokio::UdpSocket": "tokio::net::UdpSocket",
    "tokio::UnixListener": "tokio::net::UnixListener",
    "tokio::UnixStream": "tokio::net::UnixStream",
    "tokio::File": "tokio::fs::File",
    "tokio::OpenOptions": "tokio::fs::OpenOptions",
    "tokio::Runtime": "tokio::runtime::Runtime",
    "tokio::Builder": "tokio::runtime::Builder",
    "tokio::block_on": "tokio::runtime::Runtime::block_on",
    "tokio::join": "tokio::macros::join",
    "tokio::try_join": "tokio::macros::try_join",
    "tokio::pin": "tokio::macros::pin",
    # std library aliases
    "std::HashMap": "std::collections::HashMap",
    "std::HashSet": "std::collections::HashSet",
    "std::BTreeMap": "std::collections::BTreeMap",
    "std::BTreeSet": "std::collections::BTreeSet",
    "std::VecDeque": "std::collections::VecDeque",
    "std::LinkedList": "std::collections::LinkedList",
    "std::BinaryHeap": "std::collections::BinaryHeap",
    "std::Vec": "std::vec::Vec",
    "std::Result": "std::result::Result",
    "std::Option": "std::option::Option",
    "std::Arc": "std::sync::Arc",
    "std::Rc": "std::rc::Rc",
    "std::Mutex": "std::sync::Mutex",
    "std::RwLock": "std::sync::RwLock",
    "std::Condvar": "std::sync::Condvar",
    "std::Barrier": "std::sync::Barrier",
    "std::Once": "std::sync::Once",
    "std::Cell": "std::cell::Cell",
    "std::RefCell": "std::cell::RefCell",
    "std::UnsafeCell": "std::cell::UnsafeCell",
    "std::Box": "std::boxed::Box",
    "std::Cow": "std::borrow::Cow",
    "std::Path": "std::path::Path",
    "std::PathBuf": "std::path::PathBuf",
    "std::OsStr": "std::ffi::OsStr",
    "std::OsString": "std::ffi::OsString",
    "std::CString": "std::ffi::CString",
    "std::CStr": "std::ffi::CStr",
    "std::File": "std::fs::File",
    "std::OpenOptions": "std::fs::OpenOptions",
    "std::DirEntry": "std::fs::DirEntry",
    "std::ReadDir": "std::fs::ReadDir",
    "std::TcpListener": "std::net::TcpListener",
    "std::TcpStream": "std::net::TcpStream",
    "std::UdpSocket": "std::net::UdpSocket",
    "std::SocketAddr": "std::net::SocketAddr",
    "std::IpAddr": "std::net::IpAddr",
    "std::Ipv4Addr": "std::net::Ipv4Addr",
    "std::Ipv6Addr": "std::net::Ipv6Addr",
    "std::Error": "std::error::Error",
    "std::thread": "std::thread",
    "std::spawn": "std::thread::spawn",
    "std::JoinHandle": "std::thread::JoinHandle",
    "std::panic": "std::panic",
    "std::catch_unwind": "std::panic::catch_unwind",
    "std::Pin": "std::pin::Pin",
    "std::Future": "std::future::Future",
    "std::IntoFuture": "std::future::IntoFuture",
    # async-std aliases (common async runtime)
    "async_std::task::spawn": "async_std::task::spawn",
    "async_std::task::block_on": "async_std::task::block_on",
    # futures aliases
    "futures::Future": "futures::future::Future",
    "futures::Stream": "futures::stream::Stream",
    "futures::Sink": "futures::sink::Sink",
    "futures::join": "futures::future::join",
    "futures::select": "futures::select",
    # reqwest aliases
    "reqwest::Client": "reqwest::Client",
    "reqwest::Response": "reqwest::Response",
    "reqwest::get": "reqwest::get",
    "reqwest::post": "reqwest::post",
    # actix-web aliases
    "actix_web::App": "actix_web::App",
    "actix_web::HttpServer": "actix_web::HttpServer",
    "actix_web::web": "actix_web::web",
    "actix_web::HttpResponse": "actix_web::HttpResponse",
    # hyper aliases
    "hyper::Client": "hyper::Client",
    "hyper::Server": "hyper::Server",
    "hyper::Body": "hyper::Body",
    "hyper::Request": "hyper::Request",
    "hyper::Response": "hyper::Response",
}

# Simple path cache with TTL
_path_cache: dict[str, tuple[float, list[str]]] = {}
PATH_CACHE_TTL = 300  # 5 minutes

# Cache for discovered re-exports
_reexport_cache: dict[str, tuple[float, dict[str, str]]] = {}
REEXPORT_CACHE_TTL = 300  # 5 minutes


def normalize_query(query: str) -> str:
    """
    Normalize Unicode characters in query for consistent matching.

    Args:
        query: The input query string

    Returns:
        Normalized query string
    """
    # Normalize to NFC (composed form) for consistency
    normalized = unicodedata.normalize("NFC", query)
    # Also handle common preprocessing
    return default_process(normalized) if normalized else ""


def composite_path_score(
    query: str, candidate: str, score_cutoff: float | None = None
) -> float:
    """
    Calculate a composite similarity score using multiple RapidFuzz algorithms.

    This combines token_set_ratio (handles subsets), token_sort_ratio (handles order),
    and partial_ratio (substring matching) for more accurate path matching.

    Args:
        query: The user's search query
        candidate: The candidate path to compare against
        score_cutoff: Optional minimum score threshold for early termination

    Returns:
        Normalized similarity score between 0.0 and 1.0
    """
    # Use global score cutoff if not provided
    if score_cutoff is None:
        score_cutoff = FUZZY_SCORE_CUTOFF / 100.0  # Convert from percentage

    # Normalize both strings for consistent comparison
    query_norm = normalize_query(query)
    candidate_norm = normalize_query(candidate)

    # Early termination: Check prefix match for performance
    if FUZZY_PREFIX_LENGTH > 0 and len(query_norm) >= FUZZY_PREFIX_LENGTH:
        prefix = query_norm[:FUZZY_PREFIX_LENGTH].lower()
        if not candidate_norm.lower().startswith(prefix):
            # Quick rejection if prefix doesn't match (unless it's a substring)
            if prefix not in candidate_norm.lower():
                return 0.0

    # Calculate individual scores with score_cutoff for optimization
    # RapidFuzz can use score_cutoff internally for early termination
    cutoff_percentage = score_cutoff * 100 if score_cutoff else 0

    token_set = (
        fuzz.token_set_ratio(query_norm, candidate_norm, score_cutoff=cutoff_percentage)
        / 100.0
    )
    token_sort = (
        fuzz.token_sort_ratio(
            query_norm, candidate_norm, score_cutoff=cutoff_percentage
        )
        / 100.0
    )
    partial = (
        fuzz.partial_ratio(query_norm, candidate_norm, score_cutoff=cutoff_percentage)
        / 100.0
    )

    # Weighted combination using configurable weights
    score = (
        FUZZY_WEIGHTS["token_set_ratio"] * token_set
        + FUZZY_WEIGHTS["token_sort_ratio"] * token_sort
        + FUZZY_WEIGHTS["partial_ratio"] * partial
    )

    # Normalize to ensure score is within bounds
    return max(0.0, min(1.0, score))


def calculate_path_bonus(query: str, candidate: str) -> float:
    """
    Calculate bonus score for exact path component matches.

    This rewards candidates where the final component (most important part)
    matches exactly or partially with the query.

    Args:
        query: The user's search query
        candidate: The candidate path to compare against

    Returns:
        Bonus score between 0.0 and 0.15
    """
    # Split paths into components
    query_parts = query.split("::")
    candidate_parts = candidate.split("::")

    # Remove empty parts
    query_parts = [p for p in query_parts if p]
    candidate_parts = [p for p in candidate_parts if p]

    if not query_parts or not candidate_parts:
        return 0.0

    # Get the final components (most important)
    query_final = query_parts[-1].lower()
    candidate_final = candidate_parts[-1].lower()

    # Exact match of final component gets highest bonus
    if query_final == candidate_final:
        return FUZZY_WEIGHTS["path_component_bonus"]
    # Partial match (query is substring of candidate) gets moderate bonus
    elif query_final in candidate_final:
        return FUZZY_WEIGHTS["partial_component_bonus"]
    # Check if candidate final is in query (reverse partial match)
    elif candidate_final in query_final:
        return (
            FUZZY_WEIGHTS["partial_component_bonus"] * 0.6
        )  # Slightly lower for reverse match

    return 0.0


def get_adaptive_threshold(query: str) -> float:
    """
    Calculate an adaptive similarity threshold based on query characteristics.

    Shorter queries are more forgiving (lower threshold) while longer queries
    should be more specific (higher threshold).

    Args:
        query: The user's search query

    Returns:
        Adaptive threshold between 0.55 and 0.65
    """
    query_length = len(query.strip())

    if query_length <= 5:
        # Very short queries need lower threshold (more forgiving)
        return 0.55
    elif query_length <= 10:
        # Default threshold for medium queries
        return 0.60
    elif query_length <= 20:
        # Slightly higher for longer queries
        return 0.63
    else:
        # Long queries should be more specific
        return 0.65


async def resolve_path_alias(
    crate_name: str, item_path: str, db_path: str | None = None
) -> str:
    """
    Resolve common path aliases to their actual rustdoc paths.

    Priority order:
    1. Check discovered re-exports from database
    2. Check static PATH_ALIASES
    3. Return original path

    Args:
        crate_name: Name of the crate being searched
        item_path: The path to resolve
        db_path: Optional path to database for discovered re-exports

    Returns:
        The resolved path or original if no alias found
    """
    # Handle crate-level paths
    if item_path == "crate":
        return item_path

    # First, check discovered re-exports from database if available
    if db_path:
        try:
            # Check cache first
            cache_key = f"{crate_name}_reexports"
            reexports = None

            if cache_key in _reexport_cache:
                timestamp, cached_reexports = _reexport_cache[cache_key]
                if time.time() - timestamp < REEXPORT_CACHE_TTL:
                    reexports = cached_reexports
                else:
                    # Expired, remove from cache
                    del _reexport_cache[cache_key]

            # Load from database if not cached
            if reexports is None:
                # Include cross-references in the resolution
                reexports = await get_discovered_reexports(
                    db_path, crate_name, include_crossrefs=True
                )

                # Cache the results
                if reexports:
                    _reexport_cache[cache_key] = (time.time(), reexports)

            # Check with crate prefix
            crate_qualified = f"{crate_name}::{item_path}"
            if crate_qualified in reexports:
                resolved = reexports[crate_qualified]
                logger.debug(
                    f"Resolved via discovered re-export: {item_path} -> {resolved}"
                )
                return resolved

            # Check without crate prefix
            if item_path in reexports:
                resolved = reexports[item_path]
                logger.debug(
                    f"Resolved via discovered re-export: {item_path} -> {resolved}"
                )
                return resolved
        except Exception as e:
            logger.debug(f"Error checking discovered re-exports: {e}")

    # Fallback to static PATH_ALIASES
    # Check for direct alias with crate prefix
    crate_qualified = f"{crate_name}::{item_path}"
    if crate_qualified in PATH_ALIASES:
        resolved = PATH_ALIASES[crate_qualified]
        logger.debug(f"Resolved via static alias: {item_path} -> {resolved}")
        return resolved

    # Try without crate prefix for common patterns
    if item_path in PATH_ALIASES:
        resolved = PATH_ALIASES[item_path]
        logger.debug(f"Resolved via static alias: {item_path} -> {resolved}")
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

    Uses enhanced composite scoring with multiple algorithms for better accuracy.

    Args:
        query: The user's query path that wasn't found
        db_path: Path to the SQLite database
        crate_name: Name of the crate being searched
        version: Version of the crate
        limit: Maximum number of suggestions to return (default: 3)
        threshold: Minimum similarity score threshold (default: 0.6, but adaptive)

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

    # Normalize the query once
    normalized_query = normalize_query(query)

    # Use adaptive threshold based on query characteristics
    adaptive_threshold = get_adaptive_threshold(query)
    # Use the minimum of provided threshold and adaptive threshold
    effective_threshold = min(threshold, adaptive_threshold)

    # Calculate scores for all candidates with enhanced scoring
    scored_candidates = []

    # Limit the number of candidates to evaluate for performance
    # Use FUZZY_MAX_EXPANSIONS to cap the number of paths checked
    paths_to_check = (
        cached_paths[: FUZZY_MAX_EXPANSIONS * 10]
        if len(cached_paths) > FUZZY_MAX_EXPANSIONS * 10
        else cached_paths
    )

    for candidate in paths_to_check:
        # Calculate composite similarity score with cutoff for early termination
        base_score = composite_path_score(
            normalized_query, candidate, score_cutoff=effective_threshold
        )

        # Skip if base score is 0 (early termination)
        if base_score == 0:
            continue

        # Add path component bonus for better Rust path matching
        path_bonus = calculate_path_bonus(query, candidate)

        # Combined score with bonus (capped at 1.0)
        final_score = min(1.0, base_score + path_bonus)

        # Only keep candidates above threshold
        if final_score >= effective_threshold:
            scored_candidates.append((candidate, final_score))

            # Stop early if we have enough high-quality matches
            if len(scored_candidates) >= FUZZY_MAX_EXPANSIONS:
                break

    # Sort by score (descending) and take top matches
    scored_candidates.sort(key=lambda x: x[1], reverse=True)
    suggestions = [candidate for candidate, _ in scored_candidates[:limit]]

    logger.info(
        f"Found {len(suggestions)} fuzzy suggestions for '{query}' "
        f"(threshold: {effective_threshold:.2f}, candidates evaluated: {len(cached_paths)})"
    )
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
