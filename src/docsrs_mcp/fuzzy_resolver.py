"""Fuzzy path resolution for improved UX when exact paths aren't found."""

import logging
import time

import aiosqlite
from rapidfuzz import fuzz, process
from rapidfuzz.utils import default_process

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
                reexports = await get_discovered_reexports(db_path, crate_name)

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
