"""Vector search operations using sqlite-vec."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import sqlite_vec
import structlog

from .. import config as app_config
from ..cache import get_search_cache
from ..config import DB_TIMEOUT
from .connection import load_sqlite_vec_extension

if TYPE_CHECKING:
    pass

# Use structlog for structured logging when available, fallback to standard logging
try:
    logger = structlog.get_logger(__name__)
except AttributeError:
    logger = logging.getLogger(__name__)


def _apply_mmr_diversification(
    ranked_results: list[tuple[float, str, str, str, str]],
    embeddings: list[np.ndarray],
    k: int,
    lambda_param: float,
) -> list[tuple[float, str, str, str]]:
    """Apply Maximum Marginal Relevance (MMR) for result diversification.

    MMR balances relevance and diversity using the formula:
    MMR = λ * relevance + (1-λ) * diversity

    Enhanced with semantic similarity between embeddings for better diversity.

    Args:
        ranked_results: List of (score, path, header, content, item_type) tuples
        embeddings: List of numpy arrays containing embeddings for each result
        k: Number of results to select
        lambda_param: Balance between relevance (1.0) and diversity (0.0)

    Returns:
        List of diversified results as (score, path, header, content) tuples
    """
    if not ranked_results or k <= 0:
        return []

    # Initialize selected items with the top-scoring result
    selected_indices = [0]
    selected_results = [
        (
            ranked_results[0][0],
            ranked_results[0][1],
            ranked_results[0][2],
            ranked_results[0][3],
        )
    ]

    # Precompute similarities between all pairs of embeddings
    n = len(embeddings)
    similarities = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            # Compute cosine similarity
            sim = np.dot(embeddings[i], embeddings[j]) / (
                np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j])
            )
            similarities[i][j] = sim
            similarities[j][i] = sim

    # Iteratively select items that maximize MMR
    while len(selected_indices) < k and len(selected_indices) < len(ranked_results):
        best_score = -1
        best_idx = -1

        for i in range(len(ranked_results)):
            if i in selected_indices:
                continue

            # Relevance: original ranking score
            relevance = ranked_results[i][0]

            # Diversity: minimum similarity to already selected items
            max_sim = 0
            for j in selected_indices:
                max_sim = max(max_sim, similarities[i][j])

            diversity = 1 - max_sim

            # MMR score
            mmr_score = lambda_param * relevance + (1 - lambda_param) * diversity

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = i

        if best_idx != -1:
            selected_indices.append(best_idx)
            selected_results.append(
                (
                    ranked_results[best_idx][0],
                    ranked_results[best_idx][1],
                    ranked_results[best_idx][2],
                    ranked_results[best_idx][3],
                )
            )

    return selected_results


async def search_embeddings(
    db_path: Path,
    query_embedding: list[float],
    k: int = 5,
    type_filter: str | None = None,
    crate_filter: str | None = None,
    module_path: str | None = None,
    has_examples: bool | None = None,
    min_doc_length: int | None = None,
    visibility: str | None = None,
    deprecated: bool | None = None,
    stability_filter: str | None = None,
) -> list[tuple[float, str, str, str]]:
    """Search for similar embeddings using k-NN with enhanced ranking and caching.

    Implements progressive filtering with selectivity analysis for optimal performance.

    Returns:
        List of (score, item_path, header, content) tuples
    """
    import aiosqlite

    if not db_path.exists():
        return []

    start_time = time.time()

    # Check cache first
    cache = get_search_cache()
    cached_results = cache.get(
        query_embedding,
        k,
        type_filter,
        crate_filter,
        module_path,
        has_examples,
        min_doc_length,
        visibility,
        deprecated,
        stability_filter,
    )
    if cached_results is not None:
        logger.debug(f"Cache hit for search with k={k}")
        return cached_results

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Load sqlite-vec extension
        await load_sqlite_vec_extension(db)

        # Prepare patterns for filtering
        crate_pattern = f"{crate_filter}::%%" if crate_filter else None
        crate_name = db_path.parent.name if db_path.parent.name != "cache" else None

        # Fix: Allow both the module itself AND items within it
        module_patterns = []
        if module_path and crate_name:
            module_patterns = [
                f"{crate_name}::{module_path}",  # The module itself
                f"{crate_name}::{module_path}::%",  # Items in the module
            ]

        # Build module filter clause
        module_filter_clause = "1=1"
        module_params = []
        if module_patterns:
            module_conditions = " OR ".join(
                ["e.item_path LIKE ?" for _ in module_patterns]
            )
            module_filter_clause = f"({module_conditions})"
            module_params = module_patterns

        # Dynamic fetch_k for over-fetching
        safe_k = min(k, 20)
        fetch_k = min(safe_k + 10, 50)

        # Perform vector search
        query = f"""
            SELECT
                v.distance,
                e.item_path,
                e.header,
                e.content,
                e.item_type,
                LENGTH(e.content) as doc_length,
                e.examples,
                e.visibility,
                e.deprecated,
                e.embedding
            FROM vec_embeddings v
            JOIN embeddings e ON v.rowid = e.id
            WHERE v.embedding MATCH ? AND k = ?
                AND (? IS NULL OR e.item_type = ?)
                AND (? IS NULL OR e.item_path LIKE ?)
                AND {module_filter_clause}
                AND (? IS NULL OR e.visibility = ?)
                AND (? IS NULL OR e.deprecated = ?)
                AND (? IS NULL OR (? = 0 OR e.examples IS NOT NULL))
                AND (? IS NULL OR LENGTH(e.content) >= ?)
                AND (? IS NULL OR ? = 'all' OR e.stability_level = ?)
            ORDER BY v.distance
            """

        params = (
            [
                bytes(sqlite_vec.serialize_float32(query_embedding)),
                fetch_k,
                type_filter,
                type_filter,
                crate_pattern,
                crate_pattern,
            ]
            + module_params
            + [
                visibility,
                visibility,
                deprecated,
                deprecated,
                has_examples,
                has_examples,
                min_doc_length,
                min_doc_length,
                stability_filter,
                stability_filter,
                stability_filter,
            ]
        )

        cursor = await db.execute(query, params)
        results = await cursor.fetchall()

        # Apply enhanced ranking
        ranked_results = []
        embeddings = []

        for row in results:
            (
                distance,
                item_path,
                header,
                content,
                item_type,
                doc_length,
                examples,
                visibility,
                deprecated,
                embedding_blob,
            ) = row

            # Deserialize embedding for MMR
            embedding = np.frombuffer(embedding_blob, dtype=np.float32)
            embeddings.append(embedding)

            # Base vector similarity score
            base_score = 1.0 - distance

            # Type-specific weight
            type_weight = (
                app_config.TYPE_WEIGHTS.get(item_type, 1.0) if item_type else 1.0
            )

            # Documentation quality score
            doc_quality = min(1.0, (doc_length or 0) / 1000)

            # Examples presence boost
            has_examples_boost = 1.2 if examples else 1.0

            # Compute composite score
            final_score = (
                app_config.RANKING_VECTOR_WEIGHT * base_score
                + app_config.RANKING_TYPE_WEIGHT * (base_score * type_weight)
                + app_config.RANKING_QUALITY_WEIGHT * doc_quality
                + app_config.RANKING_EXAMPLES_WEIGHT * has_examples_boost
            )

            # Ensure score stays in [0, 1] range
            final_score = max(0.0, min(1.0, final_score))

            ranked_results.append((final_score, item_path, header, content, item_type))

        # Sort by final score
        paired_results = list(zip(ranked_results, embeddings, strict=False))
        paired_results.sort(key=lambda x: x[0][0], reverse=True)

        ranked_results = [r for r, _ in paired_results]
        embeddings = [e for _, e in paired_results]

        # Apply MMR diversification if enabled
        if len(ranked_results) > k and app_config.RANKING_DIVERSITY_WEIGHT > 0:
            top_results = _apply_mmr_diversification(
                ranked_results,
                embeddings,
                k,
                app_config.RANKING_DIVERSITY_LAMBDA,
            )
        else:
            top_results = [
                (score, path, header, content)
                for score, path, header, content, _ in ranked_results[:k]
            ]

        # Store in cache
        cache.set(
            query_embedding,
            k,
            top_results,
            type_filter,
            crate_filter,
            module_path,
            has_examples,
            min_doc_length,
            visibility,
            deprecated,
            stability_filter,
        )

        return top_results


async def get_see_also_suggestions(
    db_path: Path,
    query_embedding: list[float],
    original_item_paths: set[str],
    k: int = 8,
    similarity_threshold: float = 0.7,
    max_suggestions: int = 5,
) -> list[str]:
    """Find related items for see-also suggestions using vector similarity.

    Args:
        db_path: Path to the database
        query_embedding: Embedding vector to find similar items for
        original_item_paths: Set of item paths to exclude from suggestions
        k: Number of candidates to fetch (over-fetch for filtering)
        similarity_threshold: Minimum similarity score (1.0 - distance) for suggestions
        max_suggestions: Maximum number of suggestions to return

    Returns:
        List of item paths for see-also suggestions
    """
    import aiosqlite

    if not db_path.exists():
        return []

    # Defensive bounds check for k parameter
    safe_k = min(k, 20)

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            await db.execute("PRAGMA journal_mode = WAL")
            await db.execute("PRAGMA busy_timeout = 5000")
            await db.execute("PRAGMA synchronous = NORMAL")
            await db.execute("PRAGMA cache_size = -64000")

            # Load sqlite-vec extension
            await load_sqlite_vec_extension(db)

            # Perform vector search with explicit k parameter
            cursor = await db.execute(
                """
                SELECT
                    v.distance,
                    e.item_path,
                    e.item_type
                FROM vec_embeddings v
                JOIN embeddings e ON v.rowid = e.id
                WHERE v.embedding MATCH ? AND k = ?
                ORDER BY v.distance
                """,
                (bytes(sqlite_vec.serialize_float32(query_embedding)), safe_k),
            )

            suggestions = []
            async for row in cursor:
                distance, item_path, item_type = row
                similarity = 1.0 - distance

                # Skip if below threshold or in original results
                if (
                    similarity < similarity_threshold
                    or item_path in original_item_paths
                ):
                    continue

                suggestions.append(item_path)

                if len(suggestions) >= max_suggestions:
                    break

            return suggestions

    except Exception as e:
        logger.warning(f"Failed to get see-also suggestions: {e}")
        return []


async def search_example_embeddings(
    db_path: Path,
    query_embedding: list[float],
    k: int = 5,
    crate_filter: str | None = None,
    language_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Search for similar code examples using dedicated example embeddings table.

    Args:
        db_path: Path to the database
        query_embedding: Query embedding vector
        k: Number of results to return
        crate_filter: Optional crate name filter
        language_filter: Optional language filter (e.g., 'rust', 'python')

    Returns:
        List of dictionaries with example information
    """
    import aiosqlite

    if not db_path.exists():
        return []

    safe_k = min(k, 20)
    fetch_k = min(safe_k + 5, 30)

    try:
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            await load_sqlite_vec_extension(db)

            # Search in dedicated example embeddings table
            query = """
                SELECT
                    v.distance,
                    e.example_text as code,
                    e.language,
                    e.item_path,
                    e.context,
                    e.example_hash
                FROM vec_example_embeddings v
                JOIN example_embeddings e ON v.rowid = e.id
                WHERE v.example_embedding MATCH ? AND k = ?
                    AND (? IS NULL OR e.crate_name = ?)
                    AND (? IS NULL OR e.language = ?)
                ORDER BY v.distance
            """

            cursor = await db.execute(
                query,
                (
                    bytes(sqlite_vec.serialize_float32(query_embedding)),
                    fetch_k,
                    crate_filter,
                    crate_filter,
                    language_filter,
                    language_filter,
                ),
            )

            results = []
            seen_hashes = set()

            async for row in cursor:
                distance, code, language, item_path, context, example_hash = row

                # Skip duplicates based on hash
                if example_hash in seen_hashes:
                    continue
                seen_hashes.add(example_hash)

                score = 1.0 - distance
                results.append(
                    {
                        "code": code,
                        "language": language,
                        "item_path": item_path,
                        "context": context,
                        "score": score,
                    }
                )

                if len(results) >= k:
                    break

            return results

    except Exception as e:
        logger.warning(f"Example search failed: {e}")
        return []


async def regex_search(
    db_path: Path,
    pattern: str,
    k: int = 10,
    type_filter: str | None = None,
    crate_filter: str | None = None,
    module_path: str | None = None,
    has_examples: bool | None = None,
    min_doc_length: int | None = None,
    visibility: str | None = None,
    deprecated: bool | None = None,
    stability_filter: str | None = None,
) -> list[tuple[float, str, str, str]]:
    """Search for items matching a regex pattern with ReDoS protection.

    Implements timeout-based ReDoS protection and pattern complexity analysis.

    Args:
        db_path: Path to the database
        pattern: Regex pattern to match against item paths and content
        k: Maximum number of results to return
        type_filter: Filter by item type (function, struct, trait, etc.)
        crate_filter: Filter to specific crate
        module_path: Filter to specific module path
        has_examples: Filter to items with examples
        min_doc_length: Minimum documentation length
        visibility: Filter by visibility level
        deprecated: Filter by deprecation status
        stability_filter: Filter by stability level

    Returns:
        List of (score, item_path, header, content) tuples
    """
    import re
    import signal
    from contextlib import contextmanager

    import aiosqlite

    # Validate pattern for ReDoS vulnerability
    def is_redos_vulnerable(pattern: str) -> bool:
        """Check if pattern may cause ReDoS."""
        # Common ReDoS patterns to avoid
        dangerous_patterns = [
            r"\(.*\+\)\+",  # Nested quantifiers like (a+)+
            r"\(.*\*\)\*",  # Nested quantifiers like (a*)*
            r"\(.*\?\)\?",  # Nested quantifiers like (a?)?
            r"\(.*\{.*\}\)\+",  # Nested quantifiers with ranges
            r".*\+.*\+",  # Multiple consecutive quantifiers
            r".*\*.*\*",  # Multiple consecutive quantifiers
        ]

        for dangerous in dangerous_patterns:
            if re.search(dangerous, pattern):
                return True
        return False

    # Timeout context manager for regex execution
    @contextmanager
    def timeout(duration):
        """Context manager for timing out operations."""

        def timeout_handler(signum, frame):
            raise TimeoutError("Regex operation timed out")

        # Set the timeout handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(duration)

        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    if not db_path.exists():
        return []

    # Check for ReDoS vulnerability
    if is_redos_vulnerable(pattern):
        raise ValueError(
            f"Regex pattern '{pattern}' may cause ReDoS (Regular Expression Denial of Service). "
            f"Avoid nested quantifiers like (a+)+ or (a*)*. "
            f"Use simpler patterns or consider using fuzzy search instead."
        )

    try:
        # Compile pattern with timeout protection
        with timeout(1):  # 1 second timeout for compilation
            compiled_pattern = re.compile(pattern, re.IGNORECASE)
    except TimeoutError:
        raise ValueError(
            "Regex pattern compilation timed out. Pattern may be too complex."
        )
    except re.error as e:
        raise ValueError(f"Invalid regex pattern: {e}")

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Build query with filters
        query_parts = [
            "SELECT item_path, header, content, item_type FROM embeddings WHERE 1=1"
        ]
        params = []

        # Apply filters
        if type_filter:
            query_parts.append("AND item_type = ?")
            params.append(type_filter)

        if crate_filter:
            query_parts.append("AND item_path LIKE ?")
            params.append(f"{crate_filter}::%")

        if module_path:
            query_parts.append("AND item_path LIKE ?")
            params.append(f"%::{module_path}::%")

        if has_examples is not None:
            if has_examples:
                query_parts.append("AND examples IS NOT NULL")
            else:
                query_parts.append("AND examples IS NULL")

        if min_doc_length:
            query_parts.append("AND LENGTH(content) >= ?")
            params.append(min_doc_length)

        if visibility:
            query_parts.append("AND visibility = ?")
            params.append(visibility)

        if deprecated is not None:
            query_parts.append("AND deprecated = ?")
            params.append(1 if deprecated else 0)

        if stability_filter and stability_filter != "all":
            query_parts.append("AND stability_level = ?")
            params.append(stability_filter)

        query = " ".join(query_parts)
        cursor = await db.execute(query, params)

        results = []
        async for row in cursor:
            item_path, header, content, item_type = row

            # Check pattern match with timeout
            try:
                with timeout(1):  # 1 second timeout per item
                    # Check if pattern matches path or content
                    path_match = compiled_pattern.search(item_path)
                    content_match = compiled_pattern.search(content)

                    if path_match or content_match:
                        # Calculate relevance score based on match location
                        if path_match:
                            # Path matches are more relevant
                            score = 1.0
                        else:
                            # Content matches are less relevant
                            score = 0.7

                        # Boost score for exact matches
                        if path_match and path_match.group() == item_path:
                            score = 1.5

                        results.append((score, item_path, header, content))

                        if len(results) >= k:
                            break
            except TimeoutError:
                logger.warning(f"Regex matching timed out for item: {item_path}")
                continue

        # Sort by score (descending)
        results.sort(key=lambda x: x[0], reverse=True)

        return results[:k]


async def cross_crate_search(
    crate_paths: list[Path],
    query_embedding: list[float],
    k: int = 5,
    type_filter: str | None = None,
    has_examples: bool | None = None,
    min_doc_length: int | None = None,
    visibility: str | None = None,
    deprecated: bool | None = None,
    stability_filter: str | None = None,
) -> list[tuple[float, str, str, str]]:
    """Search across multiple crates simultaneously with result aggregation.

    Uses parallel queries and Reciprocal Rank Fusion (RRF) for ranking.

    Args:
        crate_paths: List of paths to crate databases (max 5)
        query_embedding: Query embedding vector
        k: Number of results to return per crate
        type_filter: Filter by item type
        has_examples: Filter to items with examples
        min_doc_length: Minimum documentation length
        visibility: Filter by visibility level
        deprecated: Filter by deprecation status
        stability_filter: Filter by stability level

    Returns:
        Aggregated list of (score, item_path, header, content) tuples
    """
    import asyncio

    if len(crate_paths) > 5:
        raise ValueError(
            "Cross-crate search limited to 5 crates maximum for performance"
        )

    # Run searches in parallel
    search_tasks = []
    for db_path in crate_paths:
        if db_path.exists():
            task = search_embeddings(
                db_path=db_path,
                query_embedding=query_embedding,
                k=k * 2,  # Get more results for better aggregation
                type_filter=type_filter,
                has_examples=has_examples,
                min_doc_length=min_doc_length,
                visibility=visibility,
                deprecated=deprecated,
            )
            search_tasks.append(task)

    if not search_tasks:
        return []

    # Gather all results
    all_results = await asyncio.gather(*search_tasks)

    # Apply Reciprocal Rank Fusion (RRF) for result aggregation
    rrf_scores = {}
    rrf_k = 60  # RRF constant (typically 60)

    for crate_results in all_results:
        for rank, (score, path, header, content) in enumerate(crate_results):
            # Use path as unique identifier
            if path not in rrf_scores:
                rrf_scores[path] = {
                    "score": 0,
                    "header": header,
                    "content": content,
                    "original_score": score,
                }

            # RRF formula: 1 / (k + rank)
            rrf_scores[path]["score"] += 1 / (rrf_k + rank + 1)

    # Convert to list and sort by RRF score
    aggregated_results = [
        (
            data["score"],
            path,
            data["header"],
            data["content"],
        )
        for path, data in rrf_scores.items()
    ]

    aggregated_results.sort(key=lambda x: x[0], reverse=True)

    return aggregated_results[:k]


__all__ = [
    "search_embeddings",
    "_apply_mmr_diversification",
    "get_see_also_suggestions",
    "search_example_embeddings",
    "regex_search",
    "cross_crate_search",
]
