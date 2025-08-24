"""Database queries for pattern analysis in workflow enhancement.

This module provides specialized database queries for Phase 7 pattern extraction
and usage analysis features.
"""

import logging
import re
from collections import Counter
from typing import Any

import aiosqlite

from ..database import DB_TIMEOUT

logger = logging.getLogger(__name__)


async def analyze_usage_patterns(
    db_path: str,
    crate_name: str,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Analyze usage patterns from documentation and examples.

    Args:
        db_path: Path to the database
        crate_name: Name of the crate to analyze
        limit: Maximum number of patterns to return

    Returns:
        List of usage patterns with metadata
    """
    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Query for items with examples and signatures
        query = """
            SELECT
                item_path,
                signature,
                examples,
                item_type,
                content,
                header
            FROM embeddings
            WHERE examples IS NOT NULL
               OR signature IS NOT NULL
               OR content LIKE '%```%'
            ORDER BY LENGTH(examples) DESC
            LIMIT 1000
        """

        cursor = await db.execute(query)

        # Pattern counters
        method_patterns = Counter()
        generic_patterns = Counter()
        error_patterns = Counter()
        trait_patterns = Counter()

        async for row in cursor:
            path, sig, examples, item_type, content, header = row

            # Extract patterns from signatures
            if sig:
                # Method patterns
                method_matches = re.findall(r"fn\s+(\w+)", sig)
                for method in method_matches:
                    method_patterns[f"fn {method}"] += 1

                # Generic patterns
                generic_matches = re.findall(r"<([^>]+)>", sig)
                for generic in generic_matches:
                    if "," not in generic:  # Simple generics only
                        generic_patterns[f"<{generic}>"] += 1

                # Result/Option patterns
                if "Result<" in sig:
                    error_patterns["Result<T, E>"] += 1
                if "Option<" in sig:
                    error_patterns["Option<T>"] += 1

            # Extract patterns from examples
            if examples:
                example_text = str(examples)

                # Trait impl patterns
                trait_impls = re.findall(r"impl\s+(\w+)\s+for\s+(\w+)", example_text)
                for trait_name, _type_name in trait_impls:
                    trait_patterns[f"impl {trait_name} for Type"] += 1

                # Common method calls
                method_calls = re.findall(r"\.(\w+)\(", example_text)
                for method in method_calls[:5]:  # Limit per example
                    method_patterns[f".{method}()"] += 1

        # Combine all patterns
        all_patterns = []

        # Add method patterns
        for pattern, count in method_patterns.most_common(limit // 2):
            all_patterns.append(
                {
                    "pattern": pattern,
                    "type": "method",
                    "frequency": count,
                    "crate": crate_name,
                }
            )

        # Add generic patterns
        for pattern, count in generic_patterns.most_common(limit // 4):
            all_patterns.append(
                {
                    "pattern": pattern,
                    "type": "generic",
                    "frequency": count,
                    "crate": crate_name,
                }
            )

        # Add error handling patterns
        for pattern, count in error_patterns.most_common(limit // 4):
            all_patterns.append(
                {
                    "pattern": pattern,
                    "type": "error_handling",
                    "frequency": count,
                    "crate": crate_name,
                }
            )

        # Add trait patterns
        for pattern, count in trait_patterns.most_common(limit // 4):
            all_patterns.append(
                {
                    "pattern": pattern,
                    "type": "trait_impl",
                    "frequency": count,
                    "crate": crate_name,
                }
            )

        # Sort by frequency and return top patterns
        all_patterns.sort(key=lambda x: x["frequency"], reverse=True)
        return all_patterns[:limit]


async def get_api_evolution(
    db_path: str,
    crate_name: str,
    versions: list[str],
) -> dict[str, Any]:
    """Track API evolution across versions.

    Args:
        db_path: Path to the database
        crate_name: Name of the crate
        versions: List of versions to compare

    Returns:
        Dictionary with API evolution data
    """
    evolution = {
        "crate": crate_name,
        "versions": versions,
        "changes": [],
        "stability": {},
    }

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Query for version-specific information
        query = """
            SELECT
                cm.version,
                COUNT(DISTINCT e.item_path) as item_count,
                COUNT(DISTINCT CASE WHEN e.deprecated = 1 THEN e.item_path END) as deprecated_count,
                COUNT(DISTINCT e.item_type) as type_diversity
            FROM crate_metadata cm
            LEFT JOIN embeddings e ON cm.id = e.crate_id
            WHERE cm.name = ?
            GROUP BY cm.version
        """

        cursor = await db.execute(query, (crate_name,))

        version_stats = {}
        async for row in cursor:
            version, item_count, deprecated_count, type_diversity = row
            version_stats[version] = {
                "item_count": item_count,
                "deprecated_count": deprecated_count,
                "type_diversity": type_diversity,
                "stability_score": 1.0 - (deprecated_count / max(item_count, 1)),
            }

        evolution["version_stats"] = version_stats

        # Analyze changes between consecutive versions
        if len(versions) > 1:
            for i in range(len(versions) - 1):
                v1, v2 = versions[i], versions[i + 1]
                if v1 in version_stats and v2 in version_stats:
                    change = {
                        "from": v1,
                        "to": v2,
                        "item_change": version_stats[v2]["item_count"]
                        - version_stats[v1]["item_count"],
                        "deprecated_change": version_stats[v2]["deprecated_count"]
                        - version_stats[v1]["deprecated_count"],
                        "growth_rate": (
                            version_stats[v2]["item_count"]
                            / max(version_stats[v1]["item_count"], 1)
                        )
                        - 1,
                    }
                    evolution["changes"].append(change)

    return evolution


async def find_common_examples(
    db_path: str,
    pattern: str,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Find common examples matching a pattern.

    Args:
        db_path: Path to the database
        pattern: Pattern to search for
        limit: Maximum number of examples

    Returns:
        List of examples with context
    """
    examples = []

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Search for examples containing the pattern
        query = """
            SELECT
                item_path,
                examples,
                signature,
                item_type
            FROM embeddings
            WHERE examples LIKE ?
               OR content LIKE ?
            LIMIT ?
        """

        search_pattern = f"%{pattern}%"
        cursor = await db.execute(query, (search_pattern, search_pattern, limit * 2))

        async for row in cursor:
            path, example_text, sig, item_type = row

            if example_text and pattern.lower() in str(example_text).lower():
                # Extract relevant snippet
                text = str(example_text)
                pattern_pos = text.lower().find(pattern.lower())

                if pattern_pos >= 0:
                    # Get surrounding context
                    start = max(0, pattern_pos - 100)
                    end = min(len(text), pattern_pos + len(pattern) + 100)
                    snippet = text[start:end]

                    examples.append(
                        {
                            "item": path,
                            "type": item_type,
                            "signature": sig,
                            "snippet": snippet,
                            "pattern_match": pattern,
                        }
                    )

                    if len(examples) >= limit:
                        break

    return examples


async def get_pattern_frequencies(
    db_path: str,
    patterns: list[str],
) -> dict[str, int]:
    """Get frequencies for specific patterns.

    Args:
        db_path: Path to the database
        patterns: List of patterns to check

    Returns:
        Dictionary mapping patterns to frequencies
    """
    frequencies = {}

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        for pattern in patterns:
            # Count occurrences in examples and content
            query = """
                SELECT COUNT(*)
                FROM embeddings
                WHERE examples LIKE ?
                   OR content LIKE ?
                   OR signature LIKE ?
            """

            search_pattern = f"%{pattern}%"
            cursor = await db.execute(
                query, (search_pattern, search_pattern, search_pattern)
            )
            row = await cursor.fetchone()
            frequencies[pattern] = row[0] if row else 0

    return frequencies


async def analyze_module_patterns(
    db_path: str,
    module_path: str,
) -> dict[str, Any]:
    """Analyze patterns within a specific module.

    Args:
        db_path: Path to the database
        module_path: Path to the module

    Returns:
        Dictionary with module-specific patterns
    """
    analysis = {
        "module": module_path,
        "patterns": [],
        "statistics": {},
    }

    async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
        # Get module items
        query = """
            SELECT
                item_path,
                item_type,
                signature,
                deprecated,
                visibility
            FROM embeddings
            WHERE item_path LIKE ?
            ORDER BY item_path
        """

        module_pattern = f"{module_path}%"
        cursor = await db.execute(query, (module_pattern,))

        # Analyze module structure
        type_counts = Counter()
        visibility_counts = Counter()
        deprecated_count = 0
        signatures = []

        async for row in cursor:
            path, item_type, sig, deprecated, visibility = row

            type_counts[item_type or "unknown"] += 1
            visibility_counts[visibility or "public"] += 1
            if deprecated:
                deprecated_count += 1
            if sig:
                signatures.append(sig)

        # Extract common patterns from signatures
        if signatures:
            # Generic usage
            generic_usage = Counter()
            for sig in signatures:
                generics = re.findall(r"<([^>]+)>", sig)
                for generic in generics:
                    if "," not in generic:
                        generic_usage[generic] += 1

            # Common return types
            return_types = Counter()
            for sig in signatures:
                if " -> " in sig:
                    ret_type = sig.split(" -> ")[-1].strip()
                    return_types[ret_type] += 1

            analysis["patterns"] = {
                "common_generics": dict(generic_usage.most_common(5)),
                "common_return_types": dict(return_types.most_common(5)),
            }

        analysis["statistics"] = {
            "total_items": sum(type_counts.values()),
            "type_distribution": dict(type_counts),
            "visibility_distribution": dict(visibility_counts),
            "deprecated_ratio": deprecated_count / max(sum(type_counts.values()), 1),
        }

    return analysis
