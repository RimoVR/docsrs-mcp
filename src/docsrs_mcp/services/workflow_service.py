"""Workflow enhancement service for Phase 7 features.

This module provides progressive detail levels, usage pattern extraction,
and learning path generation for enhanced documentation workflows.
"""

import logging
import math
import re
from collections import Counter, defaultdict
from typing import Any

import aiosqlite

from ..database import DB_TIMEOUT
from ..ingest import ingest_crate
from ..models import ChangeCategory, CompareVersionsRequest
from ..version_diff import get_diff_engine

logger = logging.getLogger(__name__)


class DetailLevel:
    """Progressive detail level definitions."""

    SUMMARY = "summary"
    DETAILED = "detailed"
    EXPERT = "expert"

    ALL_LEVELS = [SUMMARY, DETAILED, EXPERT]


class WorkflowService:
    """Service for workflow enhancement features."""

    def __init__(self, db_path: str | None = None):
        """Initialize workflow service.

        Args:
            db_path: Optional database path for direct operations
        """
        self.db_path = db_path
        self._detail_cache: dict[str, Any] = {}
        self._pattern_cache: dict[str, list[dict]] = {}
        self._learning_cache: dict[str, dict] = {}

    async def get_documentation_with_detail_level(
        self,
        crate_name: str,
        item_path: str,
        detail_level: str = DetailLevel.SUMMARY,
        version: str | None = None,
    ) -> dict[str, Any]:
        """Get documentation at specified detail level.

        Args:
            crate_name: Name of the crate
            item_path: Path to the item
            detail_level: Level of detail (summary/detailed/expert)
            version: Optional version

        Returns:
            Documentation with appropriate detail level
        """
        # Validate detail level
        if detail_level not in DetailLevel.ALL_LEVELS:
            logger.warning(f"Invalid detail level {detail_level}, using summary")
            detail_level = DetailLevel.SUMMARY

        # Check cache
        cache_key = f"{crate_name}:{item_path}:{detail_level}:{version or 'latest'}"
        if cache_key in self._detail_cache:
            logger.debug(f"Cache hit for {cache_key}")
            return self._detail_cache[cache_key]

        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        # Query documentation based on detail level
        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Base query for all levels
            base_query = """
                SELECT item_path, header, content, signature,
                       item_type, examples, visibility, deprecated,
                       generic_params, trait_bounds
                FROM embeddings
                WHERE item_path = ?
            """

            cursor = await db.execute(base_query, (item_path,))
            row = await cursor.fetchone()

            if not row:
                return {
                    "error": f"Item {item_path} not found",
                    "detail_level": detail_level,
                    "available_levels": DetailLevel.ALL_LEVELS,
                }

            # Parse row data
            (
                path,
                header,
                content,
                signature,
                item_type,
                examples,
                visibility,
                deprecated,
                generics,
                bounds,
            ) = row

            # Build response based on detail level
            result = {
                "item_path": path,
                "detail_level": detail_level,
                "available_levels": DetailLevel.ALL_LEVELS,
            }

            if detail_level == DetailLevel.SUMMARY:
                # Summary: Just essential information
                result.update(
                    {
                        "summary": header or "No summary available",
                        "signature": signature,
                        "type": item_type or "unknown",
                        "visibility": visibility or "public",
                        "deprecated": bool(deprecated),
                    }
                )

            elif detail_level == DetailLevel.DETAILED:
                # Detailed: Full documentation without implementation details
                result.update(
                    {
                        "summary": header or "No summary available",
                        "documentation": content or "No documentation available",
                        "signature": signature,
                        "type": item_type or "unknown",
                        "visibility": visibility or "public",
                        "deprecated": bool(deprecated),
                        "examples": examples if examples else None,
                    }
                )

            else:  # DetailLevel.EXPERT
                # Expert: Everything including implementation details
                result.update(
                    {
                        "summary": header or "No summary available",
                        "documentation": content or "No documentation available",
                        "signature": signature,
                        "type": item_type or "unknown",
                        "visibility": visibility or "public",
                        "deprecated": bool(deprecated),
                        "examples": examples if examples else None,
                        "generic_params": generics,
                        "trait_bounds": bounds,
                    }
                )

                # Add related items for expert level
                related_query = """
                    SELECT item_path, signature, item_type
                    FROM embeddings
                    WHERE parent_id = (
                        SELECT parent_id FROM embeddings WHERE item_path = ?
                    )
                    AND item_path != ?
                    LIMIT 10
                """
                cursor = await db.execute(related_query, (item_path, item_path))
                related = []
                async for rel_row in cursor:
                    related.append(
                        {
                            "path": rel_row[0],
                            "signature": rel_row[1],
                            "type": rel_row[2],
                        }
                    )
                if related:
                    result["related_items"] = related

        # Cache the result
        self._detail_cache[cache_key] = result
        return result

    async def extract_usage_patterns(
        self,
        crate_name: str,
        version: str | None = None,
        limit: int = 10,
        min_frequency: int = 2,
    ) -> list[dict[str, Any]]:
        """Extract common usage patterns from documentation and examples.

        Args:
            crate_name: Name of the crate
            version: Optional version
            limit: Maximum patterns to return
            min_frequency: Minimum pattern frequency

        Returns:
            List of usage patterns with frequency and examples
        """
        # Check cache
        cache_key = f"{crate_name}:{version or 'latest'}:patterns"
        if cache_key in self._pattern_cache:
            patterns = self._pattern_cache[cache_key]
            return patterns[:limit]

        # Ensure crate is ingested
        db_path = await ingest_crate(crate_name, version)

        patterns = []
        pattern_counter = Counter()
        pattern_examples = defaultdict(list)

        async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
            # Query examples and signatures
            query = """
                SELECT item_path, signature, examples, item_type
                FROM embeddings
                WHERE examples IS NOT NULL
                   OR signature IS NOT NULL
            """

            cursor = await db.execute(query)

            # Pattern extraction regex patterns
            method_call_pattern = re.compile(r"\b(\w+)\.(\w+)\s*\(")
            generic_pattern = re.compile(r"<([^>]+)>")
            result_pattern = re.compile(r"Result<([^,>]+)(?:,\s*([^>]+))?>")

            async for row in cursor:
                path, signature, examples, item_type = row

                # Extract patterns from signature
                if signature:
                    # Method call patterns
                    for match in method_call_pattern.finditer(signature):
                        pattern = f"{match.group(1)}.{match.group(2)}()"
                        pattern_counter[pattern] += 1
                        pattern_examples[pattern].append(
                            {
                                "item": path,
                                "context": signature[:100],
                            }
                        )

                    # Generic patterns
                    for match in generic_pattern.finditer(signature):
                        pattern = f"Generic<{match.group(1)}>"
                        pattern_counter[pattern] += 1
                        pattern_examples[pattern].append(
                            {
                                "item": path,
                                "context": signature[:100],
                            }
                        )

                    # Result patterns
                    for match in result_pattern.finditer(signature):
                        ok_type = match.group(1)
                        err_type = match.group(2) or "Error"
                        pattern = f"Result<{ok_type}, {err_type}>"
                        pattern_counter[pattern] += 1
                        pattern_examples[pattern].append(
                            {
                                "item": path,
                                "context": signature[:100],
                            }
                        )

                # Extract patterns from examples
                if examples:
                    # Parse examples as string
                    example_text = str(examples) if examples else ""

                    # Method calls in examples
                    for match in method_call_pattern.finditer(example_text):
                        pattern = f"{match.group(1)}.{match.group(2)}()"
                        pattern_counter[pattern] += 1
                        if len(pattern_examples[pattern]) < 3:
                            pattern_examples[pattern].append(
                                {
                                    "item": path,
                                    "context": example_text[
                                        max(0, match.start() - 50) : match.end() + 50
                                    ],
                                }
                            )

        # Build pattern results
        for pattern, count in pattern_counter.most_common():
            if count >= min_frequency:
                patterns.append(
                    {
                        "pattern": pattern,
                        "frequency": count,
                        "confidence": min(1.0, count / 10.0),  # Simple confidence score
                        "examples": pattern_examples[pattern][:3],  # Limit examples
                        "pattern_type": self._categorize_pattern(pattern),
                    }
                )

        # Sort by frequency and limit
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        patterns = patterns[:limit]

        # Cache results
        self._pattern_cache[cache_key] = patterns
        return patterns

    def _categorize_pattern(self, pattern: str) -> str:
        """Categorize a pattern type."""
        # Check for method call patterns (e.g., ".method()", "obj.method()")
        if (
            "." in pattern
            and "(" in pattern
            and pattern.index(".") < pattern.index("(")
        ):
            return "method_call"
        elif "Result<" in pattern:
            return "error_handling"
        elif "Option<" in pattern:
            return "optional_value"
        elif "Generic<" in pattern or "<" in pattern:
            return "generic_type"
        elif "impl" in pattern:
            return "trait_implementation"
        else:
            return "other"

    async def generate_learning_path(
        self,
        crate_name: str,
        from_version: str | None = None,
        to_version: str | None = None,
        focus_areas: list[str] | None = None,
    ) -> dict[str, Any]:
        """Generate learning path for API migration or onboarding.

        Args:
            crate_name: Name of the crate
            from_version: Starting version (None for new users)
            to_version: Target version
            focus_areas: Optional list of focus areas

        Returns:
            Structured learning path with steps and resources
        """
        # Cache key
        cache_key = f"{crate_name}:{from_version or 'new'}:{to_version or 'latest'}"
        if cache_key in self._learning_cache:
            return self._learning_cache[cache_key]

        learning_path = {
            "crate_name": crate_name,
            "from_version": from_version or "new_user",
            "to_version": to_version or "latest",
            "steps": [],
            "estimated_effort": "unknown",
        }

        if from_version and to_version:
            # Migration path between versions
            learning_path["type"] = "migration"

            # Use version diff engine for migration analysis
            diff_engine = get_diff_engine()

            request = CompareVersionsRequest(
                crate_name=crate_name,
                version_a=from_version,
                version_b=to_version,
                categories=[
                    ChangeCategory.BREAKING,
                    ChangeCategory.DEPRECATED,
                    ChangeCategory.ADDED,
                ],
            )

            try:
                diff_result = await diff_engine.compare_versions(request)

                # Generate migration steps from diff
                steps = []

                # Step 1: Handle breaking changes
                if diff_result.summary.breaking_changes > 0:
                    steps.append(
                        {
                            "order": 1,
                            "title": "Address Breaking Changes",
                            "description": f"Fix {diff_result.summary.breaking_changes} breaking changes",
                            "items": [
                                {
                                    "path": hint.affected_path,
                                    "action": hint.suggested_fix,
                                    "severity": hint.severity.value,
                                }
                                for hint in diff_result.migration_hints[:5]
                            ],
                            "estimated_time": f"{diff_result.summary.breaking_changes * 15} minutes",
                        }
                    )

                # Step 2: Update deprecated items
                if diff_result.summary.deprecated_items > 0:
                    steps.append(
                        {
                            "order": 2,
                            "title": "Replace Deprecated Items",
                            "description": f"Update {diff_result.summary.deprecated_items} deprecated items",
                            "estimated_time": f"{diff_result.summary.deprecated_items * 10} minutes",
                        }
                    )

                # Step 3: Adopt new features
                if diff_result.summary.added_items > 0:
                    steps.append(
                        {
                            "order": 3,
                            "title": "Explore New Features",
                            "description": f"Learn about {diff_result.summary.added_items} new items",
                            "estimated_time": f"{diff_result.summary.added_items * 5} minutes",
                        }
                    )

                learning_path["steps"] = steps

                # Estimate total effort
                total_minutes = (
                    diff_result.summary.breaking_changes * 15
                    + diff_result.summary.deprecated_items * 10
                    + diff_result.summary.added_items * 5
                )

                if total_minutes < 60:
                    learning_path["estimated_effort"] = f"{total_minutes} minutes"
                elif total_minutes < 480:
                    learning_path["estimated_effort"] = f"{total_minutes // 60} hours"
                else:
                    learning_path["estimated_effort"] = f"{total_minutes // 480} days"

            except Exception as e:
                logger.error(f"Failed to generate migration path: {e}")
                learning_path["error"] = str(e)

        else:
            # Onboarding path for new users
            learning_path["type"] = "onboarding"

            # Ensure crate is ingested
            db_path = await ingest_crate(crate_name, to_version)

            async with aiosqlite.connect(db_path, timeout=DB_TIMEOUT) as db:
                # Get key modules and types
                modules_query = """
                    SELECT DISTINCT item_path, item_type, header
                    FROM embeddings
                    WHERE item_type IN ('module', 'struct', 'trait', 'function')
                    AND parent_id IS NULL
                    ORDER BY item_path
                    LIMIT 20
                """

                cursor = await db.execute(modules_query)
                core_items = []
                async for row in cursor:
                    core_items.append(
                        {
                            "path": row[0],
                            "type": row[1],
                            "description": row[2] or "",
                        }
                    )

                # Build onboarding steps
                steps = [
                    {
                        "order": 1,
                        "title": "Core Concepts",
                        "description": "Understand fundamental types and modules",
                        "items": [
                            item
                            for item in core_items
                            if item["type"] in ["module", "trait"]
                        ][:5],
                        "estimated_time": "30 minutes",
                    },
                    {
                        "order": 2,
                        "title": "Key Data Structures",
                        "description": "Learn about main structs and enums",
                        "items": [
                            item
                            for item in core_items
                            if item["type"] in ["struct", "enum"]
                        ][:5],
                        "estimated_time": "45 minutes",
                    },
                    {
                        "order": 3,
                        "title": "Common Functions",
                        "description": "Explore frequently used functions",
                        "items": [
                            item for item in core_items if item["type"] == "function"
                        ][:5],
                        "estimated_time": "30 minutes",
                    },
                ]

                # Add focus area steps if specified
                if focus_areas:
                    for i, area in enumerate(focus_areas[:3], start=4):
                        area_query = """
                            SELECT item_path, item_type, header
                            FROM embeddings
                            WHERE item_path LIKE ?
                               OR header LIKE ?
                            LIMIT 5
                        """
                        cursor = await db.execute(
                            area_query, (f"%{area}%", f"%{area}%")
                        )
                        area_items = []
                        async for row in cursor:
                            area_items.append(
                                {
                                    "path": row[0],
                                    "type": row[1],
                                    "description": row[2] or "",
                                }
                            )

                        if area_items:
                            steps.append(
                                {
                                    "order": i,
                                    "title": f"Focus: {area.title()}",
                                    "description": f"Deep dive into {area}",
                                    "items": area_items,
                                    "estimated_time": "60 minutes",
                                }
                            )

                learning_path["steps"] = steps
                learning_path["estimated_effort"] = f"{len(steps) * 45} minutes"

        # Cache the result
        self._learning_cache[cache_key] = learning_path
        return learning_path

    def _calculate_confidence_score(self, frequency: int, total: int) -> float:
        """Calculate pattern confidence score."""
        if total == 0:
            return 0.0
        base_score = frequency / total
        # Apply logarithmic scaling for better distribution
        scaled_score = math.log(frequency + 1) / math.log(total + 1)
        return min(1.0, (base_score + scaled_score) / 2)
