"""Service layer for cross-reference operations."""

import logging
import re
import time

import aiosqlite

from .. import fuzzy_resolver
from ..models.cross_references import (
    MigrationSuggestion,
    MigrationSuggestionsResponse,
)

logger = logging.getLogger(__name__)


class CircuitOpenError(Exception):
    """Exception raised when circuit breaker is open."""


class SimpleCircuitBreaker:
    """Simple circuit breaker for resilience."""

    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        """Initialize circuit breaker.

        Args:
            failure_threshold: Number of failures before opening circuit
            timeout: Seconds to wait before attempting reset
        """
        self.failures = 0
        self.threshold = failure_threshold
        self.timeout = timeout
        self.last_failure_time: float | None = None
        self.is_open_flag = False

    def is_open(self) -> bool:
        """Check if circuit is open."""
        if not self.is_open_flag:
            return False

        # Check if timeout has passed
        if (
            self.last_failure_time
            and (time.time() - self.last_failure_time) > self.timeout
        ):
            self.reset()
            return False

        return True

    def record_failure(self):
        """Record a failure."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= self.threshold:
            self.is_open_flag = True
            logger.warning(f"Circuit breaker opened after {self.failures} failures")

    def reset(self):
        """Reset the circuit breaker."""
        self.failures = 0
        self.is_open_flag = False
        self.last_failure_time = None

    async def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.is_open():
            raise CircuitOpenError("Circuit breaker is open")

        try:
            result = await func(*args, **kwargs)
            self.reset()
            return result
        except Exception:
            self.record_failure()
            raise


class CrossReferenceService:
    """Service for cross-reference operations."""

    def __init__(self, db_path: str):
        """Initialize cross-reference service.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path
        self._graph_cache: dict[str, dict] = {}
        self._cache_ttl = 300  # 5 minutes
        self._cache_timestamps: dict[str, float] = {}
        self.circuit_breaker = SimpleCircuitBreaker()

    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached entry is still valid."""
        if key not in self._cache_timestamps:
            return False
        return (time.time() - self._cache_timestamps[key]) < self._cache_ttl

    async def resolve_import(
        self, crate_name: str, import_path: str, include_alternatives: bool = False
    ) -> dict:
        """Resolve import paths and suggest alternatives.

        Args:
            crate_name: Name of the crate
            import_path: Import path to resolve
            include_alternatives: Whether to include alternative suggestions

        Returns:
            Dictionary with resolved path and alternatives
        """
        cache_key = f"import:{crate_name}:{import_path}"

        # Check cache
        if cache_key in self._graph_cache and self._is_cache_valid(cache_key):
            return self._graph_cache[cache_key]

        # Try standard resolution first
        resolved_path = await fuzzy_resolver.resolve_path_alias(
            crate_name, import_path, self.db_path
        )

        result = {
            "resolved_path": "",
            "confidence": 0.0,
            "alternatives": [],
            "source_crate": crate_name,
        }

        # If we got a resolved path that's different from the input, use it
        if resolved_path and resolved_path != import_path:
            result["resolved_path"] = resolved_path
            result["confidence"] = 0.9

        # Get alternatives if requested or if no exact match
        if include_alternatives or not resolved_path:
            # Query database for alternative import paths
            alternatives = []
            try:
                import aiosqlite
                async with aiosqlite.connect(self.db_path) as conn:
                    # Look for items with similar names in the database
                    query = """
                    SELECT DISTINCT e.item_path, 
                           r.alias_path,
                           CASE 
                               WHEN e.item_path = ? THEN 1.0
                               WHEN e.item_path LIKE ? THEN 0.8
                               WHEN e.item_path LIKE ? THEN 0.6
                               ELSE 0.4
                           END as confidence
                    FROM embeddings e
                    LEFT JOIN reexports r ON e.item_path = r.actual_path
                    WHERE e.item_path LIKE ?
                       OR r.alias_path LIKE ?
                    ORDER BY confidence DESC
                    LIMIT 5
                    """
                    
                    # Create search patterns
                    exact_pattern = import_path
                    prefix_pattern = f"{crate_name}::{import_path.split('::')[-1] if '::' in import_path else import_path}"
                    suffix_pattern = f"%::{import_path.split('::')[-1] if '::' in import_path else import_path}"
                    general_pattern = f"%{import_path.split('::')[-1] if '::' in import_path else import_path}%"
                    
                    cursor = await conn.execute(
                        query,
                        (exact_pattern, prefix_pattern, suffix_pattern, general_pattern, general_pattern)
                    )
                    rows = await cursor.fetchall()
                    
                    for path, alias, conf in rows:
                        # Skip the already resolved path
                        if path == resolved_path:
                            continue
                        
                        alt = {
                            "path": path,
                            "confidence": conf,
                            "link_type": "direct" if not alias else "reexport",
                        }
                        if alias:
                            alt["alias"] = alias
                        alternatives.append(alt)
                    
            except Exception as e:
                logger.warning(f"Failed to get alternatives for {import_path}: {e}")
            
            result["alternatives"] = alternatives

            # If no resolved path was found, return the original path
            if not result["resolved_path"]:
                result["resolved_path"] = import_path
                result["confidence"] = 0.5

        # Cache result
        self._graph_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        return result

    async def get_dependency_graph(
        self, crate_name: str, max_depth: int = 3, include_versions: bool = True
    ) -> dict:
        """Get dependency graph with version constraints.

        Args:
            crate_name: Name of the crate
            max_depth: Maximum depth to traverse
            include_versions: Whether to include version information

        Returns:
            Dictionary with dependency graph
        """
        cache_key = f"graph:{crate_name}:{max_depth}"

        # Check cache
        if cache_key in self._graph_cache and self._is_cache_valid(cache_key):
            return self._graph_cache[cache_key]

        # Build dependency graph using recursive CTE
        graph = {}
        visited_nodes = set()
        has_cycles = False

        async with aiosqlite.connect(self.db_path) as conn:
            # Get crate ID
            cursor = await conn.execute(
                "SELECT id FROM crate_metadata WHERE name = ? LIMIT 1", (crate_name,)
            )
            row = await cursor.fetchone()
            if not row:
                return {
                    "root": {"name": crate_name, "version": None, "dependencies": []},
                    "total_nodes": 0,
                    "max_depth": 0,
                    "has_cycles": False,
                }

            crate_id = row[0]

            # Use recursive CTE for efficient traversal
            # Modified to use actual schema with embeddings table and path-based relationships
            query = """
            WITH RECURSIVE dependency_tree AS (
                -- Base case: direct dependencies
                SELECT
                    r.alias_path as source_path,
                    r.actual_path as target_path,
                    r.confidence_score,
                    1 as depth,
                    r.alias_path || ',' as path,
                    e1.item_path as source_name,
                    e2.item_path as target_name,
                    cm2.version as target_version
                FROM reexports r
                LEFT JOIN embeddings e1 ON r.alias_path = e1.item_path
                LEFT JOIN embeddings e2 ON r.actual_path = e2.item_path
                LEFT JOIN crate_metadata cm2 ON 
                    SUBSTR(e2.item_path, 1, INSTR(e2.item_path || '::', '::') - 1) = cm2.name
                WHERE r.crate_id = ?
                  AND r.link_type IN ('dependency', 'crossref', 'reexport')

                UNION ALL

                -- Recursive case with cycle detection
                SELECT
                    r.alias_path,
                    r.actual_path,
                    r.confidence_score,
                    dt.depth + 1,
                    dt.path || r.alias_path || ',',
                    e1.item_path as source_name,
                    e2.item_path as target_name,
                    cm2.version as target_version
                FROM reexports r
                JOIN dependency_tree dt ON r.alias_path = dt.target_path
                LEFT JOIN embeddings e1 ON r.alias_path = e1.item_path
                LEFT JOIN embeddings e2 ON r.actual_path = e2.item_path
                LEFT JOIN crate_metadata cm2 ON 
                    SUBSTR(e2.item_path, 1, INSTR(e2.item_path || '::', '::') - 1) = cm2.name
                WHERE dt.depth < ?
                  AND dt.path NOT LIKE '%' || r.actual_path || ',%'
                  AND r.crate_id = ?
            )
            SELECT DISTINCT source_name, target_name, target_version, depth
            FROM dependency_tree
            WHERE source_name IS NOT NULL AND target_name IS NOT NULL
            ORDER BY depth, source_name, target_name;
            """

            cursor = await conn.execute(query, (crate_id, max_depth, crate_id))
            rows = await cursor.fetchall()

            # Build graph structure
            for source_name, target_name, target_version, _depth in rows:
                if source_name not in graph:
                    graph[source_name] = []

                dep_info = {"name": target_name}
                if include_versions and target_version:
                    dep_info["version"] = target_version

                if dep_info not in graph[source_name]:
                    graph[source_name].append(dep_info)

                visited_nodes.add(source_name)
                visited_nodes.add(target_name)

        # Check for cycles using DFS
        has_cycles = self._detect_cycles(graph)

        # Build hierarchical structure
        root_node = self._build_hierarchy(crate_name, graph, max_depth)

        result = {
            "root": root_node,
            "total_nodes": len(visited_nodes),
            "max_depth": max_depth,
            "has_cycles": has_cycles,
        }

        # Cache result
        self._graph_cache[cache_key] = result
        self._cache_timestamps[cache_key] = time.time()

        return result

    def _detect_cycles(self, graph: dict) -> bool:
        """Detect cycles in dependency graph using DFS.

        Args:
            graph: Adjacency list representation of graph

        Returns:
            True if cycles detected, False otherwise
        """
        visited = set()
        rec_stack = set()

        def dfs(node):
            visited.add(node)
            rec_stack.add(node)

            for neighbor_info in graph.get(node, []):
                neighbor = (
                    neighbor_info["name"]
                    if isinstance(neighbor_info, dict)
                    else neighbor_info
                )
                if neighbor not in visited:
                    if dfs(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True

            rec_stack.remove(node)
            return False

        for node in graph:
            if node not in visited:
                if dfs(node):
                    return True
        return False

    def _build_hierarchy(self, root_name: str, graph: dict, max_depth: int) -> dict:
        """Build hierarchical dependency structure.

        Args:
            root_name: Root crate name
            graph: Flat graph structure
            max_depth: Maximum depth to build

        Returns:
            Hierarchical node structure
        """
        visited = set()

        def build_node(name: str, depth: int) -> dict:
            if depth > max_depth or name in visited:
                return {"name": name, "dependencies": []}

            visited.add(name)
            node = {"name": name, "dependencies": []}

            if name in graph and depth < max_depth:
                for dep_info in graph[name]:
                    dep_name = (
                        dep_info["name"] if isinstance(dep_info, dict) else dep_info
                    )
                    dep_node = build_node(dep_name, depth + 1)
                    if isinstance(dep_info, dict) and "version" in dep_info:
                        dep_node["version"] = dep_info["version"]
                    node["dependencies"].append(dep_node)

            return node

        return build_node(root_name, 0)

    async def suggest_migrations(
        self, crate_name: str, from_version: str, to_version: str
    ) -> list[dict]:
        """Suggest migration paths between versions.
        
        Args:
            crate_name: Name of the crate
            from_version: Source version
            to_version: Target version
            
        Returns:
            List of suggestion dictionaries with keys: old_path, new_path, change_type, confidence, notes
        """
        suggestions = []
        
        async with aiosqlite.connect(self.db_path) as conn:
            # Simplified query that works with the actual schema
            # Since each database is per crate, we just need to compare items between versions
            query = """
            WITH old_version_items AS (
                SELECT DISTINCT e.item_path, e.signature, e.item_type
                FROM embeddings e
                JOIN crate_metadata cm ON cm.id = (
                    SELECT id FROM crate_metadata 
                    WHERE name = ? AND version = ?
                    LIMIT 1
                )
            ),
            new_version_items AS (
                SELECT DISTINCT e.item_path, e.signature, e.item_type
                FROM embeddings e
                JOIN crate_metadata cm ON cm.id = (
                    SELECT id FROM crate_metadata 
                    WHERE name = ? AND version = ?
                    LIMIT 1
                )
            )
            SELECT * FROM (
                -- Items that were removed
                SELECT 
                    ovi.item_path as old_path,
                    NULL as new_path,
                    'removed' as change_type,
                    0.9 as confidence
                FROM old_version_items ovi
                LEFT JOIN new_version_items nvi ON ovi.item_path = nvi.item_path
                WHERE nvi.item_path IS NULL
                
                UNION
                
                -- Items that were added
                SELECT 
                    NULL as old_path,
                    nvi.item_path as new_path,
                    'added' as change_type,
                    0.9 as confidence
                FROM new_version_items nvi
                LEFT JOIN old_version_items ovi ON nvi.item_path = ovi.item_path
                WHERE ovi.item_path IS NULL
                
                UNION
                
                -- Items with signature changes
                SELECT 
                    ovi.item_path as old_path,
                    nvi.item_path as new_path,
                    'modified' as change_type,
                    0.7 as confidence
                FROM old_version_items ovi
                JOIN new_version_items nvi ON ovi.item_path = nvi.item_path
                WHERE ovi.signature != nvi.signature OR 
                      (ovi.signature IS NULL AND nvi.signature IS NOT NULL) OR
                      (ovi.signature IS NOT NULL AND nvi.signature IS NULL)
            ) LIMIT 100
            """
            
            try:
                cursor = await conn.execute(
                    query, (crate_name, from_version, crate_name, to_version)
                )
                rows = await cursor.fetchall()
                
                for old_path, new_path, change_type, confidence in rows:
                    suggestion_dict = {
                        "old_path": old_path,
                        "new_path": new_path,
                        "change_type": change_type,
                        "confidence": confidence,
                        "notes": self._generate_migration_notes(change_type, old_path, new_path)
                    }
                    suggestions.append(suggestion_dict)
                    
            except Exception as e:
                logger.warning(f"Database query failed, using pattern-based fallback: {e}")
                # Fallback to pattern-based suggestions
                suggestions = await self._get_pattern_based_suggestions(
                    crate_name, from_version, to_version
                )
        
        return suggestions

    def _pattern_based_migrations(
        self, crate_name: str, from_version: str, to_version: str
    ) -> list[dict]:
        """Generate pattern-based migration suggestions.

        Args:
            crate_name: Name of the crate
            from_version: Starting version
            to_version: Target version

        Returns:
            List of pattern-based suggestions
        """
        suggestions = []

        # Common migration patterns
        patterns = [
            {
                "pattern": r"^(\d+)\.(\d+)\.\d+$",  # Major.Minor.Patch
                "check": lambda f, t: f.split(".")[0]
                != t.split(".")[0],  # Major version change
                "suggestions": [
                    {
                        "old_path": f"{crate_name}::Error",
                        "new_path": f"{crate_name}::error::Error",
                        "change_type": "moved",
                        "confidence": 0.7,
                    },
                    {
                        "old_path": f"{crate_name}::Result",
                        "new_path": f"{crate_name}::result::Result",
                        "change_type": "moved",
                        "confidence": 0.7,
                    },
                ],
            }
        ]

        for pattern_info in patterns:
            pattern = pattern_info["pattern"]
            if re.match(pattern, from_version) and re.match(pattern, to_version):
                if pattern_info["check"](from_version, to_version):
                    suggestions.extend(pattern_info["suggestions"])

        return suggestions

    def _generate_migration_notes(
        self, change_type: str, old_path: str | None, new_path: str | None
    ) -> str:
        """Generate helpful notes for a migration suggestion.

        Keeps guidance concise and safe for display in clients.
        """
        try:
            if change_type == "removed":
                return (
                    f"The item '{old_path}' was removed. Remove usages or find an alternative."
                )
            if change_type == "added":
                return f"A new item '{new_path}' was added. Consider adopting it if needed."
            if change_type in {"renamed", "moved"}:
                return (
                    f"The item '{old_path}' was {change_type} to '{new_path}'. Update imports and references."
                )
            if change_type == "modified":
                return f"The API of '{old_path}' changed. Review and update call sites."
        except Exception:
            pass
        return "A change was detected. Review migration guidance."

    async def _get_pattern_based_suggestions(
        self, crate_name: str, from_version: str, to_version: str
    ) -> list[dict]:
        """Async wrapper to provide pattern-based suggestions as dicts.

        This matches the tests' expectation that suggest_migrations returns a list.
        """
        return self._pattern_based_migrations(crate_name, from_version, to_version)

    async def trace_reexports(self, crate_name: str, item_path: str) -> dict:
        """Trace re-exported items to original source.

        Args:
            crate_name: Name of the crate
            item_path: Path of the item to trace

        Returns:
            Dictionary with re-export chain and original source
        """
        chain = []
        original_source = ""
        original_crate = crate_name

        async with aiosqlite.connect(self.db_path) as conn:
            # Follow re-export chain
            current_path = item_path
            current_crate = crate_name
            visited = set()
            max_iterations = 10  # Prevent infinite loops

            for _ in range(max_iterations):
                if (current_crate, current_path) in visited:
                    break  # Cycle detected
                visited.add((current_crate, current_path))

                # Modified query to use actual schema with path-based relationships
                query = """
                SELECT
                    r.actual_path as target_path,
                    cm.name as target_crate,
                    r.link_type
                FROM reexports r
                JOIN crate_metadata cm ON r.crate_id = cm.id
                WHERE r.alias_path = ? 
                  AND cm.name = ? 
                  AND r.link_type = 'reexport'
                LIMIT 1;
                """

                cursor = await conn.execute(query, (current_path, current_crate))
                row = await cursor.fetchone()

                if not row:
                    # No more re-exports, this is the original
                    original_source = current_path
                    original_crate = current_crate
                    break

                target_path, target_crate, _ = row
                chain.append(
                    f"{current_crate}::{current_path} -> {target_crate}::{target_path}"
                )
                current_path = target_path
                current_crate = target_crate

        return {
            "chain": chain,
            "original_source": original_source,
            "original_crate": original_crate,
        }

    def _generate_migration_notes(
        self, change_type: str, old_path: str | None, new_path: str | None
    ) -> str:
        """Generate helpful migration notes for a suggestion.

        Args:
            change_type: Type of change (renamed, moved, removed, added)
            old_path: Path in old version (may be None for additions)
            new_path: Path in new version (may be None for removals)

        Returns:
            Helpful migration guidance text
        """
        if change_type == "removed" and old_path:
            return f"The item '{old_path}' has been removed. Check release notes for alternatives or deprecation notices."
        elif change_type == "added" and new_path:
            return f"New item '{new_path}' was added. Consider using it if it replaces deprecated functionality."
        elif change_type == "moved" and old_path and new_path:
            return f"Item was moved from '{old_path}' to '{new_path}'. Update your imports accordingly."
        elif change_type == "renamed" and old_path and new_path:
            return f"Item was renamed from '{old_path}' to '{new_path}'. Update all references."
        elif change_type == "modified" and old_path and new_path:
            return f"The signature of '{old_path}' has changed. Review the new API at '{new_path}' for compatibility."
        else:
            return f"Change detected: {change_type}. Review documentation for migration guidance."

    async def _get_pattern_based_suggestions(
        self, crate_name: str, from_version: str, to_version: str
    ) -> list[dict]:
        """Get pattern-based migration suggestions as dict list.

        Args:
            crate_name: Name of the crate
            from_version: Starting version
            to_version: Target version

        Returns:
            List of suggestion dictionaries with required keys
        """
        pattern_suggestions = self._pattern_based_migrations(
            crate_name, from_version, to_version
        )
        
        # Convert to dict format with notes
        result = []
        for suggestion in pattern_suggestions:
            suggestion_dict = {
                "old_path": suggestion["old_path"],
                "new_path": suggestion["new_path"],
                "change_type": suggestion["change_type"],
                "confidence": suggestion["confidence"],
            }
            # Add notes using our generator
            suggestion_dict["notes"] = self._generate_migration_notes(
                suggestion["change_type"],
                suggestion["old_path"],
                suggestion["new_path"]
            )
            result.append(suggestion_dict)
        
        return result
