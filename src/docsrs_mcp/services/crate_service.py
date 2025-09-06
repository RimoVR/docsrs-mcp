"""Service layer for crate-related operations."""

import logging

import aiohttp
import aiosqlite

from .. import config

from ..database import (
    get_module_tree as get_module_tree_from_db,
)
from ..database import (
    get_see_also_suggestions,
    search_embeddings,
    search_example_embeddings,
)
from ..fuzzy_resolver import get_fuzzy_suggestions_with_fallback, resolve_path_alias
from ..ingest import get_embedding_model, ingest_crate
from ..models import (
    CodeExample,
    CrateModule,
    ModuleTreeNode,
    SearchResult,
)
from ..version_diff import get_diff_engine

logger = logging.getLogger(__name__)


class CrateService:
    """Service for crate-related operations."""

    async def get_crate_summary(
        self, crate_name: str, version: str | None = None
    ) -> dict:
        """Get summary information about a Rust crate.

        Args:
            crate_name: Name of the crate
            version: Optional version (defaults to latest)

        Returns:
            Dictionary containing crate metadata and modules
        """
        # Ingest crate if not already done
        db_path = await ingest_crate(crate_name, version)

        # Fetch crate metadata from database
        async with aiosqlite.connect(db_path) as db:
            cursor = await db.execute(
                "SELECT name, version, description, repository, documentation FROM crate_metadata LIMIT 1"
            )
            row = await cursor.fetchone()

            if not row:
                raise ValueError("Crate not found")

            name, version, description, repository, documentation = row

            # Fetch modules with hierarchy information, filtering out noise
            cursor = await db.execute(
                """
                SELECT name, path, parent_id, depth, item_count 
                FROM modules 
                WHERE depth <= 3
                  AND (item_count >= 2 OR depth = 0)
                  AND path NOT LIKE '%target::%'
                  AND path NOT LIKE '%.cargo::%'
                  AND path NOT LIKE '%build::%'
                  AND path NOT LIKE '%out::%'
                  AND path NOT LIKE '%tests::%'
                  AND path NOT LIKE '%benches::%'
                  AND path NOT LIKE '%examples::%'
                  AND path NOT LIKE '%deps::%'
                  AND path NOT LIKE '%__pycache__%'
                ORDER BY depth ASC, path ASC
                """
            )

            all_rows = await cursor.fetchall()

            # Additional filtering in Python for more complex patterns
            filtered_modules = []
            for row in all_rows:
                module_name, path, parent_id, depth, item_count = row

                # Skip internal/private modules (often start with underscore)
                if module_name.startswith("_") and depth > 0:
                    continue

                # Skip generated modules
                if any(
                    pattern in path.lower()
                    for pattern in ["generated", "autogen", ".pb.", "proto"]
                ):
                    continue

                # Include the module
                filtered_modules.append(
                    CrateModule(
                        name=module_name,
                        path=path,
                        parent_id=parent_id,
                        depth=depth,
                        item_count=item_count,
                    )
                )

            # If we filtered out too many modules, include some key ones back
            if len(filtered_modules) < 5 and len(all_rows) > 10:
                cursor = await db.execute(
                    """
                    SELECT name, path, parent_id, depth, item_count 
                    FROM modules 
                    WHERE (depth <= 1 OR item_count >= 5)
                      AND path NOT LIKE '%target::%'
                      AND path NOT LIKE '%.cargo::%'
                    ORDER BY depth ASC, item_count DESC
                    LIMIT 20
                    """
                )
                filtered_modules = [
                    CrateModule(
                        name=row[0],
                        path=row[1],
                        parent_id=row[2],
                        depth=row[3],
                        item_count=row[4],
                    )
                    for row in await cursor.fetchall()
                ]

            return {
                "name": name,
                "version": version,
                "description": description or "",
                "modules": filtered_modules,
                "repository": repository,
                "documentation": documentation,
            }

    async def search_items(
        self,
        crate_name: str,
        query: str,
        version: str | None = None,
        k: int = 5,
        item_type: str | None = None,
        crate_filter: str | None = None,
        module_path: str | None = None,
        has_examples: bool | None = None,
        min_doc_length: int | None = None,
        visibility: str | None = None,
        deprecated: bool | None = None,
    ) -> list[SearchResult]:
        """Search for items in a crate's documentation.

        Args:
            crate_name: Name of the crate
            query: Search query
            version: Optional version
            k: Number of results
            item_type: Filter by item type
            crate_filter: Filter by crate
            module_path: Filter by module path
            has_examples: Filter by having examples
            min_doc_length: Minimum documentation length
            visibility: Filter by visibility
            deprecated: Filter by deprecation status

        Returns:
            List of search results
        """
        # Ingest crate if not already done
        db_path = await ingest_crate(crate_name, version)

        # Generate embedding for query
        model = get_embedding_model()
        query_embedding = list(model.embed([query]))[0]

        # Search embeddings with filters
        results = await search_embeddings(
            db_path,
            query_embedding,
            k=k,
            type_filter=item_type,
            crate_filter=crate_filter,
            module_path=module_path,
            has_examples=has_examples,
            min_doc_length=min_doc_length,
            visibility=visibility,
            deprecated=deprecated,
        )

        # Get see-also suggestions if we have results
        suggestions = []
        if results:
            # Get the item paths from the main results to exclude from suggestions
            original_paths = {item_path for _, item_path, _, _ in results}

            # Get suggestions using the same query embedding
            suggestions = await get_see_also_suggestions(
                db_path,
                query_embedding,
                original_paths,
                k=10,  # Over-fetch to allow filtering
                similarity_threshold=0.7,
                max_suggestions=5,
            )

        # Convert to response format
        search_results = []
        for i, (score, item_path, header, content) in enumerate(results):
            from .. import config
            from ..app import extract_smart_snippet

            # Check if item is from stdlib or dependency
            is_stdlib = crate_name in config.STDLIB_CRATES
            is_dependency = False

            # Check if this is a dependency item (if dependency filter is enabled)
            if config.DEPENDENCY_FILTER_ENABLED:
                from ..dependency_filter import get_dependency_filter

                dep_filter = get_dependency_filter()
                is_dependency = dep_filter.is_dependency(item_path, crate_name)

            result = SearchResult(
                score=score,
                item_path=item_path,
                header=header,
                snippet=extract_smart_snippet(content),
                is_stdlib=is_stdlib,
                is_dependency=is_dependency,
            )
            # Add suggestions only to the first result to avoid redundancy
            if i == 0 and suggestions:
                result.suggestions = suggestions
            search_results.append(result)

        return search_results

    async def get_item_doc(
        self, crate_name: str, item_path: str, version: str | None = None
    ) -> dict:
        """Get complete documentation for a specific item.

        Args:
            crate_name: Name of the crate
            item_path: Path to the item
            version: Optional version

        Returns:
            Dictionary with item documentation
        """
        # Ingest crate if not already done
        db_path = await ingest_crate(crate_name, version)

        # Try to resolve the path using alias resolution
        resolved_item_path = await resolve_path_alias(db_path, item_path)

        async with aiosqlite.connect(db_path) as db:
            # Fetch item documentation with flexible path matching
            cursor = await db.execute(
                """
                SELECT item_path, content 
                FROM embeddings 
                WHERE item_path = ? OR item_path = ? OR item_path LIKE ?
                """,
                (
                    resolved_item_path,
                    item_path,
                    f"%::{item_path.replace('::', '%::').split('::')[-1]}",
                ),
            )
            row = await cursor.fetchone()

            if row:
                from .. import config

                # Check if item is from stdlib or dependency
                is_stdlib = crate_name in config.STDLIB_CRATES
                is_dependency = False

                # Check if this is a dependency item (if dependency filter is enabled)
                if config.DEPENDENCY_FILTER_ENABLED:
                    from ..dependency_filter import get_dependency_filter

                    dep_filter = get_dependency_filter()
                    is_dependency = dep_filter.is_dependency(row[0], crate_name)

                return {
                    "item_path": row[0],
                    "documentation": row[1],
                    "is_stdlib": is_stdlib,
                    "is_dependency": is_dependency,
                }

            # If not found, try fuzzy suggestions
            suggestions = await get_fuzzy_suggestions_with_fallback(
                query=item_path,
                db_path=str(db_path),
                crate_name=crate_name,
                version=version or "latest",
            )

            if suggestions:
                suggestion_text = "\n".join(
                    [f"- `{s}` (item)" for s in suggestions[:5]]
                )
                error_message = (
                    f"Item '{item_path}' not found. Did you mean one of these?\n\n"
                    f"{suggestion_text}\n\n"
                    f"Use the exact path from the suggestions above."
                )
            else:
                error_message = (
                    f"Item '{item_path}' not found in crate '{crate_name}'. "
                    f"Try searching with search_items first."
                )

            raise ValueError(error_message)

    def _build_module_tree(
        self, flat_modules: list[dict], crate_name: str
    ) -> ModuleTreeNode:
        """Build a hierarchical tree structure from flat module list.

        Args:
            flat_modules: List of module dictionaries with parent_id relationships
            crate_name: Name of the crate (used for root node if empty)

        Returns:
            Root ModuleTreeNode with nested children
        """
        if not flat_modules:
            # Return empty root node for empty crate
            return ModuleTreeNode(
                name=crate_name,
                path=crate_name,
                depth=0,
                item_count=0,
                children=[],
            )

        # Build parent-child mapping
        children_map: dict[int | None, list[dict]] = {}
        modules_by_id: dict[int, dict] = {}
        root_modules = []

        for module in flat_modules:
            module_id = module.get("id")
            parent_id = module.get("parent_id")
            
            if module_id is not None:
                modules_by_id[module_id] = module
            
            if parent_id is None:
                root_modules.append(module)
            else:
                children_map.setdefault(parent_id, []).append(module)

        def build_node(module: dict) -> ModuleTreeNode:
            """Recursively build tree node from module dict."""
            module_id = module.get("id")
            children_list = children_map.get(module_id, [])
            
            return ModuleTreeNode(
                name=module.get("name", ""),
                path=module.get("path", ""),
                depth=module.get("depth", 0),
                item_count=module.get("item_count", 0),
                children=[build_node(child) for child in children_list],
            )

        # Handle edge cases
        if len(root_modules) == 1:
            # Single root - use it directly
            return build_node(root_modules[0])
        elif len(root_modules) > 1:
            # Multiple roots - create synthetic root
            return ModuleTreeNode(
                name=crate_name,
                path=crate_name,
                depth=0,
                item_count=sum(m.get("item_count", 0) for m in root_modules),
                children=[build_node(root) for root in root_modules],
            )
        else:
            # No root modules found - shouldn't happen but handle gracefully
            # Try to find the module with the smallest depth
            if flat_modules:
                min_depth_module = min(flat_modules, key=lambda m: m.get("depth", 0))
                return build_node(min_depth_module)
            
            # Fallback to empty root
            return ModuleTreeNode(
                name=crate_name,
                path=crate_name,
                depth=0,
                item_count=0,
                children=[],
            )

    async def get_module_tree(
        self, crate_name: str, version: str | None = None
    ) -> dict:
        """Get the module hierarchy tree for a crate.

        Args:
            crate_name: Name of the crate
            version: Optional version

        Returns:
            Dictionary with module tree
        """
        # Ingest crate if not already done
        db_path = await ingest_crate(crate_name, version)

        # Get flat module list from database
        flat_modules = await get_module_tree_from_db(db_path)
        
        # Build hierarchical tree structure
        tree = self._build_module_tree(flat_modules, crate_name)

        return {
            "crate_name": crate_name,
            "version": version or "latest",
            "tree": tree,
        }

    async def search_examples(
        self,
        crate_name: str,
        query: str,
        version: str | None = None,
        k: int = 5,
        language: str | None = None,
    ) -> dict:
        """Search for code examples in crate documentation.

        Args:
            crate_name: Name of the crate
            query: Search query
            version: Optional version
            k: Number of results
            language: Filter by programming language

        Returns:
            Dictionary with examples
        """
        # Ingest crate if not already done
        db_path = await ingest_crate(crate_name, version)

        # Generate embedding for query
        model = get_embedding_model()
        query_embedding = list(model.embed([query]))[0]

        # Search example embeddings
        results = await search_example_embeddings(
            db_path, query_embedding, k=k, language_filter=language
        )

        # Format results - results is a list of dictionaries
        examples = []
        for result in results:
            # Extract fields from dictionary
            code = result.get("code", "")
            score = result.get("score", 0.0)
            item_path = result.get("item_path", "")
            language = result.get("language", "unknown")
            context = result.get("context", "")
            
            # Join code list into a single string if it's a list
            if isinstance(code, list):
                code = "".join(code)

            examples.append(
                CodeExample(
                    code=code,
                    language=language,
                    detected=True,  # We always detect language from the database
                    item_path=item_path,  # Use the correct field name
                    context=context,
                    score=score,
                )
            )

        return {
            "crate_name": crate_name,
            "version": version or "latest",
            "query": query,
            "examples": examples,
            "total_count": len(examples),
        }

    async def list_versions(self, crate_name: str) -> dict:
        """List all available versions of a crate.

        Args:
            crate_name: Name of the crate

        Returns:
            Dictionary with version information
        """
        try:
            # Query crates.io API for all versions
            url = f"https://crates.io/api/v1/crates/{crate_name}/versions"
            headers = {
                "User-Agent": f"docsrs-mcp/{config.VERSION} (https://github.com/anthropics/docsrs-mcp)"
            }

            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status == 404:
                        raise ValueError(f"Crate '{crate_name}' not found on crates.io")
                    elif resp.status != 200:
                        raise Exception(
                            f"Failed to fetch versions for {crate_name}: HTTP {resp.status}"
                        )

                    data = await resp.json()
                    versions_data = data.get("versions", [])

                    if not versions_data:
                        raise ValueError(f"No versions found for crate '{crate_name}'")

                    # Process versions and find latest
                    versions = []
                    latest_version = None
                    latest_timestamp = 0

                    for version_info in versions_data:
                        version_str = version_info.get("num", "unknown")
                        yanked = version_info.get("yanked", False)
                        created_at = version_info.get("created_at", "")
                        
                        # Parse timestamp to find latest non-yanked version
                        try:
                            import dateutil.parser
                            timestamp = dateutil.parser.parse(created_at).timestamp()
                            if not yanked and (latest_version is None or timestamp > latest_timestamp):
                                latest_version = version_str
                                latest_timestamp = timestamp
                        except:
                            # If timestamp parsing fails, still include the version
                            if not yanked and latest_version is None:
                                latest_version = version_str

                        versions.append({
                            "version": version_str,
                            "yanked": yanked,
                            "is_latest": False,  # Will be set correctly below
                        })

                    # Mark the latest version
                    if latest_version:
                        for version in versions:
                            if version["version"] == latest_version:
                                version["is_latest"] = True
                                break

                    # Sort versions by semver if possible, fallback to string sort
                    try:
                        import semver
                        # Filter to valid semver versions for sorting
                        semver_versions = []
                        other_versions = []
                        
                        for version in versions:
                            try:
                                semver.Version.parse(version["version"])
                                semver_versions.append(version)
                            except:
                                other_versions.append(version)
                        
                        # Sort semver versions in descending order (newest first)
                        semver_versions.sort(
                            key=lambda v: semver.Version.parse(v["version"]), 
                            reverse=True
                        )
                        
                        # Sort other versions by string (newest first)
                        other_versions.sort(key=lambda v: v["version"], reverse=True)
                        
                        # Combine with semver versions first
                        versions = semver_versions + other_versions
                        
                    except ImportError:
                        # Fallback to simple string sorting
                        versions.sort(key=lambda v: v["version"], reverse=True)

                    return {
                        "crate_name": crate_name,
                        "versions": versions,
                        "latest": latest_version,
                    }

        except ValueError:
            # Re-raise validation errors as-is
            raise
        except Exception as e:
            # For network or parsing errors, fall back to local version if possible
            logger.warning(f"Failed to fetch versions from crates.io for {crate_name}: {e}")
            
            try:
                # Try to get version from local cache as fallback
                db_path = await ingest_crate(crate_name, "latest")
                
                async with aiosqlite.connect(db_path) as db:
                    cursor = await db.execute("SELECT version FROM crate_metadata LIMIT 1")
                    row = await cursor.fetchone()
                    
                    if row:
                        current_version = row[0]
                        return {
                            "crate_name": crate_name,
                            "versions": [
                                {
                                    "version": current_version,
                                    "yanked": False,
                                    "is_latest": True,
                                }
                            ],
                            "latest": current_version,
                        }
                    
            except Exception as fallback_error:
                logger.error(f"Fallback failed for {crate_name}: {fallback_error}")
            
            # If all else fails, raise the original error
            raise Exception(f"Unable to fetch versions for {crate_name}: {e}")

    async def compare_versions(
        self,
        crate_name: str,
        version_a: str,
        version_b: str,
        categories: str | list[str] | None = None,
        include_unchanged: bool = False,
        max_results: int = 1000,
    ) -> dict:
        """Compare two versions of a crate.

        Args:
            crate_name: Name of the crate
            version_a: First version
            version_b: Second version
            categories: Categories to include
            include_unchanged: Include unchanged items
            max_results: Maximum results

        Returns:
            Dictionary with version diff
        """
        # Ingest both versions (ensure they're available)
        await ingest_crate(crate_name, version_a)
        await ingest_crate(crate_name, version_b)

        # Create comparison request
        from ..models import CompareVersionsRequest

        request = CompareVersionsRequest(
            crate_name=crate_name,
            version_a=version_a,
            version_b=version_b,
            categories=categories,
            include_unchanged=include_unchanged,
            max_results=max_results,
        )

        # Compute diff using correct method
        engine = get_diff_engine()
        diff_result = await engine.compare_versions(request)

        # Convert response to dict format
        return {
            "crate_name": diff_result.crate_name,
            "version_a": diff_result.version_a,
            "version_b": diff_result.version_b,
            "summary": diff_result.summary,
            "changes": diff_result.changes,
        }
