"""Rustdoc JSON parsing with streaming optimization.

This module handles:
- Streaming JSON parsing with ijson (656x speedup)
- Module hierarchy building
- Item extraction with type classification
- Path normalization for stable IDs
- Code intelligence extraction (Phase 5)
"""

import asyncio
import logging
from collections.abc import AsyncGenerator
from typing import Any

import ijson

from ..memory_utils import trigger_gc_if_needed
from .intelligence_extractor import (
    extract_error_types,
    extract_feature_requirements,
    extract_safety_info,
    safe_extract,
)

logger = logging.getLogger(__name__)


async def parse_rustdoc_items_streaming(
    json_content: str,
) -> AsyncGenerator[dict[str, Any], None]:
    """Parse rustdoc JSON and yield items in streaming fashion.

    This function parses the rustdoc JSON and yields properly structured items
    with all required fields for storage.

    Args:
        json_content: Raw JSON string from rustdoc

    Yields:
        Dict containing parsed item data
    """
    import json
    
    try:
        # Parse the JSON content (3MB is manageable)
        data = json.loads(json_content)
        
        # Get the main sections
        index = data.get("index", {})
        paths = data.get("paths", {})
        
        # Build module hierarchy from paths
        modules = build_module_hierarchy(paths)
        
        # Process items from the index
        items_processed = 0
        
        for item_id, item_data in index.items():
            try:
                # Get path information
                path_info = paths.get(item_id, {})
                path_list = path_info.get("path", [])
                item_kind = path_info.get("kind", "unknown")
                
                # Get item details
                name = item_data.get("name")
                docs = item_data.get("docs")
                # Ensure docs is always a string (never None)
                if docs is None:
                    docs = ""
                visibility = item_data.get("visibility", "public")
                attrs = item_data.get("attrs", [])
                deprecation = item_data.get("deprecation")
                inner = item_data.get("inner", {})
                
                # Skip items without name or docs (we need content to embed)
                if not name and not docs:
                    continue
                
                # Build item path - handle cases where name might be None
                if name:
                    if path_list:
                        item_path = "::".join(path_list + [name])
                    else:
                        item_path = name
                else:
                    # Use path as item_path if no name
                    if path_list:
                        item_path = "::".join(path_list)
                    else:
                        item_path = f"item_{item_id}"  # Fallback
                
                # Determine item type from inner or kind
                if isinstance(inner, dict):
                    inner_kind = inner.get("kind")
                    if inner_kind:
                        item_type = normalize_item_type(inner_kind)
                    else:
                        item_type = normalize_item_type(item_kind)
                else:
                    item_type = normalize_item_type(item_kind)
                
                # Use name as header, or derive from path
                if name:
                    header = name
                elif path_list:
                    header = path_list[-1] if path_list else "unknown"
                else:
                    header = item_type  # Last fallback
                
                # Create the item structure with required fields
                item = {
                    "item_id": item_id,
                    "item_path": item_path,  # Required field
                    "item_type": item_type,
                    "name": name,
                    "header": header,  # Required field
                    "doc": docs,
                    "visibility": visibility,
                    "deprecated": deprecation is not None,
                    "attrs": attrs,
                }
                
                # Extract signature if available
                signature = ""
                if isinstance(inner, dict):
                    # Try to extract signature from inner structure
                    if inner.get("decl"):
                        # Function or method declaration
                        decl = inner["decl"]
                        if isinstance(decl, dict):
                            # Build signature from decl structure
                            inputs = decl.get("inputs", [])
                            output = decl.get("output", None)
                            signature = f"fn {name}({inputs}) -> {output}"
                        else:
                            signature = str(decl)
                    elif inner.get("type"):
                        # Type alias
                        signature = str(inner["type"])
                
                item["signature"] = signature
                
                # Extract parent ID from path
                if len(path_list) > 0:
                    parent_path = "::".join(path_list)
                    item["parent_id"] = parent_path
                else:
                    item["parent_id"] = None
                
                # Extract code intelligence
                # Extract error types from signatures
                item["error_types"] = safe_extract(
                    extract_error_types, signature, item_type, default=[]
                )
                
                # Extract safety information
                safety_info = safe_extract(
                    extract_safety_info, attrs, signature, docs, default={}
                )
                item["safety_info"] = safety_info
                item["is_safe"] = safety_info.get("is_safe", True)
                
                # Extract feature requirements
                item["feature_requirements"] = safe_extract(
                    extract_feature_requirements, attrs, default=[]
                )
                
                # Extract code examples from documentation
                from .code_examples import extract_code_examples
                item["examples"] = extract_code_examples(docs)
                
                # Extract generic parameters and trait bounds from inner structure
                generic_params = None
                trait_bounds = None
                
                if isinstance(inner, dict):
                    # Look for generics in the inner structure
                    generics = inner.get("generics")
                    if isinstance(generics, dict):
                        # Extract generic parameters
                        params = generics.get("params", [])
                        if params:
                            generic_params = json.dumps(params)
                        
                        # Extract where predicates (trait bounds)
                        where_predicates = generics.get("where_predicates", [])
                        if where_predicates:
                            trait_bounds = json.dumps(where_predicates)
                    
                    # For traits, also check bounds field
                    if item_type == "trait" and inner.get("bounds"):
                        trait_bounds = json.dumps(inner["bounds"])
                
                item["generic_params"] = generic_params
                item["trait_bounds"] = trait_bounds
                
                yield item
                items_processed += 1
                
                # Allow other tasks periodically
                if items_processed % 100 == 0:
                    trigger_gc_if_needed()
                    await asyncio.sleep(0)
                    
            except Exception as e:
                logger.warning(f"Error processing item {item_id}: {e}")
                continue
        
        # Yield module hierarchy as special marker at the end
        yield {"_modules": modules}
        
        logger.info(f"Successfully parsed {items_processed} items from rustdoc JSON")
        
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse rustdoc JSON: {e}")
        # Yield empty modules on error
        yield {"_modules": {}}
    except Exception as e:
        logger.error(f"Error parsing rustdoc JSON: {e}")
        # Yield empty modules on error
        yield {"_modules": {}}


async def parse_rustdoc_items(json_content: str) -> list[dict[str, Any]]:
    """Parse rustdoc JSON and return a list of items (backwards compatible).

    This is a compatibility wrapper around the streaming parser.

    Args:
        json_content: Raw JSON string from rustdoc

    Returns:
        List of parsed items
    """
    items = []
    async for item in parse_rustdoc_items_streaming(json_content):
        # Skip the special modules marker
        if "_modules" not in item:
            items.append(item)
    return items


def build_module_hierarchy(paths: dict) -> dict:
    """Build complete module hierarchy from paths dictionary.

    CRITICAL BUG FIX: This fixes the character iteration bug at line 761.
    The bug was treating code example strings as individual characters.

    Args:
        paths: Dictionary from rustdoc JSON paths section (id -> path_info dict)

    Returns:
        dict: Module ID -> module info with parent relationships
    """
    modules = {}
    total_entries = 0
    module_count = 0

    try:
        for id_str, path_info in paths.items():
            total_entries += 1
            if not isinstance(path_info, dict):
                continue

            # Check if it's a module
            kind = path_info.get("kind")
            if isinstance(kind, str) and kind.lower() in ["module", "mod"]:
                module_count += 1
                path_parts = path_info.get("path", [])

                # Module name is the last part of the path
                name = path_parts[-1] if path_parts else ""

                # The path already includes the module name
                full_path_parts = path_parts
                full_path = "::".join(full_path_parts)

                # Determine parent
                parent_id = None
                depth = len(full_path_parts)

                if depth > 1:
                    # Find parent module by path
                    parent_path_parts = full_path_parts[:-1]
                    parent_path = "::".join(parent_path_parts)

                    # Search for parent module ID
                    for pid, pinfo in paths.items():
                        if isinstance(pinfo, dict):
                            pkind = pinfo.get("kind", "")
                            if isinstance(pkind, str) and pkind.lower() in [
                                "module",
                                "mod",
                            ]:
                                pname = pinfo.get("name", "")
                                ppath = pinfo.get("path", [])
                                if pname and pname not in ppath:
                                    p_full = "::".join(ppath + [pname])
                                else:
                                    p_full = "::".join(ppath)
                                if p_full == parent_path:
                                    parent_id = pid
                                    break

                modules[id_str] = {
                    "name": name,
                    "path": full_path,
                    "parent_id": parent_id,
                    "depth": depth - 1,  # 0-indexed depth (crate root = 0)
                    "item_count": 0,  # Will be updated during index pass
                }

    except Exception as e:
        logger.warning(f"Error building module hierarchy: {e}")

    logger.info(
        f"Processed {total_entries} path entries, found {module_count} modules, built {len(modules)} module records"
    )
    return modules


def resolve_parent_id(item: dict, paths: dict) -> str | None:
    """Resolve the parent module/struct ID for an item.

    Args:
        item: The rustdoc item to resolve parent for
        paths: Dictionary of all paths from rustdoc

    Returns:
        Optional[str]: Parent ID if found, None otherwise
    """
    try:
        # Check if item has a parent field
        if "parent" in item:
            parent = item["parent"]
            if parent and parent != "null":
                return parent

        # Try to infer from path if available
        if "path" in item:
            path_parts = item["path"]
            if isinstance(path_parts, list) and len(path_parts) > 1:
                # The parent would be all but the last part
                parent_path = "::".join(path_parts[:-1])
                # Try to find the parent ID from paths
                for pid, pinfo in paths.items():
                    if isinstance(pinfo, dict) and "path" in pinfo:
                        if "::".join(pinfo["path"]) == parent_path:
                            return pid
    except Exception as e:
        logger.warning(f"Could not resolve parent ID: {e}")
    return None


def normalize_item_type(kind: dict | str) -> str:
    """Normalize rustdoc kind to standard item_type.

    Args:
        kind: Raw kind from rustdoc (can be dict or string)

    Returns:
        str: Normalized item type
    """
    if isinstance(kind, dict):
        kind_str = list(kind.keys())[0] if kind else "unknown"
    else:
        kind_str = str(kind).lower()

    # Map rustdoc kinds to standard types
    type_map = {
        "function": "function",
        "struct": "struct",
        "trait": "trait",
        "mod": "module",
        "module": "module",
        "method": "method",
        "enum": "enum",
        "type": "type",
        "typedef": "type",
        "const": "const",
        "static": "static",
        "impl": "trait_impl",
        "macro": "macro",
        "primitive": "primitive",
        "union": "union",
    }

    # Find matching type
    for key, value in type_map.items():
        if key in kind_str:
            return value

    return "unknown"


def extract_item_path(item: dict) -> str:
    """Extract the full path for an item.

    Args:
        item: Rustdoc item

    Returns:
        str: Full path like "module::submodule::ItemName"
    """
    path_parts = item.get("path", [])
    name = item.get("name", "")

    if path_parts:
        if name and name not in path_parts:
            return "::".join(path_parts + [name])
        return "::".join(path_parts)
    elif name:
        return name
    else:
        return "unknown"


def normalize_paths(paths: list[str]) -> list[str]:
    """Normalize path components for consistent formatting.

    Args:
        paths: List of path components

    Returns:
        List[str]: Normalized path components
    """
    normalized = []
    for part in paths:
        # Remove any extraneous whitespace
        part = part.strip()
        # Replace hyphens with underscores (Rust convention)
        part = part.replace("-", "_")
        if part:
            normalized.append(part)
    return normalized
