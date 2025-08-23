"""Signature and metadata extraction from rustdoc items.

This module handles:
- Function/method signature formatting
- Type extraction and normalization
- Generic parameter handling
- Visibility and deprecation status extraction
"""

import logging

logger = logging.getLogger(__name__)


def format_signature(decl: dict, generics: dict | None = None) -> str:
    """Format a function/method declaration into a readable signature with generics.

    Args:
        decl: Declaration dict from rustdoc
        generics: Optional generics dict

    Returns:
        str: Formatted signature like "<T: Display>(arg: T) -> Result<T>"
    """
    try:
        # Extract generics if present
        generic_str = ""
        if generics and isinstance(generics, dict):
            params = generics.get("params", [])
            if params:
                generic_parts = []
                for param in params:
                    if isinstance(param, dict):
                        name = param.get("name", "")
                        kind = param.get("kind", {})

                        # Handle different generic parameter kinds
                        if isinstance(kind, dict):
                            if "lifetime" in kind:
                                generic_parts.append(f"'{name}")
                            elif "const" in kind:
                                const_type = (
                                    kind["const"].get("type")
                                    if isinstance(kind["const"], dict)
                                    else None
                                )
                                if const_type:
                                    type_name = extract_type_name(const_type)
                                    generic_parts.append(f"const {name}: {type_name}")
                                else:
                                    generic_parts.append(f"const {name}")
                            elif "type" in kind:
                                # Check for bounds on the type parameter
                                type_info = kind["type"]
                                if (
                                    isinstance(type_info, dict)
                                    and "bounds" in type_info
                                ):
                                    bounds = []
                                    for bound in type_info["bounds"]:
                                        if isinstance(bound, dict):
                                            bound_str = extract_type_name(
                                                bound.get("trait")
                                            )
                                            if bound_str:
                                                bounds.append(bound_str)
                                    if bounds:
                                        generic_parts.append(
                                            f"{name}: {' + '.join(bounds)}"
                                        )
                                    else:
                                        generic_parts.append(name)
                                else:
                                    generic_parts.append(name)
                            else:
                                generic_parts.append(name)
                        else:
                            generic_parts.append(name)

                if generic_parts:
                    generic_str = f"<{', '.join(generic_parts)}>"

        inputs = decl.get("inputs", [])
        output = decl.get("output")

        # Format parameters
        params = []
        for param in inputs:
            if isinstance(param, dict):
                param_name = param.get("name", "_")
                param_type = param.get("type", {})
                # Simple type extraction - can be enhanced
                if isinstance(param_type, dict):
                    type_str = param_type.get("name", "Unknown")
                else:
                    type_str = str(param_type)
                params.append(f"{param_name}: {type_str}")
            else:
                params.append(str(param))

        signature = f"{generic_str}({', '.join(params)})"

        # Add return type if present
        if output and output != "unit":
            if isinstance(output, dict):
                return_type = output.get("name", "")
            else:
                return_type = str(output)
            if return_type:
                signature += f" -> {return_type}"

        return signature
    except Exception:
        return ""


def extract_signature(item: dict) -> str | None:
    """Extract function/method signature with generics.

    Args:
        item: Rustdoc item

    Returns:
        Optional[str]: Formatted signature if extractable
    """
    try:
        inner = item.get("inner", {})

        # Check for function
        if isinstance(inner, dict) and "function" in inner:
            func_data = inner["function"]
            decl = func_data.get("decl", {})
            generics = func_data.get("generics")
            if decl:
                return format_signature(decl, generics)

        # Check for method
        if isinstance(inner, dict) and "method" in inner:
            method_data = inner["method"]
            decl = method_data.get("decl", {})
            generics = method_data.get("generics")
            if decl:
                return format_signature(decl, generics)

        # Check for other inner types that might have signatures
        for key in ["assoc_const", "assoc_type"]:
            if isinstance(inner, dict) and key in inner:
                # These might have type information
                type_info = inner[key].get("type", {})
                if isinstance(type_info, dict) and "name" in type_info:
                    return type_info["name"]
    except Exception as e:
        logger.warning(f"Could not extract signature: {e}")
    return None


def extract_type_name(type_info: dict | str | None) -> str:
    """Extract readable type name from rustdoc Type object.

    Args:
        type_info: Type info from rustdoc JSON (can be dict, string, or None)

    Returns:
        str: Extracted type name or "Unknown" if unable to extract
    """
    if not type_info:
        return "Unknown"

    if isinstance(type_info, str):
        return type_info

    if isinstance(type_info, dict):
        # Handle resolved_path type
        if "resolved_path" in type_info:
            resolved = type_info["resolved_path"]
            if isinstance(resolved, dict):
                name = resolved.get("name", "")
                if name:
                    return name

        # Handle path type
        if "path" in type_info:
            path = type_info["path"]
            if isinstance(path, dict):
                name = path.get("name", "")
                if name:
                    return name

        # Handle generic type
        if "generic" in type_info:
            return type_info["generic"]

        # Handle primitive type
        if "primitive" in type_info:
            return type_info["primitive"]

        # Try to get name directly
        if "name" in type_info:
            return type_info["name"]

    return "Unknown"


def extract_visibility(item: dict) -> str:
    """Extract visibility modifier from item.

    Args:
        item: Rustdoc item

    Returns:
        str: Visibility like "pub", "pub(crate)", "private"
    """
    try:
        visibility = item.get("visibility", {})
        if isinstance(visibility, dict):
            if visibility.get("public", False):
                return "pub"
            elif "crate" in visibility:
                return "pub(crate)"
            elif "restricted" in visibility:
                path = visibility["restricted"].get("path", "")
                return f"pub(in {path})" if path else "pub(restricted)"
        return "private"
    except Exception:
        return "private"


def extract_generics(item: dict) -> dict | None:
    """Extract generic parameters from item.

    Args:
        item: Rustdoc item

    Returns:
        Optional[Dict]: Generics dict if present
    """
    try:
        inner = item.get("inner", {})

        # Check various inner types for generics
        for key in ["function", "method", "struct", "enum", "trait", "type"]:
            if (
                isinstance(inner, dict)
                and key in inner
                and isinstance(inner[key], dict)
            ):
                generics = inner[key].get("generics")
                if generics:
                    return generics

        # Check top-level generics
        if "generics" in item:
            return item["generics"]

    except Exception as e:
        logger.warning(f"Could not extract generics: {e}")

    return None


def extract_deprecated(item: dict) -> bool:
    """Extract deprecated status from rustdoc item attributes.

    Args:
        item: Rustdoc item

    Returns:
        bool: True if item is deprecated
    """
    try:
        # Check attrs field for deprecated attribute
        attrs = item.get("attrs", [])
        if isinstance(attrs, list):
            for attr in attrs:
                if isinstance(attr, str):
                    if "deprecated" in attr.lower():
                        return True
                elif isinstance(attr, dict):
                    # Handle structured attributes
                    if attr.get("name") == "deprecated":
                        return True
                    # Check for deprecated in attribute content
                    content = attr.get("content", "")
                    if "deprecated" in str(content).lower():
                        return True

        # Check if item has deprecated field directly
        if item.get("deprecated", False):
            return True

        # Check inner for deprecated status
        inner = item.get("inner", {})
        if isinstance(inner, dict):
            if inner.get("deprecated", False):
                return True

        return False
    except Exception as e:
        logger.warning(f"Error extracting deprecated status: {e}")
        return False


def extract_where_clause(generics: dict) -> list[dict] | None:
    """Extract where clause predicates from generics.

    Args:
        generics: Generics dict from rustdoc

    Returns:
        Optional[List[Dict]]: List of where predicates
    """
    try:
        where_clause = generics.get("where_predicates", [])
        if where_clause:
            predicates = []
            for predicate in where_clause:
                parsed = extract_where_predicate(predicate)
                if parsed:
                    predicates.append(parsed)
            return predicates if predicates else None
    except Exception:
        return None


def extract_where_predicate(predicate: dict) -> dict | None:
    """Extract a single where predicate into a structured format.

    Args:
        predicate: Single where predicate from rustdoc

    Returns:
        Optional[Dict]: Structured predicate or None
    """
    try:
        pred_type = predicate.get("type", "")

        if pred_type == "bound_predicate":
            # T: Display + Debug style
            type_name = extract_type_name(predicate.get("type"))
            bounds = []
            for bound in predicate.get("bounds", []):
                if isinstance(bound, dict):
                    trait_name = extract_type_name(bound.get("trait"))
                    if trait_name:
                        bounds.append(trait_name)

            if type_name and bounds:
                return {"type": "bound", "target": type_name, "bounds": bounds}

        elif pred_type == "region_predicate":
            # 'a: 'b style lifetime bounds
            lifetime = predicate.get("lifetime")
            bounds = predicate.get("bounds", [])
            if lifetime and bounds:
                return {"type": "lifetime", "lifetime": lifetime, "bounds": bounds}

        elif pred_type == "eq_predicate":
            # T = ConcreteType style
            lhs = extract_type_name(predicate.get("lhs"))
            rhs = extract_type_name(predicate.get("rhs"))
            if lhs and rhs:
                return {"type": "equality", "left": lhs, "right": rhs}

        return None

    except Exception:
        return None
