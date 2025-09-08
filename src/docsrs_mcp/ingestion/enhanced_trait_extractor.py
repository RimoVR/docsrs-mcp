"""Enhanced trait extraction from rustdoc JSON.

This module implements comprehensive trait implementation extraction following
the codex-bridge reviewed architecture, with correct rustdoc JSON field usage.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TraitImplementation:
    """Represents a trait implementation extracted from rustdoc JSON."""
    
    trait_path: str
    impl_type_path: str
    crate_id: str
    item_id: str
    generic_params: Optional[str] = None
    where_clauses: Optional[str] = None
    is_blanket: bool = False
    is_negative: bool = False
    is_synthetic: bool = False
    impl_signature: Optional[str] = None
    stability_level: str = "stable"


@dataclass
class TraitDefinition:
    """Represents a trait definition with supertraits."""
    
    trait_path: str
    crate_id: str
    item_id: str
    supertraits: List[str]
    associated_items: List[str]
    is_unsafe: bool = False
    generic_params: Optional[str] = None
    stability_level: str = "stable"


class EnhancedTraitExtractor:
    """Enhanced trait extractor using correct rustdoc JSON field names.
    
    Based on codex-bridge review:
    - Use inner.trait and inner.for fields (not trait_ and for_)
    - Handle inherent vs trait implementations via null checking
    - Extract supertraits from trait definitions
    - Support blanket implementations and auto-traits
    """
    
    def __init__(self):
        """Initialize the enhanced trait extractor."""
        self.extracted_impls: List[TraitImplementation] = []
        self.extracted_traits: List[TraitDefinition] = []
        self.type_fingerprints: Dict[str, str] = {}
        self.paths_cache: Dict[str, Any] = {}
    
    def extract_impl_blocks(self, item_data: Dict[str, Any], item_id: str, 
                           crate_id: str, paths: Dict[str, Any]) -> List[TraitImplementation]:
        """Extract trait implementations from impl blocks.
        
        Args:
            item_data: Raw item data from rustdoc JSON
            item_id: Unique item identifier
            crate_id: Crate identifier for the item
            paths: Path information mapping
            
        Returns:
            List of extracted trait implementations
        """
        # Cache the paths for FQN resolution
        self.paths_cache = paths or {}
        implementations = []
        # Cache paths for use by _extract_trait_path
        try:
            if paths:
                self.paths_cache = paths
        except Exception:
            pass
        
        try:
            inner = item_data.get("inner", {})
            if not isinstance(inner, dict):
                logger.debug(f"Item {item_id} has no inner dict: {type(inner)}")
                return implementations
            
            # Log the inner structure for debugging
            logger.info(f"Processing impl block {item_id}, inner keys: {list(inner.keys())}")
            
            # CRITICAL FIX: Access the impl data nested inside inner["impl"]
            # Rustdoc JSON structure: {"inner": {"impl": {"trait": ..., "for": ...}}}
            if "impl" not in inner:
                logger.debug(f"Item {item_id} has no impl data in inner dict")
                return implementations
                
            impl_data = inner["impl"]
            if not isinstance(impl_data, dict):
                logger.debug(f"Item {item_id} impl data is not dict: {type(impl_data)}")
                return implementations
            
            # Extract trait and for fields from the impl data
            trait_info = impl_data.get("trait")  # None for inherent impls
            for_type = impl_data.get("for")      # The implementing type
            
            logger.info(f"Impl {item_id}: trait_info={trait_info is not None}, for_type={for_type is not None}")
            
            if for_type is None:
                logger.debug(f"Impl block {item_id} missing 'for' field, skipping")
                return implementations
            
            # Extract implementing type path
            impl_type_path = self._extract_type_path(for_type)
            if not impl_type_path:
                logger.debug(f"Could not extract implementing type for {item_id}")
                return implementations
            
            # Generate stable type fingerprint for uniqueness
            type_fingerprint = self._generate_type_fingerprint(for_type)
            self.type_fingerprints[impl_type_path] = type_fingerprint
            
            # Check if trait implementation (trait_info != None) or inherent (trait_info == None)
            if trait_info is not None:
                # This is a trait implementation: impl Trait for Type
                trait_path = self._extract_trait_path(trait_info)
                if not trait_path:
                    logger.debug(f"Could not extract trait path for {item_id}")
                    return implementations
                
                # Extract additional impl properties
                is_synthetic = impl_data.get("synthetic", False)  # Auto-traits
                is_negative = impl_data.get("negative", False)    # Negative impls
                polarity = impl_data.get("polarity")              # Alternative negative detection
                if polarity == "negative":
                    is_negative = True
                
                # Extract blanket implementation info
                blanket_impl = impl_data.get("blanket_impl")
                is_blanket = blanket_impl is not None or is_synthetic
                
                # Extract generic parameters and constraints
                generics = impl_data.get("generics", {})
                generic_params = None
                where_clauses = None
                
                if isinstance(generics, dict):
                    params = generics.get("params", [])
                    if params:
                        generic_params = json.dumps(params)
                    
                    where_predicates = generics.get("where_predicates", [])
                    if where_predicates:
                        where_clauses = json.dumps(where_predicates)
                
                # Build impl signature
                impl_signature = self._build_impl_signature(
                    trait_path, impl_type_path, generic_params, where_clauses
                )
                
                # Create trait implementation record
                impl = TraitImplementation(
                    trait_path=trait_path,
                    impl_type_path=impl_type_path,
                    crate_id=crate_id,
                    item_id=item_id,
                    generic_params=generic_params,
                    where_clauses=where_clauses,
                    is_blanket=is_blanket,
                    is_negative=is_negative,
                    is_synthetic=is_synthetic,
                    impl_signature=impl_signature,
                    stability_level=self._extract_stability(item_data)
                )
                
                implementations.append(impl)
                logger.debug(f"Extracted trait impl: {trait_path} for {impl_type_path}")
                
            else:
                # This is an inherent implementation: impl Type
                # For now, we don't store inherent impls in trait_implementations table
                # but we could track them for method resolution
                logger.debug(f"Inherent impl for {impl_type_path}, skipping trait extraction")
                
        except Exception as e:
            logger.error(f"Error extracting impl block {item_id}: {e}")
            
        return implementations
    
    def extract_trait_definition(self, item_data: Dict[str, Any], item_id: str,
                                crate_id: str, item_path: str, paths: Dict[str, Any] = None) -> Optional[TraitDefinition]:
        """Extract trait definition with supertraits.
        
        Args:
            item_data: Raw trait item data
            item_id: Unique item identifier  
            crate_id: Crate identifier
            item_path: Full path to the trait
            paths: Path information mapping for FQN resolution
            
        Returns:
            Extracted trait definition or None
        """
        # Cache the paths for FQN resolution
        if paths:
            self.paths_cache = paths
        try:
            inner = item_data.get("inner", {})
            if not isinstance(inner, dict):
                return None
            
            # Extract supertraits (trait bounds)
            supertraits = []
            bounds = inner.get("bounds", [])
            if isinstance(bounds, list):
                for bound in bounds:
                    supertrait_path = self._extract_trait_path(bound)
                    if supertrait_path:
                        supertraits.append(supertrait_path)
            
            # Extract associated items
            items = inner.get("items", [])
            associated_items = list(items) if isinstance(items, list) else []
            
            # Extract generic parameters
            generics = inner.get("generics", {})
            generic_params = None
            if isinstance(generics, dict):
                params = generics.get("params", [])
                if params:
                    generic_params = json.dumps(params)
            
            # Check if unsafe trait
            is_unsafe = inner.get("is_unsafe", False)
            
            trait_def = TraitDefinition(
                trait_path=item_path,
                crate_id=crate_id,
                item_id=item_id,
                supertraits=supertraits,
                associated_items=associated_items,
                is_unsafe=is_unsafe,
                generic_params=generic_params,
                stability_level=self._extract_stability(item_data)
            )
            
            logger.debug(f"Extracted trait definition: {item_path} with {len(supertraits)} supertraits")
            return trait_def
            
        except Exception as e:
            logger.error(f"Error extracting trait definition {item_id}: {e}")
            return None
    
    def _extract_trait_path(self, trait_info: Any) -> Optional[str]:
        """Extract fully qualified trait path from trait reference.
        
        Args:
            trait_info: Trait reference from rustdoc JSON
            
        Returns:
            Fully qualified trait path or None
        """
        if not isinstance(trait_info, dict):
            return None
        
        # Try to resolve using path ID first (most reliable for FQN)
        trait_id = None
        if "id" in trait_info:
            trait_id = trait_info["id"]
        elif "resolved_path" in trait_info and isinstance(trait_info["resolved_path"], dict):
            trait_id = trait_info["resolved_path"].get("id")
        elif "path" in trait_info and isinstance(trait_info["path"], dict):
            trait_id = trait_info["path"].get("id")
        
        # Look up in paths cache for FQN
        cache = getattr(self, "paths_cache", {})
        if trait_id and trait_id in cache:
            path_entry = cache[trait_id]
            if isinstance(path_entry, dict):
                # Try to construct FQN from path segments
                if "path" in path_entry:
                    path_segments = path_entry["path"]
                    if isinstance(path_segments, list) and path_segments:
                        return "::".join(path_segments)
                
                # Fallback to name field in path entry
                if "name" in path_entry:
                    return path_entry["name"]
        
        # Fallback to direct extraction (may not be fully qualified)
        # Handle resolved_path format
        if "resolved_path" in trait_info:
            resolved = trait_info["resolved_path"]
            if isinstance(resolved, dict) and "name" in resolved:
                name = resolved["name"]
                logger.debug(f"Using resolved_path name (may not be FQN): {name}")
                return name
        
        # Handle path format
        if "path" in trait_info:
            path_info = trait_info["path"]
            if isinstance(path_info, dict) and "name" in path_info:
                name = path_info["name"]
                logger.debug(f"Using path name (may not be FQN): {name}")
                return name
            elif isinstance(path_info, str):
                logger.debug(f"Using string path (may not be FQN): {path_info}")
                return path_info
        
        # Handle direct name with enhanced fallback for external traits
        if "name" in trait_info:
            name = trait_info["name"]
            # Try to infer FQN for common standard library traits
            fqn = self._infer_trait_fqn(name)
            if fqn != name:
                logger.debug(f"Inferred FQN for '{name}': {fqn}")
                return fqn
            else:
                logger.debug(f"Using direct name (external trait): {name}")
                return name
            
        logger.debug(f"Could not extract trait path from: {trait_info}")
        return None
    
    def _infer_trait_fqn(self, trait_name: str) -> str:
        """Infer fully qualified name for common standard library traits.
        
        Args:
            trait_name: Bare trait name (e.g., "Display", "Debug")
            
        Returns:
            Inferred FQN or the original name if no mapping exists
        """
        # Common std traits mapping - based on actual Rust std library
        std_trait_mapping = {
            # core::fmt traits
            "Display": "core::fmt::Display",
            "Debug": "core::fmt::Debug", 
            "Write": "core::fmt::Write",
            "Formatter": "core::fmt::Formatter",
            
            # core::clone and copy
            "Clone": "core::clone::Clone",
            "Copy": "core::marker::Copy",
            
            # core::cmp traits
            "PartialEq": "core::cmp::PartialEq",
            "Eq": "core::cmp::Eq",
            "PartialOrd": "core::cmp::PartialOrd",
            "Ord": "core::cmp::Ord",
            
            # core::hash
            "Hash": "core::hash::Hash",
            "Hasher": "core::hash::Hasher",
            
            # core::convert traits
            "From": "core::convert::From",
            "Into": "core::convert::Into",
            "TryFrom": "core::convert::TryFrom",
            "TryInto": "core::convert::TryInto",
            "AsRef": "core::convert::AsRef",
            "AsMut": "core::convert::AsMut",
            
            # core::ops traits
            "Add": "core::ops::Add",
            "Sub": "core::ops::Sub",
            "Mul": "core::ops::Mul",
            "Div": "core::ops::Div",
            "Rem": "core::ops::Rem",
            "Deref": "core::ops::Deref",
            "DerefMut": "core::ops::DerefMut",
            "Index": "core::ops::Index",
            "IndexMut": "core::ops::IndexMut",
            "Drop": "core::ops::Drop",
            
            # core::iter traits
            "Iterator": "core::iter::Iterator",
            "IntoIterator": "core::iter::IntoIterator",
            "Extend": "core::iter::Extend",
            "FromIterator": "core::iter::FromIterator",
            
            # core::marker traits
            "Send": "core::marker::Send",
            "Sync": "core::marker::Sync",
            "Sized": "core::marker::Sized",
            "Unpin": "core::marker::Unpin",
            
            # core::default
            "Default": "core::default::Default",
            
            # alloc and std specific
            "ToString": "alloc::string::ToString",
            "Vec": "alloc::vec::Vec",
            "String": "alloc::string::String",
        }
        
        return std_trait_mapping.get(trait_name, trait_name)
    
    def _extract_type_path(self, type_info: Any) -> Optional[str]:
        """Extract type path from type reference.
        
        Args:
            type_info: Type reference from rustdoc JSON
            
        Returns:
            Formatted type path or None
        """
        if not isinstance(type_info, dict):
            return str(type_info) if type_info else None
        
        # Handle resolved_path format (most common in stored data)
        if "resolved_path" in type_info:
            resolved = type_info["resolved_path"]
            if isinstance(resolved, dict):
                # Extract base path
                base_path = resolved.get("path", resolved.get("name", ""))
                
                # Handle generic arguments
                if "args" in resolved:
                    args_info = resolved["args"]
                    if isinstance(args_info, dict) and "angle_bracketed" in args_info:
                        angle_args = args_info["angle_bracketed"]
                        if isinstance(angle_args, dict) and "args" in angle_args:
                            arg_list = angle_args["args"]
                            if isinstance(arg_list, list) and arg_list:
                                # Extract generic argument types
                                generic_args = []
                                for arg in arg_list:
                                    if isinstance(arg, dict) and "type" in arg:
                                        arg_type = arg["type"]
                                        if isinstance(arg_type, dict):
                                            if "generic" in arg_type:
                                                generic_args.append(arg_type["generic"])
                                            elif "primitive" in arg_type:
                                                generic_args.append(arg_type["primitive"])
                                            else:
                                                # Recursively extract nested type
                                                nested = self._extract_type_path(arg_type)
                                                if nested:
                                                    generic_args.append(nested)
                                
                                if generic_args:
                                    return f"{base_path}<{', '.join(generic_args)}>"
                
                return base_path
        
        # Handle direct path format
        if "path" in type_info:
            path_info = type_info["path"]
            if isinstance(path_info, dict) and "name" in path_info:
                return path_info["name"]
            elif isinstance(path_info, str):
                return path_info
        
        # Handle tuple types
        if "tuple" in type_info:
            tuple_elements = type_info["tuple"]
            if isinstance(tuple_elements, list):
                element_types = []
                for element in tuple_elements:
                    if isinstance(element, dict):
                        if "generic" in element:
                            element_types.append(element["generic"])
                        elif "primitive" in element:
                            element_types.append(element["primitive"])
                        else:
                            # Recursively extract nested type
                            nested = self._extract_type_path(element)
                            if nested:
                                element_types.append(nested)
                
                if element_types:
                    if len(element_types) == 1:
                        return f"({element_types[0]},)"  # Single-element tuple
                    else:
                        return f"({', '.join(element_types)})"
        
        # Handle generic types
        if "generic" in type_info:
            return type_info["generic"]
        
        # Handle primitive types
        if "primitive" in type_info:
            return type_info["primitive"]
        
        # Handle array types
        if "array" in type_info:
            array_info = type_info["array"]
            if isinstance(array_info, dict):
                element_type = array_info.get("type")
                length = array_info.get("len", "N")
                if element_type:
                    element_path = self._extract_type_path(element_type)
                    if element_path:
                        return f"[{element_path}; {length}]"
        
        # Handle slice types  
        if "slice" in type_info:
            slice_type = type_info["slice"]
            if slice_type:
                element_path = self._extract_type_path(slice_type)
                if element_path:
                    return f"[{element_path}]"
        
        # Handle reference types
        if "borrowed_ref" in type_info:
            ref_info = type_info["borrowed_ref"]
            if isinstance(ref_info, dict):
                ref_type = ref_info.get("type")
                lifetime = ref_info.get("lifetime", "")
                is_mutable = ref_info.get("mutable", False)
                
                if ref_type:
                    type_path = self._extract_type_path(ref_type)
                    if type_path:
                        lifetime_str = f"'{lifetime} " if lifetime else ""
                        mut_str = "mut " if is_mutable else ""
                        return f"&{lifetime_str}{mut_str}{type_path}"
        
        # Handle function pointer types
        if "function_pointer" in type_info:
            return "fn(...)"  # Simplified representation
        
        # Last resort: try to extract any meaningful name
        if isinstance(type_info, dict):
            # Look for any name field
            for key in ["name", "id"]:
                if key in type_info and isinstance(type_info[key], str):
                    return type_info[key]
        
        # Ultimate fallback - at least don't return the entire JSON
        return "UnknownType"
    
    def _generate_type_fingerprint(self, type_info: Any) -> str:
        """Generate stable fingerprint for type identity.
        
        Args:
            type_info: Type information
            
        Returns:
            Stable type fingerprint
        """
        # Simple hash of canonical representation
        import hashlib
        canonical = json.dumps(type_info, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:16]
    
    def _build_impl_signature(self, trait_path: str, impl_type_path: str,
                             generic_params: Optional[str], where_clauses: Optional[str]) -> str:
        """Build human-readable impl signature.
        
        Args:
            trait_path: Path to the trait
            impl_type_path: Path to the implementing type  
            generic_params: Generic parameters JSON
            where_clauses: Where clauses JSON
            
        Returns:
            Formatted impl signature
        """
        signature = f"impl {trait_path} for {impl_type_path}"
        
        if generic_params:
            try:
                params = json.loads(generic_params)
                if params:
                    param_names = [p.get("name", "?") for p in params if isinstance(p, dict)]
                    signature = f"impl<{', '.join(param_names)}> {trait_path} for {impl_type_path}"
            except (json.JSONDecodeError, KeyError):
                pass
        
        if where_clauses:
            signature += " where ..."
            
        return signature
    
    def _extract_stability(self, item_data: Dict[str, Any]) -> str:
        """Extract stability level from item data.
        
        Args:
            item_data: Item data from rustdoc
            
        Returns:
            Stability level (stable, unstable, deprecated)
        """
        if item_data.get("deprecation"):
            return "deprecated"
        
        attrs = item_data.get("attrs", [])
        for attr in attrs:
            if isinstance(attr, str) and "unstable" in attr.lower():
                return "unstable"
                
        return "stable"
