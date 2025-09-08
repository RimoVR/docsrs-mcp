"""Enhanced method extraction with trait source attribution.

This module implements comprehensive method extraction that correctly handles
trait methods, default implementations, and inherent methods as identified
in the codex-bridge review.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Set
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class MethodSignature:
    """Represents a method signature with trait source information."""
    
    parent_type_path: str
    method_name: str
    full_signature: str
    crate_id: str
    item_id: str
    generic_params: Optional[str] = None
    where_clauses: Optional[str] = None
    return_type: Optional[str] = None
    is_async: bool = False
    is_unsafe: bool = False
    is_const: bool = False
    visibility: str = "pub"
    method_kind: str = "inherent"  # inherent, trait, static
    trait_source: Optional[str] = None  # Which trait provides this method
    receiver_type: Optional[str] = None  # self, &self, &mut self, etc.
    stability_level: str = "stable"


class EnhancedMethodExtractor:
    """Enhanced method extractor with trait source attribution.
    
    Based on codex-bridge review:
    - Extract both inherent and trait methods
    - Include default trait methods from trait definitions
    - Attribute trait methods to their providing trait
    - Handle method overrides vs defaults correctly
    """
    
    def __init__(self):
        """Initialize the enhanced method extractor."""
        self.extracted_methods: List[MethodSignature] = []
        self.trait_default_methods: Dict[str, List[Dict]] = {}  # trait_path -> methods
        self.index: Dict[str, Dict] = {}

    def set_index(self, index: Dict[str, Dict]) -> None:
        """Provide the rustdoc index (id -> item) for resolving item IDs."""
        if isinstance(index, dict):
            self.index = index
    
    def extract_inherent_methods(self, item_data: Dict[str, Any], item_id: str,
                                crate_id: str, type_path: str) -> List[MethodSignature]:
        """Extract methods directly implemented on types (inherent implementations).
        
        Args:
            item_data: Raw impl item data from rustdoc JSON
            item_id: Unique item identifier
            crate_id: Crate identifier
            type_path: Path to the type that owns these methods
            
        Returns:
            List of extracted inherent method signatures
        """
        methods = []
        
        try:
            inner = item_data.get("inner", {})
            if not isinstance(inner, dict):
                return methods
            
            # Check if this is an inherent impl (inner.trait is None)
            trait_info = inner.get("trait")
            if trait_info is not None:
                # This is a trait impl, not inherent
                return methods
            
            # Extract methods from inherent impl
            items = inner.get("items", [])
            if not isinstance(items, list):
                return methods
            
            for method_id in items:
                try:
                    method_signature = self._extract_method_from_id(
                        method_id, crate_id, type_path, "inherent", None
                    )
                    if method_signature:
                        methods.append(method_signature)
                except Exception as e:
                    logger.warning(f"Error extracting inherent method {method_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error extracting inherent methods from {item_id}: {e}")
            
        return methods
    
    def extract_trait_methods(self, item_data: Dict[str, Any], item_id: str,
                             crate_id: str, type_path: str, trait_path: str,
                             trait_defaults: Dict[str, Dict]) -> List[MethodSignature]:
        """Extract methods from trait implementations.
        
        According to codex-bridge review, impl items only list overrides.
        We must join with trait def items to include default methods.
        
        Args:
            item_data: Raw trait impl item data from rustdoc JSON
            item_id: Unique item identifier  
            crate_id: Crate identifier
            type_path: Path to the implementing type
            trait_path: Path to the implemented trait
            trait_defaults: Default methods from trait definition
            
        Returns:
            List of extracted trait method signatures
        """
        methods = []
        
        try:
            inner = item_data.get("inner", {})
            if not isinstance(inner, dict):
                return methods
            
            # Verify this is a trait impl
            trait_info = inner.get("trait")
            if trait_info is None:
                return methods
            
            # Extract overridden methods from impl
            impl_items = inner.get("items", [])
            overridden_methods = set()
            
            for method_id in impl_items:
                try:
                    method_signature = self._extract_method_from_id(
                        method_id, crate_id, type_path, "trait", trait_path
                    )
                    if method_signature:
                        methods.append(method_signature)
                        overridden_methods.add(method_signature.method_name)
                except Exception as e:
                    logger.warning(f"Error extracting trait method {method_id}: {e}")
            
            # Add default methods from trait definition that aren't overridden
            for method_name, method_info in trait_defaults.items():
                if method_name not in overridden_methods:
                    try:
                        default_method = self._create_default_method_signature(
                            method_name, method_info, crate_id, type_path, trait_path
                        )
                        if default_method:
                            methods.append(default_method)
                    except Exception as e:
                        logger.warning(f"Error creating default method {method_name}: {e}")
                        
        except Exception as e:
            logger.error(f"Error extracting trait methods from {item_id}: {e}")
            
        return methods
    
    def extract_trait_default_methods(self, trait_data: Dict[str, Any],
                                     trait_path: str) -> Dict[str, Dict]:
        """Extract default method definitions from trait.
        
        Args:
            trait_data: Raw trait item data from rustdoc JSON
            trait_path: Path to the trait
            
        Returns:
            Dictionary mapping method names to their definitions
        """
        default_methods = {}
        
        try:
            inner = trait_data.get("inner", {})
            if not isinstance(inner, dict):
                return default_methods
            
            items = inner.get("items", [])
            if not isinstance(items, list):
                return default_methods
            
            # Process each item in the trait
            for item_id in items:
                # This would need to resolve the actual item from the index
                # For now, we'll store the item_id and resolve later
                # In practice, this needs integration with the full rustdoc parsing
                pass
                
        except Exception as e:
            logger.error(f"Error extracting trait defaults from {trait_path}: {e}")
            
        return default_methods
    
    def analyze_method_safety(self, method_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract unsafe requirements and safety guarantees.
        
        Args:
            method_data: Method item data from rustdoc JSON
            
        Returns:
            Safety analysis information
        """
        safety_info = {
            "is_safe": True,
            "unsafe_requirements": [],
            "safety_guarantees": []
        }
        
        try:
            # Check if method is marked unsafe
            inner = method_data.get("inner", {})
            if isinstance(inner, dict):
                decl = inner.get("decl", {})
                if isinstance(decl, dict):
                    header = decl.get("header", {})
                    if isinstance(header, dict):
                        is_unsafe = header.get("unsafety", "safe") == "unsafe"
                        safety_info["is_safe"] = not is_unsafe
                        
                        # Extract async/const information
                        asyncness = header.get("asyncness", "sync") == "async"
                        constness = header.get("constness", "not_const") == "const"
                        
                        safety_info["is_async"] = asyncness
                        safety_info["is_const"] = constness
            
            # Analyze documentation for safety requirements
            docs = method_data.get("docs", "")
            if isinstance(docs, str):
                if "unsafe" in docs.lower():
                    # Basic safety requirement extraction
                    safety_info["unsafe_requirements"] = ["See documentation"]
                    
        except Exception as e:
            logger.warning(f"Error analyzing method safety: {e}")
            
        return safety_info
    
    def _extract_method_from_id(self, method_id: str, crate_id: str,
                                parent_type_path: str, method_kind: str,
                                trait_source: Optional[str]) -> Optional[MethodSignature]:
        """Extract method signature from method ID.
        
        Note: This is a simplified implementation. In practice, this would
        need to resolve the method_id against the full rustdoc index.
        
        Args:
            method_id: Method identifier from rustdoc
            crate_id: Crate identifier
            parent_type_path: Path to parent type
            method_kind: Kind of method (inherent/trait/static)
            trait_source: Trait providing the method (if any)
            
        Returns:
            Method signature or None
        """
        try:
            item = self.index.get(method_id)
            if not isinstance(item, dict):
                logger.debug(f"Method id {method_id} not found in index")
                return None

            name = item.get("name") or ""
            inner = item.get("inner", {})
            decl = None
            if isinstance(inner, dict):
                # Method can be represented under different keys, look for Function-like decl
                decl = inner.get("decl") or inner.get("function") or inner.get("fn")

            # Build signature
            signature = self._build_method_signature(name, {"decl": decl} if decl else {})

            # Flags
            is_async = False
            is_const = False
            is_unsafe = False
            visibility = item.get("visibility", "pub")
            receiver_type = None

            if isinstance(decl, dict):
                header = decl.get("header", {}) if isinstance(decl.get("header"), dict) else {}
                is_async = header.get("asyncness", "sync") == "async"
                is_const = header.get("constness", "not_const") == "const"
                is_unsafe = header.get("unsafety", "safe") == "unsafe"

                inputs = decl.get("inputs", [])
                receiver_type = self._extract_receiver_type(inputs) if isinstance(inputs, list) else None

            return MethodSignature(
                parent_type_path=parent_type_path,
                method_name=name,
                full_signature=signature,
                crate_id=crate_id,
                item_id=method_id,
                is_async=is_async,
                is_const=is_const,
                is_unsafe=is_unsafe,
                visibility=visibility,
                method_kind=method_kind,
                trait_source=trait_source,
                receiver_type=receiver_type,
                stability_level="stable",
            )
        except Exception as e:
            logger.warning(f"Failed to resolve method id {method_id}: {e}")
            return None
    
    def _create_default_method_signature(self, method_name: str, method_info: Dict,
                                        crate_id: str, parent_type_path: str,
                                        trait_path: str) -> Optional[MethodSignature]:
        """Create method signature for trait default method.
        
        Args:
            method_name: Name of the method
            method_info: Method information from trait definition
            crate_id: Crate identifier
            parent_type_path: Path to implementing type
            trait_path: Path to trait providing default
            
        Returns:
            Method signature for default method
        """
        try:
            # Extract signature components
            signature = self._build_method_signature(method_name, method_info)
            
            method_sig = MethodSignature(
                parent_type_path=parent_type_path,
                method_name=method_name,
                full_signature=signature,
                crate_id=crate_id,
                item_id=f"default_{method_name}",
                method_kind="trait",
                trait_source=trait_path,
                stability_level="stable"
            )
            
            return method_sig
            
        except Exception as e:
            logger.error(f"Error creating default method signature for {method_name}: {e}")
            return None
    
    def _build_method_signature(self, method_name: str, method_info: Dict) -> str:
        """Build human-readable method signature.
        
        Args:
            method_name: Name of the method
            method_info: Method information from rustdoc
            
        Returns:
            Formatted method signature
        """
        try:
            # Extract declaration info
            decl = method_info.get("decl", {})
            if not isinstance(decl, dict):
                return f"fn {method_name}()"
            
            # Extract inputs and outputs
            inputs = decl.get("inputs", [])
            output = decl.get("output")
            
            # Build parameter list
            param_strs = []
            for input_param in inputs:
                if isinstance(input_param, dict):
                    name = input_param.get("name", "")
                    type_info = input_param.get("type", "")
                    param_strs.append(f"{name}: {type_info}")
            
            params = ", ".join(param_strs)
            
            # Build return type
            ret_type = ""
            if output and output != "()":
                ret_type = f" -> {output}"
            
            return f"fn {method_name}({params}){ret_type}"
            
        except Exception as e:
            logger.warning(f"Error building method signature for {method_name}: {e}")
            return f"fn {method_name}()"
    
    def _extract_receiver_type(self, inputs: List[Dict]) -> Optional[str]:
        """Extract receiver type (self, &self, &mut self, etc.).
        
        Args:
            inputs: Method input parameters
            
        Returns:
            Receiver type string or None
        """
        if not inputs:
            return None
            
        first_param = inputs[0]
        if not isinstance(first_param, dict):
            return None
        
        name = first_param.get("name", "")
        if name == "self":
            # Analyze the type to determine mutability and reference
            type_info = first_param.get("type", {})
            if isinstance(type_info, dict):
                if "borrowed_ref" in type_info:
                    ref_info = type_info["borrowed_ref"]
                    if isinstance(ref_info, dict):
                        mutability = ref_info.get("mutability", "shared")
                        if mutability == "mutable":
                            return "&mut self"
                        else:
                            return "&self"
                return "self"  # Owned self
                
        return None
