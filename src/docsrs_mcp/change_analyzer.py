"""Change analyzer for detecting breaking changes in Rust APIs.

This module implements Rust-specific semver rules and breaking change detection
based on the Rust API evolution guidelines.
"""

import logging
import re

from .models import ItemChange, ItemKind, Severity

logger = logging.getLogger(__name__)


class RustBreakingChangeDetector:
    """Detects breaking changes according to Rust semver rules."""

    # Rust breaking change patterns based on cargo semver documentation
    BREAKING_PATTERNS = {
        # Structural changes
        "enum_variant_added": {
            "description": "Adding enum variant without #[non_exhaustive]",
            "severity": Severity.BREAKING,
        },
        "struct_field_added": {
            "description": "Adding field to publicly constructible struct",
            "severity": Severity.BREAKING,
        },
        "trait_method_added": {
            "description": "Adding required method to trait",
            "severity": Severity.BREAKING,
        },
        # Signature changes
        "function_signature_changed": {
            "description": "Changed function parameters or return type",
            "severity": Severity.BREAKING,
        },
        "generic_params_changed": {
            "description": "Modified generic parameters or constraints",
            "severity": Severity.BREAKING,
        },
        "trait_bounds_changed": {
            "description": "Changed trait bounds on type parameters",
            "severity": Severity.BREAKING,
        },
        # Visibility changes
        "public_to_private": {
            "description": "Changed from public to private/crate visibility",
            "severity": Severity.BREAKING,
        },
        "item_removed": {
            "description": "Removed public API item",
            "severity": Severity.BREAKING,
        },
        # Type changes
        "type_alias_changed": {
            "description": "Changed type alias definition",
            "severity": Severity.BREAKING,
        },
        "const_value_changed": {
            "description": "Changed const value (may break pattern matching)",
            "severity": Severity.MAJOR,
        },
    }

    def analyze_change(self, change: ItemChange) -> tuple[bool, list[str]]:
        """Analyze a change for breaking patterns.

        Args:
            change: The item change to analyze

        Returns:
            Tuple of (is_breaking, list_of_reasons)
        """
        is_breaking = False
        reasons = []

        # Check removed items
        if change.change_type.value == "removed" and change.details.before:
            if change.details.before.visibility == "public":
                is_breaking = True
                reasons.append("Public item removed from API")

        # Check modified items
        elif change.change_type.value == "modified":
            if change.details.before and change.details.after:
                # Check signature changes
                if self._signature_breaking_change(change):
                    is_breaking = True
                    reasons.append("Function/method signature incompatibly changed")

                # Check visibility changes
                if self._visibility_breaking_change(change):
                    is_breaking = True
                    reasons.append("Visibility reduced from public")

                # Check generic changes
                if self._generics_breaking_change(change):
                    is_breaking = True
                    reasons.append("Generic parameters or bounds changed")

                # Check enum-specific changes
                if change.kind == ItemKind.ENUM:
                    if self._enum_breaking_change(change):
                        is_breaking = True
                        reasons.append("Enum variant added without #[non_exhaustive]")

                # Check struct-specific changes
                if change.kind == ItemKind.STRUCT:
                    if self._struct_breaking_change(change):
                        is_breaking = True
                        reasons.append("Field added to publicly constructible struct")

                # Check trait-specific changes
                if change.kind == ItemKind.TRAIT:
                    if self._trait_breaking_change(change):
                        is_breaking = True
                        reasons.append("Required method added to trait")

        return is_breaking, reasons

    def _signature_breaking_change(self, change: ItemChange) -> bool:
        """Check if signature change is breaking."""
        if not (change.details.before and change.details.after):
            return False

        before_sig = change.details.before.raw_signature
        after_sig = change.details.after.raw_signature

        if not before_sig or not after_sig:
            return False

        # Parse and compare function signatures
        before_params = self._extract_parameters(before_sig)
        after_params = self._extract_parameters(after_sig)

        # Different number of required parameters is breaking
        if len(before_params) != len(after_params):
            # Check if new params have defaults (non-breaking)
            if len(after_params) > len(before_params):
                # If all new params have defaults, it's not breaking
                # This is a simplification - proper analysis would parse defaults
                return True

        # Check return type changes
        before_return = self._extract_return_type(before_sig)
        after_return = self._extract_return_type(after_sig)

        if before_return != after_return:
            return True

        return False

    def _visibility_breaking_change(self, change: ItemChange) -> bool:
        """Check if visibility change is breaking."""
        if not (change.details.before and change.details.after):
            return False

        before_vis = change.details.before.visibility
        after_vis = change.details.after.visibility

        # Public to private/crate is breaking
        if before_vis == "public" and after_vis in ["private", "crate"]:
            return True

        return False

    def _generics_breaking_change(self, change: ItemChange) -> bool:
        """Check if generic parameter changes are breaking."""
        if not (change.details.before and change.details.after):
            return False

        # Check if generics field exists and changed
        before_generics = change.details.before.generics
        after_generics = change.details.after.generics

        # Adding or removing generic parameters is usually breaking
        if before_generics != after_generics:
            # Simple check - could be more sophisticated
            return True

        return False

    def _enum_breaking_change(self, change: ItemChange) -> bool:
        """Check if enum changes are breaking.

        Adding variants to enums without #[non_exhaustive] is breaking
        because it can break exhaustive pattern matching.
        """
        # Check semantic changes for variant additions
        for semantic_change in change.details.semantic_changes:
            if semantic_change is not None and (
                "variant" in semantic_change.lower()
                and "added" in semantic_change.lower()
            ):
                # Check if enum is marked as non_exhaustive
                # This would require checking attributes in the signature
                # For now, assume adding variants is breaking unless proven otherwise
                return True

        return False

    def _struct_breaking_change(self, change: ItemChange) -> bool:
        """Check if struct changes are breaking.

        Adding fields to structs with public constructors is breaking.
        """
        # Check semantic changes for field additions
        for semantic_change in change.details.semantic_changes:
            if semantic_change is not None and (
                "field" in semantic_change.lower()
                and "added" in semantic_change.lower()
            ):
                # Check if struct has private fields (can't be constructed publicly)
                # This would require more detailed analysis
                # For now, assume adding fields is breaking for public structs
                if (
                    change.details.before
                    and change.details.before.visibility == "public"
                ):
                    return True

        return False

    def _trait_breaking_change(self, change: ItemChange) -> bool:
        """Check if trait changes are breaking.

        Adding required methods to traits is breaking for implementors.
        """
        # Check semantic changes for method additions
        for semantic_change in change.details.semantic_changes:
            if semantic_change is not None and (
                "method" in semantic_change.lower()
                and "added" in semantic_change.lower()
            ):
                # Check if method has default implementation
                # This would require parsing trait definition
                # For now, assume adding methods is breaking
                return True

        return False

    def _extract_parameters(self, signature: str) -> list[str]:
        """Extract parameter list from function signature."""
        # Simple regex to extract content between parentheses
        match = re.search(r"\((.*?)\)", signature)
        if match:
            params_str = match.group(1)
            if not params_str:
                return []
            # Split by comma, handling nested types
            params = []
            current = ""
            depth = 0
            for char in params_str:
                if char == "," and depth == 0:
                    params.append(current.strip())
                    current = ""
                else:
                    if char in "(<[{":
                        depth += 1
                    elif char in ")>]}":
                        depth -= 1
                    current += char
            if current:
                params.append(current.strip())
            return params
        return []

    def _extract_return_type(self, signature: str) -> str | None:
        """Extract return type from function signature."""
        # Look for -> pattern
        match = re.search(r"->\s*(.+?)(?:\s*(?:where|{|$))", signature)
        if match:
            return match.group(1).strip()
        return None


class MigrationSuggestionGenerator:
    """Generates helpful migration suggestions for breaking changes."""

    def generate_suggestion(self, change: ItemChange) -> dict[str, str] | None:
        """Generate migration suggestion for a breaking change.

        Args:
            change: The breaking change to generate suggestion for

        Returns:
            Dictionary with migration guidance or None
        """
        suggestion = {}

        if change.change_type.value == "removed":
            suggestion["action"] = "remove_usage"
            suggestion["description"] = f"Remove all usages of '{change.path}'"
            suggestion["hint"] = "Search for alternative APIs or refactor code"

        elif change.change_type.value == "modified":
            # Signature change
            if any(
                "signature" in s.lower()
                for s in change.details.semantic_changes
                if s is not None
            ):
                suggestion["action"] = "update_calls"
                suggestion["description"] = f"Update all calls to '{change.path}'"

                if change.details.before and change.details.after:
                    suggestion["before"] = change.details.before.raw_signature
                    suggestion["after"] = change.details.after.raw_signature
                    suggestion["hint"] = "Update function calls to match new signature"

            # Visibility change
            elif any(
                "private" in s.lower()
                for s in change.details.semantic_changes
                if s is not None
            ):
                suggestion["action"] = "find_alternative"
                suggestion["description"] = f"'{change.path}' is no longer public"
                suggestion["hint"] = "Find alternative public API or refactor"

            # Generic change
            elif any(
                "generic" in s.lower()
                for s in change.details.semantic_changes
                if s is not None
            ):
                suggestion["action"] = "update_types"
                suggestion["description"] = (
                    f"Update type parameters for '{change.path}'"
                )
                suggestion["hint"] = "Adjust generic type arguments and constraints"

        return suggestion if suggestion else None


# Global instances
_breaking_detector: RustBreakingChangeDetector | None = None
_migration_generator: MigrationSuggestionGenerator | None = None


def get_breaking_detector() -> RustBreakingChangeDetector:
    """Get or create the global breaking change detector."""
    global _breaking_detector
    if _breaking_detector is None:
        _breaking_detector = RustBreakingChangeDetector()
    return _breaking_detector


def get_migration_generator() -> MigrationSuggestionGenerator:
    """Get or create the global migration suggestion generator."""
    global _migration_generator
    if _migration_generator is None:
        _migration_generator = MigrationSuggestionGenerator()
    return _migration_generator
