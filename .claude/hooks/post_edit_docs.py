#!/usr/bin/env python3
"""
Post-edit hook to suggest documentation updates when relevant
"""

import json
import sys
from pathlib import Path


def analyze_edit_significance(file_path, edit_content):
    """
    Determine if an edit is significant enough for documentation
    """
    significance_indicators = {
        "high": [
            "pub fn",
            "pub struct",
            "pub trait",
            "pub enum",  # Rust public API
            "export",
            "interface",
            "type",
            "class",  # TypeScript exports
            "async fn",
            "impl",
            "derive",  # Rust patterns
            "useState",
            "useEffect",
            "component",  # React patterns
            "CREATE",
            "ALTER",
            "TABLE",  # SQL
            "breaking",
            "deprecated",
            "BREAKING CHANGE",  # Version markers
        ],
        "medium": [
            "TODO",
            "FIXME",
            "HACK",
            "BUG",  # Code markers
            "error",
            "panic",
            "unwrap",
            "expect",  # Error handling
            "config",
            "settings",
            "env",  # Configuration
            "test",
            "spec",
            "mock",  # Testing
        ],
    }

    content_lower = edit_content.lower() if edit_content else ""

    # Check for high significance indicators
    high_score = sum(
        1
        for indicator in significance_indicators["high"]
        if indicator.lower() in content_lower
    )

    # Check for medium significance indicators
    medium_score = sum(
        1
        for indicator in significance_indicators["medium"]
        if indicator.lower() in content_lower
    )

    # File type significance
    path = Path(file_path)
    is_api_file = any(
        part in path.parts for part in ["api", "routes", "handlers", "services"]
    )
    is_config_file = path.name in [
        "Cargo.toml",
        "package.json",
        ".env",
        "config.rs",
        "config.ts",
    ]

    return {
        "is_significant": high_score > 0
        or medium_score > 2
        or is_api_file
        or is_config_file,
        "significance_level": "high" if high_score > 0 or is_api_file else "medium",
        "reasons": [],
    }


def suggest_documentation_updates(file_path, significance):
    """
    Generate documentation update suggestions
    """
    suggestions = []
    path = Path(file_path)

    # Module README suggestion
    module_dir = path.parent
    if module_dir.name not in [".", "/", "src"]:
        suggestions.append(
            {
                "file": str(module_dir / "README.md"),
                "update_type": "module_readme",
                "reason": "Document new functionality or API changes in module README",
            }
        )

    # Architecture.md suggestion for structural changes
    if significance["significance_level"] == "high":
        suggestions.append(
            {
                "file": "Architecture.md",
                "update_type": "architecture",
                "reason": "Update component relationships or data flow in mermaid diagrams",
            }
        )

    # UsefulInformation.json for error handling or workarounds
    if any(keyword in str(path) for keyword in ["error", "fix", "workaround", "hack"]):
        suggestions.append(
            {
                "file": "UsefulInformation.json",
                "update_type": "lessons",
                "reason": "Document error solutions or workarounds discovered",
            }
        )

    # Language-specific suggestions
    if path.suffix == ".rs":
        if "api" in str(path) or "handler" in str(path):
            suggestions.append(
                {
                    "file": "docs/API.md",
                    "update_type": "api_docs",
                    "reason": "Update API documentation for Rust endpoints",
                }
            )
    elif path.suffix in [".ts", ".tsx"]:
        if "component" in str(path):
            suggestions.append(
                {
                    "file": "docs/Components.md",
                    "update_type": "component_docs",
                    "reason": "Document React component props and usage",
                }
            )

    return suggestions


def main():
    """
    Process PostToolUse hook data for Edit/Write tools
    """
    try:
        # Read JSON from stdin as per Claude Code documentation
        data = json.load(sys.stdin)

        # Extract tool information
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # Process Edit, MultiEdit, and Write tools
        if tool_name not in ["Edit", "MultiEdit", "Write"]:
            return 0

        # Get file path and content
        file_path = tool_input.get("file_path", "")
        if not file_path:
            return 0

        # Get edit content
        edit_content = ""
        if tool_name == "Edit":
            edit_content = tool_input.get("new_string", "")
        elif tool_name == "MultiEdit":
            edits = tool_input.get("edits", [])
            edit_content = " ".join(edit.get("new_string", "") for edit in edits)
        elif tool_name == "Write":
            edit_content = tool_input.get("content", "")

        # Analyze significance
        significance = analyze_edit_significance(file_path, edit_content)

        if significance["is_significant"]:
            # Generate suggestions
            suggestions = suggest_documentation_updates(file_path, significance)

            if suggestions:
                print(f"\nDocumentation Update Suggestions for {Path(file_path).name}:")
                print(f"   Significance: {significance['significance_level']}")

                for suggestion in suggestions:
                    print(f"\n   • Update {suggestion['file']}:")
                    print(f"     {suggestion['reason']}")

                print(
                    "\n   Consider updating these files to keep documentation in sync"
                )

    except Exception as e:
        # Log error but don't fail the hook
        print(f"Documentation hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
