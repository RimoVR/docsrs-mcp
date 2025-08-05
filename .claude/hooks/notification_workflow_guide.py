#!/usr/bin/env python3
"""
Notification hook to guide tool permission decisions with workflow context
"""

import json
import sys


class WorkflowGuide:
    """Provides context for tool permission decisions"""

    # Tool categories and their typical use cases
    TOOL_CATEGORIES = {
        "file_operations": {
            "tools": ["Read", "Write", "Edit", "MultiEdit"],
            "context": "File manipulation - core development work",
        },
        "search_operations": {
            "tools": ["Glob", "Grep", "LS"],
            "context": "Code discovery and exploration",
        },
        "mcp_probe": {
            "tools": [
                "mcp__probe__search_code",
                "mcp__probe__query_code",
                "mcp__probe__extract_code",
            ],
            "context": "Fast code search without indexing - USE FIRST",
        },
        "mcp_serena": {
            "tools": ["mcp__serena__find_symbol", "mcp__serena__get_symbols_overview"],
            "context": "Precise semantic operations - USE AFTER PROBE",
        },
        "mcp_external": {
            "tools": ["mcp__context7__", "mcp__octocode__"],
            "context": "External services - PREFER THROUGH AGENTS",
        },
        "execution": {
            "tools": ["Bash"],
            "context": "Command execution - be cautious with destructive operations",
        },
        "web": {
            "tools": ["WebSearch", "WebFetch"],
            "context": "Internet access for research",
        },
        "workflow": {
            "tools": ["Task", "TodoWrite"],
            "context": "Workflow management and agent orchestration",
        },
    }

    # Patterns that suggest workflow commands should be used
    WORKFLOW_TRIGGERS = {
        "complex_implementation": ["implement", "create", "build", "add feature"],
        "debugging": ["debug", "fix", "error", "issue"],
        "research": ["research", "compare", "which library", "best practice"],
    }

    def __init__(self, tool_name: str, context: dict):
        self.tool_name = tool_name
        self.context = context

    def categorize_tool(self) -> str:
        """Determine which category this tool belongs to"""
        for category, info in self.TOOL_CATEGORIES.items():
            if any(tool in self.tool_name for tool in info["tools"]):
                return category
        return "unknown"

    def check_workflow_opportunity(self) -> list[str]:
        """Check if a workflow command might be more appropriate"""
        suggestions = []

        # Check for complex multi-tool sequences
        if self.tool_name == "Task" and "general-purpose" in str(self.context):
            suggestions.append("Consider using specialized workflow commands:")
            suggestions.append("• /create-plan - For implementation planning")
            suggestions.append("• /debug-feature - For debugging workflows")

        # Check for MCP usage patterns
        if "mcp__serena__" in self.tool_name:
            suggestions.append("⚠️  Serena usage - ensure Probe was used first")

        if "mcp__context7__" in self.tool_name or "mcp__octocode__" in self.tool_name:
            suggestions.append("💡 Consider using websearch agent for external MCP")

        return suggestions

    def generate_guidance(self) -> dict[str, any]:
        """Generate contextual guidance for the tool request"""
        category = self.categorize_tool()
        category_info = self.TOOL_CATEGORIES.get(category, {})

        guidance = {
            "tool": self.tool_name,
            "category": category,
            "context": category_info.get("context", "Unknown tool category"),
            "suggestions": [],
            "warnings": [],
        }

        # Add category-specific guidance
        if category == "mcp_serena":
            guidance["warnings"].append("Serena can crash LSP with broad searches")
            guidance["suggestions"].append("Use Probe first for discovery")

        elif category == "mcp_external":
            guidance["suggestions"].append("These tools are token-heavy")
            guidance["suggestions"].append("Best used through websearch agents")

        elif category == "execution":
            if any(cmd in self.tool_name.lower() for cmd in ["rm", "delete", "drop"]):
                guidance["warnings"].append("⚠️  Destructive operation detected")

        # Check for workflow opportunities
        workflow_suggestions = self.check_workflow_opportunity()
        if workflow_suggestions:
            guidance["suggestions"].extend(workflow_suggestions)

        return guidance


def format_notification(guidance: dict) -> str:
    """Format guidance as notification message"""
    lines = []

    # Only show if we have specific guidance
    if not (guidance["suggestions"] or guidance["warnings"]):
        return ""

    lines.append(f"\n🔧 Tool: {guidance['tool']}")
    lines.append(f"📁 Category: {guidance['context']}")

    if guidance["warnings"]:
        lines.append("\n⚠️  Warnings:")
        for warning in guidance["warnings"]:
            lines.append(f"  {warning}")

    if guidance["suggestions"]:
        lines.append("\n💡 Suggestions:")
        for suggestion in guidance["suggestions"]:
            lines.append(f"  {suggestion}")

    lines.append("")  # Empty line at end

    return "\n".join(lines)


def main():
    """Process Notification hook data"""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)

        # Extract tool information
        tool_name = data.get("tool_name", "")
        context = data.get("context", {})

        if not tool_name:
            return 0

        # Generate guidance
        guide = WorkflowGuide(tool_name, context)
        guidance = guide.generate_guidance()

        # Format and display
        output = format_notification(guidance)
        if output:
            print(output)

    except Exception as e:
        # Log error but don't interfere with permissions
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
