#!/usr/bin/env python3
"""
Pre-tool hook to enforce MCP server usage patterns from CLAUDE.md
Ensures Probe is used for broad searches, Serena only for precise symbolic operations
"""

import json
import os
import sys


class MCPUsageEnforcer:
    """Enforces proper MCP server usage patterns"""

    # Serena tools that require careful usage
    SERENA_TOOLS = {
        "mcp__serena__get_symbols_overview",
        "mcp__serena__find_symbol",
        "mcp__serena__find_referencing_symbols",
        "mcp__serena__replace_symbol_body",
        "mcp__serena__insert_after_symbol",
        "mcp__serena__insert_before_symbol",
        "mcp__serena__search_for_pattern",
        "mcp__serena__list_dir",
        "mcp__serena__find_file",
    }

    # Probe tools for broad searches
    PROBE_TOOLS = {
        "mcp__probe__search_code",
        "mcp__probe__query_code",
        "mcp__probe__extract_code",
    }

    # Context7/Octocode tools (token-heavy)
    EXTERNAL_TOOLS = {
        "mcp__context7__resolve-library-id",
        "mcp__context7__get-library-docs",
        "mcp__octocode__githubSearchCode",
        "mcp__octocode__githubGetFileContent",
        "mcp__octocode__githubSearchRepositories",
        "mcp__octocode__packageSearch",
    }

    def __init__(self, data):
        self.tool_name = data.get("tool_name", "")
        self.tool_input = data.get("tool_input", {})
        self.context = data.get("context", {})
        self.session_history = self._load_session_history()

    def _load_session_history(self):
        """Load session history from temp file"""
        history_file = "/tmp/claude_mcp_usage_history.json"
        if os.path.exists(history_file):
            try:
                with open(history_file) as f:
                    return json.load(f)
            except:
                pass
        return {"probe_searches": [], "serena_searches": [], "warnings": 0}

    def _save_session_history(self):
        """Save session history to temp file"""
        history_file = "/tmp/claude_mcp_usage_history.json"
        try:
            with open(history_file, "w") as f:
                json.dump(self.session_history, f)
        except:
            pass

    def check_serena_usage(self):
        """Check if Serena is being used appropriately"""
        if self.tool_name not in self.SERENA_TOOLS:
            return True, None

        # Extract key parameters
        relative_path = self.tool_input.get("relative_path", "")
        name_path = self.tool_input.get("name_path", "")
        pattern = self.tool_input.get("substring_pattern", "")

        # Check for broad search patterns
        issues = []

        # 1. Check if using search_for_pattern for broad searches
        if self.tool_name == "mcp__serena__search_for_pattern":
            if pattern and len(pattern) < 5:
                issues.append("Pattern too short - may cause LSP overload")
            if not relative_path or relative_path == ".":
                issues.append("No path restriction - searching entire codebase")

        # 2. Check if using get_symbols_overview on large directories
        if self.tool_name == "mcp__serena__get_symbols_overview":
            if relative_path == "." or relative_path == "":
                issues.append("Attempting overview of entire codebase")

        # 3. Check if find_symbol is too broad
        if self.tool_name == "mcp__serena__find_symbol":
            if name_path and len(name_path) < 3:
                issues.append("Symbol name too short - be more specific")
            if not relative_path:
                issues.append("No path restriction for symbol search")

        # 4. Check if Probe was used first
        recent_probe = any(
            tool in self.session_history.get("probe_searches", [])[-5:]
            for tool in self.PROBE_TOOLS
        )
        if not recent_probe and self.tool_name in [
            "mcp__serena__find_symbol",
            "mcp__serena__search_for_pattern",
        ]:
            issues.append("Consider using Probe first for discovery")

        if issues:
            return False, issues
        return True, None

    def check_external_tool_usage(self):
        """Check if Context7/Octocode tools are used appropriately"""
        if self.tool_name not in self.EXTERNAL_TOOLS:
            return True, None

        # These should primarily be used within websearch agents
        if "agent" not in str(self.context).lower():
            return False, ["External MCP tools should be used through websearch agents"]

        return True, None

    def suggest_alternative(self):
        """Suggest better tool usage"""
        suggestions = []

        if self.tool_name in self.SERENA_TOOLS:
            suggestions.append("For broad code searches, use Probe MCP:")
            suggestions.append("- mcp__probe__search_code: ElasticSearch-like queries")
            suggestions.append("- mcp__probe__query_code: AST pattern matching")
            suggestions.append("- mcp__probe__extract_code: Get full context")
            suggestions.append(
                "\nUse Serena only after Probe identifies specific targets"
            )

        if self.tool_name in self.EXTERNAL_TOOLS:
            suggestions.append("Use these tools through websearch agents:")
            suggestions.append("- Invoke Task with subagent_type='websearch'")
            suggestions.append("- Let agents handle token-heavy operations")

        return suggestions

    def enforce(self):
        """Main enforcement logic"""
        # Track tool usage
        if self.tool_name in self.PROBE_TOOLS:
            self.session_history["probe_searches"].append(self.tool_name)
        elif self.tool_name in self.SERENA_TOOLS:
            self.session_history["serena_searches"].append(self.tool_name)

        # Check Serena usage
        serena_ok, serena_issues = self.check_serena_usage()

        # Check external tool usage
        external_ok, external_issues = self.check_external_tool_usage()

        # Build response
        if not serena_ok or not external_ok:
            print("\n⚠️  MCP USAGE WARNING")
            print("=" * 50)

            if serena_issues:
                print("\nSerena Usage Issues:")
                for issue in serena_issues:
                    print(f"  • {issue}")

            if external_issues:
                print("\nExternal Tool Issues:")
                for issue in external_issues:
                    print(f"  • {issue}")

            print("\nSuggestions:")
            for suggestion in self.suggest_alternative():
                print(f"  {suggestion}")

            print("\nWorkflow Reminder:")
            print("  1. Start with Probe for discovery (fast, no indexing)")
            print("  2. Use Serena for precise operations on found symbols")
            print("  3. Keep Serena searches targeted to avoid LSP crashes")
            print("=" * 50)

            self.session_history["warnings"] += 1

            # Block if too many warnings
            if self.session_history["warnings"] > 3:
                print("\n🛑 BLOCKING: Too many MCP usage warnings")
                print("Please follow the Probe → Serena workflow pattern")
                self._save_session_history()
                return {"decision": "block", "message": "Improper MCP usage pattern"}

        self._save_session_history()
        return {"decision": "allow"}


def main():
    """Process PreToolUse hook data"""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)

        # Create enforcer and run checks
        enforcer = MCPUsageEnforcer(data)
        result = enforcer.enforce()

        # Output decision
        if result.get("decision") == "block":
            print(json.dumps(result))
            return 1

    except Exception as e:
        # Log error but don't block
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
