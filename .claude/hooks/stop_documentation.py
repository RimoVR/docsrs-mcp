#!/usr/bin/env python3
"""
Enhanced stop hook for comprehensive session wrap-up and documentation check
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class SessionWrapUp:
    """Comprehensive session analysis and wrap-up"""

    def __init__(self, session_data: dict, project_dir: str):
        self.session_data = session_data
        self.project_dir = Path(project_dir)
        self.session_text = json.dumps(session_data).lower()

    def check_todo_completion(self) -> tuple[bool, list[str]]:
        """Check if todo items were properly marked complete"""
        issues = []

        # Look for todo-related patterns
        todo_patterns = ["todowrite", "todo list", "marking.*completed", "task.*done"]

        has_todos = any(pattern in self.session_text for pattern in todo_patterns)

        if has_todos:
            # Check for incomplete patterns
            incomplete_patterns = ["pending", "in_progress", "blocked", "not.*complete"]

            if any(pattern in self.session_text for pattern in incomplete_patterns):
                issues.append("Some todo items may still be pending")
                issues.append("Review todo list and mark completed items")

        return has_todos, issues

    def check_uncommitted_changes(self) -> list[str]:
        """Check for indicators of uncommitted changes"""
        reminders = []

        # File operation keywords
        file_ops = ["edit", "write", "create", "modify", "update"]
        has_file_ops = any(op in self.session_text for op in file_ops)

        # Git operation keywords
        git_ops = ["git add", "git commit", "git push"]
        has_git_ops = any(op in self.session_text for op in git_ops)

        if has_file_ops and not has_git_ops:
            reminders.append("Files were modified but no git commits detected")
            reminders.append("Consider committing changes with semantic messages")
            reminders.append("Use /commit-fast for quick commits")

        return reminders

    def analyze_session_activity(self) -> dict[str, list[str]]:
        """Analyze session to determine documentation needs"""
        needs_update = {
            "Tasks.json": [],
            "Architecture.md": [],
            "UsefulInformation.json": [],
            "ResearchFindings.json": [],
            "Module READMEs": [],
        }

        # Task-related activity
        task_keywords = ["implement", "create", "add", "fix", "refactor", "complete"]
        if any(kw in self.session_text for kw in task_keywords):
            needs_update["Tasks.json"].extend(
                [
                    "Update task completion status",
                    "Add progress percentages",
                    "Document any blockers",
                ]
            )

        # Architecture changes
        arch_keywords = ["component", "module", "service", "interface", "design"]
        if any(kw in self.session_text for kw in arch_keywords):
            needs_update["Architecture.md"].extend(
                [
                    "Update component diagrams",
                    "Document new relationships",
                    "Add design decisions",
                ]
            )

        # Error handling
        error_keywords = [
            "error",
            "exception",
            "fail",
            "issue",
            "problem",
            "workaround",
        ]
        if any(kw in self.session_text for kw in error_keywords):
            needs_update["UsefulInformation.json"].extend(
                [
                    "Document errors and solutions",
                    "Add workarounds discovered",
                    "Record debugging insights",
                ]
            )

        # Research activity
        research_keywords = [
            "library",
            "framework",
            "version",
            "dependency",
            "research",
        ]
        if any(kw in self.session_text for kw in research_keywords):
            needs_update["ResearchFindings.json"].extend(
                [
                    "Add library findings",
                    "Document version compatibility",
                    "Include best practices",
                ]
            )

        # Module changes
        if "edit" in self.session_text or "write" in self.session_text:
            needs_update["Module READMEs"].extend(
                [
                    "Update API documentation",
                    "Add usage examples",
                    "Document breaking changes",
                ]
            )

        # Filter out empty entries
        return {k: v for k, v in needs_update.items() if v}

    def check_mcp_usage_patterns(self) -> list[str]:
        """Check if MCP servers were used appropriately"""
        warnings = []

        # Check for Serena usage without Probe
        has_serena = "mcp__serena__" in self.session_text
        has_probe = "mcp__probe__" in self.session_text

        if has_serena and not has_probe:
            warnings.append("Serena was used without Probe - remember Probe first!")

        # Check for direct Context7/Octocode usage
        external_mcps = ["mcp__context7__", "mcp__octocode__"]
        if any(mcp in self.session_text for mcp in external_mcps):
            if "websearch" not in self.session_text:
                warnings.append("External MCPs used directly - prefer websearch agents")

        return warnings

    def generate_agent_suggestions(self) -> list[str]:
        """Suggest using agents for documentation updates"""
        suggestions = []

        needs_update = self.analyze_session_activity()

        if needs_update:
            suggestions.append("\n🤖 Use living-memory-updater agent:")
            for file, updates in needs_update.items():
                if updates:
                    suggestions.append(f"\nTask living-memory-updater '{file}':")
                    for update in updates[:2]:  # Show first 2 items
                        suggestions.append(f"  • {update}")

        return suggestions


def main():
    """Process Stop hook for comprehensive session wrap-up"""
    try:
        # Read JSON from stdin
        session_data = json.load(sys.stdin)
    except:
        session_data = {"timestamp": datetime.now().isoformat()}

    # Get project directory
    project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")

    # Create wrap-up analyzer
    wrap_up = SessionWrapUp(session_data, project_dir)

    print("\n" + "=" * 60)
    print("📋 SESSION WRAP-UP & DOCUMENTATION CHECK")
    print("=" * 60)

    # Check todo completion
    has_todos, todo_issues = wrap_up.check_todo_completion()
    if todo_issues:
        print("\n✅ Todo List Status:")
        for issue in todo_issues:
            print(f"  • {issue}")

    # Check uncommitted changes
    git_reminders = wrap_up.check_uncommitted_changes()
    if git_reminders:
        print("\n🔀 Version Control:")
        for reminder in git_reminders:
            print(f"  • {reminder}")

    # Check MCP usage
    mcp_warnings = wrap_up.check_mcp_usage_patterns()
    if mcp_warnings:
        print("\n⚠️  MCP Usage:")
        for warning in mcp_warnings:
            print(f"  • {warning}")

    # Analyze documentation needs
    needs_update = wrap_up.analyze_session_activity()
    if needs_update:
        print("\n📚 Documentation Updates Needed:")

        # Prioritize by importance
        priority_order = [
            "Tasks.json",
            "UsefulInformation.json",
            "ResearchFindings.json",
            "Architecture.md",
            "Module READMEs",
        ]

        for file in priority_order:
            if file in needs_update and needs_update[file]:
                print(f"\n  📄 {file}:")
                for item in needs_update[file]:
                    print(f"     • {item}")

    # Generate agent suggestions
    agent_suggestions = wrap_up.generate_agent_suggestions()
    if agent_suggestions:
        print("\n".join(agent_suggestions))

    # Final reminders
    print("\n💡 Session Summary:")
    print("  • Update living memory files to preserve knowledge")
    print("  • Commit changes with semantic messages")
    print("  • Review and complete any pending todos")
    print("  • Follow Probe → Serena workflow pattern")

    print("\n" + "=" * 60)
    print("Great work! Session context preserved for future reference.")
    print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
