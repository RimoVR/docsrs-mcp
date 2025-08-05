#!/usr/bin/env python3
"""
Session start hook to remind about workflow best practices and available commands
"""

import json
import os
import sys
from pathlib import Path


def check_project_state():
    """Check current project state and provide context"""
    insights = []

    # Check for CLAUDE.md
    claude_md = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / "CLAUDE.md"
    if claude_md.exists():
        insights.append("✅ CLAUDE.md found - project instructions loaded")
    else:
        insights.append("⚠️  No CLAUDE.md found - consider creating one")

    # Check for living memory files
    memory_files = {
        "PRD.md": "Product requirements",
        "Tasks.json": "Task management",
        "Architecture.md": "System design",
        "ResearchFindings.json": "Library research",
        "UsefulInformation.json": "Errors & solutions",
    }

    found_files = []
    for file, desc in memory_files.items():
        if (Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / file).exists():
            found_files.append(f"  • {file} ({desc})")

    if found_files:
        insights.append("\n📚 Living Memory Files Found:\n" + "\n".join(found_files))

    # Check for recent plan
    temp_plan = Path(os.environ.get("CLAUDE_PROJECT_DIR", ".")) / "TemporaryPlan.md"
    if temp_plan.exists():
        insights.append("\n📋 TemporaryPlan.md exists - use /revise-plan to execute")

    return insights


def suggest_workflow(data):
    """Suggest appropriate workflow based on context"""
    suggestions = []

    # Check if resuming a session
    is_resume = data.get("is_resume", False)

    if is_resume:
        suggestions.append("🔄 Resuming session - consider reviewing:")
        suggestions.append("  • Todo list status (if any)")
        suggestions.append("  • Recent changes made")
        suggestions.append("  • Any pending tasks")
    else:
        suggestions.append("🚀 Starting new session")

    # Always remind about key workflows
    suggestions.append("\n📖 Key Workflow Commands:")
    suggestions.append("  • /initialize-PRD - Set up new project from PRD")
    suggestions.append("  • /initialize-existing - Bootstrap existing project")
    suggestions.append("  • /create-plan - Plan implementation with research")
    suggestions.append("  • /revise-plan - Execute plan with quality checks")
    suggestions.append("  • /add-feature-to-task - Add feature with research")
    suggestions.append("  • /debug-feature - Debug with MCP assistance")

    suggestions.append("\n🔧 MCP Server Guidelines:")
    suggestions.append("  • Always start with Probe for code discovery")
    suggestions.append("  • Use Serena only for precise symbolic operations")
    suggestions.append("  • Avoid broad Serena searches (can crash LSP)")
    suggestions.append("  • Context7/Octocode through websearch agents only")

    suggestions.append("\n💡 Best Practices:")
    suggestions.append("  • Use TodoWrite for complex multi-step tasks")
    suggestions.append("  • Update living memory files as you progress")
    suggestions.append("  • Run linters after code changes")
    suggestions.append("  • Commit with semantic messages")

    return suggestions


def main():
    """Process SessionStart hook data"""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)

        print("\n" + "=" * 60)
        print("🎯 CLAUDE CODE WORKFLOW ASSISTANT")
        print("=" * 60)

        # Check project state
        insights = check_project_state()
        if insights:
            print("\n📊 Project Status:")
            for insight in insights:
                print(insight)

        # Provide workflow suggestions
        suggestions = suggest_workflow(data)
        print("\n" + "\n".join(suggestions))

        # Check for common issues
        print("\n⚠️  Reminders:")
        print("  • Don't let perfect be enemy of good")
        print("  • Keep it simple (KISS principle)")
        print("  • Fail fast with clear errors")
        print("  • Follow language idioms")

        print("\n" + "=" * 60)
        print("Ready to assist! What would you like to work on?")
        print("=" * 60 + "\n")

    except Exception as e:
        # Log error but don't fail
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
