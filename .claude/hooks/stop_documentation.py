#!/usr/bin/env python3
"""
Stop hook to ensure documentation is updated with progress and findings
"""

import json
import sys
from datetime import datetime


def check_documentation_needs(session_data):
    """
    Analyze session to determine which documentation files need updates
    """
    needs_update = []

    # Keywords that indicate different types of work
    task_keywords = [
        "implement",
        "create",
        "add",
        "fix",
        "refactor",
        "update",
        "complete",
    ]
    architecture_keywords = [
        "component",
        "module",
        "service",
        "api",
        "struct",
        "trait",
        "class",
    ]
    error_keywords = ["error", "bug", "issue", "problem", "fail", "exception"]
    research_keywords = ["search", "find", "research", "investigate", "discover"]

    # Convert session data to searchable text
    session_text = json.dumps(session_data).lower()

    # Check if Tasks.json needs update
    if any(keyword in session_text for keyword in task_keywords):
        needs_update.append(
            {
                "file": "Tasks.json",
                "reason": "Update task status, progress, or add new subtasks",
                "priority": "high",
            }
        )

    # Check if Architecture.md needs update
    if any(keyword in session_text for keyword in architecture_keywords):
        needs_update.append(
            {
                "file": "Architecture.md",
                "reason": "Update component diagrams or relationships",
                "priority": "medium",
            }
        )

    # Check if UsefulInformation.json needs update
    if any(keyword in session_text for keyword in error_keywords):
        needs_update.append(
            {
                "file": "UsefulInformation.json",
                "reason": "Document errors encountered and solutions found",
                "priority": "high",
            }
        )

    # Check if ResearchFindings.json needs update
    if any(keyword in session_text for keyword in research_keywords):
        needs_update.append(
            {
                "file": "ResearchFindings.json",
                "reason": "Add new research findings or library information",
                "priority": "medium",
            }
        )

    # Check for module updates
    if "edit" in session_text or "write" in session_text:
        needs_update.append(
            {
                "file": "Module README files",
                "reason": "Update module documentation with new functionality",
                "priority": "medium",
            }
        )

    return needs_update


def generate_update_reminders(needs_update):
    """
    Generate actionable reminders for documentation updates
    """
    if not needs_update:
        return None

    reminders = {
        "Tasks.json": [
            "- Mark completed tasks as 'completed' with progress: 100",
            "- Update in_progress tasks with current progress percentage",
            "- Add any newly discovered subtasks",
            "- Document any roadblocks encountered",
        ],
        "Architecture.md": [
            "- Update mermaid diagrams with new components",
            "- Document new relationships between modules",
            "- Add any new design decisions made",
            "- Update technology stack if changed",
        ],
        "UsefulInformation.json": [
            "- Add unexpected errors and their solutions",
            "- Document workarounds for library issues",
            "- Record performance optimization discoveries",
            "- Note any quirks or gotchas found",
        ],
        "ResearchFindings.json": [
            "- Add version-specific library information",
            "- Document best practices discovered",
            "- Include useful documentation links",
            "- Record compatibility notes",
        ],
        "Module README files": [
            "- Document new functions or APIs added",
            "- Update usage examples",
            "- Note any breaking changes",
            "- Add configuration details",
        ],
    }

    return reminders


def main():
    """
    Process Stop hook to check documentation needs
    """
    try:
        # Read JSON from stdin as per Claude Code documentation
        session_data = json.load(sys.stdin)
    except:
        # If no data available, create minimal structure
        session_data = {"timestamp": datetime.now().isoformat(), "hook_type": "Stop"}

    # Check what documentation needs updating
    needs_update = check_documentation_needs(session_data)

    if needs_update:
        print("\nMANDATORY Documentation Check")
        print("=" * 50)
        print("\nThe following documentation files may need updates:")

        # Sort by priority
        high_priority = [item for item in needs_update if item["priority"] == "high"]
        medium_priority = [
            item for item in needs_update if item["priority"] == "medium"
        ]

        reminders = generate_update_reminders(needs_update)

        if high_priority:
            print("\n[HIGH PRIORITY]:")
            for item in high_priority:
                print(f"\n• {item['file']}")
                print(f"  Reason: {item['reason']}")
                if item["file"] in reminders:
                    for reminder in reminders[item["file"]]:
                        print(f"  {reminder}")

        if medium_priority:
            print("\n[MEDIUM PRIORITY]:")
            for item in medium_priority:
                print(f"\n• {item['file']}")
                print(f"  Reason: {item['reason']}")
                if item["file"] in reminders:
                    for reminder in reminders[item["file"]]:
                        print(f"  {reminder}")

        print("\n" + "=" * 50)
        print("Update these files to maintain project memory!")
        print("   Remove outdated information and add new findings.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
