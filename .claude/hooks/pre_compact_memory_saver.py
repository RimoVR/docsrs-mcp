#!/usr/bin/env python3
"""
Pre-compact hook to save important context to living memory before compaction
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path


class MemorySaver:
    """Detects and saves important information before context compaction"""

    def __init__(self, project_dir: str):
        self.project_dir = Path(project_dir)
        self.memory_files = {
            "ResearchFindings.json": self.project_dir / "ResearchFindings.json",
            "UsefulInformation.json": self.project_dir / "UsefulInformation.json",
            "Tasks.json": self.project_dir / "Tasks.json",
            "Architecture.md": self.project_dir / "Architecture.md",
        }

    def analyze_conversation_context(self, data: dict) -> dict[str, list[str]]:
        """Analyze conversation to find unsaved information"""
        findings = {"research": [], "errors": [], "architecture": [], "tasks": []}

        # This would need access to conversation history
        # For now, we'll provide reminders based on common patterns

        # Check for research-like patterns
        research_indicators = [
            "library",
            "framework",
            "best practice",
            "documentation",
            "version",
            "dependency",
            "api",
            "pattern",
        ]

        # Check for error patterns
        error_indicators = [
            "error",
            "exception",
            "failed",
            "issue",
            "problem",
            "workaround",
            "solution",
            "fixed",
            "resolved",
        ]

        # Architecture indicators
        arch_indicators = [
            "design",
            "architecture",
            "component",
            "integration",
            "interface",
            "module",
            "service",
            "structure",
        ]

        return findings

    def check_memory_files(self) -> dict[str, any]:
        """Check when memory files were last updated"""
        status = {}

        for name, path in self.memory_files.items():
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                hours_old = (datetime.now() - mtime).total_seconds() / 3600
                status[name] = {
                    "exists": True,
                    "hours_old": round(hours_old, 1),
                    "path": str(path),
                }
            else:
                status[name] = {"exists": False, "path": str(path)}

        return status

    def generate_suggestions(self, status: dict) -> list[str]:
        """Generate suggestions based on file status"""
        suggestions = []

        # Check ResearchFindings.json
        research = status.get("ResearchFindings.json", {})
        if not research.get("exists"):
            suggestions.append(
                "ResearchFindings.json missing - consider creating if you've researched libraries"
            )
        elif research.get("hours_old", 0) > 24:
            suggestions.append(
                "ResearchFindings.json is stale - update with recent discoveries"
            )

        # Check UsefulInformation.json
        useful = status.get("UsefulInformation.json", {})
        if not useful.get("exists"):
            suggestions.append(
                "UsefulInformation.json missing - create to track errors and solutions"
            )

        # Check Tasks.json
        tasks = status.get("Tasks.json", {})
        if tasks.get("exists") and tasks.get("hours_old", 0) > 2:
            suggestions.append(
                "Tasks.json may need updating - mark completed tasks and add new ones"
            )

        return suggestions

    def create_memory_template(self, file_type: str) -> str:
        """Create template content for missing memory files"""
        templates = {
            "ResearchFindings.json": json.dumps(
                {
                    "libraries": {},
                    "patterns": {},
                    "best_practices": {},
                    "documentation_links": {},
                    "last_updated": datetime.now().isoformat(),
                },
                indent=2,
            ),
            "UsefulInformation.json": json.dumps(
                {
                    "errors_and_solutions": [],
                    "workarounds": [],
                    "lessons_learned": [],
                    "code_snippets": {},
                    "last_updated": datetime.now().isoformat(),
                },
                indent=2,
            ),
            "Tasks.json": json.dumps(
                {
                    "tasks": [],
                    "metadata": {
                        "created": datetime.now().isoformat(),
                        "last_updated": datetime.now().isoformat(),
                    },
                },
                indent=2,
            ),
        }

        return templates.get(file_type, "{}")


def main():
    """Process PreCompact hook data"""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)

        # Get project directory
        project_dir = os.environ.get("CLAUDE_PROJECT_DIR", ".")
        saver = MemorySaver(project_dir)

        print("\n🧠 MEMORY PRESERVATION CHECK")
        print("=" * 50)
        print("Context will be compacted. Checking for unsaved information...")

        # Check memory file status
        status = saver.check_memory_files()

        print("\n📊 Memory File Status:")
        for name, info in status.items():
            if info["exists"]:
                print(f"  ✅ {name} - {info['hours_old']}h old")
            else:
                print(f"  ❌ {name} - not found")

        # Generate suggestions
        suggestions = saver.generate_suggestions(status)

        if suggestions:
            print("\n💡 Suggestions:")
            for suggestion in suggestions:
                print(f"  • {suggestion}")

        # Provide quick actions
        print("\n🚀 Quick Actions:")
        print("  • Use living-memory-updater agent to update files")
        print("  • Run: Task living-memory-updater 'Update [file] with [info]'")
        print("  • Critical findings should be saved before compaction")

        # Remind about patterns
        print("\n📝 Information to Preserve:")
        print("  • Library versions and compatibility findings")
        print("  • Error messages and their solutions")
        print("  • Architectural decisions and rationale")
        print("  • Task progress and blockers")

        print("=" * 50)
        print("Proceeding with compaction in 3 seconds...\n")

    except Exception as e:
        # Log error but don't block compaction
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
