#!/usr/bin/env python3
"""
Pre-prompt hook to validate and enhance user requests with workflow suggestions
"""

import json
import re
import sys


class PromptWorkflowValidator:
    """Analyzes prompts and suggests appropriate workflows"""

    # Keywords that indicate different types of tasks
    IMPLEMENTATION_KEYWORDS = [
        "implement",
        "add",
        "create",
        "build",
        "develop",
        "write",
        "feature",
        "functionality",
        "component",
        "module",
        "service",
    ]

    DEBUG_KEYWORDS = [
        "debug",
        "fix",
        "error",
        "bug",
        "issue",
        "problem",
        "crash",
        "failing",
        "broken",
        "not working",
        "exception",
        "traceback",
    ]

    RESEARCH_KEYWORDS = [
        "how",
        "what",
        "which",
        "best practice",
        "approach",
        "pattern",
        "library",
        "framework",
        "option",
        "alternative",
        "compare",
    ]

    PLANNING_KEYWORDS = [
        "plan",
        "design",
        "architect",
        "structure",
        "organize",
        "approach",
        "strategy",
        "roadmap",
    ]

    # MCP server mentions
    MCP_PATTERNS = [
        r"mcp__serena__",
        r"mcp__probe__",
        r"mcp__context7__",
        r"mcp__octocode__",
        r"serena\s+(search|find)",
        r"probe\s+(search|query)",
        r"search\s+code",
        r"find\s+symbol",
    ]

    def __init__(self, prompt: str):
        self.prompt = prompt.lower()
        self.original_prompt = prompt

    def detect_task_type(self) -> tuple[str, float]:
        """Detect the primary task type from the prompt"""
        scores = {
            "implementation": self._score_keywords(self.IMPLEMENTATION_KEYWORDS),
            "debug": self._score_keywords(self.DEBUG_KEYWORDS),
            "research": self._score_keywords(self.RESEARCH_KEYWORDS),
            "planning": self._score_keywords(self.PLANNING_KEYWORDS),
        }

        # Get highest scoring type
        task_type = max(scores, key=scores.get)
        confidence = scores[task_type]

        return task_type, confidence

    def _score_keywords(self, keywords: list[str]) -> float:
        """Score how many keywords appear in prompt"""
        count = sum(1 for keyword in keywords if keyword in self.prompt)
        return count / len(keywords) if keywords else 0

    def detect_mcp_usage(self) -> list[str]:
        """Detect direct MCP server mentions"""
        found = []
        for pattern in self.MCP_PATTERNS:
            if re.search(pattern, self.prompt, re.IGNORECASE):
                found.append(pattern)
        return found

    def check_complexity(self) -> str:
        """Determine if task is complex enough to need planning"""
        # Count sentences/requirements
        sentences = len(re.split(r"[.!?]+", self.original_prompt))

        # Count "and" conjunctions indicating multiple requirements
        and_count = len(re.findall(r"\band\b", self.prompt))

        # Check for numbered lists
        has_list = bool(
            re.search(r"^\s*\d+\.|\n\s*[-*]", self.original_prompt, re.MULTILINE)
        )

        if sentences > 3 or and_count > 2 or has_list:
            return "complex"
        elif sentences > 1 or and_count > 0:
            return "moderate"
        else:
            return "simple"

    def suggest_workflow(self) -> dict[str, any]:
        """Suggest appropriate workflow based on analysis"""
        task_type, confidence = self.detect_task_type()
        complexity = self.check_complexity()
        mcp_mentions = self.detect_mcp_usage()

        suggestions = {
            "task_type": task_type,
            "complexity": complexity,
            "confidence": confidence,
            "workflow_suggestions": [],
            "reminders": [],
            "warnings": [],
        }

        # Task-specific suggestions
        if task_type == "implementation" and complexity != "simple":
            suggestions["workflow_suggestions"].append(
                "Consider using /create-plan to research and plan this implementation"
            )
            suggestions["workflow_suggestions"].append(
                "Then use /revise-plan to execute with quality checks"
            )
            suggestions["reminders"].append(
                "Use TodoWrite to track implementation steps"
            )

        elif task_type == "debug":
            suggestions["workflow_suggestions"].append(
                "Consider using /debug-feature for systematic debugging"
            )
            suggestions["reminders"].append(
                "Failed commands will trigger web search suggestions"
            )

        elif task_type == "planning":
            suggestions["workflow_suggestions"].append(
                "Use /create-plan for comprehensive planning with research"
            )
            suggestions["reminders"].append(
                "Planning agents will analyze PRD and architecture"
            )

        # MCP usage warnings
        if mcp_mentions:
            suggestions["warnings"].append("Direct MCP usage detected - remember:")
            suggestions["warnings"].append("• Use Probe first for broad searches")
            suggestions["warnings"].append("• Serena only for precise operations")
            suggestions["warnings"].append("• Context7/Octocode through agents")

        # Complexity-based suggestions
        if complexity == "complex":
            suggestions["reminders"].append(
                "Complex task detected - consider breaking down with TodoWrite"
            )

        # Check for missing context
        if "continue" in self.prompt or "keep going" in self.prompt:
            suggestions["reminders"].append("Check todo list for pending tasks")

        return suggestions


def format_output(suggestions: dict[str, any]) -> str:
    """Format suggestions for display"""
    output = []

    # Only show if we have high confidence or specific warnings
    if suggestions["confidence"] < 0.2 and not suggestions["warnings"]:
        return ""

    output.append("\n💡 WORKFLOW SUGGESTION")
    output.append("=" * 40)

    if suggestions["workflow_suggestions"]:
        output.append("\n📋 Recommended Workflow:")
        for suggestion in suggestions["workflow_suggestions"]:
            output.append(f"  • {suggestion}")

    if suggestions["warnings"]:
        output.append("\n⚠️  Important:")
        for warning in suggestions["warnings"]:
            output.append(f"  {warning}")

    if suggestions["reminders"]:
        output.append("\n🔔 Reminders:")
        for reminder in suggestions["reminders"]:
            output.append(f"  • {reminder}")

    output.append("=" * 40 + "\n")

    return "\n".join(output)


def main():
    """Process UserPromptSubmit hook data"""
    try:
        # Read JSON from stdin
        data = json.load(sys.stdin)

        # Extract prompt
        prompt = data.get("prompt", "")
        if not prompt:
            return 0

        # Analyze prompt
        validator = PromptWorkflowValidator(prompt)
        suggestions = validator.suggest_workflow()

        # Format and display suggestions
        output = format_output(suggestions)
        if output:
            print(output)

    except Exception as e:
        # Log error but don't fail
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
