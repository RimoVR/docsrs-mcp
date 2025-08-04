#!/usr/bin/env python3
"""
Post-command hook to suggest web searches for failed bug fixes
"""

import json
import sys


def extract_error_info(command, output):
    """
    Extract key error information from command output
    """
    error_keywords = []
    output_lower = output.lower()

    # Common error patterns to search for
    if "error" in output_lower:
        lines = output.split("\n")
        for line in lines:
            if "error" in line.lower():
                # Extract the error message
                error_keywords.append(line.strip())

    # Language-specific error patterns
    if "cargo" in command:
        if "failed to compile" in output:
            error_keywords.append("rust compilation error")
        if "unresolved import" in output:
            error_keywords.append("rust unresolved import")
        if "clippy" in command and (
            "warning" in output_lower or "error" in output_lower
        ):
            error_keywords.append("rust clippy warnings")

    elif "npm" in command or "node" in command:
        if "MODULE_NOT_FOUND" in output:
            error_keywords.append("node module not found")
        if "TypeError" in output or "ReferenceError" in output:
            error_keywords.append("javascript runtime error")
        if "lint" in command and "error" in output_lower:
            error_keywords.append("eslint errors")

    # Build tool errors
    if "docker" in command and "build" in command:
        if "error" in output_lower or "failed" in output_lower:
            error_keywords.append("docker build failed")

    return error_keywords


def suggest_search_queries(command, error_keywords):
    """
    Generate suggested search queries for the errors
    """
    suggestions = []

    # Get the tool/technology context
    tech_context = ""
    if "cargo" in command:
        tech_context = "rust"
    elif "npm" in command:
        tech_context = "nodejs npm"
    elif "docker" in command:
        tech_context = "docker"

    # Generate specific search suggestions
    for error in error_keywords[:3]:  # Limit to top 3 errors
        # Clean up the error message
        clean_error = error.replace("error:", "").strip()
        if len(clean_error) > 100:
            clean_error = clean_error[:100] + "..."

        suggestions.append(
            {
                "query": f"{tech_context} {clean_error} solution 2024",
                "purpose": "Find recent solutions and workarounds",
            }
        )

    # Add general troubleshooting query
    if tech_context:
        suggestions.append(
            {
                "query": f"{tech_context} common errors troubleshooting guide",
                "purpose": "General troubleshooting reference",
            }
        )

    return suggestions


def main():
    """
    Process PostToolUse hook data for Bash tool
    """
    try:
        # Read JSON from stdin as per Claude Code documentation
        data = json.load(sys.stdin)

        # Extract command information
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})
        tool_output = data.get("tool_output", {})

        # Only process Bash commands
        if tool_name != "Bash":
            return 0

        command = tool_input.get("command", "")
        output = tool_output if isinstance(tool_output, str) else str(tool_output)

        # Check if command indicates a bug fix attempt or compilation/test
        # More sophisticated detection based on actual commands
        bugfix_commands = [
            "cargo build",
            "cargo test",
            "cargo clippy",
            "npm test",
            "npm run lint",
            "pytest",
            "make",
            "docker build",
        ]
        is_bugfix_attempt = any(cmd in command.lower() for cmd in bugfix_commands)

        # Also check for error-fixing patterns in command history
        if not is_bugfix_attempt and "fix" in command.lower():
            is_bugfix_attempt = True

        if not is_bugfix_attempt:
            return 0

        # Check for errors in output
        error_indicators = ["error", "fail", "warning", "exception", "fatal"]
        has_error = any(indicator in output.lower() for indicator in error_indicators)

        if has_error:
            # Extract error information
            error_keywords = extract_error_info(command, output)

            if error_keywords:
                # Generate search suggestions
                suggestions = suggest_search_queries(command, error_keywords)

                # Output suggestions for Claude
                print(
                    "\nBug fix attempt encountered errors. Consider using concurrent web search agents:"
                )
                print("\nSuggested searches:")
                for i, suggestion in enumerate(suggestions, 1):
                    print(f'{i}. Query: "{suggestion["query"]}"')
                    print(f"   Purpose: {suggestion['purpose']}")

                print(
                    "\nUse Task agents to search concurrently for solutions and precedents."
                )

    except Exception as e:
        # Log error but don't fail the hook
        print(f"Hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
