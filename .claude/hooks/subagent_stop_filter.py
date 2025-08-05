#!/usr/bin/env python3
"""
Enhanced SubagentStop hook to validate agent outputs and filter information
Handles JSON validation for plan-writer agents and enforces output schemas
"""

import json
import re
import sys
from typing import Any


class SubagentOutputValidator:
    """Validates and filters subagent outputs based on agent type"""

    # Expected agent types and their output formats
    AGENT_SCHEMAS = {
        "plan-writer": {
            "required_fields": [
                "approach_name",
                "key_decisions",
                "implementation_steps",
                "files_to_modify",
                "advantages",
            ],
            "optional_fields": [
                "tradeoffs",
                "error_handling",
                "extensibility_points",
                "performance_optimizations",
                "resource_considerations",
            ],
        },
        "living-memory-analyzer": {
            "required_fields": ["type", "schema"],
            "context_specific": True,  # Schema varies by document type
        },
        "websearch": {
            "expected_content": ["search results", "documentation", "best practices"],
            "should_not_contain": ["implementation", "code changes"],
        },
        "codebase-analyzer": {
            "required_sections": [
                "summary",
                "relevantComponents",
                "patterns",
                "dependencies",
                "implementationInsights",
            ]
        },
    }

    def __init__(self, transcript: list[dict[str, Any]]):
        self.transcript = transcript
        self.agent_type = self._detect_agent_type()
        self.original_task = self._extract_original_task()

    def _detect_agent_type(self) -> str:
        """Detect the type of agent from the transcript"""
        agent_indicators = {
            "plan-writer": ["implementation sketch", "approach", "plan"],
            "living-memory-analyzer": ["extract", "memory", "document"],
            "websearch": ["search", "research", "documentation"],
            "codebase-analyzer": ["analyze", "codebase", "patterns"],
            "code-linter-formatter": ["lint", "format", "quality"],
        }

        transcript_text = json.dumps(self.transcript).lower()

        for agent_type, indicators in agent_indicators.items():
            if any(indicator in transcript_text for indicator in indicators):
                return agent_type

        return "general"

    def _extract_original_task(self) -> str:
        """Extract the original task from the first user message"""
        for entry in self.transcript:
            if entry.get("role") == "user":
                return entry.get("content", "")[:200]
        return "Unknown task"

    def validate_json_output(self, content: str) -> dict | None:
        """Validate JSON output from agents"""
        try:
            # Look for JSON blocks in the content
            json_pattern = r"```json\s*([\s\S]*?)\s*```"
            matches = re.findall(json_pattern, content)

            if matches:
                # Try to parse the first JSON block
                json_data = json.loads(matches[0])
                return json_data
            # Try to parse the entire content as JSON
            elif content.strip().startswith("{"):
                return json.loads(content)
        except json.JSONDecodeError:
            pass

        return None

    def validate_plan_writer_output(self, json_data: dict) -> list[str]:
        """Validate plan-writer agent output"""
        issues = []
        schema = self.AGENT_SCHEMAS.get("plan-writer", {})

        # Check required fields
        for field in schema.get("required_fields", []):
            if field not in json_data:
                issues.append(f"Missing required field: {field}")

        # Validate implementation_steps is a list
        if "implementation_steps" in json_data:
            if not isinstance(json_data["implementation_steps"], list):
                issues.append("implementation_steps should be a list")

        return issues

    def validate_codebase_analyzer_output(self, json_data: dict) -> list[str]:
        """Validate codebase-analyzer output"""
        issues = []
        schema = self.AGENT_SCHEMAS.get("codebase-analyzer", {})

        for section in schema.get("required_sections", []):
            if section not in json_data:
                issues.append(f"Missing section: {section}")

        return issues

    def extract_relevant_content(self) -> dict[str, Any]:
        """Extract and validate relevant content based on agent type"""
        result = {
            "agent_type": self.agent_type,
            "task": self.original_task,
            "summary": "",
            "key_findings": [],
            "validation_issues": [],
            "warnings": [],
            "structured_output": None,
        }

        # Look for structured JSON output
        for entry in self.transcript:
            if entry.get("role") == "assistant":
                content = entry.get("content", "")

                # Try to extract JSON
                json_data = self.validate_json_output(content)
                if json_data:
                    result["structured_output"] = json_data

                    # Validate based on agent type
                    if self.agent_type == "plan-writer":
                        issues = self.validate_plan_writer_output(json_data)
                        result["validation_issues"].extend(issues)
                    elif self.agent_type == "codebase-analyzer":
                        issues = self.validate_codebase_analyzer_output(json_data)
                        result["validation_issues"].extend(issues)

                # Extract key findings
                self._extract_findings(content, result)

        # Check for agent-specific issues
        if self.agent_type == "websearch":
            self._check_websearch_compliance(result)

        # Generate summary
        result["summary"] = self._generate_summary(result)

        return result

    def _extract_findings(self, content: str, result: dict):
        """Extract key findings from content"""
        # Look for bullet points or numbered lists
        list_items = re.findall(r"^\s*[-•*]\s*(.+)$", content, re.MULTILINE)
        numbered_items = re.findall(r"^\s*\d+\.\s*(.+)$", content, re.MULTILINE)

        findings = list_items + numbered_items
        result["key_findings"].extend(findings[:10])  # Limit to 10 items

    def _check_websearch_compliance(self, result: dict):
        """Check if websearch agent stayed within scope"""
        if result.get("structured_output"):
            output_text = json.dumps(result["structured_output"]).lower()

            # Check for implementation details (should not be present)
            if any(word in output_text for word in ["implement", "code", "write"]):
                result["warnings"].append(
                    "Websearch agent may have exceeded scope - found implementation details"
                )

    def _generate_summary(self, result: dict) -> str:
        """Generate a concise summary"""
        if result["validation_issues"]:
            return f"{self.agent_type} agent had {len(result['validation_issues'])} validation issues"
        elif result["structured_output"]:
            return (
                f"{self.agent_type} agent completed successfully with structured output"
            )
        elif result["key_findings"]:
            return f"{self.agent_type} agent found {len(result['key_findings'])} items"
        else:
            return f"{self.agent_type} agent completed task"


def format_validated_output(validated_data: dict[str, Any]) -> str:
    """Format the validated data into a concise message"""
    lines = []

    # Header with agent type
    lines.append(f"🤖 {validated_data['agent_type'].title()} Agent Output")
    lines.append("=" * 40)

    # Summary
    lines.append(f"\n📋 {validated_data['summary']}")

    # Validation issues (important!)
    if validated_data["validation_issues"]:
        lines.append("\n⚠️  Validation Issues:")
        for issue in validated_data["validation_issues"]:
            lines.append(f"  • {issue}")

    # Warnings
    if validated_data["warnings"]:
        lines.append("\n⚠️  Warnings:")
        for warning in validated_data["warnings"]:
            lines.append(f"  • {warning}")

    # Key findings (limited)
    if validated_data["key_findings"] and len(validated_data["key_findings"]) <= 5:
        lines.append("\n🔍 Key Points:")
        for finding in validated_data["key_findings"][:5]:
            lines.append(f"  • {finding[:100]}...")  # Truncate long findings

    # Structured output indicator
    if validated_data["structured_output"]:
        lines.append("\n✅ Structured JSON output received and validated")

        # For plan-writer, show approach names
        if validated_data["agent_type"] == "plan-writer":
            approach = validated_data["structured_output"].get(
                "approach_name", "Unknown"
            )
            lines.append(f"   Approach: {approach}")

    lines.append("\n" + "=" * 40)

    return "\n".join(lines)


def main():
    """Process SubagentStop hook to validate and filter agent outputs"""
    try:
        # Read JSON from stdin
        input_data = json.load(sys.stdin)

        # Extract transcript
        transcript = input_data.get("transcript", [])
        if not transcript:
            return 0

        # Create validator and process
        validator = SubagentOutputValidator(transcript)
        validated_data = validator.extract_relevant_content()

        # Format output
        filtered_message = format_validated_output(validated_data)

        # Output the filtered message
        print(filtered_message)

        # Also output validation results as JSON to stderr for debugging
        validation_result = {
            "continue": True,
            "agent_type": validated_data["agent_type"],
            "has_issues": bool(validated_data["validation_issues"]),
            "has_structured_output": bool(validated_data["structured_output"]),
        }

        json.dump(validation_result, sys.stderr)

        return 0

    except Exception as e:
        # On error, provide minimal output
        print(f"\n🤖 Agent Output (Error in processing: {str(e)})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
