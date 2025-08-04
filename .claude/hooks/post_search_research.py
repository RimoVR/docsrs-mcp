#!/usr/bin/env python3
"""
Post-search hook to dynamically decide if findings should be added to ResearchFindings.json
"""

import json
import sys
from datetime import datetime
from pathlib import Path


def is_relevant_finding(search_query, search_results, relevance_score_out):
    """
    Determine if search results contain information beyond Claude's knowledge scope
    or are particularly shocking/important for the project.
    """
    # Keywords that indicate relevant technical findings
    relevance_indicators = [
        "version",
        "v2",
        "v3",
        "breaking change",
        "deprecated",
        "security",
        "vulnerability",
        "CVE",
        "patch",
        "performance",
        "optimization",
        "benchmark",
        "best practice",
        "recommended",
        "anti-pattern",
        "bug",
        "issue",
        "workaround",
        "fix",
        "update",
        "release",
        "announcement",
        "experimental",
        "beta",
        "alpha",
        "preview",
    ]

    # Check if query relates to specific versions or recent updates
    query_lower = search_query.lower()
    results_lower = json.dumps(search_results).lower()

    # Count relevance indicators
    relevance_score = sum(
        1
        for indicator in relevance_indicators
        if indicator in query_lower or indicator in results_lower
    )

    relevance_score_out[0] = relevance_score

    # Consider relevant if score > 2 or contains version-specific info
    return relevance_score > 2 or any(
        term in results_lower for term in ["2024", "2025", "latest"]
    )


def update_research_findings(finding_data):
    """
    Update ResearchFindings.json with new relevant findings
    """
    findings_path = Path("ResearchFindings.json")

    # Load existing findings or create new structure
    if findings_path.exists():
        with open(findings_path) as f:
            findings = json.load(f)
    else:
        findings = {
            "findings": [],
            "lastUpdated": "",
            "categories": [
                "libraries",
                "patterns",
                "solutions",
                "version-specific",
                "security",
            ],
        }

    # Add new finding
    findings["findings"].append(finding_data)
    findings["lastUpdated"] = datetime.now().isoformat()

    # Keep only the most recent 100 findings to prevent file bloat
    if len(findings["findings"]) > 100:
        findings["findings"] = findings["findings"][-100:]

    # Write back to file
    with open(findings_path, "w") as f:
        json.dump(findings, f, indent=2)


def main():
    """
    Process PostToolUse hook data for WebSearch tool
    """
    try:
        # Read JSON from stdin as per Claude Code documentation
        data = json.load(sys.stdin)

        # Extract search information
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})
        tool_output = data.get("tool_output", "")

        # Only process WebSearch results
        if tool_name != "WebSearch":
            return 0

        search_query = tool_input.get("query", "")

        # Check if findings are relevant
        relevance_score = [0]
        if is_relevant_finding(search_query, tool_output, relevance_score):
            finding = {
                "timestamp": datetime.now().isoformat(),
                "query": search_query,
                "summary": f"Search for: {search_query}",
                "category": "version-specific"
                if any(y in search_query.lower() for y in ["2024", "2025", "version"])
                else "general",
                "relevantInfo": tool_output[:500]
                if isinstance(tool_output, str)
                else str(tool_output)[:500],
                "fullContext": {
                    "allowedDomains": tool_input.get("allowed_domains", []),
                    "blockedDomains": tool_input.get("blocked_domains", []),
                },
            }

            update_research_findings(finding)

            # Provide feedback to Claude
            print(
                f"Added search findings for '{search_query}' to ResearchFindings.json (relevance: {relevance_score[0]})"
            )

    except Exception as e:
        # Log error but don't fail the hook
        print(f"Error processing search results: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
