#!/usr/bin/env python3
"""
SubagentStop hook to filter and convey only relevant information to the main session
Prevents context overflow by removing redundant details and focusing on answers
"""
import json
import sys
import re
from typing import Dict, List, Any

def extract_relevant_content(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract only the most relevant information from subagent transcript
    """
    result = {
        "summary": "",
        "key_findings": [],
        "errors": [],
        "files_modified": [],
        "commands_run": []
    }
    
    # Track what the subagent was asked to do
    original_task = ""
    
    for entry in transcript:
        if entry.get("role") == "user" and not original_task:
            original_task = entry.get("content", "")[:200]  # First user message
        
        elif entry.get("role") == "assistant":
            content = entry.get("content", "")
            
            # Extract key findings from text
            if any(keyword in content.lower() for keyword in ["found", "discovered", "located", "identified"]):
                # Extract sentences containing findings
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    if any(kw in sentence.lower() for kw in ["found", "discovered", "located", "identified"]):
                        finding = sentence.strip()
                        if finding and len(finding) > 10:
                            result["key_findings"].append(finding)
        
        elif entry.get("role") == "tool_use":
            tool_name = entry.get("name", "")
            tool_input = entry.get("input", {})
            
            # Track file modifications
            if tool_name in ["Edit", "MultiEdit", "Write"]:
                file_path = tool_input.get("file_path", "")
                if file_path:
                    result["files_modified"].append(file_path)
            
            # Track important commands
            elif tool_name == "Bash":
                command = tool_input.get("command", "")
                if command and not any(skip in command for skip in ["ls", "pwd", "cd", "echo"]):
                    result["commands_run"].append(command[:100])
        
        elif entry.get("role") == "tool_result":
            content = entry.get("content", "")
            # Check for errors
            if "error" in content.lower() or "failed" in content.lower():
                error_lines = [line.strip() for line in content.split('\n') 
                             if "error" in line.lower() or "failed" in line.lower()]
                result["errors"].extend(error_lines[:3])  # Limit errors
    
    # Generate concise summary
    if result["key_findings"]:
        result["summary"] = f"Found {len(result['key_findings'])} relevant items for: {original_task}"
    elif result["errors"]:
        result["summary"] = f"Encountered {len(result['errors'])} errors while: {original_task}"
    elif result["files_modified"]:
        result["summary"] = f"Modified {len(result['files_modified'])} files for: {original_task}"
    else:
        result["summary"] = f"Completed task: {original_task}"
    
    # Remove duplicates and limit findings
    result["key_findings"] = list(dict.fromkeys(result["key_findings"]))[:5]
    result["files_modified"] = list(dict.fromkeys(result["files_modified"]))
    result["commands_run"] = list(dict.fromkeys(result["commands_run"]))[:3]
    result["errors"] = list(dict.fromkeys(result["errors"]))[:3]
    
    return result

def format_filtered_output(filtered_data: Dict[str, Any]) -> str:
    """
    Format the filtered data into a concise message
    """
    output_lines = []
    
    # Add summary
    output_lines.append(f"Summary: {filtered_data['summary']}")
    
    # Add key findings if any
    if filtered_data["key_findings"]:
        output_lines.append("\nKey Findings:")
        for finding in filtered_data["key_findings"]:
            output_lines.append(f"  • {finding}")
    
    # Add errors if any
    if filtered_data["errors"]:
        output_lines.append("\nErrors:")
        for error in filtered_data["errors"]:
            output_lines.append(f"  • {error}")
    
    # Add file modifications if relevant
    if filtered_data["files_modified"] and len(filtered_data["files_modified"]) <= 3:
        output_lines.append("\nFiles Modified:")
        for file_path in filtered_data["files_modified"]:
            output_lines.append(f"  • {file_path}")
    elif filtered_data["files_modified"]:
        output_lines.append(f"\nModified {len(filtered_data['files_modified'])} files")
    
    return "\n".join(output_lines)

def main():
    """
    Process SubagentStop hook to filter information
    """
    try:
        # Read JSON from stdin
        input_data = json.load(sys.stdin)
        
        # Extract transcript
        transcript = input_data.get("transcript", [])
        if not transcript:
            return 0
        
        # Filter relevant content
        filtered_data = extract_relevant_content(transcript)
        
        # Format output
        filtered_message = format_filtered_output(filtered_data)
        
        # Output the filtered message
        print(filtered_message)
        
        # Also output JSON for potential further processing
        output = {
            "continue": True,
            "filtered_summary": filtered_data
        }
        
        # Write JSON to stderr for parsing if needed
        json.dump(output, sys.stderr)
        
        return 0
        
    except Exception as e:
        # On error, just pass through without filtering
        print(f"SubagentStop filter error: {str(e)}", file=sys.stderr)
        return 0

if __name__ == "__main__":
    sys.exit(main())