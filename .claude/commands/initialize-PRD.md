---
description: Initialize a new project from PRD with living memory files, task structure, and version control
allowed_tools:
  - Read
  - Write
  - Bash
  - TodoWrite
  - WebSearch
  - Task
subagents:
  - websearch
---

{
  "command": {
    "name": "Initialize Project from PRD",
    "purpose": "Set up a new project based on the Product Requirements Document (PRD.md), creating all necessary living memory files and establishing the project structure",
    "input": "$ARGUMENTS",

    "workflow": {
      "steps": [
        {
          "id": 1,
          "name": "Create Todo List",
          "type": "todo_creation",
          "description": "Use TodoWrite to track initialization steps"
        },
        {
          "id": 2,
          "name": "Read and Analyze PRD",
          "type": "analysis",
          "actions": [
            "Read PRD.md to understand project requirements",
            "Extract key features, goals, and technical requirements",
            "Identify mentioned technologies and frameworks"
          ]
        },
        {
          "id": 3,
          "name": "Create Tasks.json",
          "type": "file_creation",
          "requirements": {
            "structure": "flexible and maintainable for task management",
            "sections": "priority-based with tasks and granular subtasks",
            "task_fields": {
              "id": "UUID format",
              "title": "descriptive name",
              "description": "detailed explanation",
              "priority": "high/medium/low",
              "status": "pending/in_progress/completed/blocked",
              "progress": "percentage (0-100)",
              "dependencies": "array of task IDs",
              "relatedTasks": "array of related task IDs",
              "roadblocks": "array added during development"
            },
            "capabilities": "support easy add/edit/move/remove operations"
          }
        },
        {
          "id": "3a",
          "name": "Research Technologies",
          "type": "concurrent_research",
          "description": "Use MCP servers to research technologies mentioned in PRD",
          "actions": [
            "Identify mentioned technologies and frameworks",
            "Note which libraries need research",
            "Use websearch subagent for package discovery and examples"
          ]
        },
        {
          "id": 4,
          "name": "Create Architecture.md",
          "type": "file_creation",
          "requirements": {
            "diagrams": [
              "System architecture overview",
              "Component relationships",
              "Data flow",
              "Technology stack"
            ],
            "format": "comprehensive mermaid diagrams",
            "design_principle": "easy updates during development"
          }
        },
        {
          "id": 5,
          "name": "Create Research and Information Files",
          "type": "file_creation",
          "files": [
            {
              "name": "ResearchFindings.json",
              "purpose": "Structure for collecting web search findings"
            },
            {
              "name": "UsefulInformation.json",
              "purpose": "Structure for errors, solutions, and lessons learned"
            }
          ]
        },
        {
          "id": 6,
          "name": "Initialize Version Control",
          "type": "implementation",
          "actions": [
            "Check if git is initialized (!git status)",
            "If not, initialize repository (!git init)",
            {
              "action": "Create project-specific .gitignore",
              "patterns": [
                "Language-specific patterns",
                "IDE files",
                "Build artifacts",
                "Environment files",
                "OS-specific files"
              ]
            }
          ]
        },
        {
          "id": 7,
          "name": "Update Todo List",
          "type": "todo_update",
          "action": "Mark all completed tasks"
        }
      ]
    },

    "important_notes": [
      "Keep structures simple but effective",
      "Design for maintainability and adaptability",
      "Avoid over-engineering",
      "Ensure all files work together as living memory",
      "Use proper JSON formatting for data files",
      "Use clear, descriptive mermaid diagrams"
    ]
  }
}

$ARGUMENTS
