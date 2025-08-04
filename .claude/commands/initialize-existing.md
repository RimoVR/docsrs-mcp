---
description: Bootstrap an existing project by analyzing current code and creating living memory files
allowed_tools: 
  - Read
  - Write
  - Bash
  - TodoWrite
  - Glob
  - Grep
  - Task
  - WebSearch
  - mcp__probe__search_code
  - mcp__probe__query_code
  - mcp__probe__extract_code
  - mcp__serena__get_symbols_overview
  - mcp__serena__find_symbol
subagents:
  - codebase-analyzer
  - websearch
---

{
  "command": {
    "name": "Initialize Existing Project",
    "purpose": "Analyze an existing codebase and set up living memory files based on the current project state, PRD, and CLAUDE.md",
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
          "name": "Deep Project Analysis",
          "type": "analysis",
          "actions": [
            "Read PRD.md and CLAUDE.md to understand project vision",
            {
              "action": "Use MCP servers for comprehensive code review",
              "mcp_workflow": [
                {
                  "tool": "mcp__probe__search_code",
                  "purpose": "Find main entry points and core modules",
                  "searches": ["main OR entry", "init OR setup", "app OR application"]
                },
                {
                  "tool": "mcp__serena__get_symbols_overview",
                  "purpose": "Understand top-level architecture",
                  "target": "src/ and app/ directories"
                },
                {
                  "tool": "Traditional file reading",
                  "targets": [
                    "@Cargo.toml @package.json - Check dependencies",
                    "@**/*.md - Review existing documentation"
                  ]
                }
              ]
            },
            {
              "action": "Identify",
              "items": [
                "Current architecture and patterns",
                "Working components",
                "Problem areas and technical debt",
                "Missing implementations"
              ]
            }
          ]
        },
        {
          "id": 3,
          "name": "Create Tasks.json",
          "type": "file_creation",
          "structure": {
            "design": "flexible structure supporting",
            "features": [
              "Task hierarchy with UUID identifiers",
              "Priority sections (high/medium/low)",
              "Granular subtasks"
            ],
            "required_fields": [
              "id, title, description",
              "priority, status, progress",
              "dependencies[], relatedTasks[]",
              "roadblocks[] (for discovered issues)"
            ],
            "populate_with": [
              "Completed tasks (from working code)",
              "In-progress tasks (partial implementations)",
              "Pending tasks (from PRD not yet started)",
              "Discovered issues as blocked tasks"
            ]
          }
        },
        {
          "id": 4,
          "name": "Create Architecture.md",
          "type": "file_creation",
          "content": {
            "diagrams": [
              "Actual current architecture",
              "Component relationships",
              "Data flow patterns",
              "Technology stack in use"
            ],
            "sections": [
              "Implemented components",
              "Planned components (from PRD)",
              "Technical debt areas"
            ]
          }
        },
        {
          "id": 5,
          "name": "Create Information Files",
          "type": "file_creation",
          "files": [
            {
              "name": "ResearchFindings.json",
              "template": {
                "findings": [],
                "lastUpdated": "ISO-date",
                "categories": ["libraries", "patterns", "solutions"]
              }
            },
            {
              "name": "UsefulInformation.json",
              "template": {
                "errors": [],
                "solutions": [],
                "lessons": [],
                "quirks": [],
                "lastUpdated": "ISO-date"
              }
            }
          ]
        },
        {
          "id": 6,
          "name": "Version Control Setup",
          "type": "implementation",
          "actions": [
            "Check git status (!git status)",
            "Initialize if needed (!git init)",
            {
              "action": "Create/update .gitignore",
              "patterns": [
                "target/ (Rust)",
                "node_modules/",
                "dist/, build/",
                ".env, .env.*",
                "*.log",
                ".DS_Store",
                "Thumbs.db",
                ".idea/, .vscode/"
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
      "Focus on understanding what exists before planning",
      "Document both successes and struggles",
      "Keep structures simple and maintainable",
      "Ensure seamless integration with existing code",
      "Preserve working functionality",
      "Be honest about technical debt"
    ]
  }
}

$ARGUMENTS