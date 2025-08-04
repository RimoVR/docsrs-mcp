---
description: Add a new feature to the project with comprehensive planning and documentation
allowed_tools:
  - TodoWrite
  - Task
  - Read
  - Write
  - WebSearch
  - mcp__probe__search_code
  - mcp__probe__query_code
  - mcp__probe__extract_code
  - mcp__serena__find_symbol
  - mcp__serena__find_referencing_symbols
subagents:
  - living-memory-analyzer
  - websearch
  - codebase-analyzer
  - living-memory-updater
---

{
  "command": {
    "name": "Add Feature to Task",
    "purpose": "PLANNING and DOCUMENTING a new feature, not implementing it",
    "important": "Follow each step carefully",
    "input": "$ARGUMENTS",
    
    "living_memory_files": [
      {
        "file": "PRD.md",
        "description": "Product requirements document"
      },
      {
        "file": "Tasks.json",
        "description": "Task management with UUID-based structure"
      },
      {
        "file": "Architecture.md",
        "description": "System design with Mermaid diagrams"
      },
      {
        "file": "ResearchFindings.json",
        "description": "External knowledge and library research"
      },
      {
        "file": "UsefulInformation.json",
        "description": "Error solutions and lessons learned"
      }
    ],
    
    "workflow": {
      "steps": [
        {
          "id": 1,
          "name": "Create Todo List for Workflow Compliance",
          "type": "todo_creation",
          "tasks": [
            "Read living documentation files concurrently",
            "Research feature implementation best practices",
            "Analyze current codebase state",
            "Design feature integration plan",
            "Present plan to user for confirmation",
            "Update living memory with approved plan"
          ]
        }

        },
        {
          "id": 2,
          "name": "Concurrent Context Gathering",
          "type": "concurrent_agents",
          "description": "Launch multiple Task agents IN PARALLEL to read living documentation"
        }
      ]
    },
    
    "agents": {
      "requirements_context": {
        "name": "Agent A - Requirements Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read PRD.md and extract information",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "existing requirements",
            "user goals",
            "system constraints"
          ],
          "include": "any related features or dependencies already documented",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "relevant_requirements": "array of requirements",
              "user_goals": "how this feature serves users",
              "constraints": "system limitations to consider",
              "related_features": "features that interact with this"
            }
          }
        }
      },
      "architecture_context": {
        "name": "Agent B - Architecture Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Architecture.md and analyze feature fit",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "module boundaries",
            "integration points",
            "design patterns"
          ],
          "include": "Mermaid diagrams showing where this feature would fit",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "module_placement": "where feature belongs",
              "integration_points": "how it connects to existing",
              "design_patterns": "patterns to follow",
              "diagram_updates": "mermaid diagram suggestions"
            }
          }
        }
      },
      "task_management_context": {
        "name": "Agent C - Task Management Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Tasks.json and identify related tasks",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "existing tasks that might conflict",
            "dependencies",
            "task structure"
          ],
          "include": "suggestions for task breakdown and dependencies",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "related_tasks": "existing tasks that relate",
              "potential_conflicts": "tasks that might conflict",
              "task_breakdown": "suggested subtasks",
              "dependencies": "task dependency structure"
            }
          }
        }
      },
      "research_solutions_context": {
        "name": "Agent D - Research and Solutions Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read ResearchFindings.json and UsefulInformation.json",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "prior research",
            "known issues",
            "lessons learned"
          ],
          "include": "relevant libraries, patterns, or solutions already documented",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "prior_research": "relevant findings",
              "known_issues": "problems to avoid",
              "lessons_learned": "insights from past work",
              "useful_libraries": "libraries that could help"
            }
          }
        }
      },

      "best_practices_research": {
        "name": "Agent E - Best Practices Research",
        "concurrent_group": "external_research",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Search the web for best practices",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "proven patterns",
            "common pitfalls",
            "performance considerations"
          ],
          "include": "version-specific information for our tech stack",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "best_practices": "array of recommended approaches",
              "common_pitfalls": "mistakes to avoid",
              "performance_tips": "optimization strategies",
              "version_specific": "tech stack considerations"
            }
          }
        }
      },
      "known_issues_research": {
        "name": "Agent F - Known Issues Research",
        "concurrent_group": "external_research",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Search for known issues and challenges",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "security concerns",
            "compatibility issues",
            "edge cases"
          ],
          "include": "solutions and workarounds from reputable sources",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "security_concerns": "potential security issues",
              "compatibility_issues": "platform/version conflicts",
              "edge_cases": "unusual scenarios to handle",
              "solutions": "workarounds and fixes"
            }
          }
        }
      },
      "codebase_analysis": {
        "name": "Agent G - Codebase Analysis",
        "concurrent_group": "external_research",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Analyze our codebase using MCP tools for comprehensive understanding",
          "feature": "$ARGUMENTS",
          "focus_areas": [
            "existing code patterns",
            "potential integration points",
            "similar features"
          ],
          "mcp_strategy": {
            "probe_for_discovery": "Use search_code to find relevant implementations",
            "serena_for_precision": "Use find_symbol for detailed analysis of key components",
            "note": "External examples will be found by websearch agent"
          },
          "include": "specific file paths and line numbers for reference",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "existing_patterns": "code patterns to follow",
              "integration_points": "where to connect new feature",
              "similar_features": "existing similar implementations",
              "file_references": "specific files and line numbers"
            }
          }
        }
      }
    },
    
    "workflow_continuation": {
      "step_3": {
        "name": "External Research and Codebase Analysis",
        "type": "concurrent_agents",
        "description": "Launch concurrent agents for research and analysis"
      },

      "step_4": {
        "name": "Feature Integration Design",
        "type": "synthesis",
        "description": "Based on all gathered information, create a comprehensive plan",
        "output_template": {
          "format": "markdown",
          "structure": [
            "# Feature Integration Plan: [Feature Name]",
            "## Executive Summary\n[Brief overview of the feature and its value]",
            "## Context Analysis",
            "### Requirements Alignment\n[How this aligns with PRD]",
            "### Architectural Integration\n[How it fits into current architecture]",
            "### Dependencies and Conflicts\n[Related tasks and potential conflicts]",
            "## Implementation Design",
            "### Approach\n[High-level implementation strategy]",
            "### Integration Points\n[Specific files and modules affected]",
            "### Technical Considerations\n[Performance, security, scalability]",
            "### Testing Strategy\n[Unit, integration, E2E testing approach]",
            "## Risk Assessment\n[Potential issues and mitigation strategies]",
            "## Task Breakdown\n[Suggested tasks for implementation]"
          ]
        }
      },
      
      "step_5": {
        "name": "User Confirmation",
        "type": "user_interaction",
        "actions": [
          "Present the plan to the user and wait for feedback",
          "If feedback provided, adapt the plan accordingly",
          "If approved, proceed to documentation",
          "Mark the todo as completed"
        ]
      },

      "step_6": {
        "name": "Update Living Memory",
        "type": "concurrent_agents",
        "condition": "Once approved",
        "description": "Launch concurrent agents to update documentation"
      }
    },
    
    "documentation_agents": {
      "update_prd": {
        "name": "Agent H - Update PRD",
        "concurrent_group": "documentation_update",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update PRD.md to include the new feature",
          "feature": "$ARGUMENTS",
          "actions": [
            "Add to appropriate sections maintaining existing structure",
            "Ensure consistency with existing requirements"
          ],
          "constraint": "DO NOT try to implement the plan, just document it",
          "output_format": {
            "type": "file_update",
            "target": "PRD.md"
          }
        }
      },
      "update_architecture": {
        "name": "Agent I - Update Architecture",
        "concurrent_group": "documentation_update",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update Architecture.md with the feature design",
          "feature": "$ARGUMENTS",
          "actions": [
            "Include Mermaid diagrams showing new components/flows",
            "Maintain consistency with existing architecture documentation"
          ],
          "constraint": "DO NOT try to implement the plan, just document it",
          "output_format": {
            "type": "file_update",
            "target": "Architecture.md"
          }
        }
      },
      "update_tasks": {
        "name": "Agent J - Update Tasks",
        "concurrent_group": "documentation_update",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update Tasks.json with new tasks for implementing",
          "feature": "$ARGUMENTS",
          "actions": [
            "Follow the existing task structure with UUIDs and dependencies",
            "Set all new tasks to 'pending' status"
          ],
          "constraint": "DO NOT try to implement the tasks, just document them",
          "output_format": {
            "type": "file_update",
            "target": "Tasks.json"
          }
        }
      },
      "update_research": {
        "name": "Agent K - Update Research",
        "concurrent_group": "documentation_update",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update ResearchFindings.json with research findings",
          "feature": "$ARGUMENTS",
          "actions": [
            "Structure findings according to existing format",
            "Include sources and version information"
          ],
          "constraint": "DO NOT try to implement anything, just document the findings",
          "output_format": {
            "type": "file_update",
            "target": "ResearchFindings.json"
          }
        }
      }
    },
    
    "completion": {
      "step_7": {
        "name": "Completion",
        "actions": [
          "Update todo list marking all tasks as completed",
          "Confirm successful documentation updates"
        ]
      }
    },
    
    "important_note": "This command is for PLANNING ONLY. Do not implement any code. Focus on thorough research, design, and documentation."
  }
}