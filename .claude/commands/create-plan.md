---
description: Create implementation plan by gathering context from memory files and generating multiple perspectives
allowed_tools:
  - Task
  - TodoWrite
  - Read
  - Write
  - Glob
  - Grep
  - mcp__probe__search_code
  - mcp__probe__query_code
  - mcp__probe__extract_code
  - mcp__serena__find_symbol
  - mcp__serena__get_symbols_overview
  - mcp__serena__find_referencing_symbols
subagents:
  - living-memory-analyzer
  - codebase-analyzer
  - plan-writer
---

{
  "command": {
    "name": "Create Implementation Plan",
    "purpose": "Gather context from living memory files and generate multiple implementation perspectives for the current development session",
    "input": "$ARGUMENTS",
    "mandatory": true,

    "workflow": {
      "steps": [
        {
          "id": 1,
          "name": "Clean Previous Plan",
          "type": "cleanup",
          "description": "Delete existing TemporaryPlan.md if it exists to avoid reading stale content",
          "action": "If TemporaryPlan.md exists, delete it before proceeding",
          "command": "!rm -f TemporaryPlan.md"
        },
        {
          "id": 2,
          "name": "Task Selection",
          "type": "task_determination",
          "logic": {
            "if_argument_provided": "Use $ARGUMENTS as the task/objective to plan",
            "if_no_argument": {
              "action": "Use living-memory-analyzer agent to retrieve current task from Tasks.json",
              "subagent_type": "living-memory-analyzer",
              "agent_prompt": {
                "task": "Retrieve task information",
                "instructions": "Retrieve the current highest-priority pending or in_progress task from Tasks.json, along with its related subtasks and dependencies",
                "output_format": {
                  "type": "structured_json",
                  "schema": {
                    "task_id": "UUID of the task",
                    "title": "Task title",
                    "description": "Task description",
                    "priority": "high/medium/low",
                    "status": "current status",
                    "subtasks": "array of subtask objects",
                    "dependencies": "array of dependency IDs"
                  }
                }
              }
            }
          }
        },
        {
          "id": 3,
          "name": "Create Workflow Todo List",
          "type": "todo_creation",
          "mandatory": true,
          "purpose": "Track this workflow - prevents skipping steps"
        },
        {
          "id": 4,
          "name": "Concurrent Context Gathering",
          "type": "concurrent_agents",
          "description": "Launch concurrent Task agents to retrieve relevant information from memory files"
        }
      ]
    },

    "agents": {
      "prd_context": {
        "name": "Agent 1 - PRD Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Extract requirements from PRD.md",
          "context": "Current objective: [current objective]",
          "focus_areas": [
            "Functional requirements",
            "Technical constraints",
            "Success criteria"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "requirements": "array of relevant requirements",
              "features": "array of related features",
              "constraints": "array of technical constraints",
              "success_criteria": "measurable success indicators"
            }
          }
        }
      },
      "architecture_context": {
        "name": "Agent 2 - Architecture Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Extract architecture information from Architecture.md",
          "context": "Current objective: [current objective]",
          "focus_areas": [
            "Architectural patterns",
            "Component relationships",
            "Design decisions",
            "Technology stack details",
            "Integration points"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "patterns": "relevant architectural patterns",
              "components": "affected components and relationships",
              "design_decisions": "relevant design choices",
              "tech_stack": "technology details",
              "integration_points": "where to integrate"
            }
          }
        }
      },
      "research_context": {
        "name": "Agent 3 - Research Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Extract research findings from ResearchFindings.json",
          "context": "Current objective: [current objective]",
          "focus_areas": [
            "Libraries that could help",
            "Patterns or solutions",
            "Version-specific information",
            "Best practices"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "libraries": "array of relevant libraries with versions",
              "patterns": "useful patterns discovered",
              "solutions": "potential solutions",
              "best_practices": "recommended approaches"
            }
          }
        }
      },
      "lessons_context": {
        "name": "Agent 4 - Lessons Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Extract lessons from UsefulInformation.json",
          "context": "Current objective: [current objective]",
          "focus_areas": [
            "Relevant errors encountered",
            "Solutions that worked",
            "Lessons learned",
            "Quirks or gotchas discovered"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "errors_to_avoid": "array of known errors",
              "proven_solutions": "what has worked before",
              "lessons": "important learnings",
              "quirks": "gotchas to watch for"
            }
          }
        }
      }

      }
    },

    "code_analysis": {
      "step_id": 5,
      "name": "Existing Code Analysis",
      "description": "Enhanced with Probe MCP for Deep Code Discovery",
      "purpose": "Discover patterns and prevent duplicate work",

      "substeps": [
        {
          "id": "4.1",
          "name": "Pattern Discovery with search_code",
          "tool": "mcp__probe__search_code",
          "usage": {
            "path": "/absolute/path/to/project",
            "query": "[relevant search terms]",
            "allowTests": false,
            "session": "new"
          },
          "query_examples": [
            {
              "scenario": "Docker integration",
              "query": "container OR docker OR lifecycle"
            },
            {
              "scenario": "Error handling",
              "query": "error AND (handle OR handler OR handling)"
            },
            {
              "scenario": "State management",
              "query": "(state OR status) AND (manage OR update OR transition)"
            }
          ]
        },
        {
          "id": "4.2",
          "name": "Structural Pattern Search with query_code",
          "tool": "mcp__probe__query_code",
          "usage": {
            "path": "/absolute/path/to/project",
            "pattern": "[ast-grep pattern]",
            "language": "rust|typescript|etc"
          },
          "pattern_examples": [
            {
              "language": "rust",
              "description": "Trait implementations",
              "pattern": "impl $TRAIT for $TYPE { $$$BODY }"
            },
            {
              "language": "typescript",
              "description": "React components",
              "pattern": "const $NAME: React.FC<$PROPS> = ($$$) => { $$$BODY }"
            },
            {
              "language": "rust",
              "description": "Error handling",
              "pattern": "match $EXPR { Err($ERR) => $HANDLER, $$$REST }"
            }
          ]
        },
        {
          "id": "4.3",
          "name": "Context Extraction with extract_code",
          "tool": "mcp__probe__extract_code",
          "usage": {
            "path": "/absolute/path/to/project",
            "files": [
              "/path/to/file.rs:100",
              "/path/to/file.ts#functionName"
            ],
            "format": "markdown"
          }
        },
        {
          "id": "4.4",
          "name": "Traditional File Reading",
          "description": "After probe analysis, read key files in their entirety",
          "targets": [
            "Module-specific README files",
            "Version files (Cargo.toml, package.json)",
            "Implementation files identified by probe",
            "Test files to understand coverage"
          ]
        }
      ],

      "workflow_integration": [
        "Use search_code to find all relevant code locations",
        "Use query_code to identify specific patterns you should follow",
        "Use extract_code to get full context of key implementations",
        "Read complete files only when necessary for full understanding"
      ],

      "session_management": "Reuse the session ID from search_code for faster subsequent searches",
      "critical_note": "This grounds the implementation in existing patterns and prevents duplicate work",

      "serena_integration": {
        "when_to_use": "For precise semantic understanding after Probe discovery",
        "tools": [
          {
            "tool": "mcp__serena__get_symbols_overview",
            "purpose": "Understand module structure before deep dive"
          },
          {
            "tool": "mcp__serena__find_symbol",
            "purpose": "Navigate to specific implementations"
          },
          {
            "tool": "mcp__serena__find_referencing_symbols",
            "purpose": "Understand usage patterns and dependencies"
          }
        ],
        "workflow": "Probe for discovery → Serena for precision"
      }

    },

    "implementation_sketches": {
      "step_id": 6,
      "name": "Generate Three Implementation Sketches",
      "critical_instruction": "You may not edit or create code or write to files at this point, only research and relay your complete plan back to the main thread",
      "warning": "It's crucial to instruct the Task agents with these instructions (otherwise they'll start implementing on their own)!",

      "agents": {
        "simple_direct": {
          "name": "Agent A - Simple & Direct",
          "concurrent_group": "implementation_sketches",
          "subagent_type": "plan-writer",
          "prompt": {
            "task": "Create implementation sketch",
            "objective": "[objective]",
            "approach": "Prioritize simplicity and directness",
            "focus_areas": [
              "Minimal viable implementation",
              "Clear, straightforward code",
              "Avoid over-engineering"
            ],
            "constraints": [
              "DO NOT write or edit any code files",
              "Only create a rough sketch/plan"
            ],
            "output_format": {
              "type": "structured_json",
              "schema": {
                "approach_name": "Simple & Direct",
                "key_decisions": "array of design choices",
                "implementation_steps": "ordered list of steps",
                "files_to_modify": "list of files and changes",
                "advantages": "benefits of this approach",
                "tradeoffs": "what we sacrifice for simplicity"
              }
            }
          }
        },
        "robust_scalable": {
          "name": "Agent B - Robust & Scalable",
          "concurrent_group": "implementation_sketches",
          "subagent_type": "plan-writer",
          "prompt": {
            "task": "Create implementation sketch",
            "objective": "[objective]",
            "approach": "Prioritize robustness and future scalability",
            "focus_areas": [
              "Error handling",
              "Extensibility",
              "Maintenance"
            ],
            "constraints": [
              "DO NOT write or edit any code files",
              "Only create a rough sketch/plan"
            ],
            "output_format": {
              "type": "structured_json",
              "schema": {
                "approach_name": "Robust & Scalable",
                "key_decisions": "array of design choices",
                "implementation_steps": "ordered list of steps",
                "files_to_modify": "list of files and changes",
                "error_handling": "error handling strategy",
                "extensibility_points": "where future features can hook in",
                "advantages": "benefits of this approach",
                "tradeoffs": "complexity vs robustness"
              }
            }
          }
        },
        "performance_efficiency": {
          "name": "Agent C - Performance & Efficiency",
          "concurrent_group": "implementation_sketches",
          "subagent_type": "plan-writer",
          "prompt": {
            "task": "Create implementation sketch",
            "objective": "[objective]",
            "approach": "Prioritize performance and efficiency",
            "focus_areas": [
              "Resource usage",
              "Algorithmic complexity",
              "Optimization opportunities"
            ],
            "constraints": [
              "DO NOT write or edit any code files",
              "Only create a rough sketch/plan"
            ],
            "output_format": {
              "type": "structured_json",
              "schema": {
                "approach_name": "Performance & Efficiency",
                "key_decisions": "array of design choices",
                "implementation_steps": "ordered list of steps",
                "files_to_modify": "list of files and changes",
                "performance_optimizations": "specific optimizations",
                "resource_considerations": "memory, CPU, I/O analysis",
                "advantages": "performance benefits",
                "tradeoffs": "what we sacrifice for performance"
              }
            }
          }
        }
      },

      "output_file": {
        "path": "TemporaryPlan.md",
        "action": "overwrite any previous content",
        "template": {
          "title": "# Implementation Planning for [Task]",
          "sections": [
            "## Context Summary\n[Brief summary of gathered context]",
            "## Sketch A: Simple & Direct\n[Implementation approach A]",
            "## Sketch B: Robust & Scalable\n[Implementation approach B]",
            "## Sketch C: Performance & Efficiency\n[Implementation approach C]"
          ],
          "note": "No recommendations should be added to preserve unbiased fresh context for revise-plan command"
        }
      }
    },

    "important_guidelines": [
      "This workflow is MANDATORY and must be followed tightly",
      "Always use concurrent agents to minimize context pollution",
      "Never skip the todo list creation - it ensures workflow compliance",
      "Read version files to ensure version-specific development",
      "The three sketches provide different viewpoints for evaluation",
      "Keep main thread context clean by using agents for file operations"
    ],

    "usage": {
      "basic": "/create-plan",
      "with_task": "/create-plan implement Docker container lifecycle management"
    }
  }
}

$ARGUMENTS
