---
description: Revise implementation plan with research, then execute with quality checks and documentation updates
allowed_tools:
  - Read
  - Write
  - Edit
  - MultiEdit
  - Bash
  - Task
  - TodoWrite
  - WebSearch
  - Glob
  - Grep
  - mcp__probe__search_code
  - mcp__probe__query_code
  - mcp__probe__extract_code
  - mcp__serena__find_symbol
  - mcp__serena__replace_symbol_body
  - mcp__serena__insert_after_symbol
  - mcp__serena__insert_before_symbol
  - mcp__context7__resolve-library-id
  - mcp__context7__get-library-docs
  - mcp__octocode__githubSearchCode
  - mcp__octocode__packageSearch
subagents:
  - websearch
  - codebase-analyzer
  - living-memory-analyzer
  - living-memory-updater
  - code-linter-formatter
---

{
  "command": {
    "name": "Revise and Execute Plan",
    "purpose": "YOU (main thread) refine the implementation sketches from /create-plan, use agents ONLY for targeted research, then YOU execute the implementation with quality checks",
    "input": "$ARGUMENTS",

    "workflow": {
      "overview": "YOU (main thread) refine the plan and implement it. Agents are used ONLY for research (gathering information) and documentation updates (updating memory files). Agents NEVER write implementation code.",
      "steps": [
        {
          "id": 1,
          "name": "Read TemporaryPlan.md",
          "type": "self_read",
          "critical": true,
          "description": "YOU (main thread) read the three sketches from TemporaryPlan.md",
          "actions": [
            "Read TemporaryPlan.md yourself",
            "Understand the three different implementation approaches"
          ]
        },
        {
          "id": 2,
          "name": "Concurrent Research for Grounding",
          "type": "concurrent_agents",
          "description": "Launch 5 concurrent Task agents ONLY for research - agents must NOT implement or modify code",
          "critical_instruction": "Agents are for RESEARCH ONLY - they gather information and return it to you"
        },
        {
          "id": 3,
          "name": "Consolidate and Merge Sketches",
          "type": "self_synthesis",
          "critical": true,
          "description": "YOU (main thread) must perform this synthesis - DO NOT delegate to agents",
          "actions": [
            "Synthesize the three sketches into one refined approach",
            "Incorporate research findings from step 2"
          ],
          "criteria": [
            "As simple as reasonably achievable",
            "NOT over-engineered (critical - avoid complexity)",
            "Effective and efficient",
            "Incorporates best aspects of each sketch"
          ],
          "output": "Overwrite TemporaryPlan.md with the refined plan"
        }
      ]
    },

    "agents": {
      "library_research": {
        "name": "Agent 1 - Library Research",
        "concurrent_group": "research",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Research libraries and dependencies - DO NOT implement or write code",
          "constraint": "Return research findings only - no code modifications",
          "search_terms": "[specific libraries/dependencies] with [language] [version]",
          "focus_areas": [
            "Official documentation",
            "Recent updates",
            "Common patterns"
          ],
          "avoid": [
            "Outdated practices",
            "Deprecated methods"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "libraries": "array of library details with versions",
              "documentation_links": "official docs URLs",
              "usage_patterns": "common implementation patterns",
              "recent_changes": "notable updates or breaking changes"
            }
          }
        }
      },
      "best_practices": {
        "name": "Agent 2 - Best Practices",
        "concurrent_group": "research",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Research implementation best practices - DO NOT implement or write code",
          "constraint": "Return research findings only - no code modifications",
          "search_terms": "[implementation pattern] best practices [year]",
          "focus_areas": [
            "Industry standards",
            "Recommended approaches",
            "Performance tips"
          ],
          "include": "Common pitfalls to avoid",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "best_practices": "array of recommended practices",
              "anti_patterns": "things to avoid",
              "performance_tips": "optimization suggestions",
              "industry_standards": "widely accepted approaches"
            }
          }
        }
      },
      "known_issues": {
        "name": "Agent 3 - Known Issues",
        "concurrent_group": "research",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Research known issues and workarounds - DO NOT implement or write code",
          "constraint": "Return research findings only - no code modifications",
          "search_terms": "[technology/library] issues quirks tips",
          "focus_areas": [
            "User-reported problems",
            "Workarounds",
            "Version-specific bugs"
          ],
          "include": [
            "Stack Overflow solutions",
            "GitHub issues"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "common_issues": "array of known problems",
              "workarounds": "solutions for each issue",
              "version_bugs": "version-specific problems",
              "community_solutions": "helpful tips from users"
            }
          }
        }
      },

      "codebase_analysis": {
        "name": "Agent 4 - Codebase Pattern Analysis",
        "concurrent_group": "research",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Analyze existing codebase for implementation patterns - DO NOT modify any code",
          "constraint": "Return analysis only - no code modifications",
          "context": "Current objective: [current objective]",
          "workflow": [
            {
              "step": "Pattern Discovery",
              "actions": [
                {
                  "tool": "mcp__probe__search_code",
                  "purpose": "Find all relevant implementations",
                  "search_for": "[key terms related to task]",
                  "focus_on": [
                    "existing patterns",
                    "integration points",
                    "similar features"
                  ]
                }
              ]
            },
            {
              "step": "Structural Analysis",
              "actions": [
                {
                  "tool": "mcp__probe__query_code",
                  "purpose": "Find specific code structures",
                  "look_for": [
                    "trait implementations",
                    "component patterns",
                    "error handling"
                  ],
                  "identify": [
                    "naming conventions",
                    "architectural patterns"
                  ]
                }
              ]
            },
            {
              "step": "Integration Points",
              "actions": [
                {
                  "tool": "mcp__probe__extract_code",
                  "purpose": "Examine discovered implementations",
                  "extract": [
                    "interfaces",
                    "APIs",
                    "data flows"
                  ],
                  "map": "where new code should integrate with existing systems"
                }
              ]
            }
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "existing_patterns": "patterns to follow in implementation",
              "integration_points": "where new code connects to existing",
              "reuse_opportunities": "code that can be reused",
              "architectural_constraints": "constraints to respect"
            }
          }
        }
      },

      "prd_requirements": {
        "name": "Agent 5 - PRD Requirements Analysis",
        "concurrent_group": "research",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Extract relevant requirements from PRD.md for the current objective",
          "constraint": "Return extracted information only - no modifications",
          "context": "Current objective: [current objective]",
          "document": "PRD.md",
          "focus_areas": [
            "Functional requirements related to the current task",
            "Technical constraints that must be respected",
            "Success criteria and acceptance criteria",
            "User stories or scenarios",
            "Performance requirements",
            "Security considerations",
            "Integration requirements"
          ],
          "special_instructions": "Extract only information directly relevant to the current implementation task",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "functional_requirements": "array of requirements directly related to task",
              "technical_constraints": "constraints that affect implementation",
              "success_criteria": "measurable criteria for task completion",
              "user_stories": "relevant user scenarios to consider",
              "performance_requirements": "any performance targets or limits",
              "security_considerations": "security requirements to implement",
              "integration_points": "how this feature integrates with others"
            }
          }
        }
      }
    },

    "critical_steps": {
      "ultra_think": {
        "step_id": 4,
        "name": "Ultra-Think and Create Granular Todo List",
        "critical": true,
        "performed_by": "YOU (main thread) - NOT delegated to agents",
        "why_not_agents": "This requires synthesis and judgment that must happen in the main context",
        "description": "YOU perform deep reflection to refine the plan using research findings",
        "actions": [
          "Synthesize all research results including codebase analysis",
          "Ground implementation in existing patterns discovered by Agent 4",
          "Identify exact integration points and files to modify",
          "Adjust implementation approach based on findings",
          "Create detailed todo list with granular steps",
          "Ensure plan remains simple, not over-engineered",
          "Include specific implementation steps with file paths",
          "Map new code to existing architectural boundaries",
          "Consider using Serena for precise semantic edits when appropriate"
        ]
      },

      "implementation": {
        "step_id": 5,
        "name": "Step-by-Step Implementation",
        "performed_by": "YOU (main thread) - execute the todo list yourself",
        "execution_guidelines": [
          "Execute the todo list methodically",
          "Follow TDD practices when applicable",
          "Write tests before/alongside implementation",
          "Implement seamlessly into existing codebase",
          "Avoid breaking existing functionality",
          "NO partial implementations - complete each step fully",
          "Update todo status in real-time",
          "Use Serena's semantic tools for complex refactoring",
          "Leverage Probe for finding all instances before bulk changes"
        ],
        "mcp_usage_patterns": {
          "probe_first": "Always use Probe to find all occurrences before making changes",
          "serena_precision": "Use Serena's replace_symbol_body for entire function/class replacements",
          "context7_docs": "Fetch API docs when using external libraries",
          "efficiency": "Batch related searches with session caching"
        }
      },

      "quality_assurance": {
        "step_id": 6,
        "name": "Quality Assurance",
        "mandatory": true,
        "note": "Keep these at end of todo list",
        "performed_by": "Either YOU directly or via code-linter-formatter agent",
        "subagent_type": "code-linter-formatter",
        "description": "Run quality checks - can use agent or run commands directly",
        "checks": {
          "rust_components": [
            "!cargo fmt",
            "!cargo clippy -- -D warnings",
            "!cargo test"
          ],
          "typescript_react_components": [
            "!npm run lint",
            "!npm run typecheck",
            "!npm test"
          ]
        },
        "requirement": "Address ALL errors and warnings before proceeding"
      }
    },

    "documentation_updates": {
      "step_id": 7,
      "name": "Concurrent Documentation Updates",
      "type": "concurrent_agents",
      "description": "Launch Task agents to update living memory files",

      "agents": {
        "tasks_update": {
          "name": "Agent 1 - Tasks.json",
          "concurrent_group": "documentation",
          "subagent_type": "living-memory-updater",
          "prompt": {
            "task": "Update Tasks.json",
            "actions": [
              "Mark completed tasks as 'completed' with 100% progress",
              "Update in_progress tasks with current progress percentage",
              "Add any new discovered subtasks",
              "Update roadblocks array with encountered issues"
            ],
            "output_format": {
              "type": "file_update",
              "target": "Tasks.json"
            }
          }
        },
        "architecture_update": {
          "name": "Agent 2 - Architecture.md",
          "concurrent_group": "documentation",
          "subagent_type": "living-memory-updater",
          "prompt": {
            "task": "Update Architecture.md",
            "actions": [
              "New component relationships in mermaid diagrams",
              "Updated data flow if changed",
              "New naming conventions used",
              "Technology decisions made"
            ],
            "output_format": {
              "type": "file_update",
              "target": "Architecture.md"
            }
          }
        },
        "research_update": {
          "name": "Agent 3 - ResearchFindings.json",
          "concurrent_group": "documentation",
          "subagent_type": "living-memory-updater",
          "prompt": {
            "task": "Update ResearchFindings.json",
            "actions": [
              "Relevant library information discovered",
              "Version-specific details",
              "Best practices learned",
              "Useful documentation links"
            ],
            "output_format": {
              "type": "file_update",
              "target": "ResearchFindings.json"
            }
          }
        },
        "useful_info_update": {
          "name": "Agent 4 - UsefulInformation.json",
          "concurrent_group": "documentation",
          "subagent_type": "living-memory-updater",
          "prompt": {
            "task": "Update UsefulInformation.json",
            "actions": [
              "Unexpected errors encountered and their solutions",
              "Workarounds for library quirks",
              "Performance optimization discoveries",
              "Lessons learned to avoid future issues"
            ],
            "output_format": {
              "type": "file_update",
              "target": "UsefulInformation.json"
            }
          }
        },
        "readme_update": {
          "name": "Agent 5 - Module README",
          "concurrent_group": "documentation",
          "subagent_type": "living-memory-updater",
          "prompt": {
            "task": "Update the module's local README.md",
            "actions": [
              "New functionality added",
              "API changes or additions",
              "Usage examples",
              "Important implementation details"
            ],
            "output_format": {
              "type": "file_update",
              "target": "module README.md"
            }
          }
        }
      }
    },

    "version_control": {
      "step_id": 8,
      "name": "Version Control",
      "commands": [
        {
          "description": "Stage all changes",
          "command": "!git add -A"
        },
        {
          "description": "Create semantic commit",
          "command": "!git commit -m \"feat/fix/refactor: [descriptive message]\n\n- [bullet point of changes]\n- [bullet point of changes]\""
        },
        {
          "description": "Push to remote",
          "command": "!git push"
        }
      ]
    },

    "cleanup": {
      "step_id": 9,
      "name": "Cleanup",
      "description": "Remove temporary files after successful completion",
      "actions": [
        {
          "description": "Remove TemporaryPlan.md",
          "command": "!rm TemporaryPlan.md",
          "condition": "Only if all previous steps completed successfully"
        }
      ]
    },

    "critical_reminders": [
      "YOU (main thread) perform ALL synthesis, planning, and implementation",
      "Agents are used ONLY for research (step 2) and documentation (step 7)",
      "NEVER delegate plan refinement or implementation to agents",
      "NEVER skip steps or mark incomplete work as done",
      "ALWAYS fix linting/testing errors before continuing",
      "AVOID over-engineering - simplicity is key",
      "MAINTAIN existing functionality",
      "UPDATE documentation concurrently",
      "COMPLETE full implementations only"
    ],

    "usage": {
      "command": "/revise-plan",
      "description": "The command will read the current TemporaryPlan.md and execute the full workflow"
    }
  }
}

$ARGUMENTS
