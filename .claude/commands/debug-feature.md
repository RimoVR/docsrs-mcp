---
description: Debug and fix a bug, error, or missing implementation with comprehensive analysis
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
  - mcp__serena__get_symbols_overview
subagents:
  - living-memory-analyzer
  - websearch
  - codebase-analyzer
  - living-memory-updater
---

{
  "command": {
    "name": "Debug Feature",
    "purpose": "PLANNING a fix, not implementing it. Use /revise-plan after this to implement",
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
      },
      {
        "file": "TemporaryPlan.md",
        "description": "Working document for planning"
      }
    ],
    
    "workflow": {
      "steps": [
        {
          "id": 1,
          "name": "Create Todo List for Debug Workflow",
          "type": "todo_creation",
          "tasks": [
            "Read living documentation for context",
            "Research known issues and solutions",
            "Analyze codebase for root cause",
            "Create three implementation approaches",
            "Document plan in TemporaryPlan.md"
          ]
        }

        },
        {
          "id": 2,
          "name": "Concurrent Context Gathering",
          "type": "concurrent_agents",
          "description": "Launch multiple Task agents IN PARALLEL to understand the issue"
        }
      ]
    },
    
    "agents": {
      "requirements_context": {
        "name": "Agent A - Requirements Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read PRD.md and extract information relevant to the bug/error",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "expected behavior",
            "feature requirements",
            "user goals"
          ],
          "include": "how this issue impacts documented requirements",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "expected_behavior": "what should happen",
              "actual_behavior": "what is happening",
              "requirements_impact": "how bug affects requirements",
              "user_impact": "how users are affected"
            }
          }
        }
      },
      "architecture_context": {
        "name": "Agent B - Architecture Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Architecture.md and analyze components",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "module responsibilities",
            "data flow",
            "integration points"
          ],
          "include": "architectural constraints that might affect the fix",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "affected_modules": "modules involved in the issue",
              "data_flow_issues": "data flow problems identified",
              "integration_problems": "integration point failures",
              "architectural_constraints": "limitations to consider"
            }
          }
        }
      },
      "task_context": {
        "name": "Agent C - Task Context",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Tasks.json and find related tasks",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "implementation history",
            "dependencies",
            "related features"
          ],
          "include": "when the feature was implemented and by which tasks",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "implementation_tasks": "tasks that implemented this feature",
              "implementation_date": "when it was implemented",
              "dependencies": "dependent tasks",
              "related_features": "connected functionality"
            }
          }
        }
      },
      "known_issues": {
        "name": "Agent D - Known Issues",
        "concurrent_group": "context_gathering",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read UsefulInformation.json for prior encounters",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "previous errors",
            "solutions tried",
            "lessons learned"
          ],
          "include": "patterns or approaches that worked or failed before",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "previous_encounters": "similar issues faced before",
              "attempted_solutions": "what was tried",
              "successful_approaches": "what worked",
              "failed_approaches": "what didn't work",
              "lessons_learned": "key insights"
            }
          }
        }
      },

      "error_research": {
        "name": "Agent E - Error Research",
        "concurrent_group": "investigation",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Search the web for solutions",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "stack overflow",
            "GitHub issues",
            "official documentation"
          ],
          "include": [
            "common causes",
            "recommended fixes",
            "version-specific solutions"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "common_causes": "typical reasons for this error",
              "recommended_fixes": "proven solutions",
              "version_specific": "fixes for our version",
              "sources": "links to helpful resources"
            }
          }
        }
      },
      "best_practices_research": {
        "name": "Agent F - Best Practices Research",
        "concurrent_group": "investigation",
        "subagent_type": "websearch",
        "prompt": {
          "task": "Search for best practices fixing issues",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "debugging techniques",
            "common pitfalls",
            "testing approaches"
          ],
          "include": "how to prevent similar issues in the future",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "debugging_techniques": "effective debugging methods",
              "common_pitfalls": "mistakes to avoid",
              "testing_strategies": "how to test the fix",
              "prevention_tips": "avoiding future occurrences"
            }
          }
        }
      },
      "codebase_analysis": {
        "name": "Agent G - Codebase Analysis",
        "concurrent_group": "investigation",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Use MCP tools strategically to locate and analyze the bug",
          "issue": "$ARGUMENTS",
          "focus_areas": [
            "error locations",
            "call stacks",
            "data flow",
            "edge cases"
          ],
          "mcp_workflow": [
            {
              "step": "Error Discovery",
              "tool": "mcp__probe__search_code",
              "purpose": "Find all error occurrences and related code"
            },
            {
              "step": "Symbol Analysis",
              "tool": "mcp__serena__find_referencing_symbols",
              "purpose": "Trace how the error propagates through the system"
            },
            {
              "step": "External Research",
              "note": "Use websearch agent for finding similar issues and solutions"
            }
          ],
          "include": "specific file paths and line numbers where the issue originates",
          "constraint": "Research and relay findings only - do not implement fixes",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "error_locations": "files and lines with errors",
              "call_stack": "execution path to error",
              "data_flow_issues": "data handling problems",
              "edge_cases": "unhandled scenarios",
              "root_cause": "underlying issue identified"
            }
          }
        }
      }
    },
    
    "workflow_continuation": {
      "step_3": {
        "name": "External Research and Codebase Analysis",
        "type": "concurrent_agents",
        "description": "Launch concurrent agents for deep investigation"
      },

      "step_4": {
        "name": "Generate Implementation Approaches",
        "type": "concurrent_agents",
        "critical_instruction": "You may not edit or create code at this point, only research and relay your complete plan",
        "description": "Launch three concurrent Task agents for diverse fix approaches"
      }
    },
    
    "fix_approaches": {
      "simple_direct": {
        "name": "Agent H - Simple & Direct Fix",
        "concurrent_group": "fix_approaches",
        "prompt": {
          "task": "Create a rough implementation sketch to fix",
          "issue": "$ARGUMENTS",
          "approach": "Prioritize simplicity and directness",
          "focus_areas": [
            "minimal changes",
            "clear logic",
            "avoiding over-engineering"
          ],
          "include": "specific code changes needed and their locations",
          "constraint": "DO NOT implement - only plan and document the approach",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "approach_name": "Simple & Direct Fix",
              "changes_required": "list of specific changes",
              "files_affected": "files and line numbers",
              "implementation_steps": "ordered steps",
              "advantages": "benefits of this approach",
              "limitations": "what this doesn't address"
            }
          }
        }
      },
      "robust_comprehensive": {
        "name": "Agent I - Robust & Comprehensive Fix",
        "concurrent_group": "fix_approaches",
        "prompt": {
          "task": "Create a rough implementation sketch to fix",
          "issue": "$ARGUMENTS",
          "approach": "Prioritize robustness and preventing recurrence",
          "focus_areas": [
            "error handling",
            "edge cases",
            "defensive programming"
          ],
          "include": "additional safeguards and validation logic",
          "constraint": "DO NOT implement - only plan and document the approach",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "approach_name": "Robust & Comprehensive Fix",
              "changes_required": "list of comprehensive changes",
              "error_handling": "error handling strategy",
              "edge_cases_covered": "edge cases addressed",
              "validation_logic": "validation to add",
              "implementation_steps": "ordered steps",
              "advantages": "robustness benefits",
              "tradeoffs": "complexity added"
            }
          }
        }
      },
      "root_cause": {
        "name": "Agent J - Root Cause Fix",
        "concurrent_group": "fix_approaches",
        "prompt": {
          "task": "Create a rough implementation sketch to fix by addressing root cause",
          "issue": "$ARGUMENTS",
          "approach": "Address the root cause",
          "focus_areas": [
            "architectural improvements",
            "refactoring if needed",
            "long-term solution"
          ],
          "include": "any structural changes that prevent the issue class entirely",
          "constraint": "DO NOT implement - only plan and document the approach",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "approach_name": "Root Cause Fix",
              "root_cause_identified": "underlying issue",
              "structural_changes": "architectural changes needed",
              "refactoring_required": "code to refactor",
              "implementation_steps": "ordered steps",
              "long_term_benefits": "future issues prevented",
              "migration_strategy": "how to transition"
            }
          }
        }
      }

    },
    
    "documentation": {
      "step_5": {
        "name": "Document Plan",
        "action": "Write all three approaches to TemporaryPlan.md, overwriting any previous content",
        "template": {
          "format": "markdown",
          "sections": [
            "# Debug Planning for: [Issue Description]",
            "## Issue Analysis",
            "### Symptoms\n[What is happening]",
            "### Root Cause\n[Why it's happening]",
            "### Impact\n[What it affects]",
            "## Context Summary",
            "### Requirements Impact\n[From PRD analysis]",
            "### Architectural Considerations\n[From Architecture analysis]",
            "### Previous Encounters\n[From UsefulInformation]",
            "### External Research Findings\n[Key solutions found]",
            "## Fix Approach A: Simple & Direct",
            "### Changes Required\n[Specific changes]",
            "### Implementation Steps\n1. [Step 1]\n2. [Step 2]",
            "### Testing Strategy\n[How to verify the fix]",
            "## Fix Approach B: Robust & Comprehensive",
            "### Changes Required\n[Specific changes]",
            "### Implementation Steps\n1. [Step 1]\n2. [Step 2]",
            "### Testing Strategy\n[How to verify the fix]",
            "## Fix Approach C: Root Cause Fix",
            "### Changes Required\n[Specific changes]",
            "### Implementation Steps\n1. [Step 1]\n2. [Step 2]",
            "### Testing Strategy\n[How to verify the fix]",
            "## Recommendation\n[Which approach is recommended and why]"
          ]
        }
      },
      
      "step_6": {
        "name": "Update Living Memory",
        "type": "agent",
        "description": "Launch an agent to update UsefulInformation.json"
      }
    },
    
    "memory_update_agent": {
      "document_learning": {
        "name": "Agent K - Document Learning",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update UsefulInformation.json with insights",
          "issue": "$ARGUMENTS",
          "include": [
            "error symptoms",
            "root cause",
            "research findings"
          ],
          "instruction": "Structure according to existing format",
          "purpose": "This helps prevent similar issues in the future",
          "output_format": {
            "type": "file_update",
            "target": "UsefulInformation.json"
          }
        }
      }
    },
    
    "completion": {
      "step_7": {
        "name": "Completion",
        "actions": [
          "Update todo list marking all tasks as completed",
          "Confirm plan is ready in TemporaryPlan.md",
          "Instruct user to run /revise-plan to review and implement the fix"
        ]
      }
    },
    
    "important_note": "This command is for DEBUGGING and PLANNING only. Do not implement any fixes. The implementation happens in /revise-plan after user approval."
  }
}