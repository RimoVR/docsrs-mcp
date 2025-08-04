---
description: Identify and guide human testing of GUI features ready for validation
allowed_tools:
  - TodoWrite
  - Task
  - Read
  - Write
  - mcp__probe__search_code
  - mcp__probe__query_code
  - mcp__probe__extract_code
  - mcp__serena__find_symbol
  - mcp__serena__get_symbols_overview
subagents:
  - living-memory-analyzer
  - codebase-analyzer
  - living-memory-updater
---

{
  "command": {
    "name": "Human in the Loop",
    "purpose": "Identify implemented features ready for human GUI testing and manage the testing feedback loop",
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
        "file": "HumanLoop.json",
        "description": "Human testing results and feedback"
      }
    ],
    
    "workflow": {
      "steps": [
        {
          "id": 1,
          "name": "Create Todo List for Testing Workflow",
          "type": "todo_creation",
          "tasks": [
            "Read living documentation to identify implemented features",
            "Check HumanLoop.json for previously tested features",
            "Analyze codebase to verify feature readiness",
            "Create detailed testing instructions",
            "Wait for user feedback",
            "Document testing results in HumanLoop.json"
          ]
        }

        },
        {
          "id": 2,
          "name": "Concurrent Documentation Analysis",
          "type": "concurrent_agents",
          "description": "Launch multiple Task agents IN PARALLEL to gather implementation status"
        }
      ]
    },
    
    "agents": {
      "task_implementation_status": {
        "name": "Agent A - Task Implementation Status",
        "concurrent_group": "documentation_analysis",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Tasks.json and identify completed tasks",
          "focus_areas": [
            "GUI-related features",
            "user-facing functionality"
          ],
          "include": [
            "task descriptions",
            "completion dates",
            "related components"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "completed_features": "list of features ready for GUI testing",
              "task_details": "descriptions and completion info",
              "gui_components": "UI elements involved",
              "testing_priority": "which to test first"
            }
          }
        }
      },
      "architecture_implementation": {
        "name": "Agent B - Architecture Implementation",
        "concurrent_group": "documentation_analysis",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read Architecture.md and identify implemented components",
          "focus_areas": [
            "UI components",
            "user interaction flows",
            "frontend features"
          ],
          "include": [
            "component status",
            "integration points",
            "UI/UX elements"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "implemented_components": "fully implemented UI components",
              "interaction_flows": "completed user flows",
              "integration_status": "backend connections",
              "ui_elements": "specific UI features"
            }
          }
        }
      },
      "testing_history": {
        "name": "Agent C - Testing History",
        "concurrent_group": "documentation_analysis",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read HumanLoop.json to identify previously tested features",
          "focus_areas": [
            "features marked as sufficiently_tested",
            "test dates",
            "user satisfaction"
          ],
          "include": "list of features that don't need retesting",
          "special_instruction": "Create the file with empty JSON object {} if it doesn't exist",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "previously_tested": "features already tested",
              "sufficiently_tested": "features needing no retest",
              "test_dates": "when features were tested",
              "satisfaction_levels": "user feedback summary"
            }
          }
        }
      },
      "prd_feature_status": {
        "name": "Agent D - PRD Feature Status",
        "concurrent_group": "documentation_analysis",
        "subagent_type": "living-memory-analyzer",
        "prompt": {
          "task": "Read PRD.md and cross-reference with completed tasks",
          "focus_areas": [
            "which requirements have been implemented",
            "priority features"
          ],
          "include": [
            "feature descriptions",
            "expected behaviors"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "implemented_requirements": "PRD features completed",
              "feature_descriptions": "what each feature does",
              "expected_behaviors": "how features should work",
              "priority_order": "testing priority based on PRD"
            }
          }
        }
      },

      "frontend_component_analysis": {
        "name": "Agent E - Frontend Component Analysis",
        "concurrent_group": "codebase_analysis",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Use MCP tools to analyze frontend components comprehensively",
          "mcp_strategy": "Start with Probe for discovery, use Serena for understanding component structure",
          "search_for": [
            "React components",
            "UI elements",
            "event handlers"
          ],
          "focus_on": "completed components with user interactions",
          "include": [
            "specific file paths",
            "component names"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "completed_components": "list of ready UI components",
              "interaction_handlers": "user interaction points",
              "file_locations": "paths to component files",
              "component_status": "implementation completeness"
            }
          }
        }
      },
      "backend_integration_analysis": {
        "name": "Agent F - Backend Integration Analysis",
        "concurrent_group": "codebase_analysis",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Use MCP tools to verify backend integration status",
          "search_for": [
            "API endpoints",
            "Tauri commands",
            "state management"
          ],
          "focus_on": "which frontend features have complete backend support",
          "include": [
            "integration status",
            "any missing pieces"
          ],
          "output_format": {
            "type": "structured_json",
            "schema": {
              "integrated_features": "features with full backend",
              "api_endpoints": "available endpoints",
              "tauri_commands": "implemented commands",
              "missing_integrations": "incomplete connections"
            }
          }
        }
      },
      "test_coverage_analysis": {
        "name": "Agent G - Test Coverage Analysis",
        "concurrent_group": "codebase_analysis",
        "subagent_type": "codebase-analyzer",
        "uses_probe_mcp": true,
        "prompt": {
          "task": "Use MCP tools to analyze test coverage gaps",
          "search_for": [
            "component tests",
            "integration tests",
            "E2E tests"
          ],
          "focus_on": "which features lack comprehensive testing and need human validation",
          "include": "test gaps that human testing should cover",
          "output_format": {
            "type": "structured_json",
            "schema": {
              "tested_features": "features with good coverage",
              "untested_features": "features needing human testing",
              "test_gaps": "specific areas lacking tests",
              "human_test_priorities": "what to focus on"
            }
          }
        }
      }
    },
    
    "workflow_continuation": {
      "step_3": {
        "name": "Codebase Readiness Analysis",
        "type": "concurrent_agents",
        "description": "Launch concurrent agents to verify implementation"
      },

      "step_4": {
        "name": "Generate Testing Instructions",
        "type": "synthesis",
        "description": "Based on the analysis, create detailed testing instructions",
        "output_template": {
          "format": "markdown",
          "structure": [
            "# GUI Testing Instructions",
            "## Testing Environment Setup",
            "- [ ] Ensure latest build is running: `npm run tauri dev`",
            "- [ ] Clear any previous test data if needed",
            "## Features Ready for Testing",
            "### Feature 1: [Feature Name]",
            "**Location**: [Where to find in GUI]",
            "**Expected Behavior**:",
            "- [Specific interaction and expected result]",
            "- [Edge cases to test]",
            "**Test Steps**:",
            "1. [Detailed step]",
            "2. [Detailed step]",
            "**Success Criteria**: [What indicates the feature works correctly]",
            "### Feature 2: [Feature Name]",
            "[Similar structure...]",
            "## Features NOT Ready for Testing",
            "- [Feature]: [Reason why not ready]",
            "## Previously Tested (Skip These)",
            "- [Feature]: Tested on [date], marked as sufficient",
            "## Testing Checklist",
            "Please test each feature and provide feedback:",
            "- Works as expected",
            "- Partially works (describe issues)",
            "- Does not work (describe errors)",
            "- UI/UX suggestions"
          ]
        }
      },
      
      "step_5": {
        "name": "Wait for User Feedback",
        "type": "user_interaction",
        "description": "Present the testing instructions and wait for the user to complete testing",
        "expected_feedback": [
          "Which features work correctly",
          "Which features have issues",
          "Any UI/UX improvements needed",
          "Overall satisfaction with each feature"
        ]
      },

      "step_6": {
        "name": "Document Testing Results",
        "type": "agent",
        "condition": "Once feedback is received",
        "description": "Launch an agent to update HumanLoop.json"
      }
    },
    
    "documentation_agent": {
      "document_results": {
        "name": "Agent H - Document Results",
        "subagent_type": "living-memory-updater",
        "prompt": {
          "task": "Update HumanLoop.json with the testing results for each feature tested",
          "structure": {
            "tested_features": {
              "[feature_name]": {
                "test_date": "[ISO date]",
                "status": "sufficiently_tested|needs_work|failed",
                "user_satisfied": "true|false",
                "feedback": "[user feedback]",
                "issues": ["issue1", "issue2"],
                "skip_future_tests": "true|false"
              }
            },
            "testing_sessions": [
              {
                "date": "[ISO date]",
                "features_tested": ["feature1", "feature2"],
                "overall_notes": "[session notes]"
              }
            ]
          },
          "instruction": "Only mark features with skip_future_tests: true if user explicitly confirms they're satisfied and don't need retesting",
          "output_format": {
            "type": "file_update",
            "target": "HumanLoop.json"
          }
        }
      }
    },
    
    "completion": {
      "step_7": {
        "name": "Completion",
        "actions": [
          "Update todo list marking all tasks as completed",
          "Summarize which features passed testing and which need more work",
          "If any features need work, suggest creating tasks for the improvements"
        ]
      }
    },
    
    "important_note": "This command focuses on coordinating human testing, not implementing fixes. Document all feedback thoroughly for future development cycles."
  }
}