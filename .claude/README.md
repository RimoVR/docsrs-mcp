# Claude Code Custom Commands and Hooks

This directory contains custom commands and hooks specifically designed for the CCCC (Claude Code Concurrent Context-Engineering) project.

## Custom Commands

### `/initialize-PRD`
Initialize a new project from the Product Requirements Document.
- Creates Tasks.json with flexible task management structure
- Generates Architecture.md with mermaid diagrams
- Sets up ResearchFindings.json and UsefulInformation.json
- Initializes git repository if needed
- Creates project-specific .gitignore

**Usage:**
```
/initialize-PRD
```

### `/initialize-existing`
Bootstrap an existing project by analyzing current code.
- Performs deep code review to understand current state
- Creates Tasks.json based on completed/pending work
- Documents existing architecture
- Identifies problem areas and technical debt

**Usage:**
```
/initialize-existing
```

### `/create-plan`
**MANDATORY WORKFLOW** - Creates implementation plan using multiple perspectives.
- Retrieves current task from Tasks.json
- Gathers context from all living memory files using concurrent agents
- Analyzes existing code to prevent duplication
- Generates three implementation sketches (Simple, Robust, Performance-focused)
- Outputs to TemporaryPlan.md

**Usage:**
```
/create-plan
/create-plan implement Docker container management
```

### `/revise-plan`
**MANDATORY WORKFLOW** - Refines plan and executes implementation.
- Merges three sketches into one refined approach
- Conducts concurrent research for grounding
- Creates granular todo list
- Implements with TDD practices
- Runs quality checks (formatting, linting, tests)
- Updates all documentation files concurrently
- Commits changes with semantic messages

**Usage:**
```
/revise-plan
```

## Hooks

### Post-Search Hook (`post_search_research.py`)
- **Trigger**: After WebSearch tool execution
- **Purpose**: Automatically adds relevant findings to ResearchFindings.json
- **Criteria**: Version-specific info, breaking changes, security updates, best practices

### Post-Command Hook (`post_command_bugfix.py`)
- **Trigger**: After Bash commands that appear to be bug fix attempts
- **Purpose**: Suggests web searches when commands fail
- **Action**: Provides specific search queries for errors encountered

### Post-Edit Linting Hook (`post_edit_lint.py`)
- **Trigger**: After Edit, MultiEdit, or Write tools
- **Purpose**: Immediate linting feedback
- **Linters**: 
  - Rust: cargo fmt, cargo clippy
  - TypeScript/JavaScript: eslint, tsc
  - Python: ruff/flake8
  - JSON: syntax validation

### Post-Edit Documentation Hook (`post_edit_docs.py`)
- **Trigger**: After Edit, MultiEdit, or Write tools
- **Purpose**: Suggests documentation updates for significant changes
- **Checks**: Public APIs, new components, error handling, configuration changes

### Stop Documentation Hook (`stop_documentation.py`)
- **Trigger**: When Claude finishes responding
- **Purpose**: Mandatory check for documentation updates
- **Files Checked**: Tasks.json, Architecture.md, ResearchFindings.json, UsefulInformation.json, module READMEs

### Subagent Stop Filter Hook (`subagent_stop_filter.py`)
- **Trigger**: When subagents finish responding
- **Purpose**: Filters subagent output to convey only relevant information
- **Features**: 
  - Extracts key findings, errors, and file modifications
  - Prevents context overflow in main session
  - Generates concise summaries
  - Limits output to essential information only

## Living Memory Files

### `Tasks.json`
- Task management with UUID identifiers
- Hierarchical structure with priorities
- Status tracking (pending/in_progress/completed/blocked)
- Dependencies and relationships
- Progress percentages and roadblocks

### `Architecture.md`
- Mermaid diagrams showing system design
- Component relationships
- Data flow patterns
- Technology stack documentation

### `ResearchFindings.json`
- Web search discoveries
- Version-specific information
- Best practices and patterns
- Library documentation

### `UsefulInformation.json`
- Error solutions
- Workarounds and quirks
- Performance optimizations
- Lessons learned

## Workflow Best Practices

1. **Always use `/create-plan` before `/revise-plan`**
   - These commands work together as a mandatory workflow
   - Ensures proper context gathering and planning

2. **Let hooks run automatically**
   - Don't disable hooks - they maintain code quality
   - Address hook feedback immediately

3. **Keep documentation updated**
   - The Stop hook will remind you
   - Update files concurrently using Task agents

4. **Use concurrent agents**
   - Prevents context pollution in main thread
   - Enables parallel processing of tasks

5. **Follow the workflow strictly**
   - Create todo lists to track steps
   - Don't skip quality checks
   - Complete full implementations only

## Configuration

All hooks are configured in `.claude/settings.json`. The configuration includes:
- Hook triggers and matchers
- Enabled tools
- Project metadata

## Troubleshooting

### Hooks not running
- Ensure scripts have execute permissions: `chmod +x .claude/hooks/*.py`
- Check paths in settings.json are absolute
- Verify Python 3 is available
- Hooks receive JSON via stdin, not environment variables

### Commands not found
- Commands must be in `.claude/commands/` directory
- File must have `.md` extension
- Use exact command name (without .md extension)

### Documentation not updating
- Stop hook runs after each session
- Manually run documentation updates if needed
- Check file permissions for JSON files

## Important Notes

- These tools are designed to work together as a system
- Avoid over-engineering - keep implementations simple
- Trust the workflow - it's designed to catch issues early
- Documentation is living memory - keep it current