# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## MCP Server Usage Guidelines

### Available MCP Servers
1. **Probe** (Fast Discovery) - Use for all initial code searches
   - Ultra-fast text/AST search without indexing
   - Always start here before using other tools
   - Supports session caching for iterative searches

2. **Serena** (Semantic Precision) - Use sparingly for specific tasks
   - LSP-powered semantic analysis
   - ⚠️ AVOID broad searches - high RAM/CPU usage
   - Use only after Probe identifies specific targets
   - Best for: precise symbol navigation, refactoring

3. **Context7 & Octocode** - Reserved for specialized agents
   - DO NOT use directly in main commands
   - Access through websearch agent only
   - Token-heavy - use judiciously

### MCP Usage Patterns
```
✅ CORRECT: Probe (broad search) → Serena (specific symbols)
❌ WRONG: Serena for exploration or broad searches
❌ WRONG: Context7/Octocode in main context
```

### Custom Slash Commands (for Claude agents)
- `/initialize-PRD` - Initialize new project from PRD
- `/initialize-existing` - Bootstrap existing project  
- `/create-plan` - Start task planning phase (uses Probe for discovery)
- `/revise-plan` - Refine plan and implement (Probe → Serena workflow)
- `/add-feature-to-task` - Plan new feature with research agents
- `/debug-feature` - Debug with strategic MCP usage
- `/human-in-the-loop` - GUI testing coordination
- `/commit-fast` - Auto-commit with semantic messages
- `/create-pr` - Create pull request

## Development Principles

1. **Early Error Detection**
   - Use assertions liberally to catch issues during development
   - Fail fast with clear error messages

2. **No Silent Failures**
   - Never use fallbacks or TODOs that hide errors. Never replace existing implementations with todo markers.
   - Raise errors explicitly - visibility over convenience
   - Each error path must be traceable and debuggable

3. **Documentation & Information Gathering**
   - Use concurrent web search agents for version-specific information
   - Web search agents have access to Context7 (docs) and Octocode (examples)
   - When asking questions, wait for user response
   - Question every requirement for clarity
   - Track system patterns in project documentation
   - Definitive documentation files: PRD.md, Tasks.json, Architecture.md, ResearchFindings.json, UsefulInformation.json, and module READMEs
   - Maintain documentation meticulously yet concisely

4. **Best Practices**
   - Follow language idioms and conventions
   - Prioritize simplicity and maintainability
   - Optimize for readability over cleverness

## Key Implementation Details

### Living Memory Files (Project Root)
- `PRD.md` - Product requirements document
- `Tasks.json` - Task management with petgraph-compatible structure
- `Architecture.md` - System design with Mermaid diagrams
- `ResearchFindings.json` - External knowledge and library research
- `UsefulInformation.json` - Error solutions and lessons learned

### MCP Server Best Practices
- **Performance**: Set Probe timeout to 120s for repos >1MLOC
- **Token Management**: Use Probe's session caching, limit results
- **Failure Handling**: If Serena LSP crashes, fallback to Probe
- **Search Strategy**: Exact queries over wildcards when possible