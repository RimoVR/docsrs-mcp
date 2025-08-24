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

### Core Philosophy
- **Don't let perfect be the enemy of good** - Ship working solutions, iterate based on real usage
- **Be as efficient and effective as reasonably achievable** - Balance optimization with development speed
- **Keep it simple, stupid (KISS)** - Complexity is the enemy of reliability

### Implementation Guidelines

1. **Early Error Detection**
   - Use assertions liberally to catch issues during development
   - Fail fast with clear error messages
   - But don't over-engineer error handling - handle what matters

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
   - Start with the simplest solution that works
   - Add complexity only when proven necessary

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

## Python Tooling

### UV-First Infrastructure
This project exclusively uses `uv` as the Python infrastructure tool:

- **Package Management**: `uv add`, `uv remove`, `uv sync` (NO pip/conda)
- **Virtual Environments**: `uv venv`, `uv run` (NO python -m venv)
- **Project Initialization**: `uv init` (NO setup.py/pip install -e)
- **Execution**: `uv run`, `uvx` for tools (NO direct python commands)
- **Development**: `uv sync --dev` for dev dependencies
- **CI/CD**: Use `uv` commands in all automation scripts

### UV Command Patterns
```bash
# Development setup
uv sync --dev                    # Install all dependencies
uv run python -m module         # Run module
uv run pytest                   # Run tests
uv add package                   # Add production dependency
uv add --dev package             # Add development dependency

# Distribution
uvx --from . docsrs-mcp         # Test local install
uvx --from git+URL docsrs-mcp   # Test from git
```

**Critical**: Never mix uv with pip, conda, or other package managers. All Python tooling must go through uv to maintain consistency and leverage its performance benefits.

### Code Quality & Formatting
This project exclusively uses **Ruff** for all linting and code formatting needs:

- **Linting**: `uv run ruff check` (replaces flake8, pylint, etc.)
- **Formatting**: `uv run ruff format` (replaces black, autopep8, etc.)
- **Import Sorting**: Built into Ruff (replaces isort)
- **Configuration**: All settings in `pyproject.toml` under `[tool.ruff]`
- **Performance**: Single Rust-based tool instead of multiple Python tools

```bash
# Code quality commands
uv run ruff check .          # Lint code
uv run ruff check --fix .    # Lint and auto-fix
uv run ruff format .         # Format code
uv run ruff check --diff .   # Show what would change
```

**Never use**: black, flake8, pylint, isort, autopep8, or any other formatters/linters. Ruff replaces all of them with a single, fast, Rust-based tool.

## Server Testing Best Practices

### Background Process Management
When testing servers with `uv run`, use background processes to avoid terminal hanging:

```bash
# Start MCP server in SDK mode (default) and capture PID
nohup uv run docsrs-mcp > server.log 2>&1 & echo $!

# Store PID for later use
SERVER_PID=$(nohup uv run docsrs-mcp > server.log 2>&1 & echo $!)

# Test with uvx (for zero-install deployment)
nohup uvx --from . docsrs-mcp > server.log 2>&1 & echo $!

# Check server logs
tail -f server.log

# Kill server when done
kill $SERVER_PID
```

**Default Mode**: MCP SDK mode is now the default (--mcp-implementation sdk). This provides full compatibility with official MCP clients.

**Important**: Never use bare `uv run docsrs-mcp &` as it requires manual interruption which can crash the terminal. Always use `nohup` with output redirection and PID capture for clean process management.

**Testing Guidelines**:
- Test MCP SDK mode (default): `uv run docsrs-mcp` or `uvx --from . docsrs-mcp`
- Test REST mode for HTTP API: `uv run docsrs-mcp --mode rest`
- Do not only run synthetic test suites but test with production code as well