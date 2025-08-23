# docsrs-mcp

A high-performance Model Context Protocol (MCP) server that provides AI agents with instant access to Rust crate documentation through vector search.

## Quick Start

```bash
# Install and run directly from GitHub (recommended)
uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp

# Or install from PyPI when published
uvx docsrs-mcp
```

## Features

- 🚀 **Zero-install launch** with `uvx` - no setup required
- 🔍 **Semantic vector search** with <50ms response times
- 📚 **Three-tier documentation system** - 80%+ crate coverage via fallback extraction
- ⚡ **Smart caching** with SQLite + sqlite-vec and LRU eviction
- 🎯 **Advanced ranking** with MMR diversification for balanced results
- 🔄 **Version comparison** - Track API changes between crate versions
- 🌍 **International support** - Handles British/American spelling variations
- 🏃 **Zero cold-start** - Embeddings warmup eliminates first-query latency
- 📦 **Popular crate pre-ingestion** - Instant responses for commonly used crates
- 🔗 **Re-export discovery** - Automatic path alias resolution from rustdoc JSON
- 🦀 **Rust stdlib support** - Priority ingestion and detection of standard library crates
- 🔍 **Dependency filtering** - Intelligent filtering of dependency documentation
- 🎯 **Advanced Search Modes** - Vector (semantic), fuzzy (typo-tolerant), regex (pattern matching), and hybrid search
- 🛡️ **ReDoS Protection** - Secure regex search with timeout and pattern validation
- 🌐 **Cross-Crate Search** - Search across up to 5 crates simultaneously with RRF ranking
- ⚖️ **Stability Filtering** - Filter results by API stability level (stable/unstable/experimental)

## Installation

### macOS/Linux
```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the server
uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp
```

### Windows
```powershell
# Install uv
irm https://astral.sh/uv/install.ps1 | iex

# Run the server
uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp
```

### Docker
```bash
docker build -t docsrs-mcp .
docker run -p 8000:8000 -v $(pwd)/cache:/app/cache docsrs-mcp
```

## MCP Configuration

### Claude Desktop

Add to your config file:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "docsrs-mcp": {
      "command": "uvx",
      "args": ["--from", "git+https://github.com/RimoVR/docsrs-mcp.git", "docsrs-mcp"],
      "env": {
        "MAX_CACHE_SIZE_GB": "2"
      }
    }
  }
}
```

### Claude Code CLI
```bash
# Add the server
claude mcp add docsrs -- uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp

# With custom settings
claude mcp add docsrs --env MAX_CACHE_SIZE_GB=5 -- uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp
```

## MCP Tools

### Core Tools

| Tool | Description |
|------|-------------|
| `get_crate_summary` | Fetch crate metadata and module structure |
| `search_items` | Semantic search with type filtering and MMR diversification |
| `get_item_doc` | Full documentation with fuzzy path matching |
| `get_module_tree` | Navigate module hierarchies |
| `search_examples` | Find code examples with language detection |
| `compare_versions` | Analyze API changes between versions |

### Advanced Tools

| Tool | Description |
|------|-------------|
| `start_pre_ingestion` | Pre-load popular crates for instant access |
| `ingest_cargo_file` | Bulk ingest dependencies from Cargo.toml/Cargo.lock with version resolution |
| `list_versions` | List all available crate versions |

## MCP Parameter Requirements

**MCP Client Compatibility**: All parameters are accepted as strings to ensure maximum compatibility with MCP clients like Claude Code. The server automatically converts these to appropriate types.

### Parameter Format Requirements

- **Numeric parameters** must be passed as strings (e.g., `k="5"`, `count="100"`)
- **Boolean parameters** must be passed as strings (e.g., `force="true"`, `deprecated="false"`)
- **String parameters** remain unchanged
- **REST API** continues to accept both strings and native types for backwards compatibility

### MCP Tool Examples

#### Search with proper parameter format:
```json
{
  "crate_name": "tokio",
  "query": "spawn async tasks",
  "k": "10",
  "item_types": "function,struct"
}
```

#### Pre-ingestion with string parameters:
```json
{
  "count": "100",
  "concurrency": "3",
  "force": "true"
}
```

#### Cargo file ingestion with string parameters:
```json
{
  "file_path": "/path/to/Cargo.toml",
  "concurrency": "5",
  "skip_existing": "true",
  "resolve_versions": "false"
}
```

## Architecture

### Three-Tier Documentation System

1. **Tier 1**: Rustdoc JSON from docs.rs (highest quality, ~15% coverage)
2. **Tier 2**: Source extraction from crates.io CDN (80%+ coverage with macro support)
3. **Tier 3**: Latest version fallback (100% guaranteed coverage)

### Performance

- **Search**: <50ms P95 warm latency
- **Fuzzy search**: <300ms with 90% accuracy for common typos
- **Regex search**: <500ms with ReDoS protection
- **Cross-crate search**: <1s for 5 crates in parallel
- **Ingestion**: ≤3s for crates up to 10MB
- **Memory**: ≤1GB RSS including embeddings
- **Cache**: Auto-evicts at 2GB with LRU

### Key Technologies

- **FastAPI** + **FastMCP** for MCP protocol
- **sqlite-vec** for vector search (<100ms for 1M+ vectors)
- **FastEmbed** with BAAI/bge-small-en-v1.5 embeddings
- **RapidFuzz** for intelligent path matching
- **ijson** for memory-efficient streaming

## Development

```bash
# Clone and setup
git clone https://github.com/RimoVR/docsrs-mcp.git
cd docsrs-mcp
uv sync --dev

# Run locally
uv run docsrs-mcp

# Run tests
uv run pytest

# Format and lint
uv run ruff format .
uv run ruff check --fix .
```

## Environment Variables

```bash
# Server configuration
PORT=8000                    # Server port
HOST=0.0.0.0                # Bind address
MAX_CACHE_SIZE_GB=2         # Cache size limit
LOG_LEVEL=INFO              # Logging level

# Performance tuning
PRE_INGEST_ENABLED=true     # Enable popular crate pre-loading
PRE_INGEST_COUNT=100        # Number of crates to pre-load
EMBEDDINGS_WARMUP=true      # Eliminate cold-start latency

# Rust stdlib support
STDLIB_PRE_CACHE=true       # Prioritize stdlib crates in pre-ingestion
RUSTUP_HOME=~/.rustup       # Override rustup location (auto-detected by default)

# Dependency filtering
DEPENDENCY_FILTER=true      # Enable dependency filtering in results
BLOOM_FILTER_CAPACITY=100000  # Expected number of dependency items
BLOOM_FILTER_ERROR_RATE=0.001 # Acceptable false positive rate

# Advanced
RANKING_DIVERSITY_WEIGHT=0.1    # MMR diversification strength
RANKING_DIVERSITY_LAMBDA=0.6    # Relevance vs diversity balance
```

## API Examples

### Search for Documentation
```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "http://localhost:8000/mcp/tools/search_items",
        json={
            "crate_name": "tokio",
            "query": "spawn async tasks",
            "k": "5"  # Note: string format for MCP compatibility
        }
    )
    print(response.json())
```

### Compare Versions
```python
response = await client.post(
    "http://localhost:8000/mcp/tools/compare_versions",
    json={
        "crate_name": "serde",
        "version_a": "1.0.0",
        "version_b": "1.0.100"
    }
)
```

### Advanced Search Examples
```python
# Fuzzy search for typo tolerance
search_items(query="tokio::spwan", search_mode="fuzzy")

# Regex pattern matching
search_items(regex_pattern="async.*spawn", search_mode="regex")

# Cross-crate search
search_items(query="error handling", crates=["tokio", "anyhow", "thiserror"])

# Stability filtering
search_items(query="async runtime", stability_filter="stable")
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "uvx not found" | Install uv: `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Port already in use | Change port: `PORT=8001 uvx docsrs-mcp` |
| Rate limiting (429) | Default: 30 req/s per IP, implement backoff |
| Memory errors | Ensure ≥256MB RAM available |
| MCP validation errors | Ensure all numeric/boolean parameters are passed as strings (e.g., `k="5"`, `force="true"`) |

## Recent Enhancements

### v1.0 (Latest)
- ✅ Three-tier fallback system for 80%+ crate coverage
- ✅ Version comparison for API evolution tracking
- ✅ MMR diversification for balanced search results
- ✅ Embeddings warmup for zero cold-start
- ✅ Popular crate pre-ingestion
- ✅ British/American spelling normalization
- ✅ Enhanced error messages with examples
- ✅ Batch processing optimizations
- ✅ Standard library support with fallbacks

### Known Limitations
- Standard library rustdoc JSON requires local generation via `rustup component add rust-docs-json`
- Some older crates may only have Tier 2/3 documentation quality

## License

MIT License - see LICENSE file

## Contributing

Contributions welcome! Please submit pull requests to the [GitHub repository](https://github.com/RimoVR/docsrs-mcp).

## Security

- **Origin allowlist**: Only fetches from docs.rs and crates.io
- **Size limits**: 30MB compressed / 100MB decompressed
- **Input validation**: Strict Pydantic models with comprehensive validation
- **Rate limiting**: Built-in protection against abuse

Report security issues to security@example.com