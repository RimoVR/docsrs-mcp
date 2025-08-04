# docsrs-mcp

A minimal, self-hostable Model Context Protocol (MCP) server that lets AI agents query Rust crate documentation without API keys.

## Overview

docsrs-mcp provides MCP endpoints for:
- Fetching crate summaries and metadata
- Semantic search within Rust documentation
- Retrieving complete rustdoc for any documented item

Built with FastAPI and sqlite-vss for efficient vector search, it caches documentation locally and serves it through the open MCP standard.

## Features

- 🚀 **Zero-install launch** with `uvx`
- 🔍 **Vector search** powered by BAAI/bge-small-en-v1.5 embeddings
- 💾 **Smart caching** with automatic LRU eviction (2GB limit)
- ⚡ **Fast performance** - <500ms search latency, <1GB memory usage
- 🔒 **Rate limiting** - 30 requests/second per IP
- 📦 **Self-contained** - Single Python process, file-backed SQLite

## Quick Start

```bash
# Run directly from GitHub (no installation needed)
uvx --from "git+https://github.com/peterkloiber/docsrs-mcp.git" docsrs-mcp

# Or from PyPI once published
uvx docsrs-mcp@latest
```

## MCP Tools

The server exposes the following MCP tools:

### `get_crate_summary`
Returns crate name, version, description, and module list.

### `search_items`
Performs vector similarity search across crate documentation.

### `get_item_doc`
Retrieves complete rustdoc markdown for a specific item.

### `list_versions`
Lists all available versions of a crate.

## Development

```bash
# Clone the repository
git clone https://github.com/peterkloiber/docsrs-mcp.git
cd docsrs-mcp

# Install dependencies (uv automatically manages virtual environment)
uv sync --dev

# Run development server
uv run python -m docsrs_mcp.cli

# Alternative: run tests
uv run pytest

# Add new dependencies
uv add package-name              # Production dependency
uv add --dev package-name        # Development dependency
```

## Architecture

- **Web Layer**: FastAPI with MCP endpoint handlers
- **Ingestion**: Async pipeline for downloading and processing rustdoc JSON
- **Storage**: SQLite with vector search extension (sqlite-vss)
- **Embedding**: FastEmbed with ONNX-optimized BAAI/bge-small-en-v1.5 model

See [Architecture.md](Architecture.md) for detailed system design.

## Requirements

- Python 3.10+
- 256MB+ RAM (1GB recommended)
- Linux, macOS, or Windows

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io/) standard
- Documentation sourced from [docs.rs](https://docs.rs/)
- Vector search powered by [sqlite-vss](https://github.com/asg017/sqlite-vss)