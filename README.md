# docsrs-mcp

A minimal, self-hostable Model Context Protocol (MCP) server that lets AI agents query Rust crate documentation with vector search capabilities.

## Overview

docsrs-mcp provides MCP endpoints for:
- Fetching crate summaries and metadata
- Semantic search within Rust crate documentation
- Retrieving documentation for specific items
- Listing available crate versions

Built with FastAPI and sqlite-vec for efficient vector search, it caches crate data locally and serves it through the MCP standard.

## Features

- 🚀 **Zero-install launch** with `uvx`
- 🔍 **Vector search** powered by BAAI/bge-small-en-v1.5 embeddings
- 📚 **Full rustdoc ingestion** - Complete documentation from docs.rs JSON files
- 🗜️ **Compression support** - Handles .json.zst (Zstandard) and .json.gz (Gzip) formats
- 💾 **Local caching** in SQLite with sqlite-vec extension and LRU eviction
- ⚡ **Fast performance** - Low latency search, efficient memory usage
- 🔒 **Concurrent safety** - Per-crate locking prevents duplicate ingestion
- 📦 **Self-contained** - Single Python process, file-backed SQLite
- 🛠️ **Easy development** - UV-based tooling, async/await architecture

## Quick Start

```bash
# Run directly from the current directory
uvx --from . docsrs-mcp

# Or clone and run
git clone https://github.com/yourusername/docsrs-mcp.git
cd docsrs-mcp
uvx --from . docsrs-mcp

# The server will start on http://localhost:8000
```

## MCP Tools

The server exposes the following MCP tools:

### `get_crate_summary`
Returns crate metadata including name, version, description, and repository info.

```json
{
  "crate_name": "tokio",
  "version": "latest"  // optional
}
```

### `search_items`
Performs vector similarity search across complete rustdoc documentation.

```json
{
  "crate_name": "tokio",
  "query": "async runtime",
  "k": 5  // optional, number of results
}
```

### `get_item_doc`
Retrieves documentation for a specific item from ingested rustdoc data.

```json
{
  "crate_name": "tokio",
  "item_path": "crate"
}
```

### Resources

- `/mcp/resources/versions` - Lists all cached versions of a crate

## Development

```bash
# Clone the repository
git clone https://github.com/peterkloiber/docsrs-mcp.git
cd docsrs-mcp

# Install dependencies (uv automatically manages virtual environment)
uv sync --dev

# Run development server
uv run docsrs-mcp

# Run with background process (avoids terminal hanging)
nohup uv run docsrs-mcp > server.log 2>&1 & echo $!

# Run tests (25 comprehensive tests covering all pipeline components)
uv run pytest

# Add new dependencies
uv add package-name              # Production dependency
uv add --dev package-name        # Development dependency
```

## Architecture

- **Web Layer**: FastAPI with MCP endpoint handlers
- **Ingestion Pipeline**: 
  - Downloads complete rustdoc JSON files from docs.rs
  - Supports compressed formats (.json.zst, .json.gz) with automatic detection
  - Memory-efficient streaming JSON parsing with ijson
  - Per-crate locking mechanism prevents duplicate concurrent ingestion
  - Version resolution via docs.rs redirects (supports "latest" version)
  - Streaming decompression with configurable size limits (100MB default)
  - Graceful fallback to crate description embedding when rustdoc unavailable
- **Storage**: SQLite with vector search extension (sqlite-vec)
- **Embedding**: FastEmbed with ONNX-optimized BAAI/bge-small-en-v1.5 model
- **Caching**: File-based SQLite databases in `./cache` directory with LRU eviction (2GB limit)

See [Architecture.md](Architecture.md) for detailed system design.

## Requirements

- Python 3.10+
- 256MB+ RAM (1GB recommended)
- Linux, macOS, or Windows

## License

MIT License - see LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Testing

The project includes comprehensive test coverage with 25 tests covering:

- **Unit Tests**: All pipeline components individually tested
- **Integration Tests**: End-to-end ingestion scenarios
- **Compression Tests**: Support for .json.zst and .json.gz formats
- **Concurrency Tests**: Per-crate locking and concurrent access
- **Cache Tests**: LRU eviction and size limit enforcement
- **Error Handling**: Graceful fallbacks and error recovery

## Acknowledgments

- Built on the [Model Context Protocol](https://modelcontextprotocol.io/) standard
- Crate data from [crates.io](https://crates.io/)
- Vector search powered by [sqlite-vec](https://github.com/asg017/sqlite-vec)