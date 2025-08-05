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

### Zero-Install Launch

The easiest way to run docsrs-mcp is with `uvx` (requires Python 3.10+):

```bash
# Install from PyPI (when published)
uvx docsrs-mcp

# Install from GitHub (latest version)
uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp

# Install specific branch/tag
uvx --from git+https://github.com/RimoVR/docsrs-mcp.git@main docsrs-mcp

# Run from local directory
uvx --from . docsrs-mcp

# The server will start in MCP mode by default (use --mode rest for HTTP API)
```

### Platform-Specific Instructions

#### macOS/Linux
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Run the server
uvx docsrs-mcp
```

#### Windows
```powershell
# Install uv using PowerShell
irm https://astral.sh/uv/install.ps1 | iex

# Run the server
uvx docsrs-mcp
```

### Docker Quick Start

```bash
# Build and run with Docker
docker build -t docsrs-mcp .
docker run -p 8000:8000 -v $(pwd)/cache:/app/cache docsrs-mcp

# Or use docker-compose
docker-compose up
```

### Environment Configuration

```bash
# Custom port
PORT=8080 uvx docsrs-mcp

# Custom host binding
HOST=0.0.0.0 uvx docsrs-mcp

# Adjust cache size limit (default 2GB)
MAX_CACHE_SIZE_GB=5 uvx docsrs-mcp

# Enable debug logging
LOG_LEVEL=DEBUG uvx docsrs-mcp
```

### Verify Installation

```bash
# Check server health
curl http://localhost:8000/health

# View API documentation
open http://localhost:8000/docs

# Get MCP manifest
curl http://localhost:8000/mcp/manifest
```

## MCP Configuration

### Server Modes

docsrs-mcp supports two operation modes:

1. **MCP mode** (default): Model Context Protocol server using STDIO transport for Claude Desktop integration
2. **REST mode**: Traditional HTTP API server for debugging and direct API access

To run the server:
```bash
# Run with MCP protocol via STDIO (default)
uvx docsrs-mcp
# or explicitly
uvx docsrs-mcp -- --mode mcp

# Run in REST mode for HTTP API
uvx docsrs-mcp -- --mode rest
```

### Claude Desktop Configuration

Add the following to your Claude Desktop configuration file:

**Location:**
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`
- Linux: `~/.config/Claude/claude_desktop_config.json`

**Configuration:**
```json
{
  "mcpServers": {
    "docsrs-mcp": {
      "command": "uvx",
      "args": ["docsrs-mcp"],
      "env": {
        "MAX_CACHE_SIZE_GB": "2"
      }
    }
  }
}
```

**Note:** The server runs in MCP mode by default, using STDIO transport with all logs sent to stderr to avoid protocol corruption. To run in REST mode for debugging, use `--mode rest`.

### Claude Code CLI

To add this server to Claude Code:

```bash
# Add the server from PyPI (when published)
claude mcp add docsrs -- uvx docsrs-mcp

# Or add from GitHub
claude mcp add docsrs -- uvx --from git+https://github.com/RimoVR/docsrs-mcp.git docsrs-mcp

# Add with custom environment variables
claude mcp add docsrs --env MAX_CACHE_SIZE_GB=5 -- uvx docsrs-mcp

# List configured servers
claude mcp list

# Remove the server
claude mcp remove docsrs
```

After configuration, restart Claude Desktop or use the `/mcp` command in Claude Code to verify the server is connected.

## MCP Tools

The server exposes the following MCP tools through the `/mcp/tools/{tool_name}` endpoint:

### `get_crate_summary`

Returns crate metadata including name, version, description, and repository info.

**Parameters:**
- `crate_name` (string, required): Name of the Rust crate
- `version` (string, optional): Specific version or "latest" (default: "latest")

**Example Request:**
```bash
curl -X POST http://localhost:8000/mcp/tools/get_crate_summary \
  -H "Content-Type: application/json" \
  -d '{
    "crate_name": "tokio",
    "version": "latest"
  }'
```

**Example Response:**
```json
{
  "name": "tokio",
  "version": "1.35.1",
  "description": "An event-driven, non-blocking I/O platform for writing asynchronous I/O backed applications.",
  "repository": "https://github.com/tokio-rs/tokio",
  "documentation": "https://docs.rs/tokio/1.35.1",
  "categories": ["asynchronous", "network-programming"],
  "keywords": ["io", "async", "non-blocking", "futures"]
}
```

**Error Response (404):**
```json
{
  "detail": "Crate 'nonexistent' not found"
}
```

### `search_items`

Performs vector similarity search across complete rustdoc documentation.

**Parameters:**
- `crate_name` (string, required): Name of the crate to search within
- `query` (string, required): Search query for semantic similarity
- `k` (integer, optional): Number of results to return (default: 5, max: 20)

**Example Request:**
```bash
curl -X POST http://localhost:8000/mcp/tools/search_items \
  -H "Content-Type: application/json" \
  -d '{
    "crate_name": "tokio",
    "query": "spawn async tasks",
    "k": 3
  }'
```

**Example Response:**
```json
{
  "results": [
    {
      "item_path": "tokio::spawn",
      "name": "spawn",
      "description": "Spawns a new asynchronous task, returning a JoinHandle for it.",
      "kind": "function",
      "score": 0.89
    },
    {
      "item_path": "tokio::task::spawn_blocking",
      "name": "spawn_blocking",
      "description": "Runs the provided blocking function on the current thread pool.",
      "kind": "function",
      "score": 0.82
    },
    {
      "item_path": "tokio::task",
      "name": "task",
      "description": "Asynchronous task utilities.",
      "kind": "module",
      "score": 0.75
    }
  ]
}
```

**Error Response (429 - Rate Limited):**
```json
{
  "detail": "Rate limit exceeded. Please retry after 1 second."
}
```

### `get_item_doc`

Retrieves documentation for a specific item from ingested rustdoc data.

**Parameters:**
- `crate_name` (string, required): Name of the crate
- `item_path` (string, required): Full path to the item (e.g., "crate", "tokio::spawn", "std::vec::Vec")

**Example Request:**
```bash
curl -X POST http://localhost:8000/mcp/tools/get_item_doc \
  -H "Content-Type: application/json" \
  -d '{
    "crate_name": "serde",
    "item_path": "serde::Deserialize"
  }'
```

**Example Response:**
```json
{
  "item_path": "serde::Deserialize",
  "name": "Deserialize",
  "kind": "trait",
  "description": "A data structure that can be deserialized from any data format supported by Serde.",
  "documentation": "# Examples\n\n```rust\nuse serde::Deserialize;\n\n#[derive(Deserialize)]\nstruct User {\n    name: String,\n    age: u32,\n}\n```\n\nThis trait can be derived using `#[derive(Deserialize)]` or implemented manually...",
  "signature": "pub trait Deserialize<'de>: Sized"
}
```

### Resources

The server also provides MCP resources for listing available data:

#### `/mcp/resources/versions`

Lists all cached versions of a crate.

**Example Request:**
```bash
curl -X POST http://localhost:8000/mcp/resources/versions \
  -H "Content-Type: application/json" \
  -d '{
    "crate_name": "tokio"
  }'
```

**Example Response:**
```json
{
  "crate_name": "tokio",
  "versions": [
    "1.35.1",
    "1.35.0",
    "1.34.0"
  ]
}
```

### Python Client Example

```python
import httpx
import asyncio

async def search_tokio_docs():
    async with httpx.AsyncClient() as client:
        # Search for spawn-related functions
        response = await client.post(
            "http://localhost:8000/mcp/tools/search_items",
            json={
                "crate_name": "tokio",
                "query": "spawn async tasks",
                "k": 5
            }
        )
        results = response.json()
        
        # Get detailed docs for the first result
        if results["results"]:
            first_item = results["results"][0]
            doc_response = await client.post(
                "http://localhost:8000/mcp/tools/get_item_doc",
                json={
                    "crate_name": "tokio",
                    "item_path": first_item["item_path"]
                }
            )
            print(doc_response.json())

asyncio.run(search_tokio_docs())
```

## Troubleshooting

### Common Issues and Solutions

#### uvx Installation Problems
- **"uvx not found"**: Install uv first with `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Python version mismatch**: Ensure Python 3.10+ is available. UV will use the appropriate version automatically
- **Permission errors**: On Unix systems, ensure `~/.local/bin` is in your PATH

#### Network Connectivity
- **Docker networking issues**: Use `--network host` or expose port 8000 properly
- **Proxy environments**: Set `HTTP_PROXY` and `HTTPS_PROXY` environment variables
- **docs.rs timeouts**: The server retries failed downloads automatically

#### Cache Directory Issues
- **Permission denied**: Ensure write access to `./cache` directory
- **Disk space**: Monitor cache size, auto-eviction maintains 2GB limit
- **Corrupted cache**: Delete specific crate cache files in `./cache/{crate}/{version}.db`

#### Rate Limiting (HTTP 429)
- **Default limit**: 30 requests/second per IP address
- **Burst handling**: Implement exponential backoff in clients
- **Monitoring**: Check server logs for rate limit violations

#### Server Startup Issues
- **Port already in use**: Change port with `PORT=8001 uvx docsrs-mcp`
- **Memory errors**: Ensure at least 256MB available RAM
- **Background process**: Use `nohup uvx docsrs-mcp > server.log 2>&1 & echo $!`

#### MCP Client Integration
- **Tool not found**: Ensure client supports MCP 2025-06-18 specification
- **Schema validation errors**: Check request format matches tool schemas exactly
- **Response timeouts**: Increase client timeout for large crate ingestion

## Development

```bash
# Clone the repository
git clone https://github.com/RimoVR/docsrs-mcp.git
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

## Performance & Resource Usage

The server is designed for efficient operation with the following characteristics:

### Performance Targets
- **Search Latency**: ≤ 500ms P95 for warm searches
  - Query execution: < 50ms
  - End-to-end response: < 500ms
- **Ingestion Speed**: ≤ 3s for crates up to 10MB compressed
- **Rate Limiting**: 30 requests/second per IP address

### Resource Requirements
- **Memory Usage**: ≤ 1GB RAM for up to 10,000 embeddings
  - ONNX model: ~100MB
  - FAISS index: Variable based on vectors
  - Server RSS: ≤ 1GB including all components
- **Disk Cache**: Auto-evicts to maintain ≤ 2GB total size
  - Per-crate SQLite databases in `./cache` directory
  - LRU eviction when cache size exceeds limit
- **CPU**: Single core sufficient for typical workloads

### Benchmarks
| Crate Size | Cold Ingest Time | Warm Search |
|------------|-----------------|-------------|
| < 1MB      | < 1s            | < 50ms      |
| 1-10MB     | 1-3s            | < 100ms     |
| 10-30MB    | 3-10s           | < 200ms     |

### Optimization Tips
- Pre-warm frequently accessed crates by querying them on startup
- Adjust cache size limit via `MAX_CACHE_SIZE_GB` environment variable
- Use background process management for testing to avoid terminal blocking
- Monitor `/health` endpoint for service status

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

## Security

### Security Features

The server implements multiple security measures to ensure safe operation:

#### Origin Allow-List
- **Restriction**: Only fetches documentation from `https://docs.rs/` URLs
- **Validation**: Strict URL validation prevents arbitrary file downloads
- **No user-supplied URLs**: All documentation sources are constructed internally

#### Size Limits
- **Compressed files**: Maximum 30MB (prevents DoS via large downloads)
- **Decompressed content**: Maximum 100MB (prevents memory exhaustion)
- **Streaming processing**: Memory-efficient handling of large files
- **Automatic cleanup**: Failed downloads are immediately purged

#### Path Safety
- **Sanitized paths**: Database files stored as `cache/{crate}/{version}.db`
- **No path traversal**: Crate and version names are sanitized
- **Isolated storage**: Each crate version has its own database file

#### Input Validation
- **Pydantic models**: All inputs validated with strict type checking
- **Extra fields forbidden**: `extra='forbid'` prevents injection attacks
- **SQL injection prevention**: Parameterized queries throughout
- **Rate limiting**: 30 requests/second per IP address

### Security Best Practices

1. **Run with minimal privileges**: Use a dedicated user account
2. **Filesystem isolation**: Configure cache directory with appropriate permissions
3. **Network isolation**: Bind to localhost unless external access required
4. **Resource limits**: Use systemd or Docker to enforce memory/CPU limits
5. **Regular updates**: Keep dependencies updated with `uv sync`

### Reporting Security Issues

If you discover a security vulnerability, please email security@example.com with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

Please do not open public issues for security vulnerabilities.