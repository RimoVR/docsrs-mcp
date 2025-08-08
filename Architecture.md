# docsrs-mcp Architecture

## System Overview

The docsrs-mcp server provides both REST API and Model Context Protocol (MCP) endpoints for querying Rust crate documentation using vector search. It features a dual-mode architecture with a FastAPI web layer that can operate in either MCP mode (default) or REST mode. The MCP mode uses STDIO transport for AI clients, while REST mode requires an explicit flag. The system includes a comprehensive asynchronous ingestion pipeline with enhanced rustdoc JSON processing for complete documentation extraction, a SQLite-based vector storage system with intelligent caching and example management, and a dedicated code example extraction and semantic search system.

## High-Level Architecture

```mermaid
graph TB
    subgraph "AI Clients"
        AI[AI Agent/LLM]
    end
    
    subgraph "Dual-Mode Server"
        CLI[CLI Entry Point<br/>--mode flag]
        API[FastAPI Application]
        MCP[MCP Server Module<br/>FastMCP wrapper]
        RL[Rate Limiter<br/>30 req/s per IP]
        IW[Ingest Worker]
        Queue[asyncio.Queue]
    end
    
    subgraph "External Services"
        DOCS[docs.rs CDN]
    end
    
    subgraph "Storage"
        CACHE[(SQLite + VSS<br/>cache/*.db)]
        META[Metadata<br/>per crate@version]
    end
    
    subgraph "ML Components"
        EMB[FastEmbed<br/>BAAI/bge-small-en-v1.5<br/>384 dimensions]
    end
    
    AI -->|MCP STDIO/REST POST| CLI
    CLI -->|--mode rest| RL
    CLI -->|MCP mode (default)| MCP
    MCP --> API
    RL --> API
    API -->|enqueue| Queue
    Queue -->|dequeue| IW
    IW -->|version resolve + download| DOCS
    IW -->|embed text| EMB
    IW -->|store vectors| CACHE
    API -->|query| CACHE
    CACHE --> META
```

## Component Architecture

```mermaid
graph LR
    subgraph "docsrs_mcp Package"
        subgraph "Web Layer"
            APP[app.py<br/>FastAPI instance<br/>OpenAPI metadata<br/>search_examples endpoint]
            ROUTES[routes.py<br/>Enhanced MCP endpoints<br/>Comprehensive docstrings<br/>Fuzzy path resolution]
            MODELS[models.py<br/>Enhanced Pydantic schemas<br/>Field validators<br/>MCP compatibility<br/>Auto-generated docs<br/>Code example data models]
            VALIDATION[validation.py<br/>Centralized validation utilities<br/>Precompiled regex patterns<br/>Type coercion functions]
            NAV[navigation.py<br/>Module tree operations<br/>Hierarchy traversal]
            MW[middleware.py<br/>Rate limiting]
        end
        
        subgraph "Ingestion Layer"
            ING[ingest.py<br/>Enhanced rustdoc pipeline<br/>Complete item extraction]
            VER[Version Resolution<br/>docs.rs redirects]
            DL[Compression Support<br/>zst, gzip, json]
            PARSE[ijson Parser<br/>Memory-efficient streaming<br/>Module hierarchy extraction]
            HIERARCHY[build_module_hierarchy()<br/>Parent-child relationships<br/>Depth calculation<br/>Item counting]
            EXTRACT[Enhanced Code Example Extractor<br/>JSON structure with metadata<br/>Language detection via pygments<br/>30% confidence threshold]
            EMBED[FastEmbed<br/>Batch processing]
            LOCK[Per-crate Locks<br/>Prevent duplicates]
        end
        
        subgraph "Storage Layer"
            DB[database.py<br/>SQLite operations<br/>get_see_also_suggestions()]
            VSS[vector_search.py<br/>k-NN queries with ranking]
            RANK[ranking.py<br/>Multi-factor scoring<br/>Type-aware weights]
            CACHE[cache_manager.py<br/>LRU eviction with TTL]
        end
        
        subgraph "Server Layer"
            MCP_SERVER[mcp_server.py<br/>FastMCP wrapper<br/>STDIO transport<br/>stderr logging]
        end
        
        subgraph "Utilities"
            CLI[cli.py<br/>Entry point<br/>--mode flag (defaults to mcp)<br/>MCP/REST selection]
            CONFIG[config.py<br/>Settings]
            ERRORS[errors.py<br/>Custom exceptions]
        end
    end
    
    APP --> ROUTES
    ROUTES --> MODELS
    MODELS --> VALIDATION
    APP --> MW
    ROUTES --> ING
    ROUTES --> NAV
    ING --> PARSE
    PARSE --> HIERARCHY
    HIERARCHY --> EXTRACT
    EXTRACT --> EMBED
    EMBED --> DB
    DB --> VSS
    VSS --> RANK
    DB --> CACHE
    CLI --> APP
    CLI --> MCP_SERVER
    MCP_SERVER --> APP
```

## Data Flow

```mermaid
sequenceDiagram
    participant Client as AI Client
    participant API as FastAPI
    participant Queue as asyncio.Queue
    participant Worker as Ingest Worker
    participant DocsRS as docs.rs
    participant Embed as FastEmbed
    participant DB as SQLite+VSS
    
    Client->>API: POST /mcp/tools/search_items
    Note over API: MCP Parameter Validation<br/>anyOf[integer,string] patterns<br/>Handle string "5" → int 5
    API->>API: Query preprocessing (Unicode normalization NFKC)
    API->>DB: Check if crate@version exists
    
    alt Cache Miss
        API->>Queue: Enqueue ingest task
        Queue->>Worker: Dequeue task
        Worker->>Worker: Acquire per-crate lock
        Worker->>DocsRS: Resolve version via redirect (or detect stdlib crate)
        DocsRS-->>Worker: Actual version + rustdoc URL (with channel resolution for stdlib)
        Worker->>DocsRS: GET compressed rustdoc (.zst/.gz/.json)
        DocsRS-->>Worker: Compressed rustdoc JSON
        Worker->>Worker: Stream decompress with size limits
        Worker->>Worker: Parse with ijson (memory-efficient)
        Worker->>Worker: Parse complete rustdoc structure
        Worker->>Worker: Extract module hierarchy (build_module_hierarchy)
        Worker->>DB: Store module hierarchy with parent_id, depth, item_count
        Worker->>Worker: Parse code examples from docs
        Note over Worker: BUG FIX: Type validation prevents<br/>character fragmentation in examples_data<br/>(ingest.py:761-769)
        Worker->>Worker: Stream items progressively (generator-based)
        Worker->>Embed: Adaptive batch embed (size=16-512 based on memory)
        Embed-->>Worker: 384-dim vectors
        Worker->>DB: Stream batch insert (size=999 with memory-aware chunking)
        Worker->>DB: Index with vss_index!
        Worker->>Worker: Check cache size (2GB limit)
        alt Cache > 2 GiB
            Worker->>DB: LRU eviction by file mtime
        end
    end
    
    alt Path Resolution Enhancement
        API->>API: Check PATH_ALIASES for common patterns
        Note over API: serde::Deserialize → serde::de::Deserialize<br/>Implemented: <1ms O(1) alias resolution
    end
    
    API->>DB: Vector search query with ranking (using normalized query)
    DB->>DB: sqlite-vec MATCH similarity
    DB->>DB: Multi-factor scoring (vector 60% + type 15% + quality 10% + examples 15%)
    DB->>DB: Type-aware boost (functions 1.2x, traits 1.15x, structs 1.1x)
    DB->>DB: Score normalization to [0, 1] range
    DB-->>API: Ranked top-k results with scores
    
    alt Results Found (Generate Suggestions)
        API->>DB: get_see_also_suggestions(query_embedding, main_result_paths)
        Note over DB: Vector similarity search<br/>0.7 threshold, k=10 over-fetch<br/>Excludes main results
        DB-->>API: Related item paths (max 5)
        API->>API: Add suggestions to first SearchResult only
    end
    
    API-->>Client: JSON response with ranking scores + see-also suggestions
```

## Database Schema

```mermaid
erDiagram
    EMBEDDINGS {
        INTEGER rowid PK
        TEXT item_id "stable rustdoc ID"
        TEXT item_path "e.g. serde::de::Deserialize"
        TEXT item_type "function, struct, trait, module, etc - DEFAULT NULL"
        TEXT signature "complete item signature - DEFAULT NULL"
        TEXT header "item signature (legacy)"
        TEXT doc "full documentation"
        INTEGER parent_id "module hierarchy parent - DEFAULT NULL"
        TEXT examples "extracted code examples - DEFAULT NULL"
        TEXT tags "searchable metadata tags"
        INTEGER char_start "original position"
        BOOLEAN deprecated "item deprecation status - DEFAULT NULL"
    }
    
    MODULES {
        INTEGER id PK
        TEXT name "module name (last component)"
        TEXT path "full module path (e.g. std::collections)"
        INTEGER parent_id "parent module ID - NULL for root"
        INTEGER depth "nesting depth from root"
        INTEGER item_count "number of items in module"
        TEXT crate_name "crate this module belongs to"
        TEXT crate_version "crate version"
    }
    
    VEC_EMBEDDINGS {
        INTEGER rowid PK
        BLOB embedding "384-dim float32 vector"
    }
    
    PASSAGES {
        INTEGER id PK
        TEXT item_id "stable rustdoc ID"
        TEXT item_path "e.g. serde::de::Deserialize"
        TEXT item_type "function, struct, trait, module, etc - DEFAULT NULL"
        TEXT signature "complete item signature - DEFAULT NULL"
        TEXT header "item signature (legacy)"
        TEXT doc "full documentation"
        INTEGER parent_id "module hierarchy parent - DEFAULT NULL"
        TEXT examples "extracted code examples - DEFAULT NULL"
        TEXT tags "searchable metadata tags"
        INTEGER char_start "original position"
        BLOB vec "384-dim float32 array"
    }
    
    VSS_PASSAGES {
        BLOB vec "FAISS index (legacy)"
    }
    
    EXAMPLES {
        INTEGER id PK
        INTEGER passage_id FK
        TEXT code "example code snippet"
        TEXT language "rust, bash, toml, etc"
        TEXT description "example description"
        INTEGER line_number "position in docs"
    }
    
    EXAMPLE_EMBEDDINGS {
        INTEGER id PK
        TEXT item_id "stable rustdoc ID"
        TEXT item_path "e.g. serde::de::Deserialize"
        TEXT crate_name "crate this example belongs to"
        TEXT version "crate version"
        TEXT example_hash "SHA256 prefix for deduplication - 16 chars"
        TEXT example_text "code example content"
        TEXT language "rust, bash, toml, etc"
        TEXT context "surrounding documentation context"
        BLOB embedding "384-dim float32 vector"
        INTEGER created_at "timestamp"
    }
    
    VEC_EXAMPLE_EMBEDDINGS {
        INTEGER rowid PK
        BLOB embedding "384-dim float32 vector for examples"
    }
    
    META {
        TEXT crate
        TEXT version
        INTEGER ts "ingestion timestamp"
        TEXT target "e.g. x86_64-unknown-linux-gnu"
    }
    
    EMBEDDINGS ||--|| VEC_EMBEDDINGS : "vector indexed by rowid"
    EMBEDDINGS ||--o{ EXAMPLES : "contains"
    EMBEDDINGS ||--o{ EMBEDDINGS : "parent_id references rowid"
    PASSAGES ||--|| VSS_PASSAGES : "legacy indexed by"
    PASSAGES ||--o{ EXAMPLES : "legacy contains"
    PASSAGES ||--o{ PASSAGES : "legacy parent_id references id"
    META ||--|| EMBEDDINGS : "describes"
    META ||--|| PASSAGES : "legacy describes"
    MODULES ||--o{ MODULES : "parent_id self-reference"
    MODULES ||--o{ EMBEDDINGS : "contains documentation items"
    MODULES ||--o{ PASSAGES : "legacy contains items"
    EXAMPLE_EMBEDDINGS ||--|| VEC_EXAMPLE_EMBEDDINGS : "vector indexed by rowid"
    EXAMPLE_EMBEDDINGS ||--o{ EXAMPLE_EMBEDDINGS : "deduplication via example_hash"
```

### Partial Indexes for Filter Optimization

The database schema includes specialized partial indexes designed to optimize common filter patterns:

- **idx_non_deprecated**: Indexes only non-deprecated items (`WHERE deprecated IS NULL OR deprecated = 0`)
- **idx_public_functions**: Indexes public functions (`WHERE item_type = 'function' AND item_path NOT LIKE '%::%'`)
- **idx_has_examples**: Indexes items with code examples (`WHERE examples IS NOT NULL AND examples != ''`)
- **idx_crate_prefix**: Enables fast crate-specific searches using prefix matching on item_path

### Example Embeddings Indexes

The example_embeddings table includes specialized indexes for efficient example search and deduplication:

- **idx_example_crate_version**: Crate-specific example queries (`CREATE INDEX idx_example_crate_version ON example_embeddings(crate_name, version)`)
- **idx_example_hash**: Hash-based deduplication (`CREATE UNIQUE INDEX idx_example_hash ON example_embeddings(crate_name, version, example_hash)`)
- **idx_example_item_path**: Item-specific example lookup (`CREATE INDEX idx_example_item_path ON example_embeddings(item_path)`)
- **UNIQUE constraint**: Enforced on (crate_name, version, example_hash) for deduplication

### Module Hierarchy Indexes

The modules table includes specialized indexes for efficient hierarchical operations:

- **idx_modules_parent_id**: Fast parent-child relationship queries (`CREATE INDEX idx_modules_parent_id ON modules(parent_id)`)
- **idx_modules_depth**: Level-based traversal optimization (`CREATE INDEX idx_modules_depth ON modules(depth)`)
- **idx_modules_crate**: Crate-specific module queries (`CREATE INDEX idx_modules_crate ON modules(crate_name, crate_version)`)
- **idx_modules_path**: Path-based module lookup (`CREATE INDEX idx_modules_path ON modules(path)`)

## Dual-Mode Architecture

```mermaid
graph TB
    subgraph "Client Interface"
        CLI_CLIENT[Claude/AI Client]
        REST_CLIENT[REST API Client]
    end
    
    subgraph "Server Modes"
        CLI_ENTRY[CLI Entry Point<br/>--mode flag]
        
        subgraph "MCP Mode (Default)"
            MCP_SERVER[mcp_server.py<br/>FastMCP wrapper]
            STDIO[STDIO Transport]
            STDERR_LOG[stderr-only logging]
        end
        
        subgraph "REST Mode (--mode rest)"
            FASTAPI[FastAPI Server<br/>HTTP transport]
            STDOUT_LOG[standard logging]
        end
    end
    
    subgraph "Shared Business Logic"
        CORE[Core FastAPI App<br/>Routes, Models, Services]
        INGEST[Ingestion Pipeline]
        STORAGE[Vector Storage]
    end
    
    CLI_CLIENT -->|STDIO| CLI_ENTRY
    REST_CLIENT -->|HTTP| CLI_ENTRY
    
    CLI_ENTRY -->|default/--mode mcp| MCP_SERVER
    CLI_ENTRY -->|--mode rest| FASTAPI
    
    MCP_SERVER --> STDIO
    MCP_SERVER --> STDERR_LOG
    FASTAPI --> STDOUT_LOG
    
    MCP_SERVER -->|FastMCP.from_fastapi()| CORE
    FASTAPI --> CORE
    
    CORE --> INGEST
    CORE --> STORAGE
```

## MCP Tool Endpoints

```mermaid
graph TD
    subgraph "Enhanced MCP Tools"
        SEARCH_DOC[search_documentation<br/>Vector similarity search with type filtering<br/>Input: query text, item_type filter<br/>Output: ranked documentation items with see-also suggestions]
        NAV_MOD[navigate_modules<br/>Module hierarchy navigation<br/>Input: crate, path<br/>Output: module tree structure]
        GET_EX[get_examples<br/>Code example retrieval<br/>Input: item_id or query<br/>Output: relevant code examples]
        SEARCH_EX[search_examples<br/>Semantic code example search<br/>Input: query, language filter<br/>Output: scored code examples with deduplication]
        GET_SIG[get_item_signature<br/>Item signature retrieval<br/>Input: item_path<br/>Output: complete signature]
        INGEST_TOOL[ingest_crate<br/>Manual crate ingestion<br/>Input: crate name/version<br/>Output: ingestion status]
        GET_MOD_TREE[get_module_tree<br/>Module hierarchy navigation<br/>Input: crate, version, module_path<br/>Output: hierarchical tree structure]
    end
    
    subgraph "MCP Protocol"
        FASTMCP[FastMCP.from_fastapi()<br/>Automatic REST → MCP conversion<br/>anyOf schema generation]
        STDIO_TRANSPORT[STDIO Transport<br/>JSON-RPC messages]
    end
    
    subgraph "Enhanced REST Endpoints"
        REST_SEARCH_DOC[POST /search_documentation<br/>Enhanced search endpoint with see-also suggestions]
        REST_NAV[POST /navigate_modules<br/>Module navigation endpoint]
        REST_EXAMPLES[POST /get_examples<br/>Example retrieval endpoint]
        REST_SEARCH_EX[POST /search_examples<br/>Code example search endpoint]
        REST_SIG[POST /get_item_signature<br/>Signature endpoint]
        REST_INGEST[POST /ingest<br/>FastAPI endpoint]
        REST_MOD_TREE[POST /get_module_tree<br/>Module tree endpoint]
        HEALTH[GET /health<br/>Liveness probe]
    end
    
    FASTMCP --> SEARCH_DOC
    FASTMCP --> NAV_MOD
    FASTMCP --> GET_EX
    FASTMCP --> GET_SIG
    FASTMCP --> INGEST_TOOL
    FASTMCP --> GET_MOD_TREE
    SEARCH_DOC -->|converts| REST_SEARCH_DOC
    NAV_MOD -->|converts| REST_NAV
    GET_EX -->|converts| REST_EXAMPLES
    SEARCH_EX -->|converts| REST_SEARCH_EX
    GET_SIG -->|converts| REST_SIG
    INGEST_TOOL -->|converts| REST_INGEST
    GET_MOD_TREE -->|converts| REST_MOD_TREE
    STDIO_TRANSPORT --> FASTMCP
```

## Documentation Architecture

```mermaid
graph TD
    subgraph "API Documentation Strategy"
        OPENAPI[FastAPI Automatic OpenAPI<br/>Swagger UI + ReDoc]
        META[Enhanced Metadata<br/>Title, version, description]
        PYDANTIC[Pydantic Models<br/>Automatic schema generation]
    end
    
    subgraph "Documentation Structure"
        README[Single README.md<br/>Comprehensive documentation]
        INLINE[Comprehensive Docstrings<br/>Functions, classes, modules]
        SECURITY[Security Documentation<br/>Integrated into README]
    end
    
    subgraph "Operational Documentation"
        PERF[Performance Metrics<br/>Benchmarks and targets]
        DEPLOY[Deployment Instructions<br/>Multiple environment options]
        API_REF[API Reference<br/>Auto-generated from code]
    end
    
    OPENAPI --> README
    META --> OPENAPI
    PYDANTIC --> OPENAPI
    INLINE --> API_REF
    README --> PERF
    README --> DEPLOY
    README --> SECURITY
```

### Documentation Architecture Decisions

**Single-File Approach**
- Consolidated README.md avoids documentation fragmentation
- Reduces maintenance overhead compared to multi-file documentation systems
- Improves discoverability for developers and operators
- Maintains consistency across installation, usage, and deployment sections

**Auto-Generated API Documentation**
- FastAPI's automatic OpenAPI schema generation eliminates manual API documentation
- Pydantic models provide comprehensive request/response schemas
- Enhanced metadata configuration improves API discoverability
- Swagger UI and ReDoc interfaces generated automatically at `/docs` and `/redoc`

**Integrated Security Documentation**
- Security considerations documented within main README for visibility
- Rate limiting, input validation, and data safety covered comprehensively
- Avoids separate security documents that may become outdated

**Performance and Operational Clarity**
- Documented performance targets and benchmarks for operational planning
- Clear deployment options with resource requirements
- Troubleshooting guidance integrated into main documentation flow

## Rate Limiting Architecture

The system implements comprehensive rate limiting using slowapi (FastAPI port of Flask-Limiter) with in-memory storage for efficient request rate control.

### Rate Limiting Implementation

```mermaid
graph TB
    subgraph "Rate Limiting Middleware"
        MW[middleware.py]
        LIMITER[slowapi.Limiter<br/>In-memory storage]
        HANDLER[rate_limit_handler<br/>HTTP 429 responses]
    end
    
    subgraph "FastAPI Application"
        APP[app.py]
        STATE[app.state.limiter]
        ROUTES[routes.py<br/>@limiter.limit decorators]
        HEALTH[/health endpoint<br/>Exempt from limiting]
    end
    
    subgraph "REST Endpoints"
        SEARCH["/mcp/tools/search_documentation"<br/>30/minute per IP]
        NAV["/mcp/tools/navigate_modules"<br/>30/minute per IP]
        EXAMPLES["/mcp/tools/get_examples"<br/>30/minute per IP]
        SIGNATURE["/mcp/tools/get_item_signature"<br/>30/minute per IP]
        INGEST["/mcp/tools/ingest_crate"<br/>30/minute per IP]
        MODULE_TREE["/mcp/tools/get_module_tree"<br/>30/minute per IP]
    end
    
    subgraph "Error Responses"
        ERROR_429[HTTP 429 Too Many Requests<br/>X-RateLimit headers<br/>JSON error format]
    end
    
    MW --> LIMITER
    MW --> HANDLER
    APP --> STATE
    STATE --> LIMITER
    ROUTES --> LIMITER
    
    ROUTES --> SEARCH
    ROUTES --> NAV
    ROUTES --> EXAMPLES
    ROUTES --> SIGNATURE
    ROUTES --> INGEST
    ROUTES --> MODULE_TREE
    
    LIMITER -.->|Rate exceeded| HANDLER
    HANDLER --> ERROR_429
    
    HEALTH -.->|Bypasses| LIMITER
```

### Rate Limiting Configuration

**Rate Limits**
- **Default Rate**: 30 requests per minute per IP address
- **Applied To**: All REST API endpoints except `/health`
- **Storage**: In-memory (non-persistent, resets on restart)
- **Identification**: Based on client IP address

**HTTP 429 Error Response Format**
```json
{
  "error": "Rate limit exceeded",
  "message": "30 per 1 minute",
  "retry_after": 45
}
```

**Response Headers**
- `X-RateLimit-Limit`: Maximum requests allowed per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when rate limit window resets
- `Retry-After`: Seconds to wait before next request (on 429 only)

### Implementation Details

**Middleware Integration Pattern**
1. **Limiter Creation**: `Limiter` instance created in `middleware.py` with in-memory storage
2. **FastAPI State**: Limiter attached to `app.state.limiter` for application-wide access
3. **Exception Handler**: `rate_limit_handler` registered for `RateLimitExceeded` exceptions
4. **Decorator Application**: `@limiter.limit("30/minute")` applied to REST endpoints
5. **Request Parameter**: `request: Request` parameter required by slowapi for client identification

**Mode-Specific Application**
- **REST Mode (`--mode rest`)**: Rate limiting fully active on all endpoints
- **MCP Mode (default)**: Rate limiting not applied (STDIO transport, no HTTP requests)
- **Health Endpoint**: Exempt from rate limiting for monitoring systems

**Storage and Persistence**
- Uses in-memory storage for request counters (faster than Redis for single-instance deployment)
- Rate limit counters reset on application restart
- No database or external dependencies required
- Suitable for single-instance deployments

### Error Handling Flow Integration

The rate limiting integrates into the existing error handling flow:

```mermaid
stateDiagram-v2
    [*] --> Request
    Request --> Validation
    
    Validation --> RateLimit: Valid
    Validation --> Error400: Invalid
    
    RateLimit --> Processing: Under limit
    RateLimit --> Error429: Over limit (NEW)
    
    Processing --> CacheCheck
    CacheCheck --> CacheHit: Found
    CacheCheck --> Ingest: Not found
    
    CacheHit --> Success
    
    Error400 --> [*]
    Error429 --> [*]
    Success --> [*]
```

**Integration Benefits**
- **Protection**: Prevents API abuse and DoS attacks
- **Resource Conservation**: Limits resource consumption per client
- **Fair Usage**: Ensures equitable access across multiple clients
- **Monitoring Integration**: Compatible with health check systems
- **Zero Configuration**: Works out-of-the-box with sensible defaults

## Technology Stack

```mermaid
mindmap
  root((docsrs-mcp))
    Python Infrastructure
      uv (exclusive package manager)
      Python 3.10+
      pyproject.toml config
      uv.lock lockfile
      
    Code Quality
      Ruff (linting & formatting)
      Replaces black/flake8/isort/pylint
      Single Rust-based tool
      pyproject.toml configuration
      
    Runtime
      uvicorn[standard]
      uvloop (event loop)
      httptools (HTTP parser)
    
    Web Framework
      FastAPI
      Pydantic (validation)
      slowapi (rate limiting)
      OpenAPI auto-generation
      
    MCP Integration
      FastMCP (REST to MCP conversion)
      STDIO transport
      JSON-RPC protocol
      stderr-only logging
      
    Storage
      SQLite
      sqlite-vec (vector extension)
      aiosqlite (async SQLite)
      File-based cache (./cache)
      
    ML/Embedding
      FastEmbed
      ONNX Runtime
      BAAI/bge-small-en-v1.5
      RapidFuzz (fuzzy string matching)
      
    HTTP Client
      aiohttp 3.9.5 (pinned for memory leak fix)
      orjson (JSON parsing)
      crates.io + docs.rs API integration
      Standard library crate detection and channel resolution
      
    JSON Processing
      ijson (streaming parser)
      Memory-efficient large file handling
      
    Compression
      zstandard (zst decompression)
      gzip (gz decompression)
      Size limits and streaming
      
    Deployment
      uvx (zero-install)
      uv build (packaging)
      Docker (optional)
      PyPI distribution
      
    Documentation
      FastAPI OpenAPI generation
      Comprehensive docstrings
      Single README.md approach
      Integrated security docs
```

## Error Handling Flow

```mermaid
stateDiagram-v2
    [*] --> Request
    Request --> Validation
    
    Validation --> RateLimit: Valid
    Validation --> Error400: Invalid
    
    RateLimit --> Processing: Under limit
    RateLimit --> Error429: Over limit
    
    Processing --> CacheCheck
    CacheCheck --> CacheHit: Found
    CacheCheck --> Ingest: Not found
    
    CacheHit --> Success
    
    Ingest --> VersionResolve
    VersionResolve --> Download: Resolved
    VersionResolve --> ChannelFallback: Stdlib channel unavailable
    VersionResolve --> Error404: Not found
    VersionResolve --> Error410: Yanked
    
    ChannelFallback --> Download: Fallback channel found
    ChannelFallback --> Error404: All channels unavailable
    
    Download --> FormatCheck: Success
    FormatCheck --> TryZst: Check .json.zst
    TryZst --> TryGz: 404 Not Found
    TryZst --> Decompress: Found
    TryGz --> TryJson: 404 Not Found  
    TryGz --> Decompress: Found
    TryJson --> Decompress: Found
    TryJson --> Error404: All formats 404
    Download --> Error504: Timeout
    
    Decompress --> Chunk
    Chunk --> Embed
    Embed --> Store
    Store --> Success
    
    Error400 --> [*]
    Error404 --> [*]
    Error410 --> [*]
    Error429 --> [*]
    Error504 --> [*]
    Success --> [*]
```

## Deployment Architecture

```mermaid
graph TB
    subgraph "Development (uv-native)"
        DEV[uv sync --dev<br/>uv run python -m docsrs_mcp.cli<br/>(MCP mode default)]
        TEST[uvx --from . docsrs-mcp<br/>uvx --from . docsrs-mcp --mode rest]
    end
    
    subgraph "Production Options"
        subgraph "Container (uv-based)"
            DOCKER[Docker Container<br/>FROM python:slim<br/>RUN pip install uv<br/>COPY . .<br/>RUN uv sync --frozen]
        end
        
        subgraph "PaaS"
            FLY[Fly.io]
            RAIL[Railway]
            RENDER[Render]
        end
        
        subgraph "VPS"
            VPS[Any VPS<br/>≥256 MiB RAM<br/>uv-managed]
        end
    end
    
    subgraph "Persistent Storage"
        VOL[Volume Mount<br/>./cache]
    end
    
    DEV --> TEST
    TEST --> DOCKER
    DOCKER --> FLY
    DOCKER --> RAIL
    DOCKER --> RENDER
    DOCKER --> VPS
    
    FLY --> VOL
    RAIL --> VOL
    RENDER --> VOL
    VPS --> VOL
```

## CI/CD Pipeline Architecture

The docsrs-mcp project uses a comprehensive CI/CD pipeline with GitHub Actions that ensures code quality, cross-platform compatibility, and automated releases.

### Pipeline Overview

```mermaid
graph TB
    subgraph "CI Workflow (ci.yml)"
        TRIGGER1[Pull Request<br/>Push to main]
        
        subgraph "Quality Gates"
            LINT[Lint Job<br/>ruff check & format]
            TEST[Test Matrix<br/>3 OS × 3 Python versions]
            INTEGRATION[Integration Job<br/>Server startup & health check]
        end
    end
    
    subgraph "Release Workflow (release.yml)"
        TRIGGER2[Git Tag v*.*.*]
        
        subgraph "Release Jobs"
            BUILD[Build Job<br/>uv build --no-sources]
            PUBLISH[Publish Job<br/>PyPI with OIDC]
        end
    end
    
    TRIGGER1 --> LINT
    TRIGGER1 --> TEST
    TRIGGER1 --> INTEGRATION
    
    TRIGGER2 --> BUILD
    BUILD --> PUBLISH
    
    style LINT fill:#e1f5fe
    style TEST fill:#f3e5f5
    style INTEGRATION fill:#e8f5e8
    style BUILD fill:#fff3e0
    style PUBLISH fill:#ffebee
```

### CI Workflow Structure

The CI pipeline (`ci.yml`) runs on pull requests and pushes to main, featuring three parallel jobs:

#### 1. Lint Job
- **Environment**: Ubuntu Latest
- **Tools**: 
  - `astral-sh/setup-uv@v6` with built-in caching
  - Ruff for linting and formatting
- **Steps**:
  - Code quality check: `uv run ruff check .`
  - Format verification: `uv run ruff format --check .`

#### 2. Test Matrix Job
- **Cross-Platform Testing**: 
  - **Operating Systems**: Ubuntu 22.04, macOS 14, Windows 2022
  - **Python Versions**: 3.10, 3.11, 3.12
  - **Total Combinations**: 9 test environments
- **Performance Optimization**: 
  - UV caching enabled across all matrix combinations
  - `uv cache prune --ci` for optimized CI storage
- **Test Execution**: `uv run pytest -q` for concise output

#### 3. Integration Job
- **Environment**: Ubuntu Latest
- **Functionality**:
  - CLI installation test: `uvx --from . docsrs-mcp --help`
  - Server startup validation with health check endpoint
  - Background process management with PID capture
  - Automated cleanup and error reporting

### Release Workflow Structure

The release pipeline (`release.yml`) is triggered by semantic version tags and consists of two sequential jobs:

#### 1. Build Job
- **Artifact Creation**: `uv build --no-sources`
- **Storage**: GitHub Actions artifacts with `actions/upload-artifact@v4`
- **Optimization**: UV caching for dependency resolution

#### 2. Publish Job
- **Security**: OIDC trusted publishing (no manual tokens)
- **Environment**: Protected `pypi` environment
- **Features**:
  - Automated PyPI publishing via `pypa/gh-action-pypi-publish@release/v1`
  - Attestation generation for supply chain security
  - Artifact download from build job

### Key Architecture Features

#### UV-First Infrastructure
- **Consistent Tooling**: All jobs use `astral-sh/setup-uv@v6`
- **Built-in Caching**: Automatic dependency and tool caching
- **Performance**: Faster dependency resolution compared to pip
- **Reliability**: Locked dependency versions with `uv sync`

#### Security Model
- **OIDC Publishing**: No long-lived PyPI tokens required
- **Minimal Permissions**: `contents: read` for CI, `id-token: write` for releases
- **Protected Environments**: PyPI publishing requires environment approval
- **Supply Chain Security**: Attestations for published packages

#### Testing Strategy
- **Matrix Coverage**: Comprehensive OS and Python version combinations
- **Integration Validation**: Real server startup and health check testing
- **Performance Testing**: Background process management prevents CI hangs
- **Error Visibility**: Comprehensive logging and failure reporting

### Performance Characteristics

- **CI Duration**: ~5-10 minutes for full matrix (parallel execution)
- **Release Duration**: ~2-3 minutes (sequential build → publish)
- **Cache Efficiency**: UV caching reduces dependency installation time by ~60%
- **Resource Usage**: Optimized for GitHub Actions resource limits

### Deployment Integration

The CI/CD pipeline seamlessly integrates with the deployment architecture:

1. **Development**: Local testing with `uv run` matches CI environment
2. **Staging**: Integration tests validate server functionality
3. **Production**: Automated PyPI releases enable immediate deployment
4. **Rollback**: Git tag-based versioning supports easy rollbacks

This architecture ensures reliable, secure, and efficient delivery of docsrs-mcp updates while maintaining high code quality standards across all supported platforms.

## Critical Bug Analysis and Fixes

### Character Fragmentation Bug in searchExamples

**Location**: `ingest.py:761-769` in `searchExamples` ingestion pipeline

**Critical Bug Description**:
The searchExamples function had a critical string iteration bug that treated string data as individual characters instead of code examples, causing severe data corruption during ingestion.

```python
# BUG: When examples_data is a string, for-loop iterates over characters
for example in examples_data:  # If examples_data="example code", iterates 'e','x','a','m'...
    if isinstance(example, str):
        example = {"code": example, "language": "rust", "confidence": None, "detection_method": "fallback"}
```

**Root Cause Analysis**:
- String objects are iterable in Python, causing `for example in "string"` to yield individual characters
- No type validation before iteration led to silent data corruption
- Each character was processed as a separate "code example" with full embedding generation
- Resulted in thousands of single-character embeddings polluting the example search index

**Fix Implementation**:
```python
# FIX: Type validation before iteration prevents character fragmentation
if isinstance(examples_data, str):
    examples_data = [examples_data]  # Wrap single string in list
    
for example in examples_data:  # Now safely iterates over list elements
    if isinstance(example, str):
        example = {"code": example, "language": "rust", "confidence": None, "detection_method": "fallback"}
```

**Impact Assessment**:
- **Pre-Fix**: Generated ~10,000+ meaningless single-character embeddings per crate
- **Post-Fix**: Correct example processing with semantic code blocks
- **Performance Impact**: 95% reduction in example embedding storage
- **Search Quality**: Dramatic improvement in example search relevance

### MCP Parameter Validation Enhancement

**Location**: `app.py:151-297` in FastAPI route parameter handling

**Issue Description**:
MCP manifest schema lacked `anyOf` patterns for numeric parameters, causing type coercion failures when MCP clients send string representations of numbers.

**Technical Problem**:
- Pydantic models include automatic string-to-number coercion
- MCP manifest only declared parameters as `{"type": "integer"}` or `{"type": "number"}`
- MCP clients sending `"k": "5"` (string) were rejected despite valid coercion capability

**Solution Implementation**:
```json
// Updated MCP manifest patterns
{
  "k": {
    "anyOf": [
      {"type": "integer"},
      {"type": "string", "pattern": "^[0-9]+$"}
    ],
    "description": "Number of results to return"
  },
  "limit": {
    "anyOf": [
      {"type": "integer"},
      {"type": "string", "pattern": "^[0-9]+$"}
    ],
    "description": "Maximum results limit"
  }
}
```

**Applied To Parameters**:
- `k` (result count)
- `limit` (result limit)
- `offset` (pagination offset)
- All other numeric parameters in MCP tool definitions

### Path Resolution Enhancement Architecture

**New Component**: Enhanced path alias resolution system integrated into fuzzy matching pipeline.

```mermaid
graph TD
    subgraph "Enhanced Path Resolution Flow"
        REQ[get_item_doc Request<br/>item_path parameter]
        EXACT[Exact Path Match<br/>SQLite query]
        ALIAS_CHECK[PATH_ALIASES Check<br/>Common pattern mapping]
        ALIAS_MATCH[Direct Alias Match<br/>Return aliased path]
        CACHE_CHECK[Path Cache Check<br/>5-minute TTL]
        FUZZY[RapidFuzz Processing<br/>Enhanced with alias awareness]
        SUGGESTIONS[Top 3 Suggestions<br/>Include alias matches]
    end
    
    REQ --> EXACT
    EXACT -->|Found| SUCCESS[Return Documentation]
    EXACT -->|Not Found| ALIAS_CHECK
    ALIAS_CHECK -->|Alias Found| ALIAS_MATCH
    ALIAS_MATCH --> EXACT
    ALIAS_CHECK -->|No Alias| CACHE_CHECK
    CACHE_CHECK --> FUZZY
    FUZZY --> SUGGESTIONS
```

**PATH_ALIASES Dictionary Implementation**:
```python
# Static dictionary in fuzzy_resolver.py with 16 common aliases
PATH_ALIASES = {
    # serde aliases
    "serde::Serialize": "serde::ser::Serialize",
    "serde::Deserialize": "serde::de::Deserialize",
    "serde::Serializer": "serde::ser::Serializer",
    "serde::Deserializer": "serde::de::Deserializer",
    # tokio aliases
    "tokio::spawn": "tokio::task::spawn",
    "tokio::JoinHandle": "tokio::task::JoinHandle",
    "tokio::select": "tokio::macros::select",
    # std aliases
    "std::HashMap": "std::collections::HashMap",
    "std::HashSet": "std::collections::HashSet",
    "std::BTreeMap": "std::collections::BTreeMap",
    "std::BTreeSet": "std::collections::BTreeSet",
    "std::VecDeque": "std::collections::VecDeque",
    "std::Vec": "std::vec::Vec",
    "std::Result": "std::result::Result",
    "std::Option": "std::option::Option"
}
```

**Implementation Details**:
- **Static Dictionary**: Implemented in fuzzy_resolver.py with 16 common aliases
- **O(1) Lookup**: Direct dictionary access before database query in get_item_doc endpoint
- **Crate Support**: Covers serde, tokio, and std library aliases
- **Resolution Types**: Both crate-prefixed and direct alias resolution
- **Performance**: <1ms overhead for alias resolution

**Enhancement Benefits**:
- **Immediate Resolution**: Common aliases resolve without fuzzy matching overhead
- **User Experience**: Handles typical import path variations automatically
- **Performance**: O(1) alias lookup before O(n) fuzzy matching
- **Extensible**: Alias dictionary can be expanded based on usage patterns

## System Components

### Ingestion Layer Details

**Version Resolution System**
- Uses docs.rs redirect mechanism to resolve version strings
- Supports "latest" and specific version identifiers
- Handles version disambiguation and canonicalization
- Constructs proper rustdoc JSON URLs with crate name transformations
- **Standard Library Support**: Detects standard library crates using STDLIB_CRATES set for special handling
- **Channel Resolution**: Maps Rust version channels (stable/beta/nightly) to appropriate documentation versions
- **Stdlib URL Construction**: Builds standard library documentation URLs from docs.rs using detected channel information

**Compression Support**
- **Zstandard (.json.zst)**: Primary format, best compression ratio
- **Gzip (.json.gz)**: Secondary format, universal support
- **Uncompressed (.json)**: Fallback format for compatibility
- Streaming decompression with configurable memory limits
- Automatic format detection and selection

**Per-Crate Locking Mechanism**
- Global asyncio.Lock registry indexed by crate@version
- Prevents duplicate ingestion across concurrent requests
- Maintains lock state throughout application lifetime
- Ensures data consistency during parallel processing
- **Standard Library Integration**: Applies same locking mechanism to stdlib crates (std, core, alloc, etc.)
- **Channel-Aware Locking**: Locks consider Rust channel versions to prevent conflicts between stable/beta/nightly docs

**Memory-Optimized Streaming Parsing with Metadata Extraction**
- **Streaming JSON Processing**: ijson event-based parser processes large files progressively
- **Generator-Based Architecture**: parse_rustdoc_items_streaming() yields items on-demand instead of collecting in memory
- **Memory-Aware Processing**: Adaptive batch sizing (16-512 items) based on memory pressure thresholds
- **Progressive Parsing Strategy**:
  - First pass: Stream paths mapping extraction without full memory load
  - Second pass: Progressive item extraction with immediate processing
  - Hierarchy building: On-demand parent-child relationship resolution
- **Enhanced Metadata Extraction Pipeline**:
  - Type normalization helper functions for consistent item classification
  - Signature extraction with full type information and generics
  - Parent ID resolution for module hierarchy relationships
  - Code example extraction from documentation comments
  - Trait implementation parsing with extract_type_name() helper for rustdoc Type objects
- **Memory Management Features**:
  - psutil-based memory monitoring with 80%/90% thresholds
  - Garbage collection triggers at chunk boundaries during high memory usage
  - Dynamic batch size adjustment (16-512 items) based on available memory
  - Backwards compatibility wrappers maintain existing API contracts
- **Performance Characteristics**:
  - Memory usage: O(1) instead of O(n) for large files
  - Processing overhead: ~10-15% for enhanced metadata extraction
  - Memory efficiency: Processes files >1GB without memory exhaustion

**LRU Cache Eviction**
- File modification time (mtime) based eviction strategy
- Configurable size limits (default 2GB total cache)
- Automatic cleanup when cache size exceeds limits
- Preserves most recently accessed crate documentation

### Standard Library Support Architecture

**Standard Library Crate Detection**
- Uses predefined STDLIB_CRATES set containing core standard library crates: `{'std', 'core', 'alloc', 'proc_macro', 'test'}`
- Enables special handling for Rust's built-in crates that require different URL construction
- Integrates seamlessly with existing crate detection pipeline with minimal code changes

**Rust Channel Version Resolution**
- **Channel Detection**: Identifies Rust version channels (stable, beta, nightly) from version strings
- **Version Mapping**: Maps channel identifiers to appropriate documentation versions on docs.rs
- **Fallback Strategy**: Defaults to stable channel when version cannot be determined
- **URL Construction**: Builds channel-specific URLs for standard library documentation access

**Integration with Existing Pipeline**
- **95% Code Reuse**: Leverages existing ingestion, parsing, and storage infrastructure unchanged
- **Minimal Architecture Changes**: Standard library support adds detection logic without modifying core data flow
- **Same Storage Schema**: Standard library documentation uses identical database structure as regular crates
- **Unified Caching**: Standard library docs cached using same LRU eviction strategy and size limits
- **Consistent API**: No changes required to MCP tools or REST endpoints for standard library support

**Fallback Mechanisms**
- **Format Fallback**: Attempts .json.zst → .json.gz → .json formats same as regular crates
- **Channel Fallback**: Falls back from nightly → beta → stable if specific channel documentation unavailable
- **Error Handling**: Graceful degradation when standard library documentation cannot be retrieved
- **Cache Resilience**: Maintains cached standard library docs even when upstream docs.rs is unavailable

## Filter Optimization Architecture

### Progressive Filtering with Selectivity Analysis

The filter optimization system implements intelligent query planning with selectivity-based optimization to maximize search performance:

```mermaid
graph TB
    subgraph "Filter Analysis Pipeline"
        QUERY[Incoming Query<br/>with filters]
        ANALYZE[Selectivity Analyzer<br/>Estimates filter cardinality]
        ORDER[Filter Ordering<br/>Most selective first]
        EXECUTE[Progressive Execution<br/>Early termination on selectivity]
    end
    
    subgraph "Selectivity Metrics"
        DEPRECATED[Deprecated Filter<br/>~5% selectivity]
        TYPE[Type Filter<br/>Functions ~40%, Traits ~5%]
        EXAMPLES[Has Examples<br/>~30% selectivity]
        CRATE[Crate Prefix<br/>Highly selective ~1%]
    end
    
    subgraph "Partial Index Strategy"
        IDX_DEPRECATED[idx_non_deprecated<br/>Non-deprecated items only]
        IDX_FUNCTIONS[idx_public_functions<br/>Public functions only]
        IDX_EXAMPLES[idx_has_examples<br/>Items with examples]
        IDX_CRATE[idx_crate_prefix<br/>Crate-specific searches]
    end
    
    QUERY --> ANALYZE
    ANALYZE --> ORDER
    ORDER --> EXECUTE
    
    ANALYZE --> DEPRECATED
    ANALYZE --> TYPE
    ANALYZE --> EXAMPLES
    ANALYZE --> CRATE
    
    EXECUTE --> IDX_DEPRECATED
    EXECUTE --> IDX_FUNCTIONS
    EXECUTE --> IDX_EXAMPLES
    EXECUTE --> IDX_CRATE
```

### Enhanced Validation with MCP Compatibility

The filter validation system provides comprehensive parameter validation while maintaining MCP client compatibility. This includes query preprocessing for search consistency:

```mermaid
graph LR
    subgraph "Filter Validation Pipeline"
        INPUT[Filter Parameters]
        VALIDATE[Pydantic Validation<br/>Type coercion & constraints]
        OPTIMIZE[Query Optimizer<br/>Selectivity analysis]
        EXECUTE[Optimized Execution<br/>Progressive filtering]
    end
    
    subgraph "MCP Parameter Handling"
        STRING_PARAMS[String Parameters<br/>from MCP clients]
        TYPE_COERCE[Field Validators<br/>mode='before']
        QUERY_PREPROCESS[Query Preprocessing<br/>Unicode normalization NFKC]
        CONSTRAINT_CHECK[Constraint Validation<br/>enum values, ranges]
    end
    
    STRING_PARAMS --> TYPE_COERCE
    TYPE_COERCE --> QUERY_PREPROCESS
    QUERY_PREPROCESS --> INPUT
    INPUT --> VALIDATE
    VALIDATE --> OPTIMIZE
    OPTIMIZE --> EXECUTE
    CONSTRAINT_CHECK --> VALIDATE
```

### Performance Timing and Metrics Collection

The system includes comprehensive performance monitoring for filter operations:

```mermaid
graph TD
    subgraph "Performance Metrics"
        TIMING[Query Timing<br/>Per-filter latency]
        SELECTIVITY[Selectivity Metrics<br/>Filter effectiveness]
        INDEX_USAGE[Index Usage Stats<br/>Partial index efficiency]
        CACHE_HITS[Cache Performance<br/>Filter cache hit rates]
    end
    
    subgraph "Monitoring Points"
        PRE_FILTER[Pre-filter Query Time]
        FILTER_APPLY[Filter Application Time]
        POST_FILTER[Post-filter Processing Time]
        TOTAL_LATENCY[Total Query Latency]
    end
    
    TIMING --> PRE_FILTER
    TIMING --> FILTER_APPLY
    TIMING --> POST_FILTER
    TIMING --> TOTAL_LATENCY
    
    SELECTIVITY --> INDEX_USAGE
    INDEX_USAGE --> CACHE_HITS
```

### Performance Characteristics for Filter Operations

| Filter Type | Selectivity | Index Used | Target Latency | Notes |
|-------------|-------------|------------|----------------|---------|
| Deprecated filter | ~5% | idx_non_deprecated | < 5ms | Most selective, applied first |
| Crate prefix | ~1% | idx_crate_prefix | < 3ms | Highly selective for single crate |
| Module path LIKE | Variable | item_path index | < 50ms | Selective for module patterns (e.g., "tokio::runtime::%") |
| Item type = function | ~40% | idx_public_functions | < 10ms | Medium selectivity |
| Has examples | ~30% | idx_has_examples | < 8ms | Medium selectivity |
| Combined filters | Variable | Multiple indexes | < 15ms | Progressive application |
| Vector search + filters | N/A | Composite strategy | < 100ms P95 | Maintains search SLA |

## Search Ranking Architecture

### Multi-Factor Scoring Algorithm

The search ranking system implements a sophisticated multi-factor scoring algorithm that combines multiple relevance signals to deliver highly relevant results, enhanced with see-also suggestions for improved discovery:

#### See-Also Suggestions Feature

The system provides intelligent see-also suggestions using the same vector embedding infrastructure to find semantically related items that complement the main search results.

**Architecture**:
- **Function**: `get_see_also_suggestions()` in database.py
- **Integration**: Computed alongside main search in search_items endpoint  
- **Reuse Strategy**: Leverages existing search_embeddings logic and query embeddings
- **Threshold**: 0.7 similarity threshold for high-quality suggestions
- **Limit**: Maximum 5 suggestions to avoid overwhelming users
- **Exclusion**: Automatically excludes items already in main search results
- **Fail-Safe**: Graceful degradation - suggestions never break main search functionality
- **Performance**: Uses same sqlite-vec MATCH operator with k=10 over-fetch for filtering

**SearchResult Model Enhancement**:
- **Field**: `suggestions: list[str] | None` - optional field for backward compatibility
- **Population**: Only first result includes suggestions to avoid redundancy
- **Content**: List of item_path strings for related documentation items

**Query Processing Flow**:
```mermaid
graph LR
    subgraph "Search with Suggestions"
        QUERY[User Query]
        EMBED[Generate Embedding]
        MAIN_SEARCH[Main Vector Search]
        SUGGESTIONS[Get See-Also Suggestions]
        EXCLUDE[Exclude Main Results]
        RESPONSE[Enhanced SearchResult]
    end
    
    QUERY --> EMBED
    EMBED --> MAIN_SEARCH
    EMBED --> SUGGESTIONS
    MAIN_SEARCH --> EXCLUDE
    EXCLUDE --> SUGGESTIONS
    SUGGESTIONS --> RESPONSE
    MAIN_SEARCH --> RESPONSE
```

```mermaid
graph TB
    subgraph "Scoring Components"
        VECTOR[Vector Similarity<br/>60% weight<br/>BAAI/bge-small-en-v1.5<br/>Reduced from 70% to 60%]
        TYPE[Type-Aware Boost<br/>15% weight<br/>Functions 1.2x, Traits 1.15x]
        QUALITY[Documentation Quality<br/>10% weight<br/>Length + examples heuristics]
        EXAMPLES[Example Presence<br/>15% weight<br/>Code example availability<br/>Enhanced from 5% to 15%]
    end
    
    subgraph "Type Weights"
        FUNC[Functions: 1.2x]
        TRAIT[Traits: 1.15x]
        STRUCT[Structs: 1.1x]
        TRAIT_IMPL[Trait Impls: 1.1x]
        MODULE[Modules: 0.9x]
    end
    
    subgraph "Score Processing"
        COMBINE[Weighted Combination<br/>Score = (V×0.60) + (T×0.15) + (Q×0.10) + (E×0.15)]
        NORMALIZE[Score Normalization<br/>Min-Max to [0, 1] range]
        RANK[Final Ranking<br/>Sorted by normalized score]
    end
    
    VECTOR --> COMBINE
    TYPE --> COMBINE
    QUALITY --> COMBINE
    EXAMPLES --> COMBINE
    
    FUNC --> TYPE
    TRAIT --> TYPE
    STRUCT --> TYPE
    MODULE --> TYPE
    
    COMBINE --> NORMALIZE
    NORMALIZE --> RANK
```

### Query Preprocessing Pipeline

The query preprocessing system ensures consistent search results through Unicode normalization and supports module path filtering:

```mermaid
graph LR
    subgraph "Query Preprocessing"
        RAW_QUERY[Raw Query Text<br/>from client request]
        UNICODE_NORM[Unicode Normalization<br/>NFKC form application]
        MODULE_PARAM[Module Path Parameter<br/>SQL LIKE pattern filtering]
        NORMALIZED[Normalized Query<br/>consistent character forms]
    end
    
    subgraph "Cache Integration"
        CACHE_KEY[Cache Key Generation<br/>query + module_path parameters]
        CACHE_LOOKUP[Cache Lookup<br/>improved hit rates]
        SEARCH_EXECUTION[Vector Search<br/>with module path filtering]
    end
    
    RAW_QUERY --> UNICODE_NORM
    MODULE_PARAM --> NORMALIZED
    UNICODE_NORM --> NORMALIZED
    NORMALIZED --> CACHE_KEY
    CACHE_KEY --> CACHE_LOOKUP
    NORMALIZED --> SEARCH_EXECUTION
```

**Key Benefits:**
- **Search Consistency**: NFKC normalization handles character variants (e.g., combining diacritics, fullwidth characters)
- **Module Path Filtering**: SQL LIKE patterns enable precise module-scoped searches (e.g., "tokio::runtime::%")
- **Cache Efficiency**: Normalized queries + module path parameters improve cache hit rates by eliminating variations
- **Cross-Platform Compatibility**: Consistent results across different operating systems and input methods
- **Transparent Processing**: Normalization and filtering happen at validation layer with no API changes
- **Performance Impact**: Minimal overhead (~1ms) for query normalization, <50ms for module filtering using existing indexes

### Caching Layer with TTL Support

The enhanced caching system provides intelligent result caching with time-to-live (TTL) support and benefits from query preprocessing with module path awareness:

```mermaid
graph LR
    subgraph "Cache Architecture"
        LRU[LRU Cache<br/>1000 entry capacity]
        TTL[TTL Support<br/>15-minute default]
        KEY_GEN[Cache Key Generation<br/>Query + module_path + parameters]
    end
    
    subgraph "Cache Operations"
        LOOKUP[Cache Lookup<br/>O(1) hash lookup]
        STORE[Cache Store<br/>Evict oldest if full]
        EXPIRE[TTL Expiration<br/>Background cleanup]
    end
    
    subgraph "Performance Monitoring"
        LATENCY[Latency Tracking<br/>Cache hit/miss timing]
        STATS[Hit Rate Statistics<br/>Cache effectiveness metrics]
        LOG[Score Distribution<br/>Ranking validation]
    end
    
    KEY_GEN --> LOOKUP
    LOOKUP --> LRU
    LOOKUP --> TTL
    STORE --> LRU
    EXPIRE --> TTL
    
    LOOKUP --> LATENCY
    LATENCY --> STATS
    STATS --> LOG
```

### Performance Optimizations

**K+10 Over-fetching Strategy**
- Fetches k+10 results from vector search to ensure sufficient candidates for re-ranking
- Prevents constraining initial retrieval while allowing sophisticated ranking
- Balances retrieval recall with ranking precision

**Score Validation and Monitoring**
- Validates score distributions to detect ranking anomalies
- Logs performance metrics for continuous optimization
- Tracks latency across cache hits, misses, and ranking operations

## Performance Characteristics

| Component | Target | Notes |
|-----------|--------|-------|
| Search latency | < 100ms P95 | Vector search with sqlite-vec MATCH + multi-factor ranking + progressive filtering + see-also suggestions |
| **See-also suggestions** | **< 15ms P95** | **Vector similarity search with 0.7 threshold, k=10 over-fetch, excludes main results** |
| Query preprocessing | < 1ms | Unicode NFKC normalization for search consistency |
| Ranking overhead | < 20ms P95 | Multi-factor scoring with type weights and normalization |
| Filter latency | < 15ms P95 | Progressive filtering with selectivity analysis and partial indexes |
| Module path filter | < 50ms P95 | SQL LIKE pattern matching leveraging existing item_path indexes |
| Deprecated filter | < 5ms P95 | High selectivity (~5%) with idx_non_deprecated partial index |
| Type filter | < 10ms P95 | Medium selectivity (~40% for functions) with idx_public_functions |
| Example filter | < 8ms P95 | Medium selectivity (~30%) with idx_has_examples partial index |
| Crate prefix filter | < 3ms P95 | High selectivity (~1%) with idx_crate_prefix optimization |
| **Example embedding generation** | **16-item batches** | **CPU-optimized batch size for memory efficiency** |
| **Example search latency** | **150-360ms warm, 1.4s cold** | **Dedicated example_embeddings search with model loading** |
| **Example memory usage** | **~421MB for tokio** | **Memory usage during example embedding generation** |
| **Deduplication rate** | **~30% reduction** | **SHA256-based deduplication effectiveness** |
| **Code example search** | **< 100ms P95** | **Vector search + language filtering + hash-based deduplication** |
| **Language detection** | **< 2ms per block** | **Pygments lexer analysis with 30% confidence threshold** |
| **Example extraction** | **< 10ms per doc** | **Enhanced JSON structure processing overhead** |
| **Rate limiting overhead** | **< 1ms P95** | **In-memory counter check and update with slowapi** |
| **Rate limit storage** | **O(active IPs)** | **In-memory storage, resets on restart** |
| **Parameter validation** | **< 2ms P95** | **Centralized validation with precompiled regex patterns** |
| **Type coercion overhead** | **< 0.5ms P95** | **String-to-type conversion for MCP client compatibility** |
| Cache hit latency | < 5ms P95 | LRU cache with TTL support |
| Cache miss latency | < 100ms P95 | Full search + ranking + cache store |
| Ingest latency | < 30s | Full rustdoc processing with streaming |
| Memory usage | < 512 MiB RSS | Streaming architecture with memory monitoring |
| Memory monitoring | psutil-based | 80%/90% thresholds with adaptive processing |
| Cache storage | ./cache directory | File-based, LRU eviction with TTL |
| Cache capacity | 1000 entries | TTL-based expiration, 15-minute default |
| Embedding model | BAAI/bge-small-en-v1.5 | 384 dimensions, adaptive batch processing |
| Async architecture | aiosqlite + asyncio | Non-blocking I/O with per-crate locks |
| Compression ratio | ~10:1 typical | .zst format for bandwidth efficiency |
| Batch sizes | 16-512 adaptive, 999 DB | Memory-aware adaptive sizing |
| Streaming memory | O(1) for large files | Generator-based progressive processing |
| Over-fetching ratio | k+10 results | Ensures sufficient candidates for re-ranking |
| Module tree retrieval | < 50ms P95 | Recursive CTE-based tree traversal |
| Module hierarchy build | < 200ms | Path analysis and parent resolution during ingestion |
| Hierarchy storage | +10% schema size | modules table with indexes on parent_id/depth |

## Performance Optimization Insights

### Critical Performance Enhancements

**sqlite-vec Optimization Strategy**
- **Chunk-Based Storage**: Use 1000-item chunks instead of individual embeddings for 40% insertion speed improvement
- **Vector Quantization**: Implement 8-bit quantization for 75% memory reduction with minimal search quality loss
- **Batch Operations**: Use `vss_index!` bulk operations instead of individual inserts for 3x throughput
- **Index Timing**: Build indexes after bulk insertion, not incrementally, for 60% faster ingestion

**RapidFuzz Performance Characteristics**
- **Speed Advantage**: 10-50x faster than FuzzyWuzzy for path matching operations
- **Bulk Processing**: Use `process.cdist()` for batch fuzzy matching instead of loops
- **Memory Efficiency**: Process 10,000+ paths with <50MB memory overhead
- **Optimization**: Pre-compile scoring options for 15% performance improvement

**FastEmbed Production Optimizations**
- **ONNX Runtime**: CPU inference with optimized thread pool configuration
- **Batch Processing**: Optimal batch_size=16 for CPU workloads balances memory/throughput
- **Session Caching**: Reuse ONNX sessions across requests for 300ms startup elimination
- **Memory Management**: Clear tensor cache every 1000 embeddings to prevent memory leaks

**Tree-sitter Future Enhancement**
- **Parsing Speed**: 36x faster than regex for structured code extraction
- **Language Support**: Native Rust, Python, JavaScript, TypeScript parsers available  
- **Memory Efficiency**: Streaming AST parsing with O(1) memory footprint
- **Integration Path**: Replace regex-based code extraction in future versions

### Architecture Pattern Optimizations

**Streaming Parser Efficiency**
- **ijson Generators**: O(1) memory usage for >1GB JSON files using yield-based processing
- **Progressive Filtering**: Apply filters during parsing, not after, reducing intermediate storage by 80%
- **Memory Monitoring**: psutil-based garbage collection triggers prevent OOM conditions
- **Chunked Processing**: 16-512 adaptive batch sizing based on available system memory

**Per-Crate Locking Benefits**
- **Concurrency**: Prevents duplicate work across simultaneous requests for same crate
- **Resource Efficiency**: Eliminates redundant downloads and parsing operations
- **Cache Coherence**: Ensures consistent database state during concurrent ingestion
- **Deadlock Prevention**: Lock ordering by crate@version string prevents circular dependencies

**Composite Scoring Performance**
- **Weight Distribution**: vector(70%) + type(15%) + quality(10%) + examples(5%) optimized through A/B testing
- **Score Normalization**: Min-max scaling to [0,1] prevents component dominance
- **Type Boosting**: Function 1.2x, trait 1.15x, struct 1.1x multipliers based on usage analysis
- **Early Termination**: Stop scoring when confidence threshold exceeded for top-k results

**Progressive Filtering Strategy**
- **Selectivity Analysis**: Apply most selective filters first (deprecated ~5%, crate ~1%, type ~40%)
- **Partial Indexes**: Specialized indexes for common filters reduce scan time by 95%
- **Filter Combination**: Short-circuit evaluation prevents unnecessary filter application
- **Query Planning**: Dynamic filter ordering based on query characteristics and database statistics

### Memory Management Architecture

**Streaming Processing Patterns**
```python
# Generator-based item processing (O(1) memory)
def process_items_streaming(items_generator):
    for batch in chunk_generator(items_generator, size=512):
        yield embed_batch(batch)  # Process and release immediately
        gc.collect() if memory_pressure() > 0.9 else None
```

**Vector Storage Optimization**
```sql
-- Quantized vector storage (75% memory reduction)
CREATE TABLE embeddings_quantized (
    id INTEGER PRIMARY KEY,
    vector_8bit BLOB,  -- 8-bit quantized instead of 32-bit float
    scale_factor REAL,  -- Reconstruction parameter
    bias_factor REAL   -- Reconstruction parameter
);
```

**Cache Management Strategy**
- **LRU Eviction**: File mtime-based eviction maintains temporal locality
- **Size Monitoring**: 2GB default limit with configurable thresholds  
- **Predictive Eviction**: Preemptive cleanup when approaching limits
- **Hot Path Optimization**: Keep popular crates (std, serde, tokio) in memory

## Validation Architecture

### Centralized Validation Module

The system implements a centralized validation module (`validation.py`) that provides reusable validation utilities across all request models:

```mermaid
graph TB
    subgraph "Validation Module (validation.py)"
        PRECOMPILED[Precompiled Regex Patterns<br/>Performance optimization]
        CRATE_VAL[validate_crate_name()<br/>Rust crate naming rules]
        VERSION_VAL[validate_version_string()<br/>SemVer + latest support]
        PATH_VAL[validate_rust_path()<br/>Module path validation]
        INT_BOUNDS[coerce_to_int_with_bounds()<br/>Type coercion + constraints]
        OPT_PATH[validate_optional_path()<br/>None-safe path validation]
    end
    
    subgraph "Request Models"
        SEARCH_REQ[SearchItemsRequest]
        NAV_REQ[NavigateModulesRequest]
        INGEST_REQ[IngestCrateRequest]
        SIG_REQ[GetItemSignatureRequest]
    end
    
    subgraph "Field Validators"
        FIELD_VAL[@field_validator(mode='before')<br/>MCP compatibility layer]
        TYPE_COERCE[String-to-type coercion]
        CONSTRAINT[Bounds checking]
    end
    
    PRECOMPILED --> CRATE_VAL
    PRECOMPILED --> VERSION_VAL
    PRECOMPILED --> PATH_VAL
    
    CRATE_VAL --> FIELD_VAL
    VERSION_VAL --> FIELD_VAL
    PATH_VAL --> FIELD_VAL
    INT_BOUNDS --> FIELD_VAL
    OPT_PATH --> FIELD_VAL
    
    FIELD_VAL --> SEARCH_REQ
    FIELD_VAL --> NAV_REQ
    FIELD_VAL --> INGEST_REQ
    FIELD_VAL --> SIG_REQ
    
    FIELD_VAL --> TYPE_COERCE
    TYPE_COERCE --> CONSTRAINT
```

### Enhanced Request Model Validation

All request models now implement comprehensive field validators with MCP client compatibility:

```mermaid
graph TD
    subgraph "MCP Client Parameter Handling"
        CLIENT[MCP Client<br/>May send strings for integers/booleans]
        SERIALIZE[JSON Serialization<br/>Type coercion needed]
    end
    
    subgraph "Pydantic Validation Pipeline"
        FIELD_VAL[Field Validators<br/>@field_validator(mode='before')]
        TYPE_COERCE[Type Coercion<br/>String → Integer/Boolean conversion]
        CONSTRAINT[Constraint Validation<br/>ge=1, le=20 bounds]
        MODEL_VAL[Model Validation<br/>extra='forbid' strict mode]
        NONE_PRESERVE[None Value Preservation<br/>Application-layer defaults]
    end
    
    subgraph "Error Handling"
        VALUE_ERR[ValueError<br/>Invalid conversion]
        VALID_ERR[ValidationError<br/>Constraint violations]
        ERROR_RESP[ErrorResponse<br/>Standardized error format]
    end
    
    CLIENT --> SERIALIZE
    SERIALIZE --> FIELD_VAL
    FIELD_VAL --> TYPE_COERCE
    TYPE_COERCE --> CONSTRAINT
    CONSTRAINT --> MODEL_VAL
    MODEL_VAL --> NONE_PRESERVE
    
    TYPE_COERCE -.->|Invalid conversion| VALUE_ERR
    CONSTRAINT -.->|Bounds check fails| VALID_ERR
    VALUE_ERR --> ERROR_RESP
    VALID_ERR --> ERROR_RESP
```

### MCP Parameter Validation Architecture

The MCP parameter validation system follows a **double-validation pattern** to ensure robust type handling and client compatibility:

1. **JSON Schema validation** with `anyOf` patterns in the manifest (app.py `get_mcp_manifest` function)
2. **Pydantic field validators** with `mode='before'` for type coercion (models.py)

This architectural decision enables consistent handling of boolean parameters alongside the existing numeric parameter pattern:

```mermaid
graph LR
    subgraph "MCP Manifest Schema"
        ANYOF[anyOf Pattern<br/>{"anyOf": [{"type": "integer"}, {"type": "string"}]}<br/>{"anyOf": [{"type": "boolean"}, {"type": "string"}]}]
        JSON_VALID[JSON Schema Validation<br/>Allows multiple types]
        PYDANTIC[Pydantic Type Coercion<br/>Handles actual conversion]
        FASTMCP[FastMCP Double Validation<br/>Passes both types through]
    end
    
    subgraph "Parameter Examples"
        INT_PARAM[k parameter<br/>Result count limit]
        BOOL_PARAM[deprecated parameter<br/>Filter deprecated items<br/>Boolean anyOf pattern]
        NUM_PARAM[min_doc_length<br/>Minimum documentation length]
    end
    
    ANYOF --> JSON_VALID
    JSON_VALID --> FASTMCP
    FASTMCP --> PYDANTIC
    
    INT_PARAM --> ANYOF
    BOOL_PARAM --> ANYOF
    NUM_PARAM --> ANYOF
```

### Validation Best Practices Implementation

The validation architecture follows these key principles:

```mermaid
graph TB
    subgraph "API Boundary Validation"
        ENTRY[Request Entry Point<br/>All validation at API boundaries only]
        EARLY[Early Error Detection<br/>Fail fast with clear messages]
        SECURITY[Security First<br/>extra='forbid' prevents injection]
    end
    
    subgraph "Error Handling Strategy"
        HELPFUL[Helpful Error Messages<br/>Include examples and guidance]
        CHAINING[Error Chaining<br/>Preserve original context]
        STANDARD[Standardized Responses<br/>Consistent error format]
    end
    
    subgraph "Value Handling"
        NONE_SAFE[None Value Preservation<br/>Maintain for default handling]
        APP_DEFAULTS[Application-Layer Defaults<br/>Business logic handles None]
        TYPE_SAFE[Type Safety<br/>Strong validation post-coercion]
    end
    
    ENTRY --> EARLY
    EARLY --> SECURITY
    
    HELPFUL --> CHAINING
    CHAINING --> STANDARD
    
    NONE_SAFE --> APP_DEFAULTS
    APP_DEFAULTS --> TYPE_SAFE
    
    SECURITY --> HELPFUL
    STANDARD --> NONE_SAFE
```

### Validation Module Implementation

The `validation.py` module provides centralized validation utilities with performance optimizations:

#### Core Validation Functions

**validate_crate_name(name: str) -> str**
- Validates Rust crate naming conventions (lowercase, hyphens, underscores)
- Uses precompiled regex pattern for performance
- Raises ValueError with helpful error messages for invalid names

**validate_version_string(version: str) -> str** 
- Supports SemVer format validation (e.g., "1.0.0", "2.1.4-alpha")
- Allows special version strings like "latest"
- Precompiled regex patterns for efficiency

**validate_rust_path(path: str) -> str**
- Validates Rust module path format (e.g., "std::collections::HashMap")
- Ensures proper "::" separator usage and valid identifiers
- Used for module filtering and navigation

**coerce_to_int_with_bounds(value, field_name: str, min_val: int = None, max_val: int = None) -> int**
- Handles string-to-integer conversion for MCP client compatibility
- Enforces optional bounds checking with clear error messages
- Preserves None values for application-layer default handling

**validate_optional_path(path: Optional[str]) -> Optional[str]**
- None-safe path validation for optional parameters
- Applies path validation only when path is not None
- Used for optional module path parameters

#### Performance Optimizations

```python
# Precompiled regex patterns for performance
CRATE_NAME_PATTERN = re.compile(r'^[a-z0-9][a-z0-9_-]*[a-z0-9]$|^[a-z0-9]$')
VERSION_PATTERN = re.compile(r'^(\d+\.\d+\.\d+(-[a-zA-Z0-9]+)?|latest)$')
RUST_PATH_PATTERN = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*(::[a-zA-Z_][a-zA-Z0-9_]*)*$')
```

### Enhanced Request Model Validation

All request models implement comprehensive field validators using the centralized validation utilities:

#### SearchItemsRequest Validation
- **query**: Unicode NFKC normalization for search consistency
- **k**: String-to-integer coercion with bounds (1-20)
- **crate**: Crate name validation using centralized function
- **version**: Version string validation with SemVer support
- **module_path**: Optional Rust path validation
- **deprecated**: String-to-boolean coercion for MCP compatibility

#### Parameter Coercion Examples

```python
@field_validator("k", mode="before")
@classmethod
def coerce_k_to_int(cls, v):
    """Convert string numbers to int for MCP client compatibility."""
    return coerce_to_int_with_bounds(v, "k", min_val=1, max_val=20)

@field_validator("crate", mode="before") 
@classmethod
def validate_crate_name_field(cls, v):
    """Validate crate name using centralized validation."""
    if v is None:
        return v
    return validate_crate_name(v)

@field_validator("deprecated", mode="before")
@classmethod  
def coerce_deprecated_to_bool(cls, v):
    """Convert various representations to boolean for MCP clients."""
    if v is None:
        return v
    if isinstance(v, str):
        if v.lower() in ('true', '1', 'yes'):
            return True
        elif v.lower() in ('false', '0', 'no'):
            return False
        else:
            raise ValueError(f"Invalid boolean value: '{v}'")
    return bool(v)
```

### MCP Manifest Schema Patterns

The MCP manifest schema uses consistent `anyOf` patterns for flexible parameter acceptance across all type-flexible parameters. This architectural decision ensures uniform handling of MCP client serialization variations:

```json
{
  "k": {
    "anyOf": [
      {"type": "integer", "minimum": 1, "maximum": 20},
      {"type": "string"}
    ],
    "description": "Number of results to return"
  },
  "deprecated": {
    "anyOf": [
      {"type": "boolean"},
      {"type": "string"}
    ],
    "description": "Filter deprecated items"
  }
}
```

**Consistent anyOf Patterns Across All Tools**

The system implements standardized `anyOf` patterns for all parameters requiring type flexibility:

- **Numeric parameters**: `[{"type": "integer"}, {"type": "string"}]`
- **Boolean parameters**: `[{"type": "boolean"}, {"type": "string"}]`  
- **Optional strings**: `[{"type": "string"}, {"type": "null"}]`

**Double Validation Architecture Benefits**
- **MCP Client Compatibility**: FastMCP validates against JSON Schema first, so `anyOf` patterns are critical for flexible type acceptance
- **Type Coercion**: Pydantic field validators handle conversion after schema validation passes
- **Client Flexibility**: MCP clients can send parameters as native types or strings regardless of implementation
- **Type Safety**: Maintains strict type validation after the coercion phase
- **Schema Consistency**: All parameters with field validators now have corresponding `anyOf` patterns in the manifest

### Validation Best Practices

**API Boundary Validation Only**
- All validation occurs at API entry points
- Business logic assumes validated data
- No redundant validation in service layers

**Error Message Standards**
- Include parameter name and invalid value in error messages
- Provide examples of valid formats
- Chain original errors using `from err` for debugging context

**None Value Handling**
- Field validators preserve None values for optional parameters
- Application layer handles default value assignment
- Enables proper OpenAPI schema generation with optional parameters

**Security Through Validation**
- `extra='forbid'` on all request models prevents parameter injection
- Strict type checking after coercion prevents type confusion attacks
- Centralized validation reduces validation bypass vulnerabilities

### Validation Flow Architecture

The **double-validation pattern** ensures MCP client compatibility through this sequential flow:

1. **MCP Client Request** → JSON-RPC serialization (may convert types to strings)
2. **FastMCP JSON Schema Validation** → `anyOf` patterns allow multiple input types to pass through
3. **Pydantic Field Validators** → `mode='before'` handles type coercion from strings to native types
4. **Centralized Validation** → Reusable functions with precompiled patterns
5. **Constraint Checking** → Bounds validation and format verification
6. **Model Validation** → Cross-field validation and `extra='forbid'` security
7. **Business Logic** → Receives validated, typed data

**Critical**: FastMCP validates against the JSON Schema first, making `anyOf` patterns essential for accepting both native types and string representations from different MCP client implementations.

### Extended Numeric Parameter Validation

The following numeric parameters now have comprehensive string-to-numeric conversion support for MCP client compatibility:

**1. SearchItemsRequest.min_doc_length**
- **Type**: integer
- **Bounds**: 100-10000  
- **Validator**: coerce_min_doc_length_to_int
- **Accepts**: both integers and numeric strings
- **Purpose**: Minimum documentation length filtering for search results

**2. ErrorResponse.status_code**
- **Type**: integer
- **Bounds**: 400-599 (HTTP error codes)
- **Validator**: coerce_status_code_to_int
- **Accepts**: both integers and numeric strings
- **Defaults**: to 500 when None
- **Purpose**: Standardized HTTP error code handling

**3. RankingConfig weights**
- **Fields**: vector_weight, type_weight, quality_weight, examples_weight
- **Type**: float
- **Bounds**: 0.0-1.0
- **Validator**: coerce_weights_to_float
- **Accepts**: strings, integers, and floats
- **Handles**: integer-to-float conversion
- **Purpose**: Search result ranking weight configuration

**4. SearchResult.score**
- **Type**: float
- **Bounds**: 0.0-1.0
- **Validator**: coerce_score_to_float
- **Accepts**: strings, integers, and floats
- **Required**: field (cannot be None)
- **Purpose**: Search relevance scoring

**Implementation Pattern**

All validators use @field_validator(mode='before') pattern to intercept and convert string inputs before Pydantic's built-in type validation:

```python
@field_validator("min_doc_length", mode="before")
@classmethod
def coerce_min_doc_length_to_int(cls, v):
    """Convert string numbers to int for MCP client compatibility."""
    if v is None:
        return v
    if isinstance(v, str):
        try:
            return int(v)
        except ValueError as err:
            raise ValueError(
                f"min_doc_length parameter must be a valid integer, got '{v}'"
            ) from err
    return v
```

**MCP Manifest Schema Updates**

The MCP manifest schema has been updated to use anyOf patterns for numeric parameters that may arrive as strings:

```json
{
  "min_doc_length": {
    "anyOf": [{"type": "integer"}, {"type": "string"}],
    "description": "Minimum documentation length",
    "minimum": 100,
    "maximum": 10000
  },
  "deprecated": {
    "anyOf": [{"type": "boolean"}, {"type": "string"}],
    "description": "Filter deprecated items"
  }
}
```

**Benefits**
- **MCP Client Compatibility**: Handles various client serialization approaches
- **Type Safety**: Maintains strict type validation after conversion
- **Error Clarity**: Provides detailed error messages for invalid conversions
- **Schema Flexibility**: anyOf pattern allows multiple input formats while preserving validation
- **Backward Compatibility**: Existing code continues to work with native types

This ensures compatibility with various MCP client implementations that may serialize parameters differently while maintaining strict type safety and validation.

## Security Model

```mermaid
graph LR
    subgraph "Input Validation"
        IV[Pydantic Models<br/>Type safety & validation<br/>Field validators for coercion]
    end
    
    subgraph "Origin Control"
        OC[HTTPS only<br/>crates.io API domain]
    end
    
    subgraph "Cache Safety"
        CS[File-based cache<br/>./cache directory<br/>Sanitized filenames]
    end
    
    subgraph "Memory Management"
        MM[aiohttp 3.9.5<br/>Fixed memory leaks<br/>Async resource cleanup]
    end
    
    IV --> OC
    OC --> CS
    CS --> MM
```

### Fuzzy Path Resolution

The system implements intelligent path suggestions using RapidFuzz library when exact path matches fail in the `get_item_doc` endpoint:

```mermaid
graph TD
    subgraph "Fuzzy Path Resolution Flow"
        REQ[get_item_doc Request<br/>item_path parameter]
        EXACT[Exact Path Match<br/>SQLite query]
        CACHE_CHECK[Path Cache Check<br/>5-minute TTL]
        CACHE_BUILD[Build Path Cache<br/>Query all item paths]
        FUZZY[RapidFuzz Processing<br/>0.6 similarity threshold]
        SUGGESTIONS[Top 3 Suggestions<br/>Ranked by similarity]
        ERROR[ItemNotFoundError<br/>with suggestions]
    end
    
    REQ --> EXACT
    EXACT -->|Found| SUCCESS[Return Documentation]
    EXACT -->|Not Found| CACHE_CHECK
    CACHE_CHECK -->|Cache Hit| FUZZY
    CACHE_CHECK -->|Cache Miss| CACHE_BUILD
    CACHE_BUILD --> FUZZY
    FUZZY --> SUGGESTIONS
    SUGGESTIONS --> ERROR
```

**Architecture Components:**

**1. Integration Point**
- **Location**: `routes.py` in `get_item_doc` endpoint
- **Trigger**: When exact SQLite path lookup fails
- **Fallback**: Automatic fuzzy matching with user-friendly error messages

**2. Path Caching System**
- **Implementation**: Simple in-memory dictionary cache
- **TTL**: 5-minute expiration for cache freshness
- **Population**: Lazy loading - cache built on first fuzzy match attempt
- **Memory Footprint**: <3MB for typical crate collections
- **Thread Safety**: Single-threaded FastAPI context (no locking required)

**3. RapidFuzz Configuration**
- **Library**: RapidFuzz library for high-performance string matching
- **Similarity Threshold**: 0.6 (60% similarity required for meaningful suggestions)
- **Result Limit**: Maximum 3 suggestions returned to users
- **Scoring Method**: Default fuzzy matching algorithm optimized for path-like strings

**4. Performance Characteristics**
- **Cold Start**: Initial cache population during first fuzzy match
- **Warm Performance**: Sub-5ms fuzzy matching after cache warmup  
- **Memory Usage**: Minimal overhead with efficient string comparison
- **Scalability**: Linear performance with crate collection size

**5. Error Response Enhancement**
- **Error Type**: `ItemNotFoundError` with enhanced context
- **User Experience**: Clear error message with "Did you mean?" suggestions
- **Suggestion Format**: Top 3 ranked alternatives presented as list
- **Fallback**: Graceful degradation if fuzzy matching fails

**Integration Benefits:**
- **No Database Changes**: Pure application-layer enhancement
- **Backward Compatible**: Existing exact match behavior preserved
- **User-Friendly**: Helpful suggestions reduce friction from typos
- **Performance Optimized**: Fast response times with intelligent caching

## Dual-Mode Server Implementation

### Architecture Overview

The docsrs-mcp server implements a dual-mode architecture that allows the same FastAPI application to operate in two distinct modes:

**MCP Mode (Default)**
- Model Context Protocol server using STDIO transport
- JSON-RPC messaging over stdin/stdout
- stderr-only logging to prevent protocol corruption
- Automatic tool generation from FastAPI endpoints via FastMCP

**REST Mode (--mode rest)**
- Standard FastAPI HTTP server with uvicorn
- Full HTTP transport with standard logging to stdout/stderr
- Compatible with web browsers, curl, and HTTP clients
- Automatic OpenAPI documentation at `/docs` and `/redoc`

### Key Implementation Details

**CLI Mode Selection**
- `--mode mcp`: Launches MCP server with STDIO transport (default behavior)
- `--mode rest`: Launches HTTP server
- Single entry point in cli.py handles mode dispatch

**FastMCP Integration**
- `FastMCP.from_fastapi()` automatically converts REST endpoints to MCP tools
- No changes required to existing FastAPI route handlers
- Preserves all business logic, validation, and error handling
- Maintains compatibility with existing FastAPI middleware
- Generates MCP-compatible JSON schemas with `anyOf` patterns for flexible parameter types
- **Boolean Parameter Support**: Manifest generation uses consistent `anyOf: [{"type": "boolean"}, {"type": "string"}]` patterns
- **Double Validation System**: FastMCP's validation layer properly allows both boolean types to pass through to Pydantic

**Protocol Isolation**
- MCP mode uses stderr exclusively for logging to avoid stdout contamination
- REST mode uses standard logging configuration
- Business logic remains completely unchanged between modes
- Same ingestion pipeline, storage, and search functionality

**Zero Duplication Architecture**
- Single FastAPI application serves both modes
- All route handlers, models, and services shared
- Configuration and error handling unified
- Maintenance overhead minimized through code reuse

## Implementation Decisions

### Recent Architectural Decisions

**Boolean Parameter Handling in MCP Manifest (2025-08-07)**
- **Decision**: Boolean parameters in MCP manifest now use `anyOf` patterns for type flexibility
- **Pattern**: `anyOf: [{"type": "boolean"}, {"type": "string"}]` consistent with existing numeric parameter handling
- **Technical Implementation**:
  - MCP manifest schema generation uses consistent `anyOf` patterns for boolean parameters
  - Pydantic field validators handle string-to-boolean conversion with `mode="before"`
  - FastMCP's double validation system properly allows both types through validation pipeline
- **Benefits**:
  - Enables MCP clients to send boolean parameters as either native booleans or strings
  - Maintains consistency with existing numeric parameter validation patterns
  - Preserves type safety through Pydantic's validation after coercion
  - Supports varied MCP client serialization approaches without breaking changes
- **Integration**: Seamless integration with existing validation architecture without code duplication

### Key Architectural Choices Made

**Vector Storage: sqlite-vec over sqlite-vss**
- sqlite-vss is deprecated, sqlite-vec is the modern successor
- Better performance and active maintenance
- Native SQLite integration with MATCH operator for similarity search

**HTTP Client: aiohttp 3.9.5 (Pinned)**
- Memory leaks discovered in aiohttp 3.10+ versions
- Version pinning ensures stability in production deployments
- Async architecture maintained with proven stable version

**Embedding Model: FastEmbed + BAAI/bge-small-en-v1.5**
- Optimized for retrieval tasks with 384-dimensional vectors
- Good balance of accuracy and performance for crate descriptions
- ONNX runtime for efficient inference without GPU requirements

**Simple Module Structure with Documentation Integration**
- Five core modules: app.py, config.py, models.py, database.py, ingest.py
- Comprehensive docstrings in all modules for auto-generated documentation
- FastAPI metadata configuration for enhanced API discoverability
- Minimal complexity, easy to understand and maintain
- Direct async/await patterns throughout

**Robust Parameter Validation with MCP Compatibility**
- Pydantic field validators with `mode='before'` for type coercion
- Handles MCP client parameter serialization differences (string-to-int, string-to-boolean conversion)
- Maintains strict validation with `extra='forbid'` to prevent parameter injection
- Graceful error handling with detailed error messages for debugging
- Supports both native types and string representations for maximum compatibility
- MCP manifest schemas use consistent `anyOf` patterns for all type-flexible parameters
- **Critical Architecture Pattern**: `anyOf: [{"type": "boolean"}, {"type": "string"}]` for booleans matches existing numeric pattern
- **Schema Consistency**: Uniform approach across integers, floats, and booleans enables predictable MCP client behavior
- **Double Validation Flow**: JSON Schema validation → FastMCP processing → Pydantic type coercion ensures compatibility

**MVP Focus: Crate Descriptions Only (Enhanced with Standard Library Support)**
- Basic ingestion pipeline processes crate metadata from crates.io API
- Enhanced with standard library crate detection and channel-specific version resolution
- Embeddings generated from crate descriptions for semantic search
- Standard library crates (std, core, alloc, proc_macro, test) supported with 95% code reuse
- Future expansion to full documentation planned for v2

**File-Based Caching**
- ./cache directory for persistent storage
- SQLite databases per crate for efficient organization
- Simple filesystem-based cache management

**Documentation Architecture**
- FastAPI automatic OpenAPI documentation for comprehensive API reference
- Enhanced metadata configuration for improved API discoverability
- Single README.md approach chosen over multi-file documentation for simplicity
- Comprehensive docstrings throughout codebase enable auto-generated documentation
- Security documentation integrated into main README to prevent fragmentation
- Performance metrics and benchmarks documented for operational clarity

### Data Flow Architecture

1. **Ingestion**: Client requests → Check cache → Fetch from crates.io API → Generate embeddings → Store in SQLite
2. **Search**: Query → Vector similarity search using sqlite-vec MATCH → Return ranked results
3. **Example Search**: Query → Primary search through dedicated example_embeddings table → Fallback to document embeddings for backward compatibility → search_example_embeddings() function with language filtering
4. **Caching**: Persistent file-based cache in ./cache directory for fast subsequent access

## Technical Implementation Details

### Compression Implementation
- **zstandard**: Uses `zstandard` library with streaming decompression
- **gzip**: Uses standard library `gzip.decompress()` with size checking
- **Size Limits**: 30MB compressed, 100MB decompressed (configurable)
- **Memory Management**: Chunked reading to prevent memory exhaustion

### Enhanced JSON Processing with Metadata Extraction
- **Streaming Parser**: Processes large files without full memory load (unchanged)
- **Multi-pass Processing**: 
  - First pass builds ID-to-path mapping from "paths" section
  - Second pass extracts documentation from "index" section with enhanced metadata
- **Enhanced Metadata Extraction**:
  - **Type Normalization**: Helper functions for consistent item type classification
  - **Signature Extraction**: Complete function signatures with generics and return types  
  - **Parent ID Resolution**: Module hierarchy relationships for navigation
  - **Code Example Parsing**: Extracts examples from documentation comments
  - **Trait Implementation Parsing**: Processes impl blocks to extract "impl TraitName for TypeName" relationships using extract_type_name() helper
- **Type Filtering**: Focuses on functions, structs, traits, modules, enums, constants, macros, and trait implementations (trait_impl)
- **Backward Compatibility**: NULL defaults for all new metadata fields
- **Memory Efficiency**: Processes items incrementally, not all at once
- **Performance Impact**: ~10-15% parsing overhead for enhanced metadata

### Module Hierarchy Implementation

The module hierarchy extraction system processes rustdoc JSON paths section to build navigable module trees:

**build_module_hierarchy() Function**
- **Path Processing**: Analyzes rustdoc JSON "paths" section for module structure
- **Parent Resolution**: Resolves parent-child relationships through path analysis (e.g., "std::collections::HashMap" → parent: "std::collections") 
- **Depth Calculation**: Determines nesting level based on "::" separators in module paths
- **Item Counting**: Aggregates number of documentation items per module for statistics
- **Database Integration**: Stores hierarchy in modules table with parent_id foreign key relationships

**API Endpoint Implementation**
- **get_module_tree**: Retrieves hierarchical module structure using recursive CTEs
- **Tree Traversal**: Efficient database queries for parent-child navigation
- **ModuleTreeNode**: Nested data structure with children collections for client consumption
- **Filtering Support**: Supports crate, version, and path-based filtering

**Storage Architecture**
- **Adjacency List Model**: modules.parent_id references modules.id for tree relationships
- **Optimized Indexes**: Specialized indexes on parent_id and depth for fast hierarchical queries
- **Memory Efficient**: Integrates with existing streaming ingestion pipeline without memory overhead

### Concurrency Architecture
- **Per-Crate Locks**: Prevents race conditions during ingestion
- **Global Lock Registry**: Maintains locks across async task lifecycle
- **Batch Processing**: Optimizes database operations and embedding generation
- **Resource Management**: Proper cleanup of connections and file handles

### Memory-Optimized Database Streaming with Metadata

The store_embeddings_streaming() function implements memory-efficient streaming processing for large datasets with enhanced metadata support:

```mermaid
sequenceDiagram
    participant Worker as Ingest Worker
    participant Monitor as Memory Monitor
    participant Serialize as sqlite_vec.serialize_float32()
    participant DB as SQLite Database
    participant Embeddings as embeddings table
    participant VecEmbed as vec_embeddings table
    
    loop For each streaming chunk (memory-aware)
        Worker->>Monitor: Check memory usage
        Monitor-->>Worker: Current memory percentage
        
        alt Memory > 90%
            Worker->>Worker: Trigger garbage collection
        end
        
        Worker->>Serialize: Stream serialize vectors (chunk-based)
        Serialize-->>Worker: Serialized vector blobs
        
        loop For each batch of 999 items within chunk
            Worker->>DB: BEGIN TRANSACTION
            Worker->>Embeddings: executemany() batch insert with metadata
            Embeddings-->>Worker: Batch inserted with type, signature, parent_id, examples
            Worker->>DB: last_insert_rowid() - get rowid range
            DB-->>Worker: Starting rowid for batch
            Worker->>VecEmbed: executemany() with calculated rowids
            VecEmbed-->>Worker: Vector index batch inserted
            Worker->>DB: COMMIT TRANSACTION
            
            alt Progress Logging (>999 total items)
                Worker->>Worker: Log batch progress with memory stats
            end
            
            alt Error in batch
                Worker->>DB: ROLLBACK TRANSACTION
                Worker->>Worker: Log error, continue next batch
            end
        end
        
        Worker->>Worker: Garbage collect at chunk boundary
    end
```

**Key Implementation Details:**
- **Streaming Architecture**: Memory-aware chunked processing with adaptive batch sizing
- **Memory Monitoring**: psutil-based monitoring with 80%/90% memory thresholds
- **Adaptive Batch Sizing**: Dynamic adjustment from 16-512 items based on memory pressure
- **Garbage Collection**: Triggered at chunk boundaries during high memory usage
- **Database Batch Size**: 999 items per transaction (SQLite parameter limit) within memory chunks
- **Vector Stream Serialization**: Chunk-based serialization with sqlite_vec.serialize_float32()
- **Two-table Strategy**: Coordinated inserts into embeddings and vec_embeddings tables
- **Enhanced Metadata Processing**: Processes item_type, signature, parent_id, examples fields with NULL defaults
- **Transaction Management**: Begin/commit per batch with rollback on errors within streaming chunks
- **Rowid Synchronization**: Uses last_insert_rowid() to maintain relationships between tables
- **Backward Compatibility**: NULL defaults and wrapper functions ensure compatibility with existing data
- **Memory Optimization**: O(chunk_size) memory usage with configurable chunk boundaries
- **Progress Logging**: Tracks progress for large datasets with memory usage statistics
- **Error Isolation**: Per-batch error handling prevents complete ingestion failure
- **Performance Impact**: ~10-15% overhead for enhanced metadata + memory monitoring

### Cache Management Strategy
- **LRU Algorithm**: Based on file system modification time (mtime)
- **Size Monitoring**: Uses `os.walk()` and `os.stat()` for efficient calculation
- **Eviction Process**: Removes oldest files first until under size limit
- **Error Handling**: Graceful handling of file system errors during cleanup

### Enhanced Rustdoc Implementation Summary

The enhanced rustdoc JSON parsing implementation provides comprehensive metadata extraction while maintaining backward compatibility and memory efficiency:

**Key Enhancements:**
- **Database Schema**: Added metadata columns (item_type, signature, parent_id, examples) with NULL defaults
- **Parsing Pipeline**: Enhanced with helper functions for type normalization, signature extraction, parent ID resolution, code example extraction, and trait implementation parsing with extract_type_name() helper
- **Memory Efficiency**: Maintained streaming approach with ijson, minimal performance impact (~10-15% overhead)
- **Batch Processing**: Continues to use 999-item transactions for optimal performance
- **Backward Compatibility**: NULL defaults ensure existing data and queries continue to work
- **Performance**: Memory usage remains under 512 MiB RSS target, search latency unchanged

### Trait Implementation Search Architecture

The system provides comprehensive trait implementation search capabilities by parsing impl blocks from rustdoc JSON and integrating them with the existing search infrastructure:

**New Item Type: trait_impl**
- Added 'trait_impl' as a distinct item type in the type system alongside existing types (function, struct, trait, module, enum, constant, macro)
- Available as a filter parameter in search queries to specifically target trait implementation searches
- Integrated into the multi-factor ranking system with a weight of 1.1x for relevance optimization

**extract_type_name() Helper Function**
- Extracts readable type names from complex rustdoc Type objects (dict, string, or None input)
- Handles various rustdoc Type structures including resolved_path, path, generic, primitive, and direct name fields
- Returns "Unknown" for unrecognized structures, ensuring robust parsing of complex type information
- Used extensively in impl block parsing to generate clean "impl TraitName for TypeName" headers

**Impl Block Parsing in parse_rustdoc_items_streaming()**
- Detects impl blocks during rustdoc JSON parsing and processes them as trait_impl items
- For trait implementations: Extracts trait and type information to generate headers like "impl Display for MyStruct"
- For inherent implementations: Generates headers like "impl MyStruct" when no trait is present
- Uses composite naming convention (e.g., "Display_for_MyStruct") to create unique identifiers
- Leverages existing infrastructure without database schema changes

**Search Integration**
- Trait implementations participate in standard vector similarity search with enhanced relevance scoring
- Type-aware ranking applies 1.1x weight boost to trait_impl items for improved search results
- Compatible with existing filtering, crate selection, and module path constraints
- No performance impact on existing search functionality - trait implementations processed in same pipeline

**Implementation Benefits**
- Zero database schema changes required - leveraged existing item_type, header, and signature fields
- Maintains backward compatibility with all existing functionality and APIs
- Reuses established parsing, embedding, and search infrastructure
- Memory efficient - trait implementations processed in same streaming pipeline with identical memory characteristics

## Memory Management Architecture

### Streaming Processing Pipeline

The memory optimization architecture implements a streaming processing pipeline that processes large rustdoc JSON files without loading them entirely into memory:

```mermaid
graph TB
    subgraph "Streaming Pipeline"
        STREAM[parse_rustdoc_items_streaming()<br/>Generator-based item streaming]
        BATCH[generate_embeddings_streaming()<br/>Adaptive batch processing]
        STORE[store_embeddings_streaming()<br/>Chunked database operations]
        MONITOR[Memory Monitor<br/>psutil-based tracking]
    end
    
    subgraph "Memory Thresholds"
        THRESH80[80% Memory<br/>Reduce batch size]
        THRESH90[90% Memory<br/>Trigger garbage collection]
        ADAPTIVE[Adaptive Batch Sizing<br/>16-512 items]
    end
    
    subgraph "Legacy Wrappers"
        COMPAT[Backwards Compatibility<br/>Non-streaming entry points]
    end
    
    STREAM --> BATCH
    BATCH --> STORE
    MONITOR --> THRESH80
    MONITOR --> THRESH90
    THRESH80 --> ADAPTIVE
    THRESH90 --> ADAPTIVE
    ADAPTIVE --> BATCH
    COMPAT --> STREAM
```

### Memory Monitoring System

The system uses psutil to monitor memory usage and implement adaptive processing:

- **Memory Thresholds**: 80% threshold triggers batch size reduction, 90% threshold forces garbage collection
- **Adaptive Batch Sizing**: Dynamically adjusts from 16-512 items based on memory pressure
- **Memory Tracking**: Uses psutil.virtual_memory() for real-time memory percentage monitoring
- **Garbage Collection**: Triggered at chunk boundaries during high memory usage

### Streaming Implementation Details

**parse_rustdoc_items_streaming()**
- Uses ijson events for progressive JSON parsing
- Yields items as generators instead of collecting in memory
- Processes items on-demand without storing intermediate collections
- Memory usage: O(1) instead of O(n) for large files

**generate_embeddings_streaming()**
- Processes items in adaptive batches (16-512 items)
- Monitors memory usage between batches
- Adjusts batch size based on memory thresholds
- Uses FastEmbed batch processing for efficiency

**store_embeddings_streaming()**
- Stores embeddings in DB_BATCH_SIZE chunks (default 999 items)
- Commits transactions at chunk boundaries
- Triggers garbage collection during high memory pressure
- Maintains backwards compatibility with existing storage logic

### Module Hierarchy Extraction Components

The system now includes comprehensive module hierarchy extraction capabilities that integrate seamlessly with the existing ingestion pipeline:

```mermaid
graph TB
    subgraph "Module Hierarchy Extraction Pipeline"
        PATHS[Paths Section Extraction<br/>rustdoc JSON "paths" field]
        BUILD[build_module_hierarchy()<br/>Parent relationship resolution]
        DEPTH[Depth Calculation<br/>Path-based depth analysis]
        COUNT[Item Count Aggregation<br/>Per-module statistics]
    end
    
    subgraph "Module Storage Schema"
        MOD_TABLE[modules table<br/>name, path, parent_id, depth, item_count]
        PARENT_IDX[Index on parent_id<br/>Fast hierarchical queries]
        DEPTH_IDX[Index on depth<br/>Level-based traversal]
        ADJ_LIST[Adjacency List Model<br/>Parent-child relationships]
    end
    
    subgraph "API Endpoints"
        TREE_API[get_module_tree<br/>Hierarchical retrieval]
        CTE[Recursive CTEs<br/>Tree traversal queries]
        TREE_NODE[ModuleTreeNode<br/>Nested children structure]
    end
    
    PATHS --> BUILD
    BUILD --> DEPTH
    DEPTH --> COUNT
    COUNT --> MOD_TABLE
    MOD_TABLE --> PARENT_IDX
    MOD_TABLE --> DEPTH_IDX
    PARENT_IDX --> ADJ_LIST
    DEPTH_IDX --> ADJ_LIST
    ADJ_LIST --> TREE_API
    TREE_API --> CTE
    CTE --> TREE_NODE
```

**Key Components:**

1. **Module Hierarchy Storage**
   - Extended modules table with parent_id, depth, item_count columns
   - Indexes on parent_id and depth for fast hierarchical traversal
   - Adjacency list model supports efficient parent-child relationship queries

2. **Module Extraction Pipeline**
   - build_module_hierarchy() function processes rustdoc JSON paths section
   - Resolves parent relationships through path analysis (e.g., "std::collections" → parent: "std")
   - Calculates depth based on path components ("::"-separated segments)
   - Counts items per module for statistics and navigation
   - Integrated into streaming ingestion pipeline with memory-efficient processing

3. **API Endpoints**
   - get_module_tree endpoint provides hierarchical module retrieval
   - Uses recursive Common Table Expressions (CTEs) for efficient tree traversal
   - Returns ModuleTreeNode structure with nested children for client consumption
   - Supports filtering by crate, version, and module path

4. **Data Flow Integration**
   - Rustdoc JSON paths section → module hierarchy building → database storage → API retrieval
   - Maintains compatibility with existing ingestion pipeline
   - No performance impact on non-hierarchical queries

## Enhanced Data Flow Architecture

```mermaid
sequenceDiagram
    participant Client as AI Client
    participant API as Enhanced FastAPI
    participant Queue as asyncio.Queue
    participant Worker as Enhanced Ingest Worker
    participant DocsRS as docs.rs
    participant Parser as Streaming Parser
    participant Embed as Adaptive Embed
    participant DB as SQLite+VSS+Examples
    participant Monitor as Memory Monitor
    
    Client->>API: POST /mcp/tools/search_documentation
    API->>DB: Check if crate@version exists
    
    alt Cache Miss
        API->>Queue: Enqueue enhanced ingest task
        Queue->>Worker: Dequeue task
        Worker->>Worker: Acquire per-crate lock
        Worker->>DocsRS: Resolve version via redirect (or detect stdlib crate)
        DocsRS-->>Worker: Actual version + rustdoc URL (with channel resolution for stdlib)
        Worker->>DocsRS: GET compressed rustdoc (.zst/.gz/.json)
        DocsRS-->>Worker: Complete rustdoc JSON
        Worker->>Worker: Stream decompress with size limits
        Worker->>Parser: parse_rustdoc_items_streaming() - progressive ijson parsing
        
        loop Progressive Processing
            Parser->>Parser: Yield items progressively (generator-based)
            Parser-->>Worker: Stream of structured items
            Worker->>Monitor: Check memory usage
            Monitor-->>Worker: Current memory percentage
            
            alt Memory > 90%
                Worker->>Worker: Trigger garbage collection
                Worker->>Worker: Reduce batch size to minimum (16)
            else Memory > 80%
                Worker->>Worker: Reduce batch size
            else Memory OK
                Worker->>Worker: Normal batch size (up to 512)
            end
            
            Worker->>Embed: generate_embeddings_streaming() - adaptive batches
            Embed-->>Worker: 384-dim vectors (batch)
            Worker->>DB: store_embeddings_streaming() - chunked storage
            Worker->>DB: Store module relationships in modules table
            Worker->>Worker: Garbage collect at chunk boundaries
        end
        
        Worker->>DB: Index with vss_index!
        Worker->>Worker: Check cache size (2GB limit)
        alt Cache > 2 GiB
            Worker->>DB: LRU eviction by file mtime
        end
    end
    
    API->>DB: Enhanced vector search with ranking
    DB->>DB: sqlite-vec MATCH (k+10 over-fetch)
    DB->>DB: Multi-factor scoring (vector + type + quality + examples)
    DB->>DB: Score normalization and ranking
    DB-->>API: Top-k ranked results with scores
    API-->>Client: Enhanced JSON response with ranking scores and metadata
```

## Enhanced MCP Tool Architecture

```mermaid
graph TD
    subgraph "Enhanced Search Tools"
        SEARCH_DOC[search_documentation<br/>• Vector similarity search<br/>• Item type filtering (struct, function, trait, etc)<br/>• Module path filtering with SQL LIKE patterns<br/>• Signature matching<br/>• Tag-based filtering<br/>• See-also suggestions (0.7 similarity threshold, max 5 items)]
        
        NAV_TREE[navigate_modules<br/>• Hierarchical module browsing<br/>• Parent-child relationships<br/>• Module content listing<br/>• Path-based navigation]
        
        GET_EXAMPLES[get_examples<br/>• Code example retrieval<br/>• Language-specific filtering<br/>• Context-aware examples<br/>• Usage pattern discovery]
        
        GET_SIG[get_item_signature<br/>• Complete function signatures<br/>• Trait definitions<br/>• Struct layouts<br/>• Type information]
    end
    
    subgraph "Enhanced Database Operations"
        HIER_QUERY[Hierarchical Queries<br/>• Recursive module traversal<br/>• Parent-child lookups<br/>• Breadcrumb generation]
        
        TYPE_FILTER[Type-based Filtering<br/>• Item type constraints<br/>• Signature pattern matching<br/>• Tag-based search]
        
        EXAMPLE_LINK[Example Relationships<br/>• Passage-to-example linking<br/>• Code snippet retrieval<br/>• Language detection]
    end
    
    subgraph "Enhanced Parsing Pipeline"
        FULL_PARSE[Complete Rustdoc Parsing<br/>• All item types extraction<br/>• Module hierarchy building<br/>• Example code parsing]
        
        HIERARCHY[Module Hierarchy<br/>• Parent-child relationships<br/>• Path reconstruction<br/>• Tree navigation support]
        
        CODE_EXTRACT[Code Example Extraction<br/>• Doc comment parsing<br/>• Language detection<br/>• Example categorization]
    end
    
    SEARCH_DOC --> TYPE_FILTER
    NAV_TREE --> HIER_QUERY
    GET_EXAMPLES --> EXAMPLE_LINK
    GET_SIG --> TYPE_FILTER
    
    TYPE_FILTER --> FULL_PARSE
    HIER_QUERY --> HIERARCHY
    EXAMPLE_LINK --> CODE_EXTRACT
```

## Enhanced Database Schema Details

### EMBEDDINGS Table Enhancements (Primary Storage)
- **item_type**: Categorizes documentation items (function, struct, trait, module, enum, constant, macro, trait_impl) - DEFAULT NULL for backward compatibility
- **signature**: Complete item signature including generics, bounds, and return types - DEFAULT NULL
- **parent_id**: Self-referencing foreign key for module hierarchy navigation - DEFAULT NULL
- **examples**: Extracted code examples from documentation comments - DEFAULT NULL
- **tags**: Comma-separated searchable metadata for enhanced filtering
- **header**: Preserved for backward compatibility with existing search functionality
- **rowid**: Primary key used for vector indexing coordination with vec_embeddings table

### PASSAGES Table (Legacy Compatibility)
- Maintains existing schema structure for backward compatibility
- Enhanced with same metadata columns as EMBEDDINGS table
- Both tables support the enhanced rustdoc parsing implementation

### EXAMPLES Table Structure
- **passage_id**: Foreign key linking examples to their parent documentation items
- **code**: The actual code snippet extracted from documentation
- **language**: Detected or specified language (rust, bash, toml, json, etc.)
- **description**: Optional description or context for the example
- **line_number**: Position within the original documentation for reference

### Navigation Support
- **Hierarchical Queries**: Self-joining PASSAGES table on parent_id for tree traversal
- **Breadcrumb Generation**: Recursive parent lookup for complete navigation paths
- **Module Content Listing**: Query children of a given module for content discovery

## Code Example Extraction System

The system includes a comprehensive code example extraction and indexing system with dedicated embeddings architecture that enables semantic search across code examples while maintaining backward compatibility. The new architecture separates example embeddings from document embeddings for improved performance and specialized example search capabilities.

### Components Architecture

```mermaid
graph TB
    subgraph "Code Example Processing Pipeline"
        PYGMENTS[Pygments Language Detection<br/>30% confidence threshold<br/>Auto-detection fallback]
        EXTRACT_FUNC[extract_code_examples()<br/>Enhanced JSON structure<br/>Backward compatible]
        METADATA[Code Metadata<br/>language, confidence, detection_method]
    end
    
    subgraph "Dedicated Example Storage"
        EXAMPLE_TABLE[example_embeddings table<br/>Dedicated storage for examples<br/>Hash-based deduplication]
        EXAMPLE_VEC[vec_example_embeddings<br/>Virtual table for vector search<br/>Separate from document embeddings]
        DEDUP_SYSTEM[SHA256-based Deduplication<br/>16-char hash prefix<br/>~30% reduction in stored embeddings]
    end
    
    subgraph "Search Infrastructure"
        SEARCH_TOOL[search_example_embeddings()<br/>Dedicated function for example search<br/>Primary search through example_embeddings]
        FALLBACK_SEARCH[Fallback to Document Embeddings<br/>Backward compatibility<br/>When dedicated examples unavailable]
        LANG_FILTER[Language Filtering<br/>rust, bash, toml, etc.<br/>SQL-based filtering]
        BATCH_GENERATE[Streaming Generation<br/>Batch size 16 for CPU<br/>Memory-efficient processing]
        SCORING[Scoring System<br/>Distance-based ranking<br/>Dedicated example scoring]
    end
    
    PYGMENTS --> EXTRACT_FUNC
    EXTRACT_FUNC --> METADATA
    METADATA --> EXAMPLE_TABLE
    EXAMPLE_TABLE --> EXAMPLE_VEC
    EXAMPLE_VEC --> SEARCH_TOOL
    SEARCH_TOOL --> FALLBACK_SEARCH
    FALLBACK_SEARCH --> LANG_FILTER
    LANG_FILTER --> BATCH_GENERATE
    BATCH_GENERATE --> SCORING
```

### Data Models

**SearchExamplesRequest**
- `query: str` - Search query for semantic matching
- `language: Optional[str]` - Filter by programming language (rust, bash, toml, etc.)
- `k: int = 5` - Number of results to return

**CodeExample**
- `code: str` - The code snippet content
- `language: str` - Detected or specified language
- `confidence: Optional[float]` - Language detection confidence score
- `detection_method: str` - How language was determined (pygments, explicit, fallback)
- `context: str` - Surrounding documentation context

**SearchExamplesResponse**
- `examples: List[CodeExample]` - List of matching code examples
- `total_count: int` - Total examples found before limiting

### Implementation Details

#### Enhanced extract_code_examples Function (ingest.py:579)

The function now returns structured JSON data while maintaining backward compatibility and includes a critical bug fix for character fragmentation:

**CRITICAL BUG FIX - Data Validation Pattern (ingest.py:761-769, app.py)**:
```python
# BUG FIX: Type validation prevents character fragmentation
if isinstance(examples_data, str):
    examples_data = [examples_data]  # Wrap single string in list

for example in examples_data:  # Now safely iterates over list elements
    if isinstance(example, str):
        example = {
            "code": example, 
            "language": "rust", 
            "confidence": None, 
            "detection_method": "fallback"
        }
```

**Data Validation Improvement**:
- **Type validation added** after JSON parsing to ensure string inputs are properly wrapped in lists
- **Consistent implementation** across both `ingest.py` and `app.py` for uniform behavior
- **Follows established `isinstance()` validation patterns** throughout the codebase
- **Prevents catastrophic character fragmentation bug** where single strings were iterated character-by-character

**Previous Bug**: When `examples_data` was a string, the for-loop iterated over individual characters, creating thousands of meaningless single-character embeddings per crate.

**New JSON structure**:
```python
{
    "code": "let x = 5;",
    "language": "rust", 
    "confidence": 0.85,
    "detection_method": "pygments"
}

# Legacy list format still supported
["let x = 5;", "cargo build"]
```

Key features:
- **Pygments Integration**: Uses lexer analysis for accurate language detection
- **Confidence Threshold**: 30% minimum confidence for auto-detection
- **Fallback Strategy**: Uses context clues when pygments fails
- **Multiple Language Support**: Handles rust, bash, toml, json, yaml, and more
- **Backward Compatibility**: Maintains support for old list format
- **Data Validation Pattern**: `isinstance()` type checking applied uniformly across `ingest.py` and `app.py` embedding pipelines
- **Character Fragmentation Prevention**: Type validation ensures proper iteration over code blocks

#### search_example_embeddings Function

Dedicated function providing semantic search capabilities for examples:
- **Dedicated Architecture**: Uses separate example_embeddings table for improved performance
- **Hash-based Deduplication**: SHA256-based deduplication using 16-character prefix reduces storage by ~30%
- **Language Filtering**: SQL-based filtering by detected language in dedicated table
- **Fallback Support**: Falls back to document embeddings for backward compatibility
- **Memory-Efficient Processing**: Batch size 16 for CPU with explicit cleanup and gc.collect()
- **Context Preservation**: Returns examples with surrounding documentation context

### Example Embeddings Pipeline

The dedicated example embeddings architecture provides a complete pipeline for processing and searching code examples:

```mermaid
sequenceDiagram
    participant Ingest as Ingestion Pipeline
    participant Extract as extract_code_examples()
    participant Hash as SHA256 Hashing
    participant Dedup as Deduplication Filter
    participant Batch as Batch Generator
    participant Embed as FastEmbed
    participant Storage as example_embeddings table
    participant Search as search_example_embeddings()
    participant Client as MCP Client
    
    Ingest->>Extract: Extract unique examples from docs
    Extract->>Hash: Generate SHA256 hash (16-char prefix)
    Hash->>Dedup: Filter already embedded examples
    Dedup->>Batch: Stream filtered examples in batches
    Batch->>Embed: Generate embeddings (batch size 16)
    Embed->>Storage: Store with transaction atomicity
    Storage->>Storage: Apply UNIQUE constraint on hash
    
    Client->>Search: Search example embeddings
    Search->>Storage: Query example_embeddings table
    Storage->>Storage: Language filtering (if specified)
    Storage->>Storage: Vector similarity search
    Storage-->>Search: Ranked examples with metadata
    Search-->>Client: Deduplicated results with context
```

### Configuration Changes

The ranking weights have been adjusted to better utilize code examples:

- **RANKING_EXAMPLES_WEIGHT**: Increased from 0.05 to 0.15 (15% vs 5%)
- **RANKING_VECTOR_WEIGHT**: Reduced from 0.70 to 0.60 (60% vs 70%)

This change reflects the increased importance and utility of code examples in search relevance.

### Backward Compatibility

The implementation maintains full backward compatibility:
- **Storage Format**: Old list format still supported and processed
- **API Changes**: No breaking changes to existing endpoints
- **Database Schema**: No changes to existing table structure
- **Search Behavior**: Existing search functionality unaffected

### Performance Characteristics

| Operation | Target | Notes |
|-----------|--------|-------|
| Language detection | < 2ms per block | Pygments lexer analysis |
| Example extraction | < 10ms per doc | Enhanced processing overhead |
| Example embedding generation | 16-item batches | CPU-optimized batch size for memory efficiency |
| Example search latency | 150-360ms warm, 1.4s cold | Dedicated example_embeddings search with model loading |
| Example memory usage | ~421MB for tokio | Memory usage during example embedding generation |
| Deduplication rate | ~30% reduction | SHA256-based deduplication effectiveness |
| Search query | < 100ms P95 | Vector search + filtering + hash-based deduplication |
| Storage overhead | +15% | JSON structure vs list format |
| Memory usage | Minimal | Uses existing embedding infrastructure |

## Enhanced Component Integration

### navigation.py Module
```mermaid
classDiagram
    class NavigationService {
        +get_module_tree(crate: str, version: str, path: str) -> ModuleTree
        +get_module_children(parent_id: int) -> List[ModuleItem]
        +get_breadcrumbs(item_id: int) -> List[BreadcrumbItem]
        +search_in_module(module_path: str, query: str) -> List[SearchResult]
        -build_tree_recursive(parent_id: int, depth: int) -> TreeNode
        -resolve_module_path(path: str) -> int
    }
    
    class ModuleTree {
        +root: ModuleItem
        +children: List[ModuleTree]
        +total_items: int
        +depth: int
    }
    
    class BreadcrumbItem {
        +name: str
        +path: str
        +item_type: str
        +is_current: bool
    }
    
    NavigationService --> ModuleTree
    NavigationService --> BreadcrumbItem
```

### Enhanced Search Capabilities
- **Multi-factor Ranking**: Combines vector similarity (70%), type-aware boost (15%), documentation quality (10%), and example presence (5%)
- **Type-aware Scoring**: Functions receive 1.2x boost, traits 1.15x, structs 1.1x, modules 0.9x for relevance optimization
- **Score Normalization**: Min-max normalization to [0, 1] range for consistent ranking across different query types
- **Intelligent Caching**: LRU cache with 15-minute TTL, 1000 entry capacity, and cache key generation from query embeddings
- **Performance Monitoring**: Latency tracking, hit rate statistics, and score distribution validation
- **Over-fetching Optimization**: K+10 strategy ensures sufficient candidates for re-ranking without constraining initial retrieval
- **Type-filtered Search**: Filter results by item type (functions only, traits only, etc.)
- **Signature-based Matching**: Search by function signatures, trait bounds, or type parameters
- **Module-scoped Search**: Limit search results to specific modules or crates using SQL LIKE patterns
- **Example-integrated Results**: Include relevant code examples with search results
- **Tag-based Discovery**: Use metadata tags for enhanced result categorization

## Performance Implications

### Enhanced Storage Requirements
| Component | Estimated Increase | Notes |
|-----------|-------------------|-------|
| EMBEDDINGS table | +40% | Additional metadata fields: item_type, signature, parent_id, examples (all DEFAULT NULL) |
| VEC_EMBEDDINGS table | Unchanged | Vector storage remains the same |
| PASSAGES table | +40% | Legacy compatibility with same enhancements |
| EXAMPLES table | +25% | New table for code examples with relationships |
| Index overhead | +15% | Additional indexes for type filtering and hierarchy |
| Total storage | +60-80% | Complete documentation vs. descriptions only |
| **Backward Compatibility** | **Maintained** | **NULL defaults ensure no breaking changes** |

### Query Performance Optimizations
- **Composite Indexes**: (crate, version, item_type) for efficient type filtering
- **Hierarchy Indexes**: (parent_id, item_path) for fast tree traversal
- **Example Indexes**: (passage_id, language) for quick example retrieval
- **Vector Search with Ranking**: sqlite-vec MATCH with multi-factor scoring overlay
- **Cache Optimization**: LRU cache with TTL reduces repeated ranking computations
- **Over-fetching Strategy**: K+10 retrieval ensures ranking quality without multiple database queries
- **Score Indexing**: Pre-computed quality and type scores for faster ranking operations

### Memory Usage Considerations
- **Streaming Memory**: O(1) memory usage for large files through progressive processing
- **Adaptive Processing**: Dynamic batch sizing (16-512 items) based on memory thresholds
- **Memory Monitoring**: Real-time tracking with psutil for proactive memory management
- **Garbage Collection**: Triggered at 90% memory threshold and chunk boundaries
- **Runtime Memory**: Optimized memory footprint through generator-based architecture
- **Cache Efficiency**: Better cache utilization due to complete documentation coverage
- **Backward Compatibility**: No memory overhead for NULL metadata fields
- **Batch Processing**: Memory efficiency enhanced with adaptive chunking strategies
- **Performance Target**: Maintains <512 MiB RSS target even with large rustdoc files (>1GB)

## Enhanced Trait Search Architecture

### Trait Resolution Enhancement

The system now includes enhanced trait search capabilities with improved path resolution and alias mapping for common Rust trait patterns:

```mermaid
graph TD
    subgraph "Enhanced Trait Search Flow"
        TRAIT_QUERY[Trait Search Query<br/>e.g., "serde::Deserialize"]
        ALIAS_MAP[PATH_ALIASES Mapping<br/>Common trait patterns]
        TRAIT_RESOLVE[Trait Resolution<br/>Full path resolution]
        TRAIT_INDEX[Trait Index Lookup<br/>Specialized trait indexes]
        IMPL_SEARCH[Implementation Search<br/>Find implementing types]
        RESULT_RANK[Result Ranking<br/>Trait boost 1.15x]
    end
    
    subgraph "Trait Alias Examples"
        SERDE_DE["serde::Deserialize<br/>→ serde::de::Deserialize"]
        SERDE_SER["serde::Serialize<br/>→ serde::ser::Serialize"] 
        STD_TRAIT["std::fmt::Display<br/>→ std::fmt::Display"]
        ASYNC_TRAIT["async_trait<br/>→ async_trait::async_trait"]
    end
    
    TRAIT_QUERY --> ALIAS_MAP
    ALIAS_MAP --> TRAIT_RESOLVE
    TRAIT_RESOLVE --> TRAIT_INDEX
    TRAIT_INDEX --> IMPL_SEARCH
    IMPL_SEARCH --> RESULT_RANK
    
    ALIAS_MAP --> SERDE_DE
    ALIAS_MAP --> SERDE_SER
    ALIAS_MAP --> STD_TRAIT
    ALIAS_MAP --> ASYNC_TRAIT
```

### Trait-Specific Optimizations

**PATH_ALIASES for Common Traits**:
```python
TRAIT_ALIASES = {
    # Serde traits - most commonly searched
    "serde::Deserialize": "serde::de::Deserialize",
    "serde::Serialize": "serde::ser::Serialize",
    "serde::Deserializer": "serde::de::Deserializer", 
    "serde::Serializer": "serde::ser::Serializer",
    
    # Standard library traits
    "std::fmt::Display": "std::fmt::Display",
    "std::fmt::Debug": "std::fmt::Debug", 
    "std::clone::Clone": "std::clone::Clone",
    "std::cmp::PartialEq": "std::cmp::PartialEq",
    
    # Async traits
    "tokio::AsyncRead": "tokio::io::AsyncRead",
    "tokio::AsyncWrite": "tokio::io::AsyncWrite",
    "futures::Future": "std::future::Future",
    
    # Error handling
    "std::error::Error": "std::error::Error",
    "anyhow::Error": "anyhow::Error"
}
```

**Enhanced Trait Ranking**:
- **Trait Boost**: 1.15x multiplier for trait items in search results
- **Implementation Priority**: Prioritize trait implementations over trait definitions
- **Usage Context**: Boost traits with more implementing types
- **Documentation Quality**: Higher ranking for well-documented traits with examples

**Trait Search Performance**:
- **Alias Resolution**: O(1) lookup for common trait patterns
- **Index Usage**: Specialized indexes for trait type items
- **Cache Efficiency**: Trait search results cached for popular patterns
- **Implementation Search**: Fast lookup of types implementing specific traits

### Trait Discovery Features

**Enhanced Error Messages**:
```python
# Example enhanced error response
{
    "error": "ItemNotFoundError",
    "message": "Trait 'serde::Deserialize' not found",
    "suggestions": [
        "serde::de::Deserialize",
        "serde::de::DeserializeOwned", 
        "serde::de::Deserializer"
    ],
    "aliases": {
        "serde::Deserialize": "serde::de::Deserialize"
    }
}
```

**Trait Relationship Discovery**:
- **Supertrait Resolution**: Find parent traits automatically
- **Associated Types**: Include associated type information in results
- **Blanket Implementations**: Surface relevant blanket implementations
- **Trait Bounds**: Show common trait bound patterns

## Implementation Phases

### Phase 1: Enhanced Database Schema
1. Add new fields to PASSAGES table with migration support
2. Create EXAMPLES table with proper foreign key relationships
3. Add composite indexes for performance optimization
4. Implement backward compatibility for existing data

### Phase 2: Enhanced Parsing Pipeline
1. Extend ingest.py to parse complete rustdoc JSON structure
2. Implement module hierarchy extraction and parent-child relationships
3. Add code example extraction from documentation comments
4. Integrate enhanced parsing with existing ingestion workflow

### Phase 3: Navigation and Enhanced Search
1. Implement navigation.py module for tree operations
2. Add enhanced MCP tools with type filtering and signature matching
3. Integrate example retrieval with search results
4. Add module-scoped search capabilities

### Phase 4: API Enhancement and Integration
1. Update Pydantic models for enhanced request/response schemas
2. Implement new REST endpoints for enhanced functionality
3. Update FastMCP integration for new tool definitions
4. Add comprehensive testing for all enhanced features

## Future Considerations (Enhanced)

- Cross-crate dependency search capabilities with enhanced metadata
- GPU acceleration for embedding generation of complete documentation
- Multi-tenant quota management with per-crate rate limiting
- Distributed caching with Redis for horizontal scaling of enhanced data
- Real-time incremental updates via docs.rs webhooks with example tracking
- Advanced search features (semantic similarity across code examples)
- Analytics and usage tracking for enhanced feature optimization
- Authentication and authorization for enterprise deployments with enhanced APIs