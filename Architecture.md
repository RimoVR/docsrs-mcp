# docsrs-mcp Architecture

## System Overview

The docsrs-mcp server provides Model Context Protocol (MCP) endpoints for querying Rust crate documentation using vector search. It consists of a FastAPI web layer, an asynchronous ingestion pipeline, and a SQLite-based vector storage system.

## High-Level Architecture

```mermaid
graph TB
    subgraph "AI Clients"
        AI[AI Agent/LLM]
    end
    
    subgraph "MCP Server"
        API[FastAPI Application]
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
    
    AI -->|MCP POST| RL
    RL --> API
    API -->|enqueue| Queue
    Queue -->|dequeue| IW
    IW -->|fetch rustdoc JSON| DOCS
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
            APP[app.py<br/>FastAPI instance]
            ROUTES[routes.py<br/>MCP endpoints]
            MODELS[models.py<br/>Pydantic schemas]
            MW[middleware.py<br/>Rate limiting]
        end
        
        subgraph "Ingestion Layer"
            ING[ingest.py<br/>Download & process]
            CHUNK[chunker.py<br/>Extract items]
            EMBED[embedder.py<br/>Text to vectors]
        end
        
        subgraph "Storage Layer"
            DB[database.py<br/>SQLite operations]
            VSS[vector_search.py<br/>k-NN queries]
            CACHE[cache_manager.py<br/>LRU eviction]
        end
        
        subgraph "Utilities"
            CLI[cli.py<br/>Entry point]
            CONFIG[config.py<br/>Settings]
            ERRORS[errors.py<br/>Custom exceptions]
        end
    end
    
    APP --> ROUTES
    ROUTES --> MODELS
    APP --> MW
    ROUTES --> ING
    ING --> CHUNK
    CHUNK --> EMBED
    EMBED --> DB
    DB --> VSS
    DB --> CACHE
    CLI --> APP
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
    API->>DB: Check if crate@version exists
    
    alt Cache Miss
        API->>Queue: Enqueue ingest task
        Queue->>Worker: Dequeue task
        Worker->>Worker: Acquire per-crate lock
        Worker->>DocsRS: GET /crate/{name}/{version}/json.zst
        DocsRS-->>Worker: Compressed rustdoc JSON
        Worker->>Worker: Decompress & parse
        Worker->>Worker: Chunk into items
        Worker->>Embed: Batch embed (size=32)
        Embed-->>Worker: 384-dim vectors
        Worker->>DB: Batch insert (size=1000)
        Worker->>DB: Index with vss_index!
        Worker->>Worker: Check cache size
        alt Cache > 2 GiB
            Worker->>DB: Delete oldest DBs
        end
    end
    
    API->>DB: Vector search query
    DB-->>API: Top-k results
    API-->>Client: JSON response
```

## Database Schema

```mermaid
erDiagram
    PASSAGES {
        INTEGER id PK
        TEXT item_id "stable rustdoc ID"
        TEXT item_path "e.g. serde::de::Deserialize"
        TEXT header "item signature"
        TEXT doc "full documentation"
        INTEGER char_start "original position"
        BLOB vec "384-dim float32 array"
    }
    
    VSS_PASSAGES {
        BLOB vec "FAISS index"
    }
    
    META {
        TEXT crate
        TEXT version
        INTEGER ts "ingestion timestamp"
        TEXT target "e.g. x86_64-unknown-linux-gnu"
    }
    
    PASSAGES ||--|| VSS_PASSAGES : "indexed by"
    META ||--|| PASSAGES : "describes"
```

## MCP Tool Endpoints

```mermaid
graph TD
    subgraph "MCP Tools"
        MANIFEST[GET /mcp/manifest<br/>Tool definitions]
        SUMMARY[POST /mcp/tools/get_crate_summary<br/>Crate overview]
        SEARCH[POST /mcp/tools/search_items<br/>Vector search]
        DOC[POST /mcp/tools/get_item_doc<br/>Full rustdoc]
        VERSIONS[GET /mcp/resources/versions<br/>List versions]
    end
    
    subgraph "Support Endpoints"
        HEALTH[GET /health<br/>Liveness probe]
    end
    
    MANIFEST --> SUMMARY
    MANIFEST --> SEARCH
    MANIFEST --> DOC
    MANIFEST --> VERSIONS
```

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
      
    Storage
      SQLite
      sqlite-vss (vector extension)
      FAISS (vector index backend)
      
    ML/Embedding
      FastEmbed
      ONNX Runtime
      BAAI/bge-small-en-v1.5
      
    HTTP Client
      aiohttp
      zstandard (decompression)
      orjson (JSON parsing)
      
    Deployment
      uvx (zero-install)
      uv build (packaging)
      Docker (optional)
      PyPI distribution
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
    VersionResolve --> Error404: Not found
    VersionResolve --> Error410: Yanked
    
    Download --> Decompress: Success
    Download --> Error504: Timeout
    Download --> Error404: No rustdoc JSON
    
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
        DEV[uv sync --dev<br/>uv run python -m docsrs_mcp.cli]
        TEST[uvx --from . docsrs-mcp]
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

## Performance Characteristics

| Component | Target | Notes |
|-----------|--------|-------|
| Warm search latency | < 500ms P95 | Vector search + result formatting |
| Cold ingest | < 3s | For crates up to 10 MiB compressed |
| Memory usage | < 1 GiB RSS | Including ONNX model + FAISS indices |
| Cache size | < 2 GiB | Auto-evicted LRU |
| Rate limit | 30 req/s per IP | Via slowapi middleware |
| Concurrent ingests | 1 per crate@version | asyncio.Lock prevents duplicates |

## Security Model

```mermaid
graph LR
    subgraph "Input Validation"
        IV[Pydantic Models<br/>extra=forbid]
    end
    
    subgraph "Origin Control"
        OC[HTTPS only<br/>docs.rs domain]
    end
    
    subgraph "Size Limits"
        SL[30 MiB compressed<br/>100 MiB decompressed]
    end
    
    subgraph "Path Safety"
        PS[Sanitized filenames<br/>cache/{crate}/{version}.db]
    end
    
    subgraph "Rate Limiting"
        RL[30 req/s per IP<br/>slowapi middleware]
    end
    
    IV --> OC
    OC --> SL
    SL --> PS
    PS --> RL
```

## Future Considerations (Out of Scope v1)

- Cross-crate search capabilities
- GPU acceleration for embeddings
- Multi-tenant quota management
- Distributed caching with Redis
- Analytics and usage tracking
- Authentication and authorization
- Popularity-based ranking
- Real-time updates via webhooks