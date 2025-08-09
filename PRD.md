# Product & Technical Requirements Document (PRD + TRD)

**Project:** **docsrs-mcp** — a minimal, self-hostable Model Context Protocol (MCP) server that lets AI agents query Rust crate documentation without API keys.
**Revision:** v1.0 (2025-08-04)
**Owner:** Solo hobby developer

---

## 1 · Purpose

Let local AI coding assistants

1. fetch a crate's high-level description and complete item documentation index,
2. run semantic (vector-based) search across all documented items with type filtering,
3. navigate the full module hierarchy with structured tree access,
4. extract and search code examples from documentation,
5. retrieve the full rustdoc for any documented item –

all through the open MCP tool-calling standard, **without** proprietary services or cloud credentials.

---

## 2 · Design principles

* **Good enough, right now.** Ship the smallest feature set that satisfies the three goals above.
* **Efficient & simple.** Target ≤ 500 ms P95 latency (warm search), ≤ 1 GiB RAM under 10 k embeddings, one Python process, and a file-backed SQLite cache.
* **KISS.** No optional knobs unless strictly required.

---

## 3 · Scope

| Capability                | Notes                                                                                                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `get_crate_summary`       | Return crate name, resolved version, README/description (≤ 1 k chars), comprehensive item index, module hierarchy, and tool links.                             |
| `search_items`            | k-NN cosine search across all documented items with type filtering (functions, structs, traits, modules). Default k = 5.                                        |
| `get_item_doc`            | Complete markdown/HTML for a single item, incl. code blocks. Supports fuzzy path matching with helpful suggestions when exact paths fail. **⚠️ Path resolution requires exact module paths - needs common alias mapping (e.g., serde::Deserialize → serde::de::Deserialize)**. |
| `get_module_tree`         | Navigate module hierarchy with structured tree representation and item counts.                                                                                   |
| `search_examples`         | 🚨 **CRITICAL BUG**: Search and retrieve code examples from documentation with context. **Currently returns individual characters instead of complete code blocks due to string iteration bug in ingest.py:761**. |
| `list_versions`           | List all published versions (`yanked` flag, latest default).                                                                                                     |
| MCP discovery             | `/mcp/manifest`, `/mcp/tools/*` per MCP 2025-06-18 spec with enhanced boolean parameter declarations using anyOf patterns for better client compatibility. **⚠️ Current MCP parameter validation rejects numeric values like k=2 - needs anyOf pattern fixes**. |
| On-disk cache             | One **SQLite + VSS** file per `crate@version`, auto-evicted LRU when total > 2 GiB.                                                                              |
| **Reliability hardening** | \* Per-crate ingest lock prevents duplicate downloads. \* Graceful fallback when rustdoc JSON is unavailable (see §6.2). \* Batch writes keep peak RAM in check. |

**Out of scope (v1):** cross-crate search, authentication, analytics, popularity ranking, GPU off-load, multi-tenant quotas.

**Planned features (v1.1+):**
- Optional pre-ingestion of popular Rust crates for improved cold-start performance
- Enhanced MCP tool descriptions with embedded tutorials for better AI agent usability
- Fuzzy path matching improvements with enhanced scoring algorithms for better item path resolution

**Planned features (v1.2+):**
- Documentation snippets with context (200+ char with surrounding content)
- Cross-reference support (parse and resolve intra-doc links)
- Version diff support (compare documentation between crate versions)
- Export capabilities (JSON, Markdown formats)
- Enhanced batch operations (memory-aware, transaction-safe)

## 3.1 · Critical Bug Fixes Required

**🚨 URGENT - BROKEN FUNCTIONALITY:**

1. **searchExamples Character Fragmentation Bug** (CRITICAL - BLOCKING)
   - **Problem**: `ingest.py:761` iterates over code example strings as individual characters instead of treating them as complete code blocks
   - **Impact**: searchExamples returns `["[", "{", "\""]` instead of actual code examples
   - **Root Cause**: String iteration in code example processing logic
   - **Fix Required**: Treat code example strings as single entities, not character sequences

2. **MCP Parameter Type Validation** (HIGH)
   - **Problem**: MCP parameter validation rejects numeric values like `k=2`
   - **Impact**: Tool calls with numeric parameters fail unexpectedly
   - **Fix Required**: Add anyOf patterns in MCP manifest for consistent type handling

3. **Pre-ingestion Parameter Synchronization** (MEDIUM)
   - **Problem**: CLI/MCP mode parameter synchronization issue causing pre-ingestion conflicts
   - **Impact**: Pre-ingestion settings may not be properly synchronized between CLI and MCP modes
   - **Fix Required**: Ensure consistent parameter handling across CLI and MCP interfaces

4. **Path Alias Resolution** (MEDIUM)
   - **Problem**: Path resolution requires exact module paths, no common aliases supported
   - **Impact**: Users must know precise internal paths (e.g., `serde::de::Deserialize` not `serde::Deserialize`)
   - **Fix Required**: Add path alias mapping for common patterns

**ACTUAL FEATURE STATUS (Live Testing):**
- ❌ searchExamples: BROKEN (character fragmentation bug)
- ✅ Fuzzy path resolution: WORKING (provides suggestions)
- ✅ Error messages: GOOD (helpful suggestions)  
- ✅ Cache warming: EXISTS (LRU with TTL)
- ⚠️ Trait search: PARTIAL (foundation exists)
- ❌ See-also suggestions: MISSING
- ❌ Dependency graph: MISSING

---

## 4 · User stories

**Current (v1.0):**
1. **AI assistant:** "I know only *serde* – give me a JSON summary of the latest docs so I can explain them."
2. **AI assistant:** "Provide rustdoc for `serde::de::Deserialize` in v1.0.197 so I can craft an example."
3. **Developer:** "I want to run the server with a single `uvx` command from a GitHub URL."

**Planned (v1.1+):**
4. **Server operator:** "I want the server to pre-ingest the top 100 most-used crates on startup so common queries are instant."
5. **AI agent:** "I need tutorial-style examples embedded in the MCP tool descriptions to understand how to use each tool effectively."
6. **Developer:** "I need zero cold-start latency through embeddings warmup for immediate responses."
7. **AI assistant:** "I want improved path resolution accuracy through enhanced fuzzy matching when exact paths aren't found."
8. **Documentation explorer:** "I need richer documentation context in search results with surrounding content."
9. **Version analyst:** "I want to compare documentation between different crate versions to track API evolution."

---

## 5 · High-level architecture

```
          AI agent  (MCP POSTs)
                 ▲
                 │
   ┌─ docsrs_mcp.app ──┐
   │  MCP tools layer  │
   │  ─ get_crate_summary
   │  ─ search_items
   │  ─ get_item_doc
   │  ─ list_versions   │
   └────────┬───────────┘
            │   asyncio.Queue
            ▼
  docs.rs CDN ───► ingest worker
                   • download + decompress
                   • chunk & embed
                   • write SQLite+VSS
            │
            ▼
        cache/*.db
```

**Runtime stack**: Python 3.10+, FastAPI, Uvicorn + uvloop, `sqlite-vss` (FAISS), FastEmbed (ONNX model `BAAI/bge-small-en-v1.5`, 384 d), RapidFuzz (fuzzy path matching).
**Launch**: `uvx` executes console-script directly from PyPI or a Git ref (zero install).

---

## 6 · Component design (technical)

### 6.0 Performance Improvements

#### 6.0.1 Embeddings Warmup

**Purpose:** Eliminate cold-start latency through ONNX offline optimization and model pre-warming.

**Technical Implementation:**
- **Warmup Strategy:** Default enabled embeddings warmup using ONNX offline optimization
- **Performance Gain:** ~400ms cold-start latency reduction through model pre-warming
- **Optimization:** Offline model serialization provides 60-80% startup time reduction
- **Implementation:** First dummy embed at startup warms model components and initializes ONNX runtime
- **Memory Impact:** Minimal additional memory overhead for pre-warmed model state
- **Result:** Warm embeddings achieve <100ms for all operations vs. previous cold-start penalties

**Value Proposition:**
- **Immediate Responsiveness:** Zero cold-start latency for embedding operations
- **Consistent Performance:** Predictable response times from first request
- **User Experience:** AI agents receive instant responses without initial delays

### 6.1 Planned Feature Specifications (v1.1+)

#### 6.1.1 Optional Pre-ingestion of Popular Crates

**Purpose:** Eliminate cold-start latency for the most commonly queried Rust crates by pre-loading them during server startup.

**Technical Requirements:**
- **Scope:** Top 100-500 most-used crates based on crates.io download statistics
- **Implementation:** Background ingestion worker with configurable concurrency (default: 3 parallel downloads)
- **Performance:** Non-blocking startup - server accepts requests immediately while pre-ingestion runs
- **Configuration:** CLI flag `--pre-ingest` to enable/disable (default: disabled for lean startup)
- **Cache Management:** Priority-aware LRU eviction - pre-ingested crates have higher retention priority
- **Resource Control:** Respects existing memory/disk limits (≤ 1 GiB RAM, ≤ 2 GiB cache)
- **Monitoring:** Progress logging and `/health` endpoint reports pre-ingestion status

**Value Proposition:**
- **Performance:** Sub-100ms response times for popular crates (tokio, serde, clap, etc.)
- **User Experience:** AI agents get instant responses for common documentation queries
- **Predictable Latency:** Eliminates the 2-3 second cold-start penalty for frequently used crates

**Implementation Strategy:**
```python
# CLI integration
@click.option('--pre-ingest', is_flag=True, help='Pre-ingest popular crates on startup')

# Background worker
async def pre_ingest_popular_crates(crate_list: List[str], concurrency: int = 3):
    semaphore = asyncio.Semaphore(concurrency)
    tasks = [ingest_crate_with_semaphore(semaphore, crate) for crate in crate_list]
    await asyncio.gather(*tasks, return_exceptions=True)
```

#### 6.1.2 Enhanced Fuzzy Path Matching

**Purpose:** Improve item path resolution accuracy through enhanced scoring algorithms while maintaining current working implementation.

**Technical Requirements:**
- **Current Status:** RapidFuzz implementation works but needs improvement for better accuracy
- **Enhancement Scope:** Extend existing RapidFuzz implementation with enhanced scoring algorithms
- **Compatibility:** Maintain backward compatibility with current fuzzy matching behavior
- **Performance:** Preserve existing performance characteristics while improving accuracy
- **Scoring Improvements:** Enhanced similarity scoring for better path suggestion relevance

**Value Proposition:**
- **Accuracy:** Better path resolution reduces user frustration with typos and variations
- **Discoverability:** More intelligent suggestions help users find intended documentation
- **Consistency:** Improved scoring provides more predictable and relevant results

#### 6.1.3 Enhanced MCP Tool Descriptions with Tutorials

**Purpose:** Provide AI agents with embedded, token-efficient tutorials within MCP tool schemas to improve tool usage accuracy and reduce trial-and-error.

**Technical Requirements:**
- **Format:** Structured examples embedded in JSON schema descriptions using compact, LLM-optimized format
- **Content:** Common usage patterns, parameter combinations, and expected response structures
- **Compatibility:** Backward-compatible with existing MCP clients - tutorials are additive metadata
- **Size Optimization:** Concise format designed for LLM token efficiency (≤ 200 tokens per tutorial)
- **Coverage:** All six core tools (get_crate_summary, search_items, get_item_doc, search_examples, get_module_tree, list_versions)

**Value Proposition:**
- **Accuracy:** Reduces incorrect tool usage by providing clear examples in context
- **Efficiency:** Eliminates need for separate documentation lookups or trial-and-error
- **Adoption:** Lowers barrier to entry for new MCP clients and AI agents
- **Self-Documenting:** Tools become self-explanatory through embedded guidance

**Implementation Strategy:**
```python
# Enhanced tool schema example
{
  "name": "search_items",
  "description": "Vector search across crate items with type filtering.\\n\\n📖 Tutorial:\\n• Basic: search_items(crate='tokio', query='spawn task')\\n• Filtered: search_items(crate='serde', query='deserialize', type_filter='trait')\\n• Tuned: search_items(crate='clap', query='argument parsing', k=10)\\n→ Returns: [{score, item_path, item_type, header, snippet}, ...]",
  "inputSchema": { ... }
}
```

---

### 6.2 Planned Feature Specifications (v1.2+)

#### 6.2.1 Documentation Snippets with Context

**Purpose:** Provide richer documentation context in search results with surrounding content for better comprehension.

**Technical Requirements:**
- **Context Length:** 200+ character snippets with surrounding content
- **Contextual Boundaries:** Intelligent snippet extraction respecting paragraph and section boundaries
- **Relevance Preservation:** Maintain search relevance while providing expanded context
- **Format Consistency:** Structured snippet format with clear context indicators

#### 6.2.2 Cross-Reference Support

**Purpose:** Parse and resolve intra-doc links for comprehensive documentation navigation.

**Technical Requirements:**
- **Link Resolution:** Parse rustdoc cross-references and resolve to target items
- **Navigation Support:** Provide structured access to related documentation items
- **Integrity Validation:** Verify link targets exist and are accessible

#### 6.2.3 Version Diff Support

**Purpose:** Compare documentation between crate versions to track API evolution.

**Technical Requirements:**
- **Version Comparison:** Side-by-side documentation comparison between versions
- **Change Detection:** Identify added, removed, and modified documentation items
- **Diff Visualization:** Structured diff output highlighting changes

#### 6.2.4 Export Capabilities

**Purpose:** Enable documentation export in multiple formats for integration workflows.

**Technical Requirements:**
- **Format Support:** JSON and Markdown export formats
- **Completeness:** Export complete documentation with metadata preservation
- **Structure Preservation:** Maintain hierarchical organization in exports

#### 6.2.5 Enhanced Batch Operations

**Purpose:** Memory-aware, transaction-safe batch operations for improved reliability.

**Technical Requirements:**
- **Memory Management:** Intelligent batching based on available memory
- **Transaction Safety:** ACID-compliant batch operations with rollback support
- **Progress Tracking:** Detailed progress reporting for long-running batch operations

### 6.3 FastAPI application (`docsrs_mcp.app`)

| Route                          | Method | Purpose                                           | Response                                         |
| ------------------------------ | ------ | ------------------------------------------------- | ------------------------------------------------ |
| `/mcp/manifest`                | GET    | Static MCP manifest listing tools & JSON schemas. | JSON                                             |
| `/mcp/tools/get_crate_summary` | POST   | Resolve version & return comprehensive summary.   | JSON with item index & module tree              |
| `/mcp/tools/search_items`      | POST   | Vector search with type filtering.                | JSON array `{score, item_path, item_type, header, snippet}` |
| `/mcp/tools/get_item_doc`      | POST   | Full rustdoc for `item_path` with fuzzy matching. | `text/markdown` or suggestions JSON on path miss |
| `/mcp/tools/get_module_tree`   | POST   | Navigate module hierarchy.                        | JSON tree structure with item counts            |
| `/mcp/tools/search_examples`   | POST   | 🚨 **BROKEN**: Search code examples in documentation. **Returns individual chars ["[", "{", "\""] instead of code blocks**.           | JSON array `{score, item_path, example_text, context}` |
| `/mcp/resources/versions`      | GET    | List published versions.                          | JSON                                             |
| `/health`                      | GET    | Liveness/readiness probe.                         | `200 OK`                                         |

Responses validate against in-repo JSON Schema before send (pydantic). MCP manifest uses anyOf patterns for boolean parameters to ensure consistent client compatibility across different MCP implementations.

---

### 6.4 Ingestion pipeline (`ingest.py`)

| Step                                                                                                                                                                                                                             | Detail                                                                                                                                                                                                                                                |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **0. Acquire lock**                                                                                                                                                                                                              | `asyncio.Lock` keyed by `crate@version` ensures a single ingest per crate version.                                                                                                                                                                    |
| **1. Resolve version**                                                                                                                                                                                                           | `GET https://docs.rs/crate/{crate}/latest/json` (or explicit version) → 302 target file. Honour `~semver` selectors & optional `target=` triple.                                                                                                      |
| **1b. Fallback**                                                                                                                                                                                                                 | If rustdoc JSON is `404` or missing (older crates never rebuilt) → return `crate_not_documented` error **without** building locally, keeping MVP lean. *Docs.rs hosts rustdoc-JSON for the majority of modern crates but not all.* ([docs.rs][1])     |
| **2. Download & decompress**                                                                                                                                                                                                     | Supports `.json`, `.json.zst` (preferred), `.json.gz`. Enforce max compressed 30 MiB and max decompressed 100 MiB; only `https://docs.rs/` URLs accepted.                                                                                             |
| **3. Chunk**                                                                                                                                                                                                                     | One passage per item (`fn/struct/trait/mod`) with comprehensive item indexing. Embedded text = `header + "\n\n" + docstring` (code blocks excluded from embedding but stored separately for example search). Store `item_path`, stable `item_id`, `item_type`, and extracted code examples from rustdoc JSON. **🚨 CRITICAL: ingest.py:761 has character iteration bug - treats code example strings as individual characters instead of complete blocks**. |
| **4. Embed**                                                                                                                                                                                                                     | FastEmbed ONNX, batch-size 32. Embeddings warmup (default enabled) using ONNX offline optimization eliminates cold-start latency (~400ms reduction). Offline model serialization provides 60-80% startup time reduction for consistent <100ms warm performance.                                                                                                                                                                   |
| **5. Persist**                                                                                                                                                                                                                   | Enhanced SQLite schema per crate-db:  \`\`\`sql                                                                                                                                                                                                       |
| CREATE TABLE passages(                                                                                                                                                                                                           |                                                                                                                                                                                                                                                       |
| id INTEGER PRIMARY KEY,                                                                                                                                                                                                          |                                                                                                                                                                                                                                                       |
| item\_id TEXT,           -- stable rustdoc identifier                                                                                                                                                                            |                                                                                                                                                                                                                                                       |
| item\_path TEXT NOT NULL,                                                                                                                                                                                                        |                                                                                                                                                                                                                                                       |
| item\_type TEXT,         -- function, struct, trait, module, etc                                                                                                                                                                |                                                                                                                                                                                                                                                       |
| header TEXT,                                                                                                                                                                                                                     |                                                                                                                                                                                                                                                       |
| doc TEXT,                                                                                                                                                                                                                        |                                                                                                                                                                                                                                                       |
| char\_start INTEGER,     -- first char of original doc                                                                                                                                                                           |                                                                                                                                                                                                                                                       |
| vec BLOB                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                       |
| );                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                       |
| CREATE TABLE code\_examples(                                                                                                                                                                                                     |                                                                                                                                                                                                                                                       |
| id INTEGER PRIMARY KEY,                                                                                                                                                                                                          |                                                                                                                                                                                                                                                       |
| item\_id TEXT,                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                       |
| item\_path TEXT,                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                       |
| example\_text TEXT,                                                                                                                                                                                                              |                                                                                                                                                                                                                                                       |
| context TEXT,                                                                                                                                                                                                                    |                                                                                                                                                                                                                                                       |
| vec BLOB                                                                                                                                                                                                                         |                                                                                                                                                                                                                                                       |
| );                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                       |
| CREATE TABLE module\_tree(                                                                                                                                                                                                       |                                                                                                                                                                                                                                                       |
| path TEXT PRIMARY KEY,                                                                                                                                                                                                           |                                                                                                                                                                                                                                                       |
| parent\_path TEXT,                                                                                                                                                                                                               |                                                                                                                                                                                                                                                       |
| item\_count INTEGER                                                                                                                                                                                                              |                                                                                                                                                                                                                                                       |
| );                                                                                                                                                                                                                               |                                                                                                                                                                                                                                                       |
| CREATE VIRTUAL TABLE vss\_passages USING vss0(vec);                                                                                                                                                                              |                                                                                                                                                                                                                                                       |
| CREATE VIRTUAL TABLE vss\_examples USING vss0(vec);                                                                                                                                                                              |                                                                                                                                                                                                                                                       |
| CREATE TABLE meta(crate TEXT, version TEXT, ts INTEGER, target TEXT);                                                                                                                                                            |                                                                                                                                                                                                                                                       |
| PRAGMA user\_version = 2;                                                                                                                                                                                                        |                                                                                                                                                                                                                                                       |
| \`\`\`  Insert vectors in batches of **1 000** rows, then call `vss_index!` to keep FAISS peak RAM well under 1 GiB even on big crates. FAISS keeps the index in memory; batching prevents a momentary spike. ([Hacker News][2]) |                                                                                                                                                                                                                                                       |
| **6. Cache eviction**                                                                                                                                                                                                            | After a successful ingest, if `cache/*/*.db` totals > 2 GiB, delete oldest databases by `meta.ts` until under limit.                                                                                                                                  |

---

### 6.5 Vector search query

```sql
SELECT id,
       item_path,
       item_type,
       header,
       doc,
       1.0 - vss_distance(vec, :qv) AS score
FROM passages
WHERE (:type_filter IS NULL OR item_type = :type_filter)
ORDER BY score DESC
LIMIT :k;
```

`score ∈ [0, 1]` (higher = better). Type filtering enables targeted search across functions, structs, traits, or modules.

---

### 6.6 Concurrency & performance

* Single Uvicorn worker avoids duplicating the 45 MiB ONNX model.
* All embedding calls `asyncio.to_thread`, keeping the event loop non-blocking.
* Per-IP rate-limit **30 req/s** via *slowapi* middleware.
* Target warm search: < 50 ms query, < 500 ms end-to-end.

---

### 6.7 Security

* **Origin allow-list:** only fetch from `https://docs.rs/…`.
* **Size caps:** 30 MiB compressed / 100 MiB decompressed.
* **Path safety:** databases saved as `cache/{crate}/{version}.db` using sanitised names.
* **Input validation:** strict pydantic models with comprehensive parameter validation; reject unknown fields (`extra="forbid"`). **⚠️ NEEDS FIX**: All numeric parameters (k, limit, offset) must validate against reasonable bounds and accept both integer and string types using anyOf schema patterns. Boolean parameters (has_examples, deprecated) use consistent anyOf declarations matching numeric parameter handling.
* **Fuzzy path resolution:** when exact item paths are not found, RapidFuzz provides intelligent path suggestions with similarity scoring to guide users to the intended documentation items.
* **Error model**

| Condition            | HTTP | Body                               |
| -------------------- | ---- | ---------------------------------- |
| crate unknown        | 404  | `{"error":"crate_not_found"}`      |
| version yanked       | 410  | `{"error":"version_yanked"}`       |
| rustdoc JSON missing | 404  | `{"error":"crate_not_documented"}` |
| upstream timeout     | 504  | `{"error":"upstream_timeout"}`     |
| rate-limit           | 429  | `{"error":"too_many_requests"}`    |
| bad input            | 400  | pydantic validation obj            |
| item path not found  | 404  | `{"error":"item_not_found", "suggestions":["similar_path1", "similar_path2"]}` |

---

### 6.8 CLI entry point

```python
# Current implementation
def main() -> None:
    import uvicorn, os
    uvicorn.run(
        "docsrs_mcp.app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        loop="uvloop",
        workers=1,
    )

# Planned (v1.1+) with pre-ingestion support
def main() -> None:
    import click, uvicorn, os, asyncio
    
    @click.command()
    @click.option('--port', default=8000, help='Server port')
    @click.option('--pre-ingest', is_flag=True, help='Pre-ingest popular crates')
    def cli(port: int, pre_ingest: bool):
        if pre_ingest:
            asyncio.create_task(pre_ingest_popular_crates())
        
        uvicorn.run(
            "docsrs_mcp.app:app",
            host="0.0.0.0",
            port=port,
            loop="uvloop",
            workers=1,
        )
    
    cli()
```

### 6.9 Packaging (uv-native)

```toml
[project]
name = "docsrs-mcp"
version = "0.1.0"
dependencies = [
  "fastapi>=0.111",
  "uvicorn[standard]>=0.30",
  "sqlite-vss>=0.3.3,<0.4",
  "fastembed>=0.7.0",
  "aiohttp>=3.9",
  "orjson>=3.9",
  "zstandard>=0.22",
  "slowapi>=0.1",
  "rapidfuzz>=3.0"
]

[project.scripts]
docsrs-mcp = "docsrs_mcp.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
  "pytest>=7.0",
  "ruff>=0.1.0"
]
```

**Package Management**: All dependencies managed exclusively through `uv`:
- `uv add package` - Add production dependency
- `uv add --dev package` - Add development dependency  
- `uv sync` - Install locked dependencies
- `uv build` - Create wheel/sdist for PyPI

### 6.10 Zero-install launch

```bash
uvx --from "git+https://github.com/<user>/docsrs-mcp.git" docsrs-mcp --port 8000
# or
uvx docsrs-mcp@latest
```

---

## 7 · Non-functional requirements

| Category          | Requirement                                                                            |
| ----------------- | -------------------------------------------------------------------------------------- |
| **Portability**   | CPython 3.10+; CI on Ubuntu 22.04, macOS 14, Windows 11 (GitHub Actions).              |
| **Performance**   | All search operations: ≤ 500 ms P95 (warm) across expanded dataset; ≤ 3 s cold ingest for crates ≤ 10 MiB compressed with full item indexing. Warm embeddings: <100ms for all operations through pre-warming optimization. |
| **Memory**        | ≤ 1 GiB RSS incl. ONNX & FAISS under 10 k vectors (with pre-ingestion enabled).       |
| **Disk**          | Cache auto-evicts to ≤ 2 GiB (priority-aware LRU for pre-ingested crates).            |
| **Availability**  | Stateless web layer; cache can be rebuilt.                                             |
| **Observability** | `/health` endpoint; debug logs to stdout.                                              |
| **Security**      | See §6.7.                                                                              |
| **License**       | MIT for server code and dependencies (FastEmbed/BGE model is MIT).                     |

---

## 8 · Deployment & CI

| Stage             | Action                                                                          |
| ----------------- | ------------------------------------------------------------------------------- |
| PR checks         | `uv sync --dev`; `uv run ruff check`; `uv run ruff format --check`; `uv run pytest -q`; `uvx --from . docsrs-mcp --help` |
| Release           | `git tag vX.Y.Z` triggers GH Action → `uv build` → upload to PyPI.      |
| Docker (uv-based) | `FROM python:slim` → `RUN pip install uv` → `COPY . .` → `RUN uv sync --frozen` → `CMD ["uv", "run", "docsrs-mcp"]`. |
| Runtime targets   | Fly.io, Railway, Render, or any VPS ≥ 256 MiB RAM (all uv-compatible).                              |

---

## 9 · Acceptance criteria

1. `uvx docsrs-mcp` starts on Linux, macOS, and Windows (zero-install).
2. `uv sync --dev && uv run python -m docsrs_mcp.cli` works for development.
3. `POST /mcp/tools/get_crate_summary` for crate **tokio** (version omitted) returns latest version & overview.
4. `POST /mcp/tools/search_items` with query `"spawn task"` and optional type filter returns ≥ 1 passage containing `tokio::spawn` with item type information.
5. `POST /mcp/tools/get_module_tree` for crate **tokio** returns structured hierarchy with item counts per module.
6. **🚨 CRITICAL FIX REQUIRED**: `POST /mcp/tools/search_examples` with query `"async"` returns ≥ 1 **complete code block** from tokio documentation (not individual characters like `["[", "{", "\""]`).
7. `POST /mcp/tools/get_item_doc` with `item_path="tokio::spawn"` returns markdown including a runnable example.
7b. `POST /mcp/tools/get_item_doc` with a slightly misspelled `item_path="tokio::spwan"` returns fuzzy match suggestions including `"tokio::spawn"`.
7c. **⚠️ PATH ALIAS FIX REQUIRED**: `POST /mcp/tools/get_item_doc` with common alias `item_path="serde::Deserialize"` automatically resolves to `serde::de::Deserialize` without user needing exact internal path.
8. Cold ingest of crate **serde** (< 5 MiB compressed) with full item indexing and example extraction completes in ≤ 2 s on an M1/Ryzen 5; crates up to 10 MiB complete in ≤ 3 s.
9. After ingesting 50 average crates with comprehensive indexing, `cache/` ≤ 2 GiB and server RSS ≤ 1 GiB.
10. A single IP exceeding 30 requests/s receives HTTP 429.
11. **⚠️ MCP VALIDATION FIX REQUIRED**: All numeric parameters (k=2, limit=10, type filters) are properly validated and **accept both integer and string types** using anyOf patterns with helpful error messages.
12. Boolean parameters in MCP manifest use anyOf patterns consistent with numeric parameters for improved client compatibility.
13. Fuzzy path matching provides helpful suggestions when exact item paths are not found, with similarity scores ≥ 0.6.
14. All package management operations use `uv` exclusively (no pip/conda mixing).
15. **🚨 CRITICAL**: searchExamples must return complete code blocks, not individual characters - fix ingest.py:761 character iteration bug.
16. **⚠️ REQUIRED**: MCP parameter validation accepts numeric values like k=2 through proper anyOf schema patterns.
17. **⚠️ REQUIRED**: Common path aliases (serde::Deserialize → serde::de::Deserialize) resolve automatically without user intervention.

**Planned acceptance criteria (v1.1+):**
18. **Pre-ingestion**: `docsrs-mcp --pre-ingest` successfully pre-loads top 100 crates within 5 minutes on standard hardware while serving requests.
19. **Tutorial integration**: All MCP tool descriptions include embedded tutorials accessible via `/mcp/manifest` with ≤ 200 tokens per tool.
20. **Performance**: Pre-ingested popular crates (tokio, serde, clap) respond in ≤ 100ms for search and summary operations.
21. **Cache priority**: Pre-ingested crates are retained longer during LRU eviction compared to on-demand crates.

---

## 10 · Glossary

| Term             | Definition                                                                                   |
| ---------------- | -------------------------------------------------------------------------------------------- |
| **MCP**          | *Model Context Protocol* (2025-06-18): JSON manifest + HTTP tool endpoints callable by LLMs. |
| **FastEmbed**    | Minimal Python wrapper shipping small ONNX embedding models.                                 |
| **sqlite-vss**   | SQLite extension that embeds a FAISS vector index inside SQLite tables.                      |
| **rustdoc JSON** | Structured docs emitted by `cargo +nightly rustdoc --output-format json`; hosted by docs.rs. |
| **uv**           | Fast Python package installer and resolver - exclusive infrastructure tool for this project. |
| **uvx**          | `uv` tool's *run* sub-command: executes console-scripts directly from PyPI or Git (zero-install). |

---

### End of document.

[1]: https://docs.rs/about/rustdoc-json?utm_source=chatgpt.com "Rustdoc JSON"
[2]: https://news.ycombinator.com/item?id=40243168 "I’m writing a new vector search SQLite Extension | Hacker News"
