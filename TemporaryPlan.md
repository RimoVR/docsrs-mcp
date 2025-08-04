# Refined Implementation Plan for docsrs-mcp

## Context
Building MCP server functionality for querying Rust crate documentation with vector search capabilities. The project has completed initial Python package setup (core-1) and is ready for core MCP implementation.

## Guiding Principles (KISS-Aligned)
- **Don't let perfect be the enemy of good** - Get a working MCP server first, optimize later
- **Be efficient and effective as reasonably achievable** - Focus on what delivers value
- **Keep it simple, stupid** - Start minimal, add complexity only when proven necessary

## Key Decisions from Research

### Critical Library Changes
- **MUST use sqlite-vec instead of sqlite-vss** - sqlite-vss is deprecated
- **Pin aiohttp to 3.9.5** - versions 3.10+ have severe memory leaks
- **Pin onnxruntime < 1.20** - compatibility issues with FastEmbed
- **Consider FastMCP 2.0** - but start with basic FastAPI first (KISS)

### Simplified Architecture Approach
Following KISS principles:
1. **Start with minimal viable MCP** - Basic endpoints that work
2. **Consolidate initially** - Fewer files, clearer flow
3. **Add modules only when needed** - Don't pre-optimize structure
4. **Focus on core functionality** - Vector search that actually works

## Simplified Implementation Architecture

### Initial Module Structure (KISS)
```
src/docsrs_mcp/
├── __init__.py          # Package exports
├── app.py               # FastAPI app with ALL MCP logic initially
├── cli.py               # CLI entry point (existing)
├── config.py            # Simple configuration
└── models.py            # Pydantic schemas only
```

### Later Refactoring (When Proven Necessary)
Only extract these when app.py becomes unwieldy:
- `database.py` - When DB logic exceeds ~200 lines
- `ingest.py` - When ingestion becomes complex
- `search.py` - When search needs optimization
- Others as needed

### Simplified Technical Decisions

1. **Start Simple**:
   - Basic SQLite with sqlite-vec (no complex pooling initially)
   - Simple FastEmbed usage (worry about memory leaks when they happen)
   - Direct aiohttp usage (add pooling when needed)
   - Basic error handling (enhance based on actual failures)

2. **Essential Constraints Only**:
   - sqlite-vec (not sqlite-vss)
   - aiohttp 3.9.5 (known issue)
   - onnxruntime < 1.20 (known issue)
   - Everything else: use latest stable

## Simplified Implementation Steps

### Phase 1: Get It Working (Core MCP)
1. **Minimal Config** (`config.py`)
   ```python
   CACHE_DIR = os.getenv("CACHE_DIR", "./cache")
   MODEL_NAME = "BAAI/bge-small-en-v1.5"
   # Add others as needed
   ```

2. **Essential Models** (`models.py`)
   - Just the Pydantic models for MCP requests/responses
   - Use `extra='forbid'` for safety
   - Don't over-engineer schemas

3. **MCP Endpoints in app.py**
   - Start with all logic in route handlers
   - `/mcp/manifest` - Hardcode the JSON
   - `/mcp/tools/get_crate_summary` - Basic implementation
   - `/mcp/tools/search_items` - Simple vector search
   - Extract functions only when handlers get too long

### Phase 2: Make It Work Well
4. **Add Database Support**
   - Initialize sqlite-vec in app startup
   - Simple schema: embeddings table, metadata table
   - Direct SQL queries (no ORM complexity)

5. **Add Ingestion**
   - Download from docs.rs
   - Parse JSON (don't over-optimize initially)
   - Generate embeddings in batches of 32
   - Store in database

6. **Add Search**
   - Basic k-NN query with sqlite-vec
   - Return top 5 results
   - Simple scoring

### Phase 3: Make It Production-Ready (Only What's Needed)
7. **Error Handling**
   - Add only after seeing actual errors
   - Simple try/except where failures occur
   - Clear error messages

8. **Performance**
   - Add connection pooling if/when timeouts occur
   - Add caching if/when repeated queries slow down
   - Monitor and optimize based on real usage

9. **Rate Limiting**
   - Add slowapi when actually needed
   - Start with simple in-memory counter if necessary

## What We're NOT Doing (KISS)

### Avoiding Premature Optimization
- ❌ Complex circuit breakers (until we see failures)
- ❌ Elaborate monitoring (until we need metrics)
- ❌ Multiple database connections (until we hit limits)
- ❌ Extensive middleware (until problems arise)
- ❌ Pluggable architectures (YAGNI)

### Focusing on What Matters
- ✅ Working MCP endpoints
- ✅ Actual vector search functionality
- ✅ Basic error responses
- ✅ Simple deployment with uvx

## Simplified Error Handling

### Start With Basics
```python
try:
    # do something
except Exception as e:
    logger.error(f"Failed: {e}")
    raise HTTPException(status_code=500, detail=str(e))
```

### Enhance Based on Actual Failures
- Add specific exceptions when we see patterns
- Add retry logic when we see transient failures
- Add circuit breakers when we see cascading failures

## Testing Strategy (Simple but Effective)

### Phase 1: Basic Functionality
- Test each endpoint with curl/httpie
- Verify with real crate (tokio)
- Check memory usage informally

### Phase 2: Automated Tests
- Add pytest tests for happy paths
- Add tests for known edge cases
- Skip complex mocking initially

## Deployment (Zero-Install Focus)

### Keep It Simple
- Ensure `uvx docsrs-mcp` just works
- Test on Linux/macOS/Windows
- Document any issues found

## Success Metrics (What Actually Matters)

### Must Work
- ✅ Can query tokio docs
- ✅ Search returns relevant results
- ✅ Doesn't crash under normal use
- ✅ Installs with single command

### Nice to Have (Later)
- Sub-500ms response (optimize if slow)
- Under 1GB memory (monitor and fix if exceeded)
- Handle 50 crates (test and adjust)

## Next Steps (KISS Implementation)

1. **Start Coding** - Get basic MCP manifest working
2. **Add One Endpoint** - Get crate summary working
3. **Test with Real Data** - Try with tokio crate
4. **Iterate** - Add features based on what breaks

Remember: **Perfect is the enemy of good**. Ship something that works, then improve based on real usage and feedback.