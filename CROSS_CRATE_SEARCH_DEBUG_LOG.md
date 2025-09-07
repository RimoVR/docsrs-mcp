# Cross-Crate Search Bug - Debug Log and Failed Fixes

## Problem Statement
**Issue**: "Cross-Crate Search - Cannot Search Across Multiple Crates"
**Symptom**: MCP clients cannot perform cross-crate searches using the `crates` parameter due to schema validation requiring `crate_name` parameter.
**Error Message**: `Input validation error: 'crate_name' is a required property`

## Root Cause Analysis (Systematic Investigation)

### Investigation Methodology
Used systematic agent-based analysis:
1. **Codebase Analysis Agent**: Analyzed MCP schema handling and validation layers
2. **Web Search Agent**: Researched MCP best practices for tool schema validation
3. **Codex-Bridge**: Generated comprehensive fix plan based on findings

### Key Findings
1. **Schema Definition**: Located in `src/docsrs_mcp/mcp_tools_config.py` - defines `search_items` with `required: ["crate_name", "query"]`
2. **Runtime Implementation**: Located in `src/docsrs_mcp/mcp_sdk_server.py` - `search_items()` function validates parameters
3. **MCP Tool Registration**: Located in `src/docsrs_mcp/mcp_sdk_server.py` - `handle_list_tools()` defines schema for MCP protocol
4. **Root Cause**: Runtime validation contradicts desired schema by requiring `crate_name` OR `crates` parameters, but schema only allows `query` for cross-crate search

### Validation Layers Identified
1. **JSON Schema Validation** (MCP SDK layer)
2. **Runtime Parameter Validation** (Application layer) 
3. **Service Layer Validation** (Backend layer)

## Attempted Fixes (All Failed)

### Fix Attempt #1: Schema Configuration Update
**Approach**: Modified `mcp_tools_config.py` to change required fields from `["crate_name", "query"]` to `["query"]`

**Files Modified**:
- `src/docsrs_mcp/mcp_tools_config.py:113` - Changed `"required": ["crate_name", "query"]` to `"required": ["query"]`

**Expected Result**: Schema would allow query-only searches
**Actual Result**: No change - MCP tool listing still showed `"required": ["crate_name", "query"]`
**Reason for Failure**: `get_tool_schema()` function not being called by MCP SDK server

### Fix Attempt #2: Runtime Validation Alignment  
**Approach**: Updated runtime validation in `search_items()` to align with permissive schema

**Files Modified**:
- `src/docsrs_mcp/mcp_sdk_server.py:262` - Removed error for missing crate parameters
- `src/docsrs_mcp/mcp_sdk_server.py:264` - Added default behavior for cross-crate search
- `src/docsrs_mcp/mcp_sdk_server.py:294-337` - Updated routing logic for empty crate list

**Changes Made**:
```python
# OLD (restrictive)
else:
    return {"error": {"code": "missing_parameter", "message": "Either crate_name or crates parameter required"}}

# NEW (permissive)  
else:
    # Default behavior: cross-crate search when no crate filters provided
    final_crates = []  # Empty list indicates all crates (cross-crate search)
```

**Expected Result**: Query-only searches would work with cross-crate default
**Actual Result**: Schema validation still failed before runtime code was reached
**Reason for Failure**: MCP SDK validation happening before runtime validation

### Fix Attempt #3: get_tool_schema() Function Usage
**Approach**: Used centralized schema function instead of hardcoded schemas

**Files Modified**:
- `src/docsrs_mcp/mcp_sdk_server.py:1279` - Changed from hardcoded schema to `get_tool_schema("search_items")`

**Expected Result**: Centralized schema config would be used consistently
**Actual Result**: No change in behavior
**Reason for Failure**: Function not being called (confirmed via debug logging)

### Fix Attempt #4: Direct Schema Hardcoding
**Approach**: Hardcoded correct schema directly in tool registration to bypass function calls

**Files Modified**:
- `src/docsrs_mcp/mcp_sdk_server.py:1283-1326` - Replaced function call with hardcoded schema containing `"required": ["query"]`

**Expected Result**: Direct schema definition would override all other layers
**Actual Result**: Still got same validation error
**Reason for Failure**: Unknown deeper MCP SDK validation layer

### Fix Attempt #5: anyOf Schema Validation
**Approach**: Used JSON Schema `anyOf` to allow either `crate_name` OR `crates` to be required

**Files Modified**:
- `src/docsrs_mcp/mcp_tools_config.py` - Added `anyOf` validation logic

**Expected Result**: JSON Schema would accept either parameter combination
**Actual Result**: MCP SDK ignored `anyOf` constraints  
**Reason for Failure**: MCP SDK doesn't support complex JSON Schema validation

## Investigation Results

### What We Confirmed Works
1. **get_tool_schema() Function**: Returns correct schema with `required: ["query"]`
2. **mcp_tools_config.py Updates**: File contains correct schema definition
3. **Runtime Validation Logic**: Fixed to handle empty crate lists with cross-crate default
4. **Backend Functionality**: `cross_crate_search()` service works correctly

### What We Confirmed Doesn't Work  
1. **Function-based Schema Loading**: `get_tool_schema()` never called during tool registration
2. **Schema Configuration Changes**: MCP tool listing ignores config file changes
3. **Direct Schema Hardcoding**: Even direct schema replacement fails
4. **anyOf Schema Constraints**: MCP SDK doesn't support complex JSON Schema features

### Unidentified Issues
1. **Hidden Validation Layer**: Unknown MCP SDK validation happening before our code
2. **Schema Caching**: Possible schema caching at MCP protocol level
3. **Types.Tool Behavior**: `types.Tool` constructor may have hidden validation
4. **MCP Protocol Issues**: Possible protocol-level schema enforcement

## Debug Evidence

### Schema Verification Tests
```bash
# Confirmed get_tool_schema returns correct schema
uv run python debug_schema.py
# Output: Required fields: ['query'], Has crates property: True

# Confirmed MCP listing shows wrong schema  
uv run python test_mcp_tools_listing.py
# Output: "required": ["crate_name", "query"]
```

### Function Call Testing
Added debug logging to `get_tool_schema()`:
```python
if tool_name == "search_items":
    print(f"DEBUG: get_tool_schema called for {tool_name}, required={schema.get('required', [])}")
```
**Result**: No debug output = function never called

### Direct Validation Testing
Created `test_query_only.py` to test query-only search:
```json
{
  "arguments": {
    "query": "deserialize"
    // No crate_name or crates parameters
  }
}
```
**Result**: Still got `'crate_name' is a required property` error

## Environment Details
- **Python**: Using `uv` package manager
- **MCP SDK**: Default implementation (`--mcp-implementation sdk`)
- **Server Mode**: STDIO transport via `uvx --from . docsrs-mcp`
- **Protocol**: MCP Protocol version "2024-11-05"

## Cleanup Actions Taken
- Cleared Python caches: `find . -name "__pycache__" -exec rm -rf {} +`
- Killed all server processes: `pkill -f "docsrs-mcp"`  
- Restarted server multiple times with `nohup uvx --from . docsrs-mcp`
- Verified file changes with `grep` and `cat`

## Recommended Next Steps (Fresh Approach)
1. **MCP SDK Deep Dive**: Investigate MCP SDK source code for hidden validation layers
2. **Protocol Analysis**: Examine MCP protocol specification for schema constraints
3. **Alternative Approach**: Consider bypassing schema validation entirely
4. **Minimal Reproduction**: Create minimal test case to isolate the issue
5. **MCP SDK Alternatives**: Investigate different MCP implementations or versions

## Files That Need Rollback
1. `src/docsrs_mcp/mcp_tools_config.py` - Schema changes
2. `src/docsrs_mcp/mcp_sdk_server.py` - Runtime validation changes, schema changes, debug code
3. `debug_schema.py` - Debug script (can be deleted)
4. `test_simple_mcp_calls.py` - Test script (can be deleted)  
5. `test_mcp_tools_listing.py` - Test script (can be deleted)
6. `test_mcp_functionality.py` - Test script (can be deleted)
7. `test_query_only.py` - Test script (can be deleted)

## Summary
Despite systematic investigation and multiple fix attempts targeting different validation layers, the core issue persists. The MCP SDK has an unidentified validation layer that enforces schema requirements independently of our code changes. A completely different approach or deeper MCP SDK investigation is needed.