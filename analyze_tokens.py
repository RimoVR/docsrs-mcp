#!/usr/bin/env python
"""Analyze current token usage in MCP tool tutorials."""

import tiktoken

# Current tutorials from app.py
tutorials = {
    "getCrateSummary": (
        "Fetches crate metadata including version, description, and module structure.\n"
        "Automatically downloads and ingests crates on first access with smart caching.\n"
        "First-time ingestion: 1-10 seconds depending on crate size.\n"
        "Use for initial crate exploration before diving into specific documentation."
    ),
    "searchItems": (
        "Performs semantic search across all crate documentation using embeddings.\n"
        "Results ranked by relevance using BAAI/bge-small-en-v1.5 model.\n"
        "Snippets use smart extraction (200-400 chars) with boundary detection.\n"
        "Warm searches complete in <50ms. Rate limit: 30 req/sec.\n"
        "Best for finding functionality when you don't know exact item names."
    ),
    "getItemDoc": (
        "Retrieves complete documentation for a specific item by path.\n"
        "Returns markdown-formatted docs with examples and descriptions.\n"
        "Use after search_items to get full details on discovered items.\n"
        "Tip: Use 'crate' as item_path for crate-level documentation."
    ),
    "searchExamples": (
        "Searches code examples extracted from crate documentation.\n"
        "Includes language detection and filtering capabilities.\n"
        "Returns runnable code snippets with smart context (200-400 chars).\n"
        "Perfect for learning implementation patterns."
    ),
    "getModuleTree": (
        "Returns hierarchical module structure with parent-child relationships.\n"
        "Shows item counts and depth levels for navigation.\n"
        "First-time processing: 1-10 seconds for crate ingestion.\n"
        "Essential for understanding crate organization."
    ),
    "startPreIngestion": (
        "The pre-ingestion system caches popular Rust crates to eliminate cold-start latency.\n\n"
        "**How it works:**\n"
        "1. Fetches the list of most-downloaded crates from crates.io\n"
        "2. Downloads and processes them in parallel (configurable concurrency)\n"
        "3. Builds search indices and caches documentation\n"
        "4. Runs in background without blocking other operations\n\n"
        "**Monitoring:**\n"
        "- Check `/health` for overall system status\n"
        "- Use `/health/pre-ingestion` for detailed progress\n"
        "- Look for 'pre_ingestion' section showing processed/total counts\n\n"
        "**Performance Impact:**\n"
        "- Popular crates respond in <100ms after pre-ingestion\n"
        "- System remains responsive during pre-ingestion\n"
        "- Memory usage increases gradually as crates are cached\n\n"
        "**Best Practices:**\n"
        "- Start pre-ingestion during low-traffic periods\n"
        "- Monitor memory usage via health endpoints\n"
        "- Allow 5-10 minutes for full completion\n"
        "- Use force=true sparingly to avoid redundant work"
    ),
}

# Optimized tutorials
optimized_tutorials = {
    "getCrateSummary": (
        "**Purpose:** Fetch crate metadata & structure\n"
        "**Input:** crate_name (req), version (opt)\n"
        "**Returns:** name, version, description, modules, repository\n"
        "**Examples:**\n"
        "• tokio → async runtime info\n"
        "• serde/1.0.195 → specific version\n"
        "• std → stdlib summary"
    ),
    "searchItems": (
        "**Purpose:** Semantic search in crate docs\n"
        "**Input:** crate_name, query (req); k, filters (opt)\n"
        "**Returns:** Ranked results with snippets\n"
        "**Examples:**\n"
        '• tokio + "spawn tasks" → async functions\n'
        '• serde + "deserialize JSON" → trait methods\n'
        "**Filters:** item_type, module_path, deprecated"
    ),
    "getItemDoc": (
        "**Purpose:** Full documentation for specific item\n"
        "**Input:** crate_name, item_path (req); version (opt)\n"
        "**Returns:** Complete markdown docs\n"
        "**Examples:**\n"
        "• tokio + tokio::spawn → function docs\n"
        "• serde + Deserialize → trait docs\n"
        "**Note:** Use searchItems first if unsure of path"
    ),
    "searchExamples": (
        "**Purpose:** Find code examples in docs\n"
        "**Input:** crate_name, query (req); k, language (opt)\n"
        "**Returns:** Code snippets with context\n"
        "**Examples:**\n"
        '• tokio + "async runtime" → usage examples\n'
        '• actix-web + "middleware" → integration code'
    ),
    "getModuleTree": (
        "**Purpose:** Crate module hierarchy\n"
        "**Input:** crate_name (req); version (opt)\n"
        "**Returns:** Tree structure with item counts\n"
        "**Use:** Navigate crate organization"
    ),
    "startPreIngestion": (
        "**Purpose:** Cache popular crates in background\n"
        "**Input:** count (10-500), concurrency (1-10), force (bool)\n"
        "**Process:** Fetches crates.io list → parallel download → index build\n"
        "**Monitor:** /health/pre-ingestion for progress\n"
        "**Performance:** <100ms response after caching\n"
        "**Tips:** Start count=100, concurrency=3; force=true to restart"
    ),
}


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """Count tokens using tiktoken."""
    enc = tiktoken.get_encoding(encoding)
    return len(enc.encode(text))


print("=== Current Tutorial Token Analysis ===\n")
for name, tutorial in tutorials.items():
    tokens = count_tokens(tutorial)
    chars = len(tutorial)
    print(f"{name}:")
    print(f"  Characters: {chars}")
    print(f"  Tokens: {tokens}")
    print(f"  Status: {'❌ EXCEEDS' if tokens > 200 else '✅ OK'}")
    print()

print("\n=== Optimized Tutorial Token Analysis ===\n")
for name, tutorial in optimized_tutorials.items():
    tokens = count_tokens(tutorial)
    chars = len(tutorial)
    print(f"{name}:")
    print(f"  Characters: {chars}")
    print(f"  Tokens: {tokens}")
    print(f"  Status: {'❌ EXCEEDS' if tokens > 200 else '✅ OK'}")
    print(f"  Reduction: {100 - (tokens / count_tokens(tutorials[name])) * 100:.1f}%")
    print()

print("\n=== Summary ===")
current_total = sum(count_tokens(t) for t in tutorials.values())
optimized_total = sum(count_tokens(t) for t in optimized_tutorials.values())
print(f"Current total tokens: {current_total}")
print(f"Optimized total tokens: {optimized_total}")
print(f"Total reduction: {100 - (optimized_total / current_total) * 100:.1f}%")
print(f"Average tokens per tool: {optimized_total / len(optimized_tutorials):.0f}")
