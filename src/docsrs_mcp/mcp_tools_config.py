"""MCP tool definitions and schemas for the manifest endpoint."""

from typing import Any

# Tool definitions with their input schemas
MCP_TOOLS_CONFIG: list[dict[str, Any]] = [
    {
        "name": "get_crate_summary",
        "description": "Get summary information about a Rust crate",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
            },
            "required": ["crate_name"],
        },
    },
    {
        "name": "search_items",
        "description": "Search for items in crate documentation with advanced modes",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the crate to search within",
                },
                "query": {
                    "type": "string",
                    "description": "Natural language search query",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
                "k": {
                    "type": "string",
                    "description": "Number of results to return (1-20)",
                },
                "type_filter": {
                    "type": "string",
                    "description": "Filter by item type (struct, trait, function, etc.)",
                },
                "module_path": {
                    "type": "string",
                    "description": "Filter results to specific module path",
                },
                "crate_filter": {
                    "type": "string",
                    "description": "Filter results to specific crate",
                },
                "visibility": {
                    "type": "string",
                    "description": "Filter by visibility (public, crate, private)",
                },
                "deprecated": {
                    "type": "boolean",
                    "description": "Include deprecated items in results",
                },
                "has_examples": {
                    "type": "boolean",
                    "description": "Only return items with code examples",
                },
                "min_doc_length": {
                    "type": "integer",
                    "description": "Minimum documentation length filter",
                },
                "search_mode": {
                    "type": "string",
                    "description": "Search mode: 'vector' (default), 'fuzzy', 'regex', or 'hybrid'",
                    "enum": ["vector", "fuzzy", "regex", "hybrid"],
                },
                "fuzzy_tolerance": {
                    "type": "number",
                    "description": "Fuzzy match threshold (0.0-1.0, default: 0.7)",
                },
                "regex_pattern": {
                    "type": "string",
                    "description": "Regex pattern for pattern matching mode",
                },
                "stability_filter": {
                    "type": "string",
                    "description": "Filter by stability: 'stable', 'unstable', 'experimental', or 'all'",
                    "enum": ["stable", "unstable", "experimental", "all"],
                },
                "crates": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of crates for cross-crate search (max 5)",
                    "maxItems": 5,
                },
            },
            "required": ["crate_name", "query"],
        },
    },
    {
        "name": "get_item_doc",
        "description": "Get complete documentation for a specific item",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "item_path": {
                    "type": "string",
                    "description": "Full path to the item (e.g., 'std::vec::Vec')",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
            },
            "required": ["crate_name", "item_path"],
        },
    },
    {
        "name": "search_examples",
        "description": "Search for code examples in crate documentation",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the crate to search",
                },
                "query": {
                    "type": "string",
                    "description": "Search query for finding relevant examples",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
                "k": {
                    "type": "integer",
                    "description": "Number of examples to return (1-10)",
                },
                "language": {
                    "type": "string",
                    "description": "Filter by programming language",
                },
            },
            "required": ["crate_name", "query"],
        },
    },
    {
        "name": "get_module_tree",
        "description": "Get the module hierarchy tree for a Rust crate",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
            },
            "required": ["crate_name"],
        },
    },
    {
        "name": "start_pre_ingestion",
        "description": "Start background pre-ingestion of popular Rust crates",
        "input_schema": {
            "type": "object",
            "properties": {
                "force": {
                    "type": "boolean",
                    "description": "Force restart even if already running",
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Number of concurrent workers (1-10)",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of crates to pre-ingest",
                },
            },
            "required": [],
        },
    },
    {
        "name": "ingest_cargo_file",
        "description": "Ingest crates from a Cargo.toml or Cargo.lock file",
        "input_schema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to Cargo.toml or Cargo.lock file",
                },
                "resolve_versions": {
                    "type": "boolean",
                    "description": "Resolve version specifications to exact versions",
                },
                "skip_existing": {
                    "type": "boolean",
                    "description": "Skip crates that are already cached",
                },
                "concurrency": {
                    "type": "integer",
                    "description": "Number of concurrent ingestion workers",
                },
            },
            "required": ["file_path"],
        },
    },
    {
        "name": "control_pre_ingestion",
        "description": "Control the pre-ingestion worker (pause/resume/stop)",
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["pause", "resume", "stop"],
                    "description": "Control action to perform",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "compare_versions",
        "description": "Compare two versions of a crate for API changes",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "version1": {
                    "type": "string",
                    "description": "First version to compare",
                },
                "version2": {
                    "type": "string",
                    "description": "Second version to compare",
                },
                "include_details": {
                    "type": "boolean",
                    "description": "Include detailed change information",
                },
                "breaking_only": {
                    "type": "boolean",
                    "description": "Show only breaking changes",
                },
            },
            "required": ["crate_name", "version1", "version2"],
        },
    },
    {
        "name": "getDocumentationDetail",
        "description": "Get documentation with progressive detail levels (summary/detailed/expert)",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "item_path": {
                    "type": "string",
                    "description": "Path to the item (e.g., 'serde::Serialize')",
                },
                "detail_level": {
                    "type": "string",
                    "description": "Level of detail: summary, detailed, or expert",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
            },
            "required": ["crate_name", "item_path"],
        },
    },
    {
        "name": "extractUsagePatterns",
        "description": "Extract common usage patterns from documentation and examples",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of patterns to return",
                },
                "min_frequency": {
                    "type": "integer",
                    "description": "Minimum frequency for a pattern to be included",
                },
            },
            "required": ["crate_name"],
        },
    },
    {
        "name": "generateLearningPath",
        "description": "Generate learning path for API migration or onboarding",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "from_version": {
                    "type": "string",
                    "description": "Starting version (omit for new users)",
                },
                "to_version": {
                    "type": "string",
                    "description": "Target version",
                },
                "focus_areas": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of focus areas",
                },
            },
            "required": ["crate_name"],
        },
    },
]

# Resource definitions
MCP_RESOURCES_CONFIG = [
    {
        "name": "crate-cache-status",
        "uri": "cache://status",
        "description": "Current cache status and statistics",
    },
    {
        "name": "popular-crates-list",
        "uri": "cache://popular",
        "description": "List of popular crates available for ingestion",
    },
]
