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
            "description": "Search within a single crate (crate_name) or across crates (crates). All parameters are strings for MCP compatibility.",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (min 2 chars)"
                },
                "crate_name": {
                    "type": "string",
                    "description": "Single crate to search"
                },
                "crates": {
                    "type": "string",
                    "description": "Comma-separated crates for cross-crate search"
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
                    "default": "latest"
                },
                "k": {
                    "type": "string",
                    "description": "Number of results (1-50)",
                    "default": "5"
                },
                "item_type": {
                    "type": "string",
                    "description": "Filter by item type"
                },
                "module_path": {
                    "type": "string",
                    "description": "Filter to module path (single-crate only)"
                },
                "has_examples": {
                    "type": "string",
                    "description": "Only items with examples (true/false)"
                },
                "min_doc_length": {
                    "type": "string",
                    "description": "Minimum documentation length"
                },
                "visibility": {
                    "type": "string",
                    "description": "public | private | crate"
                },
                "deprecated": {
                    "type": "string",
                    "description": "Include deprecated (true/false)"
                }
            },
            "required": ["query"]
        },
        "examples": [
            {
                "query": "deserialize",
                "k": "10",
                "description": "Cross-crate search for deserialize functionality across all crates"
            },
            {
                "query": "async",
                "crates": ["tokio", "async-std"],
                "k": "5",
                "description": "Cross-crate search for async functionality in specific crates"
            },
            {
                "crate_name": "serde",
                "query": "serialize",
                "k": "10",
                "description": "Single-crate search within serde crate"
            }
        ]
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
                    "type": "string",
                    "description": "Number of examples to return (1-10, as string)",
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
                    "type": "string",
                    "description": "Force restart even if already running (true/false)",
                },
                "concurrency": {
                    "type": "string",
                    "description": "Number of concurrent workers (1-10, as string)",
                },
                "count": {
                    "type": "string",
                    "description": "Number of crates to pre-ingest (as string)",
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
                    "type": "string",
                    "description": "Resolve version specifications to exact versions (true/false)",
                },
                "skip_existing": {
                    "type": "string",
                    "description": "Skip crates that are already cached (true/false)",
                },
                "concurrency": {
                    "type": "string",
                    "description": "Number of concurrent ingestion workers (as string)",
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
                "version_a": {
                    "type": "string",
                    "description": "First version to compare",
                },
                "version_b": {
                    "type": "string",
                    "description": "Second version to compare",
                },
                "include_details": {
                    "type": "string",
                    "description": "Include detailed change information (true/false)",
                },
                "breaking_only": {
                    "type": "string",
                    "description": "Show only breaking changes (true/false)",
                },
            },
            "required": ["crate_name", "version_a", "version_b"],
        },
    },
    {
        "name": "get_documentation_detail",
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
        "name": "extract_usage_patterns",
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
                    "type": "string",
                    "description": "Maximum number of patterns to return (as string)",
                },
                "min_frequency": {
                    "type": "string",
                    "description": "Minimum frequency for a pattern to be included (as string)",
                },
            },
            "required": ["crate_name"],
        },
    },
    {
        "name": "generate_learning_path",
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
    {
        "name": "get_code_intelligence",
        "description": "Get comprehensive code intelligence for a specific item including safety info, error types, and feature requirements",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "item_path": {
                    "type": "string",
                    "description": "Full path to the item (e.g., 'tokio::spawn')",
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
        "name": "get_error_types",
        "description": "List all error types in a crate or matching a pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "pattern": {
                    "type": "string",
                    "description": "Optional pattern to filter error types",
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
        "name": "get_unsafe_items",
        "description": "List all unsafe items in a crate with optional safety documentation",
        "input_schema": {
            "type": "object",
            "properties": {
                "crate_name": {
                    "type": "string",
                    "description": "Name of the Rust crate",
                },
                "include_reasons": {
                    "type": "string",
                    "description": "Include detailed unsafe reasons and safety documentation (true/false)",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version or 'latest'",
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
