"""Simple configuration for docsrs-mcp."""

import os
from pathlib import Path

# Cache configuration
CACHE_DIR = Path(os.getenv("DOCSRS_CACHE_DIR", "./cache"))
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Model configuration
MODEL_NAME = os.getenv("DOCSRS_MODEL_NAME", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIM = 384  # for bge-small-en-v1.5

# Database configuration
DB_TIMEOUT = float(os.getenv("DOCSRS_DB_TIMEOUT", "30.0"))

# HTTP configuration
HTTP_TIMEOUT = float(os.getenv("DOCSRS_HTTP_TIMEOUT", "120.0"))
MAX_DOWNLOAD_SIZE = int(
    os.getenv("DOCSRS_MAX_DOWNLOAD_SIZE", str(30 * 1024 * 1024))
)  # 30MB
MAX_DECOMPRESSED_SIZE = int(
    os.getenv("DOCSRS_MAX_DECOMPRESSED_SIZE", str(100 * 1024 * 1024))
)  # 100MB

# Search configuration
DEFAULT_K = int(os.getenv("DOCSRS_DEFAULT_K", "5"))

# Batch processing
EMBEDDING_BATCH_SIZE = int(os.getenv("DOCSRS_EMBEDDING_BATCH_SIZE", "32"))
DB_BATCH_SIZE = int(os.getenv("DOCSRS_DB_BATCH_SIZE", "999"))  # SQLite parameter limit

# Memory management
MEMORY_THRESHOLD_HIGH = int(
    os.getenv("DOCSRS_MEMORY_THRESHOLD_HIGH", "80")
)  # Percentage
MEMORY_THRESHOLD_CRITICAL = int(
    os.getenv("DOCSRS_MEMORY_THRESHOLD_CRITICAL", "90")
)  # Percentage
MIN_BATCH_SIZE = int(os.getenv("DOCSRS_MIN_BATCH_SIZE", "16"))
MAX_BATCH_SIZE = int(os.getenv("DOCSRS_MAX_BATCH_SIZE", "512"))
PARSE_CHUNK_SIZE = int(
    os.getenv("DOCSRS_PARSE_CHUNK_SIZE", "100")
)  # Items to process before yielding

# Rate limiting
RATE_LIMIT_PER_SECOND = int(os.getenv("DOCSRS_RATE_LIMIT", "30"))

# Cache management
CACHE_MAX_SIZE_BYTES = int(
    os.getenv("DOCSRS_CACHE_MAX_SIZE_BYTES", str(2 * 1024**3))
)  # 2GB

# Rustdoc processing
RUSTDOC_CHUNK_MAX_SIZE = int(os.getenv("DOCSRS_RUSTDOC_CHUNK_MAX_SIZE", "4096"))

# Download configuration
DOWNLOAD_CHUNK_SIZE = int(os.getenv("DOCSRS_DOWNLOAD_CHUNK_SIZE", "8192"))

# Standard library configuration
STDLIB_CRATES = {"std", "core", "alloc", "proc_macro", "test"}
RUST_CHANNEL_BASE = "https://static.rust-lang.org"
RUST_VERSION_MANIFEST_URL = f"{RUST_CHANNEL_BASE}/version"
