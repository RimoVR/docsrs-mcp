"""Code example extraction and processing from documentation.

This module handles:
- Code block extraction from markdown
- Language detection for code blocks
- Code normalization for deduplication
- Example embedding generation
"""

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import AsyncGenerator
from pathlib import Path

import aiosqlite
import sqlite_vec
from pygments.lexers import guess_lexer
from pygments.util import ClassNotFound

logger = logging.getLogger(__name__)


def extract_code_examples(docstring: str) -> str | None:
    """Extract code blocks from documentation with language detection.

    Returns JSON string with structure:
    [{"code": str, "language": str, "detected": bool}]

    Args:
        docstring: Documentation string containing code blocks

    Returns:
        Optional[str]: JSON string of code examples or None if no examples found
    """
    if not docstring:
        return None

    try:
        # Match code blocks with optional language tags
        # Captures: 1) optional language tag, 2) code content
        pattern = r"```(\w*)\s*\n(.*?)```"
        matches = re.findall(pattern, docstring, re.DOTALL | re.MULTILINE)

        if not matches:
            return None

        examples = []
        for lang_tag, code in matches:
            code = code.strip()
            if not code:  # Skip empty examples
                continue

            # Determine language
            detected = False
            if lang_tag:
                # Explicit language tag provided
                language = lang_tag.lower()
            else:
                # Try to detect language using pygments
                try:
                    lexer = guess_lexer(code)
                    # Use confidence threshold - pygments returns confidence score
                    if hasattr(lexer, "analyse_text"):
                        confidence = lexer.analyse_text(code)
                        if confidence and confidence > 0.3:  # 30% confidence threshold
                            language = (
                                lexer.aliases[0]
                                if lexer.aliases
                                else lexer.name.lower()
                            )
                            detected = True
                        else:
                            language = "rust"  # Default for Rust crates
                            detected = True
                    else:
                        language = "rust"
                        detected = True
                except (ClassNotFound, Exception):
                    # Default to Rust if detection fails
                    language = "rust"
                    detected = True

            examples.append({"code": code, "language": language, "detected": detected})

        return json.dumps(examples) if examples else None

    except Exception as e:
        logger.warning(f"Error extracting code examples: {e}")
        return None


def normalize_code(code: str) -> str:
    """Normalize code for consistent hashing and deduplication.

    Removes comments and normalizes whitespace to detect duplicate examples.

    Args:
        code: Raw code string

    Returns:
        str: Normalized code string
    """
    lines = code.strip().split("\n")
    normalized_lines = []

    for line in lines:
        stripped = line.strip()
        # Skip various comment types
        if stripped and not any(
            [
                stripped.startswith("#"),  # Python comments
                stripped.startswith("//"),  # Rust/C++ comments
                stripped.startswith("/*"),  # Block comments
                stripped.startswith("*"),  # Continuation of block comments
                stripped.startswith("--"),  # SQL/Lua comments
            ]
        ):
            # Normalize whitespace but preserve structure
            normalized_lines.append(" ".join(stripped.split()))

    return "\n".join(normalized_lines)


def calculate_example_hash(example_text: str, language: str) -> str:
    """Generate hash for deduplication of code examples.

    Includes language in hash to avoid cross-language collisions.

    Args:
        example_text: Code example text
        language: Programming language

    Returns:
        str: First 16 characters of SHA256 hash
    """
    # Normalize the code
    normalized = normalize_code(example_text)
    # Include language in hash to avoid cross-language collisions
    content = f"{language}:{normalized}"
    # Return first 16 chars of SHA256 hash for reasonable uniqueness
    return hashlib.sha256(content.encode()).hexdigest()[:16]


async def batch_examples(
    examples: list[dict], batch_size: int
) -> AsyncGenerator[list[dict], None]:
    """Yield batches of examples for processing.

    Memory-efficient batching for embedding generation.

    Args:
        examples: List of example dictionaries
        batch_size: Size of each batch

    Yields:
        List[Dict]: Batch of examples
    """
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        yield batch
        # Allow other async operations between batches
        await asyncio.sleep(0)


def format_example_for_embedding(example: dict) -> str:
    """Format a code example for embedding generation.

    Combines code with context for better semantic search.

    Args:
        example: Example dictionary

    Returns:
        str: Formatted text for embedding
    """
    # Include language as context
    language = example.get("language", "unknown")
    code = example.get("example_text", example.get("code", ""))
    context = example.get("context", "")

    # Combine elements for embedding
    parts = []
    if language and language != "unknown":
        parts.append(f"Language: {language}")
    if context:
        parts.append(f"Context: {context[:200]}")  # Limit context length
    parts.append(code)

    return "\n\n".join(parts)


async def generate_example_embeddings(
    db_path: Path, crate_name: str, version: str
) -> None:
    """Generate embeddings for code examples with deduplication.

    CRITICAL BUG FIX: This function includes the fix for character iteration bug.
    When examples_data is a string, it's wrapped in a list to prevent iteration
    over individual characters.

    Extracts examples from existing embeddings table and generates
    dedicated embeddings for semantic search.

    Args:
        db_path: Path to the database
        crate_name: Name of the crate
        version: Version of the crate
    """
    logger.info(f"Generating example embeddings for {crate_name}@{version}")

    async with aiosqlite.connect(db_path) as db:
        # Enable sqlite-vec extension
        await db.enable_load_extension(True)
        await db.execute(f"SELECT load_extension('{sqlite_vec.loadable_path()}')")
        await db.enable_load_extension(False)

        # Extract examples from embeddings table
        # Fixed: Use 'id' column instead of non-existent 'item_id'
        cursor = await db.execute("""
            SELECT id, item_path, examples, content
            FROM embeddings
            WHERE examples IS NOT NULL AND examples != ''
        """)

        all_examples = []
        async for row in cursor:
            item_id, item_path, examples_json, content = row

            try:
                examples_data = json.loads(examples_json)

                # CRITICAL BUG FIX: Handle string input - wrap in list to prevent character iteration
                # This fixes the bug where code examples were being treated as individual characters
                if isinstance(examples_data, str):
                    examples_data = [examples_data]
                elif not examples_data:
                    logger.warning(
                        f"Empty examples_data for {crate_name}/{version} at {item_path}"
                    )
                    continue

                # Handle both old list format and new dict format
                if isinstance(examples_data, list) and all(
                    isinstance(e, str) for e in examples_data
                ):
                    examples_data = [
                        {"code": e, "language": "rust", "detected": False}
                        for e in examples_data
                    ]

                # Process each example as a complete block (not individual characters)
                for example in examples_data:
                    if isinstance(example, str):
                        example = {
                            "code": example,
                            "language": "rust",
                            "detected": False,
                        }

                    code = example.get("code", "")
                    if not code:
                        continue

                    # Calculate hash for deduplication
                    language = example.get("language", "rust")
                    example_hash = calculate_example_hash(code, language)

                    all_examples.append(
                        {
                            "item_id": str(item_id),  # Convert to string for consistency
                            "item_path": item_path,
                            "crate_name": crate_name,
                            "version": version,
                            "example_hash": example_hash,
                            "example_text": code,
                            "language": language,
                            "context": content[:500]
                            if content
                            else None,  # Store context
                        }
                    )

            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse examples JSON for {item_path}: {e}")
                continue
            except Exception as e:
                logger.warning(f"Error processing examples for {item_path}: {e}")
                continue

        if not all_examples:
            logger.info(f"No code examples found for {crate_name}@{version}")
            return

        logger.info(
            f"Found {len(all_examples)} code examples for {crate_name}@{version}"
        )

        # Generate embeddings in batches
        from .embedding_manager import get_embedding_model

        model = get_embedding_model()

        batch_size = 32  # Optimal batch size for embedding generation
        async for batch in batch_examples(all_examples, batch_size):
            # Format examples for embedding
            texts = [format_example_for_embedding(ex) for ex in batch]

            # Generate embeddings
            embeddings = await asyncio.to_thread(model.embed, texts)

            # Store in database
            for example, embedding in zip(batch, embeddings, strict=False):
                await db.execute(
                    """
                    INSERT OR REPLACE INTO example_embeddings
                    (item_id, item_path, crate_name, version, example_hash,
                     example_text, language, context, embedding, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                    """,
                    (
                        example["item_id"],
                        example["item_path"],
                        example["crate_name"],
                        example["version"],
                        example["example_hash"],
                        example["example_text"],
                        example["language"],
                        example["context"],
                        bytes(sqlite_vec.serialize_float32(embedding)),
                    ),
                )

            await db.commit()

        logger.info(
            f"Successfully generated embeddings for {len(all_examples)} examples"
        )
