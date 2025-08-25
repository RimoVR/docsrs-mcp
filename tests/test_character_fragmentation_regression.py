"""Regression tests for the character fragmentation bug in example processing.

This test suite ensures that code examples are properly processed as complete strings
rather than being fragmented into individual characters.

Bug Context:
- Location: src/docsrs_mcp/ingestion/code_examples.py:234-242
- Issue: String examples were being iterated character-by-character
- Fix: Wrap string input in list to prevent character iteration
"""

import asyncio
import json
import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from docsrs_mcp.ingestion.code_examples import (
    calculate_example_hash,
    extract_code_examples,
    format_example_for_embedding,
    generate_example_embeddings,
    normalize_code,
)


class TestCharacterFragmentationRegression:
    """Test suite to prevent regression of the character fragmentation bug."""

    def test_extract_code_examples_returns_json_string(self):
        """Test that extract_code_examples returns a JSON string, not individual characters."""
        docstring = '''
        This is documentation with a code example:
        
        ```rust
        fn main() {
            println!("Hello, world!");
        }
        ```
        '''
        
        result = extract_code_examples(docstring)
        
        # Should return a JSON string
        assert isinstance(result, str)
        
        # Parse the JSON
        examples = json.loads(result)
        
        # Should be a list with one example
        assert isinstance(examples, list)
        assert len(examples) == 1
        
        # The example should contain the complete code, not individual characters
        example = examples[0]
        # Check that it contains the complete function, not individual characters
        assert "fn main()" in example["code"]
        assert "println!" in example["code"]
        assert "Hello, world!" in example["code"]
        assert example["language"] == "rust"
        
        # Verify it's not fragmented - the code should be longer than a few characters
        assert len(example["code"]) > 10
        # Verify no single-character "examples" exist
        assert "f" not in [ex.get("code") for ex in examples if len(ex.get("code", "")) == 1]

    def test_extract_code_examples_multiple_blocks(self):
        """Test extraction of multiple code blocks."""
        docstring = '''
        Multiple examples:
        
        ```rust
        let x = 5;
        ```
        
        And another:
        
        ```bash
        cargo build
        ```
        '''
        
        result = extract_code_examples(docstring)
        examples = json.loads(result)
        
        assert len(examples) == 2
        assert examples[0]["code"] == "let x = 5;"
        assert examples[0]["language"] == "rust"
        assert examples[1]["code"] == "cargo build"
        assert examples[1]["language"] == "bash"

    @pytest.mark.asyncio
    async def test_generate_example_embeddings_handles_string_input(self):
        """Test that generate_example_embeddings properly handles string examples_data.
        
        This is the CRITICAL test that verifies the bug fix.
        """
        # Create a temporary database
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Initialize database with example_embeddings table
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Create embeddings table with examples column
            cursor.execute("""
                CREATE TABLE embeddings (
                    item_id TEXT,
                    item_path TEXT,
                    examples TEXT,
                    content TEXT
                )
            """)
            
            # Create example_embeddings table
            cursor.execute("""
                CREATE TABLE example_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    item_path TEXT NOT NULL,
                    crate_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    example_hash TEXT NOT NULL,
                    example_text TEXT NOT NULL,
                    language TEXT,
                    context TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert a row with STRING examples (simulating the bug condition)
            # In the buggy case, a string was stored directly instead of JSON
            examples_string = '"fn main() { println!(\\"test\\"); }"'  # JSON string
            cursor.execute(
                "INSERT INTO embeddings (item_id, item_path, examples, content) VALUES (?, ?, ?, ?)",
                ("test_id", "test::path", examples_string, "Test content")
            )
            
            conn.commit()
            conn.close()
            
            # Mock the embedding model
            with patch("docsrs_mcp.ingestion.embedding_manager.get_embedding_model") as mock_model:
                # The embed method is called synchronously via asyncio.to_thread
                mock_instance = MagicMock()
                mock_instance.embed.return_value = [[0.1] * 384]  # 384-dim vector
                mock_model.return_value = mock_instance
                
                # Run the function that had the bug
                await generate_example_embeddings(db_path, "test_crate", "1.0.0")
                
                # Verify the examples were stored correctly
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT example_text FROM example_embeddings")
                results = cursor.fetchall()
                conn.close()
                
                # Should have one complete example, not 34 individual characters
                assert len(results) == 1
                assert results[0][0] == 'fn main() { println!("test"); }'
                
                # Verify it wasn't fragmented into characters
                assert results[0][0] != "f"  # Would be first character if fragmented
                assert len(results[0][0]) > 10  # Complete code, not single char

    @pytest.mark.asyncio
    async def test_generate_example_embeddings_handles_list_input(self):
        """Test that generate_example_embeddings properly handles list examples_data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Initialize database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE embeddings (
                    item_id TEXT,
                    item_path TEXT,
                    examples TEXT,
                    content TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE example_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    item_path TEXT NOT NULL,
                    crate_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    example_hash TEXT NOT NULL,
                    example_text TEXT NOT NULL,
                    language TEXT,
                    context TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert a row with LIST examples (proper format)
            examples_list = json.dumps(["fn first() {}", "fn second() {}"])
            cursor.execute(
                "INSERT INTO embeddings (item_id, item_path, examples, content) VALUES (?, ?, ?, ?)",
                ("test_id", "test::path", examples_list, "Test content")
            )
            
            conn.commit()
            conn.close()
            
            # Mock the embedding model
            with patch("docsrs_mcp.ingestion.embedding_manager.get_embedding_model") as mock_model:
                # The embed method is called synchronously via asyncio.to_thread
                mock_instance = MagicMock()
                mock_instance.embed.return_value = [[0.1] * 384, [0.2] * 384]
                mock_model.return_value = mock_instance
                
                await generate_example_embeddings(db_path, "test_crate", "1.0.0")
                
                # Verify the examples were stored correctly
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT example_text FROM example_embeddings ORDER BY id")
                results = cursor.fetchall()
                conn.close()
                
                # Should have two complete examples
                assert len(results) == 2
                assert results[0][0] == "fn first() {}"
                assert results[1][0] == "fn second() {}"

    @pytest.mark.asyncio
    async def test_generate_example_embeddings_handles_dict_format(self):
        """Test that generate_example_embeddings handles new dict format with metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = Path(tmpdir) / "test.db"
            
            # Initialize database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE embeddings (
                    item_id TEXT,
                    item_path TEXT,
                    examples TEXT,
                    content TEXT
                )
            """)
            
            cursor.execute("""
                CREATE TABLE example_embeddings (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    item_id TEXT NOT NULL,
                    item_path TEXT NOT NULL,
                    crate_name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    example_hash TEXT NOT NULL,
                    example_text TEXT NOT NULL,
                    language TEXT,
                    context TEXT,
                    embedding BLOB NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Insert a row with new dict format
            examples_dict = json.dumps([
                {"code": "fn main() {}", "language": "rust", "detected": False},
                {"code": "#!/bin/bash\necho test", "language": "bash", "detected": True}
            ])
            cursor.execute(
                "INSERT INTO embeddings (item_id, item_path, examples, content) VALUES (?, ?, ?, ?)",
                ("test_id", "test::path", examples_dict, "Test content")
            )
            
            conn.commit()
            conn.close()
            
            # Mock the embedding model
            with patch("docsrs_mcp.ingestion.embedding_manager.get_embedding_model") as mock_model:
                # The embed method is called synchronously via asyncio.to_thread
                mock_instance = MagicMock()
                mock_instance.embed.return_value = [[0.1] * 384, [0.2] * 384]
                mock_model.return_value = mock_instance
                
                await generate_example_embeddings(db_path, "test_crate", "1.0.0")
                
                # Verify the examples were stored with language metadata
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT example_text, language FROM example_embeddings ORDER BY id")
                results = cursor.fetchall()
                conn.close()
                
                assert len(results) == 2
                assert results[0][0] == "fn main() {}"
                assert results[0][1] == "rust"
                assert results[1][0] == "#!/bin/bash\necho test"
                assert results[1][1] == "bash"

    def test_normalize_code_preserves_complete_strings(self):
        """Test that normalize_code works on complete code, not characters."""
        code = "fn main() {\n    // Comment\n    println!(\"test\");\n}"
        normalized = normalize_code(code)
        
        # Should preserve the structure, just normalize whitespace
        assert "fn main" in normalized
        assert "println" in normalized
        # Comments should be removed
        assert "// Comment" not in normalized
        
        # Should not be a single character
        assert len(normalized) > 1

    def test_calculate_example_hash_complete_string(self):
        """Test that hash is calculated on complete example, not characters."""
        example = "fn main() { println!(\"Hello\"); }"
        hash1 = calculate_example_hash(example, "rust")
        
        # Hash should be consistent
        hash2 = calculate_example_hash(example, "rust")
        assert hash1 == hash2
        
        # Hash should be 16 characters (as per implementation)
        assert len(hash1) == 16
        
        # Different code should produce different hash
        different = "fn test() { }"
        hash3 = calculate_example_hash(different, "rust")
        assert hash1 != hash3

    def test_format_example_for_embedding_complete(self):
        """Test that formatting preserves complete code examples."""
        example = {
            "code": "fn main() { println!(\"test\"); }",
            "language": "rust",
            "context": "This is a test function"
        }
        
        formatted = format_example_for_embedding(example)
        
        # Should contain the complete code
        assert "fn main() { println!(\"test\"); }" in formatted
        assert "Language: rust" in formatted
        assert "Context: This is a test function" in formatted
        
        # Should not be fragmented
        assert len(formatted) > 10  # Much longer than a single character


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])