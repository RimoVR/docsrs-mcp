"""
Crates.io source code extractor for documentation fallback.

Extracts documentation from Rust source archives when rustdoc JSON is unavailable.
Uses the crates.io CDN for rate-limit-free downloads and memory-efficient streaming.
"""

import asyncio
import io
import logging
import re
import tarfile

import aiohttp

logger = logging.getLogger(__name__)


class CratesIoSourceExtractor:
    """
    Extracts documentation from crate source archives when rustdoc JSON is unavailable.

    This extractor:
    1. Downloads .crate files from the crates.io CDN (no rate limits)
    2. Streams through tar.gz archives memory-efficiently
    3. Extracts documentation comments and public API signatures
    4. Returns chunks compatible with the existing ingestion format
    """

    # CDN URL pattern - uses static CDN to avoid API rate limits
    CDN_URL = "https://static.crates.io/crates/{name}/{name}-{version}.crate"

    # Size limits matching existing infrastructure
    MAX_COMPRESSED_SIZE = 30 * 1024 * 1024  # 30 MB compressed
    MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024  # 100 MB decompressed
    MAX_FILE_SIZE = 1 * 1024 * 1024  # 1 MB per file

    # Regex patterns for Rust documentation extraction
    PATTERNS = {
        "doc_line": re.compile(r"^\s*///\s?(.*)$"),
        "doc_inner": re.compile(r"^\s*//!\s?(.*)$"),
        "doc_block": re.compile(r"/\*\*(.*?)\*/", re.DOTALL),
        "pub_fn": re.compile(
            r'pub\s+(?:async\s+)?(?:unsafe\s+)?(?:const\s+)?(?:extern\s+(?:"[^"]*"\s+)?)?fn\s+(\w+)'
        ),
        "pub_struct": re.compile(r"pub\s+struct\s+(\w+)"),
        "pub_enum": re.compile(r"pub\s+enum\s+(\w+)"),
        "pub_trait": re.compile(r"pub\s+(?:unsafe\s+)?trait\s+(\w+)"),
        "pub_type": re.compile(r"pub\s+type\s+(\w+)"),
        "pub_mod": re.compile(r"pub\s+mod\s+(\w+)"),
        "pub_const": re.compile(r"pub\s+const\s+(\w+)"),
        "pub_static": re.compile(r"pub\s+static\s+(\w+)"),
        # Macro patterns - crucial for macro crates
        "macro_rules": re.compile(r"#?\[?macro_export\]?\s*\n?\s*macro_rules!\s+(\w+)"),
        "pub_macro": re.compile(r"pub\s+macro\s+(\w+)"),
        "macro_use": re.compile(r"#\[macro_use(?:\([^\)]*\))?\]"),
    }

    def __init__(
        self,
        session: aiohttp.ClientSession | None = None,
        memory_monitor: object | None = None,
        timeout: int = 30,
    ):
        """
        Initialize the extractor.

        Args:
            session: Aiohttp session for downloads (will create if None)
            memory_monitor: Memory monitor for resource management
            timeout: Download timeout in seconds
        """
        self.session = session
        self.memory_monitor = memory_monitor
        self.timeout = timeout
        self._own_session = False

    async def __aenter__(self):
        """Async context manager entry."""
        if self.session is None:
            self.session = aiohttp.ClientSession()
            self._own_session = True
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._own_session and self.session:
            await self.session.close()

    async def extract_from_source(self, name: str, version: str) -> list[dict]:
        """
        Main extraction pipeline.

        Downloads and extracts documentation from a crate's source archive.

        Args:
            name: Crate name
            version: Crate version

        Returns:
            List of extracted documentation items in chunk format
        """
        logger.info(f"Starting source extraction for {name}@{version}")

        try:
            # Download the crate archive
            archive_data = await self.download_crate_archive(name, version)

            # Extract documentation from the archive
            items = await self.extract_from_archive(archive_data, name)

            logger.info(f"Extracted {len(items)} items from {name}@{version}")
            return items

        except Exception as e:
            logger.error(f"Source extraction failed for {name}@{version}: {e}")
            raise

    async def download_crate_archive(self, name: str, version: str) -> bytes:
        """
        Download crate archive from CDN.

        Args:
            name: Crate name
            version: Crate version

        Returns:
            Downloaded archive data as bytes

        Raises:
            Exception: If download fails or size limits exceeded
        """
        url = self.CDN_URL.format(name=name, version=version)
        logger.debug(f"Downloading crate from: {url}")

        if not self.session:
            self.session = aiohttp.ClientSession()
            self._own_session = True

        try:
            timeout_config = aiohttp.ClientTimeout(total=self.timeout)
            async with self.session.get(url, timeout=timeout_config) as response:
                response.raise_for_status()

                # Check content length
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > self.MAX_COMPRESSED_SIZE:
                    raise ValueError(f"Archive too large: {content_length} bytes")

                # Download with size limit
                data = bytearray()
                async for chunk in response.content.iter_chunked(8192):
                    data.extend(chunk)
                    if len(data) > self.MAX_COMPRESSED_SIZE:
                        raise ValueError("Archive exceeds size limit during download")

                return bytes(data)

        except aiohttp.ClientError as e:
            logger.error(f"Failed to download {name}@{version}: {e}")
            raise

    async def extract_from_archive(
        self, archive_data: bytes, crate_name: str
    ) -> list[dict]:
        """
        Extract documentation from a crate archive.

        Uses memory-efficient streaming to process tar.gz archives.

        Args:
            archive_data: The .crate archive data
            crate_name: Name of the crate for path construction

        Returns:
            List of extracted documentation items
        """
        items = []

        # Process in thread pool to avoid blocking
        loop = asyncio.get_event_loop()

        def process_archive():
            """Process archive in separate thread."""
            extracted_items = []

            try:
                # Open tar.gz archive
                with tarfile.open(fileobj=io.BytesIO(archive_data), mode="r:gz") as tar:
                    # Process each member
                    for member in tar:
                        # Only process Rust source files
                        if not member.name.endswith(".rs"):
                            continue

                        # Skip huge files
                        if member.size > self.MAX_FILE_SIZE:
                            logger.debug(
                                f"Skipping large file: {member.name} ({member.size} bytes)"
                            )
                            continue

                        # Skip test files and examples (focus on main API)
                        if (
                            "/tests/" in member.name
                            or "/examples/" in member.name
                            or "/benches/" in member.name
                        ):
                            continue

                        # Extract and process file
                        file_obj = tar.extractfile(member)
                        if file_obj:
                            try:
                                content = file_obj.read().decode(
                                    "utf-8", errors="ignore"
                                )
                                file_items = self.extract_from_rust_file(
                                    content, member.name, crate_name
                                )
                                extracted_items.extend(file_items)
                            except Exception as e:
                                logger.debug(f"Failed to process {member.name}: {e}")
                            finally:
                                file_obj.close()

                        # Clear member cache to prevent memory growth (Python <3.13)
                        tar.members = []

                        # Check memory if monitor available
                        if self.memory_monitor:
                            self.memory_monitor.check_memory()

            except Exception as e:
                logger.error(f"Archive processing failed: {e}")
                raise

            return extracted_items

        # Run extraction in thread pool
        items = await loop.run_in_executor(None, process_archive)

        return items

    def extract_from_rust_file(
        self, content: str, filepath: str, crate_name: str
    ) -> list[dict]:
        """
        Extract documentation from a single Rust source file.

        Uses regex patterns to find public items and their documentation.
        Includes enhanced macro extraction for better coverage.

        Args:
            content: File content
            filepath: Path within the archive
            crate_name: Name of the crate

        Returns:
            List of extracted items with documentation
        """
        items = []
        lines = content.split("\n")

        # Try enhanced macro extraction first
        try:
            from .enhanced_macro_extractor import EnhancedMacroExtractor

            macro_extractor = EnhancedMacroExtractor()
            macro_items = macro_extractor.extract_macros(content, filepath, crate_name)
            if macro_items:
                logger.debug(
                    f"Enhanced extractor found {len(macro_items)} macros in {filepath}"
                )
                items.extend(macro_items)
        except Exception as e:
            logger.debug(f"Enhanced macro extraction failed, using basic: {e}")

        # Clean up filepath for item paths
        # Remove crate-version prefix and .rs extension
        clean_path = filepath
        if "/" in clean_path:
            # Remove the crate-version directory prefix
            parts = clean_path.split("/", 1)
            if len(parts) > 1:
                clean_path = parts[1]
        clean_path = clean_path.replace(".rs", "").replace("/", "::")

        # Special handling for lib.rs and mod.rs
        if clean_path.endswith("::lib"):
            clean_path = crate_name
        elif clean_path.endswith("::mod"):
            clean_path = clean_path[:-5]  # Remove ::mod

        # Track module-level documentation
        module_docs = []

        # Track which macros we already found with enhanced extraction
        found_macros = set()
        for item in items:
            if item.get("item_type") == "macro":
                # Extract macro name from path
                name = item["item_path"].split("::")[-1]
                found_macros.add(name)

        for i, line in enumerate(lines):
            # Check for module-level documentation at the start
            if i < 50:  # Only check first 50 lines for module docs
                if match := self.PATTERNS["doc_inner"].match(line):
                    module_docs.append(match.group(1))

            # Check for public items and macros
            for item_type, pattern in self.PATTERNS.items():
                if item_type.startswith(("pub_", "macro")):
                    if match := pattern.search(line):
                        # For macro patterns, we might not have a capturing group
                        if match.groups():
                            item_name = match.group(1)
                        elif item_type == "macro_use":
                            continue  # Skip macro_use, it's just an attribute
                        else:
                            # Try to extract macro name from the line
                            macro_match = re.search(r"macro_rules!\s+(\w+)", line)
                            if macro_match:
                                item_name = macro_match.group(1)
                            else:
                                continue

                        # Skip if this macro was already found by enhanced extractor
                        if item_type.startswith("macro") and item_name in found_macros:
                            continue

                        # Extract preceding doc comments
                        doc_lines = []
                        j = i - 1
                        while j >= 0:
                            doc_match = self.PATTERNS["doc_line"].match(lines[j])
                            if doc_match:
                                doc_lines.insert(0, doc_match.group(1))
                                j -= 1
                            elif lines[j].strip() == "":
                                # Allow blank lines in doc comments
                                j -= 1
                            else:
                                break

                        # Build item path
                        if clean_path == crate_name:
                            item_path = f"{crate_name}::{item_name}"
                        else:
                            item_path = f"{clean_path}::{item_name}"

                        # Get the full signature (current line plus continuations)
                        signature = line.strip()
                        if item_type.startswith("macro"):
                            # For macros, try to capture the pattern
                            # Look ahead for the opening brace and first pattern
                            for k in range(i + 1, min(i + 10, len(lines))):
                                signature += " " + lines[k].strip()
                                if "{" in lines[k]:
                                    # Try to get first pattern
                                    for m in range(k + 1, min(k + 3, len(lines))):
                                        if lines[m].strip():
                                            signature += " " + lines[m].strip()[:50]
                                            break
                                    break
                        elif not line.rstrip().endswith((";", "{", "}")):
                            # Multi-line signature for non-macros
                            for k in range(i + 1, min(i + 5, len(lines))):
                                signature += " " + lines[k].strip()
                                if lines[k].rstrip().endswith((";", "{", "}")):
                                    break

                        # Clean up signature
                        signature = re.sub(r"\s+", " ", signature)
                        if len(signature) > 200:
                            signature = signature[:197] + "..."

                        # Determine item type for storage
                        if item_type.startswith("pub_"):
                            stored_type = item_type.replace("pub_", "")
                        elif item_type in ["macro_rules", "pub_macro"]:
                            stored_type = "macro"
                        else:
                            stored_type = item_type

                        # Store item if it has documentation or is a function/macro
                        if doc_lines or item_type in [
                            "pub_fn",
                            "macro_rules",
                            "pub_macro",
                        ]:
                            items.append(
                                {
                                    "item_path": item_path,
                                    "header": item_name,
                                    "item_type": stored_type,
                                    "signature": signature,
                                    "docstring": "\n".join(doc_lines),
                                    "visibility": "public",
                                }
                            )

        # Add module-level documentation if present
        if module_docs and clean_path:
            module_path = clean_path if clean_path != crate_name else crate_name
            items.insert(
                0,
                {
                    "item_path": module_path,
                    "header": module_path.split("::")[-1],
                    "item_type": "module",
                    "signature": f"mod {module_path.split('::')[-1]}",
                    "docstring": "\n".join(module_docs),
                    "visibility": "public",
                },
            )

        return items

    def extract_code_examples(self, docstring: str) -> list[dict]:
        """
        Extract code examples from documentation.

        Finds ```rust code blocks in documentation.

        Args:
            docstring: Documentation string

        Returns:
            List of code examples with metadata
        """
        examples = []

        # Find code blocks
        code_block_pattern = re.compile(r"```(?:rust|rs)?\n(.*?)\n```", re.DOTALL)

        for match in code_block_pattern.finditer(docstring):
            code = match.group(1).strip()
            if code:
                examples.append(
                    {
                        "description": "Code example from documentation",
                        "code": code,
                        "language": "rust",
                    }
                )

        return examples
