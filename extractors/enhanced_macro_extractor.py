"""
Enhanced macro extraction for Rust source code.

Based on research of macro structures and best practices for extraction.
Handles macro_rules!, procedural macros, derive macros, and attribute macros.
"""

import logging
import re

logger = logging.getLogger(__name__)


class EnhancedMacroExtractor:
    """
    Specialized extractor for Rust macros with advanced pattern recognition.

    Handles:
    - macro_rules! with various patterns and fragment specifiers
    - #[macro_export] and visibility attributes
    - Procedural macros with #[proc_macro]
    - Derive macros with #[proc_macro_derive]
    - Attribute macros with #[proc_macro_attribute]
    - Complex nested macro patterns
    - Documentation extraction for macros
    """

    # Enhanced regex patterns for macro detection
    MACRO_PATTERNS = {
        # Basic macro_rules! with or without macro_export
        "macro_rules_basic": re.compile(
            r"(?:#\[macro_export(?:\([^)]*\))?\]\s*\n\s*)?"  # Optional macro_export
            r"macro_rules!\s+(\w+)\s*\{",  # macro_rules! name {
            re.MULTILINE,
        ),
        # Standalone macro_export attribute (to find macros that follow)
        "macro_export_attr": re.compile(
            r"^#\[macro_export(?:\([^)]*\))?\]", re.MULTILINE
        ),
        # Procedural macro types
        "proc_macro": re.compile(
            r"#\[proc_macro\]\s*\n\s*pub\s+fn\s+(\w+)", re.MULTILINE
        ),
        "proc_macro_derive": re.compile(
            r"#\[proc_macro_derive\((\w+)(?:,\s*attributes\([^)]*\))?\)\]\s*\n\s*pub\s+fn\s+(\w+)",
            re.MULTILINE,
        ),
        "proc_macro_attribute": re.compile(
            r"#\[proc_macro_attribute\]\s*\n\s*pub\s+fn\s+(\w+)", re.MULTILINE
        ),
        # Public macro (2.0 style)
        "pub_macro": re.compile(
            r"pub\s+macro\s+(\w+)\s*(?:\([^)]*\))?\s*\{", re.MULTILINE
        ),
        # Documentation patterns
        "doc_comment": re.compile(r"^\s*///\s?(.*)$"),
        "doc_inner": re.compile(r"^\s*//!\s?(.*)$"),
        "doc_block": re.compile(r"/\*\*(.*?)\*/", re.DOTALL),
    }

    # Fragment specifier patterns for macro_rules!
    FRAGMENT_SPECIFIERS = {
        "expr": "expression",
        "ident": "identifier",
        "pat": "pattern",
        "ty": "type",
        "stmt": "statement",
        "block": "block",
        "item": "item",
        "meta": "meta attribute",
        "tt": "token tree",
        "vis": "visibility",
        "literal": "literal",
        "path": "path",
    }

    def extract_macros(
        self, content: str, filepath: str, crate_name: str
    ) -> list[dict]:
        """
        Extract all macro definitions from Rust source content.

        Args:
            content: Source file content
            filepath: Path to the file
            crate_name: Name of the crate

        Returns:
            List of extracted macro items with documentation and signatures
        """
        items = []
        lines = content.split("\n")

        # Clean filepath for module path
        module_path = self._clean_filepath(filepath, crate_name)

        # Extract different macro types
        items.extend(self._extract_macro_rules(lines, module_path, crate_name))
        items.extend(self._extract_proc_macros(content, lines, module_path, crate_name))
        items.extend(self._extract_pub_macros(lines, module_path, crate_name))

        return items

    def _clean_filepath(self, filepath: str, crate_name: str) -> str:
        """Clean filepath to create module path."""
        clean = filepath
        if "/" in clean:
            parts = clean.split("/", 1)
            if len(parts) > 1:
                clean = parts[1]
        clean = clean.replace(".rs", "").replace("/", "::")

        if clean.endswith("::lib"):
            return crate_name
        elif clean.endswith("::mod"):
            return clean[:-5]
        return clean

    def _extract_macro_rules(
        self, lines: list[str], module_path: str, crate_name: str
    ) -> list[dict]:
        """Extract macro_rules! definitions with enhanced pattern recognition."""
        items = []
        i = 0

        while i < len(lines):
            line = lines[i]

            # Check for macro_export attribute
            has_export = bool(self.MACRO_PATTERNS["macro_export_attr"].match(line))
            if has_export:
                # Look ahead for macro_rules!
                for j in range(i + 1, min(i + 5, len(lines))):
                    if "macro_rules!" in lines[j]:
                        i = j
                        line = lines[i]
                        break

            # Check for macro_rules!
            if "macro_rules!" in line:
                match = re.search(r"macro_rules!\s+(\w+)", line)
                if match:
                    macro_name = match.group(1)

                    # Extract documentation
                    doc_lines = self._extract_preceding_docs(lines, i)

                    # Extract macro signature with patterns
                    signature, patterns = self._extract_macro_signature(lines, i)

                    # Determine visibility
                    visibility = "public" if has_export else "crate"

                    # Build item path
                    if module_path == crate_name:
                        item_path = f"{crate_name}::{macro_name}"
                    else:
                        item_path = f"{module_path}::{macro_name}"

                    # Create the item
                    item = {
                        "item_path": item_path,
                        "header": macro_name,
                        "item_type": "macro",
                        "signature": signature,
                        "docstring": "\n".join(doc_lines),
                        "visibility": visibility,
                    }

                    # Add pattern information if available
                    if patterns:
                        item["macro_patterns"] = patterns

                    items.append(item)

                    # Skip to end of macro definition
                    i = self._find_macro_end(lines, i)

            i += 1

        return items

    def _extract_proc_macros(
        self, content: str, lines: list[str], module_path: str, crate_name: str
    ) -> list[dict]:
        """Extract procedural macro definitions."""
        items = []

        # Find proc_macro functions
        for pattern_name, pattern in [
            ("proc_macro", self.MACRO_PATTERNS["proc_macro"]),
            ("proc_macro_attribute", self.MACRO_PATTERNS["proc_macro_attribute"]),
        ]:
            for match in pattern.finditer(content):
                macro_name = match.group(1)

                # Find line number
                line_num = content[: match.start()].count("\n")

                # Extract documentation
                doc_lines = self._extract_preceding_docs(lines, line_num)

                # Build item
                item_path = f"{crate_name}::{macro_name}"

                items.append(
                    {
                        "item_path": item_path,
                        "header": macro_name,
                        "item_type": pattern_name.replace("_", " "),
                        "signature": match.group(0).replace("\n", " "),
                        "docstring": "\n".join(doc_lines),
                        "visibility": "public",
                    }
                )

        # Find derive macros
        for match in self.MACRO_PATTERNS["proc_macro_derive"].finditer(content):
            derive_name = match.group(1)
            fn_name = match.group(2)

            # Find line number
            line_num = content[: match.start()].count("\n")

            # Extract documentation
            doc_lines = self._extract_preceding_docs(lines, line_num)

            # Build item
            item_path = f"{crate_name}::{derive_name}"

            items.append(
                {
                    "item_path": item_path,
                    "header": derive_name,
                    "item_type": "derive macro",
                    "signature": match.group(0).replace("\n", " "),
                    "docstring": "\n".join(doc_lines),
                    "visibility": "public",
                    "impl_fn": fn_name,
                }
            )

        return items

    def _extract_pub_macros(
        self, lines: list[str], module_path: str, crate_name: str
    ) -> list[dict]:
        """Extract pub macro definitions (macro 2.0)."""
        items = []

        for i, line in enumerate(lines):
            if match := self.MACRO_PATTERNS["pub_macro"].search(line):
                macro_name = match.group(1)

                # Extract documentation
                doc_lines = self._extract_preceding_docs(lines, i)

                # Extract signature
                signature = self._extract_pub_macro_signature(lines, i)

                # Build item path
                if module_path == crate_name:
                    item_path = f"{crate_name}::{macro_name}"
                else:
                    item_path = f"{module_path}::{macro_name}"

                items.append(
                    {
                        "item_path": item_path,
                        "header": macro_name,
                        "item_type": "macro",
                        "signature": signature,
                        "docstring": "\n".join(doc_lines),
                        "visibility": "public",
                    }
                )

        return items

    def _extract_preceding_docs(self, lines: list[str], line_num: int) -> list[str]:
        """Extract documentation comments preceding a line."""
        doc_lines = []
        j = line_num - 1

        while j >= 0:
            line = lines[j]

            # Check for doc comment
            if match := self.MACRO_PATTERNS["doc_comment"].match(line):
                doc_lines.insert(0, match.group(1))
                j -= 1
            elif line.strip() == "":
                # Allow blank lines
                j -= 1
            elif line.strip().startswith("#["):
                # Skip other attributes
                j -= 1
            else:
                break

        return doc_lines

    def _extract_macro_signature(
        self, lines: list[str], start_line: int
    ) -> tuple[str, list[dict]]:
        """
        Extract macro_rules! signature with pattern analysis.

        Returns:
            Tuple of (signature_string, patterns_list)
        """
        signature_lines = []
        patterns = []
        brace_count = 0
        in_macro = False

        for i in range(start_line, min(start_line + 50, len(lines))):
            line = lines[i]

            if "{" in line:
                in_macro = True
                brace_count += line.count("{")

            if in_macro:
                signature_lines.append(line.strip())

                # Extract pattern information
                if "=>" in line:
                    # Extract matcher pattern
                    matcher = line.split("=>")[0].strip()

                    # Find fragment specifiers
                    fragments = re.findall(r"\$(\w+):(\w+)", matcher)
                    if fragments:
                        pattern_info = {
                            "matcher": matcher[:100],  # Truncate long patterns
                            "parameters": [
                                {
                                    "name": name,
                                    "type": self.FRAGMENT_SPECIFIERS.get(spec, spec),
                                }
                                for name, spec in fragments
                            ],
                        }
                        patterns.append(pattern_info)

            brace_count -= line.count("}")

            if in_macro and brace_count <= 0:
                break

            # Limit signature length
            if len(signature_lines) > 10:
                signature_lines.append("...")
                break

        signature = " ".join(signature_lines)
        signature = re.sub(r"\s+", " ", signature)

        if len(signature) > 300:
            signature = signature[:297] + "..."

        return signature, patterns

    def _extract_pub_macro_signature(self, lines: list[str], start_line: int) -> str:
        """Extract pub macro signature."""
        signature_lines = []
        brace_count = 0

        for i in range(start_line, min(start_line + 20, len(lines))):
            line = lines[i]
            signature_lines.append(line.strip())

            brace_count += line.count("{")
            brace_count -= line.count("}")

            if brace_count <= 0 and "{" in line:
                break

        signature = " ".join(signature_lines)
        signature = re.sub(r"\s+", " ", signature)

        if len(signature) > 200:
            signature = signature[:197] + "..."

        return signature

    def _find_macro_end(self, lines: list[str], start_line: int) -> int:
        """Find the end of a macro definition."""
        brace_count = 0
        in_macro = False

        for i in range(start_line, len(lines)):
            line = lines[i]

            if "{" in line:
                in_macro = True
                brace_count += line.count("{")

            if in_macro:
                brace_count -= line.count("}")

                if brace_count <= 0:
                    return i

        return start_line
