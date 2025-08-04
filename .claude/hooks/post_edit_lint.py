#!/usr/bin/env python3
"""
Post-edit hook to run linters and make errors/warnings visible immediately
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def get_file_type(file_path):
    """
    Determine the file type based on extension
    """
    path = Path(file_path)
    extension = path.suffix.lower()

    # Map extensions to file types
    type_map = {
        ".rs": "rust",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".py": "python",
        ".json": "json",
        ".md": "markdown",
    }

    return type_map.get(extension, "unknown")


def check_tool_available(tool_name):
    """
    Check if a tool is available in PATH
    """
    return shutil.which(tool_name) is not None


def run_linter(file_path, file_type):
    """
    Run appropriate linter for the file type with rock-solid error handling
    """
    results = []

    if file_type == "rust":
        # Check if cargo is available
        if check_tool_available("cargo"):
            try:
                # Find the workspace root by looking for Cargo.toml
                workspace_root = Path(file_path).parent
                while workspace_root != workspace_root.parent:
                    if (workspace_root / "Cargo.toml").exists():
                        break
                    workspace_root = workspace_root.parent

                # Run cargo fmt check
                fmt_result = subprocess.run(
                    ["cargo", "fmt", "--", "--check"],
                    check=False, capture_output=True,
                    text=True,
                    cwd=workspace_root,
                )
                if fmt_result.returncode != 0:
                    results.append(
                        {
                            "tool": "cargo fmt",
                            "status": "error",
                            "message": "File needs formatting. Run: cargo fmt",
                        }
                    )

                # Run clippy with all targets
                clippy_result = subprocess.run(
                    ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
                    check=False, capture_output=True,
                    text=True,
                    cwd=workspace_root,
                    timeout=30,
                )
                if clippy_result.returncode != 0:
                    # Parse clippy output for specific file
                    if str(file_path) in clippy_result.stderr:
                        results.append(
                            {
                                "tool": "cargo clippy",
                                "status": "error",
                                "message": clippy_result.stderr[:1000],
                            }
                        )

                # Run cargo check for type errors
                check_result = subprocess.run(
                    ["cargo", "check", "--all-targets"],
                    check=False, capture_output=True,
                    text=True,
                    cwd=workspace_root,
                    timeout=30,
                )
                if check_result.returncode != 0:
                    if str(file_path) in check_result.stderr:
                        results.append(
                            {
                                "tool": "cargo check",
                                "status": "error",
                                "message": check_result.stderr[:1000],
                            }
                        )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "tool": "cargo",
                        "status": "warning",
                        "message": "Linting timed out after 30 seconds",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "tool": "cargo",
                        "status": "warning",
                        "message": f"Linting error: {str(e)}",
                    }
                )
        else:
            results.append(
                {
                    "tool": "rust",
                    "status": "warning",
                    "message": "cargo not found. Install Rust toolchain for linting.",
                }
            )

    elif file_type in ["typescript", "javascript"]:
        # Check for npm/npx
        if check_tool_available("npm") or check_tool_available("npx"):
            try:
                # Find package.json root
                project_root = Path(file_path).parent
                while project_root != project_root.parent:
                    if (project_root / "package.json").exists():
                        break
                    project_root = project_root.parent

                # Check if eslint is available
                eslint_config_exists = any(
                    (project_root / f).exists()
                    for f in [
                        ".eslintrc",
                        ".eslintrc.js",
                        ".eslintrc.json",
                        ".eslintrc.yml",
                        "eslint.config.js",
                    ]
                )

                if eslint_config_exists:
                    # Run eslint with --max-warnings 0 for strict checking
                    lint_result = subprocess.run(
                        ["npx", "eslint", "--max-warnings", "0", str(file_path)],
                        check=False, capture_output=True,
                        text=True,
                        cwd=project_root,
                        timeout=20,
                    )
                    if lint_result.returncode != 0:
                        results.append(
                            {
                                "tool": "eslint",
                                "status": "error",
                                "message": lint_result.stdout[:1000]
                                if lint_result.stdout
                                else lint_result.stderr[:1000],
                            }
                        )

                # Run TypeScript compiler for .ts/.tsx files
                if (
                    file_type == "typescript"
                    and (project_root / "tsconfig.json").exists()
                ):
                    tsc_result = subprocess.run(
                        ["npx", "tsc", "--noEmit", "--pretty", "false"],
                        check=False, capture_output=True,
                        text=True,
                        cwd=project_root,
                        timeout=20,
                    )
                    if tsc_result.returncode != 0:
                        # Filter for specific file errors
                        if str(file_path) in tsc_result.stdout:
                            results.append(
                                {
                                    "tool": "typescript",
                                    "status": "error",
                                    "message": tsc_result.stdout[:1000],
                                }
                            )

                # Run prettier check if available
                prettier_config_exists = any(
                    (project_root / f).exists()
                    for f in [
                        ".prettierrc",
                        ".prettierrc.js",
                        ".prettierrc.json",
                        "prettier.config.js",
                    ]
                )

                if prettier_config_exists:
                    prettier_result = subprocess.run(
                        ["npx", "prettier", "--check", str(file_path)],
                        check=False, capture_output=True,
                        text=True,
                        cwd=project_root,
                        timeout=10,
                    )
                    if prettier_result.returncode != 0:
                        results.append(
                            {
                                "tool": "prettier",
                                "status": "warning",
                                "message": "File needs formatting. Run: npm run format or npx prettier --write",
                            }
                        )
            except subprocess.TimeoutExpired:
                results.append(
                    {
                        "tool": "node",
                        "status": "warning",
                        "message": "Linting timed out",
                    }
                )
            except Exception as e:
                results.append(
                    {
                        "tool": "node",
                        "status": "warning",
                        "message": f"Linting error: {str(e)}",
                    }
                )
        else:
            results.append(
                {
                    "tool": "node",
                    "status": "warning",
                    "message": "npm/npx not found. Install Node.js for JavaScript/TypeScript linting.",
                }
            )

    elif file_type == "python":
        try:
            # Check for ruff first (fastest and most comprehensive)
            if check_tool_available("ruff"):
                # Run ruff check
                ruff_result = subprocess.run(
                    ["ruff", "check", str(file_path)],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10,
                )
                if ruff_result.returncode != 0:
                    results.append(
                        {
                            "tool": "ruff",
                            "status": "error",
                            "message": ruff_result.stdout[:1000],
                        }
                    )

                # Run ruff format check
                ruff_format_result = subprocess.run(
                    ["ruff", "format", "--check", str(file_path)],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10,
                )
                if ruff_format_result.returncode != 0:
                    results.append(
                        {
                            "tool": "ruff format",
                            "status": "warning",
                            "message": "File needs formatting. Run: ruff format",
                        }
                    )
            elif check_tool_available("flake8"):
                # Fallback to flake8
                flake8_result = subprocess.run(
                    ["flake8", str(file_path)],
                    check=False, capture_output=True,
                    text=True,
                    timeout=10,
                )
                if flake8_result.returncode != 0:
                    results.append(
                        {
                            "tool": "flake8",
                            "status": "error",
                            "message": flake8_result.stdout[:1000],
                        }
                    )

            # Check for mypy if available and .py file
            if check_tool_available("mypy"):
                # Find project root (where pyproject.toml or setup.py exists)
                project_root = Path(file_path).parent
                while project_root != project_root.parent:
                    if any(
                        (project_root / f).exists()
                        for f in ["pyproject.toml", "setup.py", "mypy.ini"]
                    ):
                        break
                    project_root = project_root.parent

                mypy_result = subprocess.run(
                    ["mypy", str(file_path)],
                    check=False, capture_output=True,
                    text=True,
                    cwd=project_root,
                    timeout=15,
                )
                if mypy_result.returncode != 0:
                    results.append(
                        {
                            "tool": "mypy",
                            "status": "warning",
                            "message": mypy_result.stdout[:1000],
                        }
                    )
        except subprocess.TimeoutExpired:
            results.append(
                {"tool": "python", "status": "warning", "message": "Linting timed out"}
            )
        except Exception as e:
            results.append(
                {
                    "tool": "python",
                    "status": "warning",
                    "message": f"Linting error: {str(e)}",
                }
            )

        if not (check_tool_available("ruff") or check_tool_available("flake8")):
            results.append(
                {
                    "tool": "python",
                    "status": "warning",
                    "message": "No Python linters found. Install ruff (recommended) or flake8.",
                }
            )

    elif file_type == "json":
        # Simple JSON validation
        try:
            with open(file_path) as f:
                json.load(f)
        except json.JSONDecodeError as e:
            results.append(
                {
                    "tool": "json",
                    "status": "error",
                    "message": f"Invalid JSON: {str(e)}",
                }
            )

    return results


def main():
    """
    Process PostToolUse hook data for Edit/Write tools
    """
    try:
        # Read JSON from stdin as per Claude Code documentation
        data = json.load(sys.stdin)

        # Extract tool information
        tool_name = data.get("tool_name", "")
        tool_input = data.get("tool_input", {})

        # Process Edit, MultiEdit, and Write tools
        if tool_name not in ["Edit", "MultiEdit", "Write"]:
            return 0

        # Get file path(s)
        file_paths = []
        if tool_name in ["Edit", "Write"]:
            file_path = tool_input.get("file_path", "")
            if file_path:
                file_paths.append(file_path)
        elif tool_name == "MultiEdit":
            file_path = tool_input.get("file_path", "")
            if file_path:
                file_paths.append(file_path)

        # Run linters for each file
        all_results = []
        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            file_type = get_file_type(file_path)
            if file_type != "unknown":
                results = run_linter(file_path, file_type)
                if results:
                    all_results.extend(results)

        # Output results
        if all_results:
            print("\nLinting Results:")
            for result in all_results:
                status_marker = (
                    "[ERROR]" if result["status"] == "error" else "[WARNING]"
                )
                print(f"\n{status_marker} {result['tool']}:")
                print(result["message"][:500])  # Limit output length
                if len(result["message"]) > 500:
                    print("... (truncated)")

            # Suggest immediate fixes
            has_errors = any(r["status"] == "error" for r in all_results)
            if has_errors:
                print("\nFix these issues immediately to maintain code quality!")
        else:
            # Silent success - no issues found
            pass

    except Exception as e:
        # Log error but don't fail the hook
        print(f"Linting hook error: {str(e)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
