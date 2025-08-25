#!/usr/bin/env python3
"""Test script to verify fuzzy path resolution fix via MCP stdio protocol."""

import asyncio
import json
import subprocess
import sys
from typing import Any, Dict

# ANSI color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
RESET = "\033[0m"


class MCPTestClient:
    """Simple MCP client for testing docsrs-mcp server."""

    def __init__(self, command: list[str]):
        self.command = command
        self.process = None
        self.reader = None
        self.writer = None
        self.id_counter = 0

    async def start(self):
        """Start the MCP server process."""
        print(f"{YELLOW}Starting MCP server with command: {' '.join(self.command)}{RESET}")
        self.process = await asyncio.create_subprocess_exec(
            *self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.reader = self.process.stdout
        self.writer = self.process.stdin

        # Initialize the connection
        await self._send_request("initialize", {"protocolVersion": "2024-11-05"})
        response = await self._read_response()
        if "result" in response:
            print(f"{GREEN}✓ Server initialized successfully{RESET}")
            return True
        else:
            print(f"{RED}✗ Failed to initialize server: {response}{RESET}")
            return False

    async def stop(self):
        """Stop the MCP server process."""
        if self.process:
            self.process.terminate()
            await self.process.wait()
            print(f"{YELLOW}Server stopped{RESET}")

    async def _send_request(self, method: str, params: Dict[str, Any]):
        """Send a JSON-RPC request to the server."""
        self.id_counter += 1
        request = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
            "id": self.id_counter,
        }
        message = json.dumps(request)
        self.writer.write(message.encode() + b"\n")
        await self.writer.drain()

    async def _read_response(self) -> Dict[str, Any]:
        """Read a JSON-RPC response from the server."""
        line = await self.reader.readline()
        if not line:
            raise Exception("Server closed connection")
        return json.loads(line.decode())

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call an MCP tool and return the result."""
        await self._send_request(
            "tools/call", {"name": tool_name, "arguments": arguments}
        )
        response = await self._read_response()
        return response


async def test_fuzzy_resolution():
    """Test the fuzzy path resolution fix."""
    print(f"\n{YELLOW}=== Testing Fuzzy Path Resolution Fix ==={RESET}\n")

    # Test cases that should trigger fuzzy suggestions
    test_cases = [
        {
            "description": "Incorrect path for serde::Serialize",
            "crate": "serde",
            "path": "Serialize",  # Should suggest serde::ser::Serialize
            "expected_suggestion": "serde::ser::Serialize",
        },
        {
            "description": "Incorrect path for serde::Deserialize",
            "crate": "serde",
            "path": "Deserialize",  # Should suggest serde::de::Deserialize
            "expected_suggestion": "serde::de::Deserialize",
        },
        {
            "description": "Typo in tokio function",
            "crate": "tokio",
            "path": "spaw",  # Should suggest tokio::spawn
            "expected_suggestion": "tokio::spawn",
        },
    ]

    # Start the server using uvx with local directory
    client = MCPTestClient(["uvx", "--from", ".", "docsrs-mcp"])

    try:
        # Initialize the server
        if not await client.start():
            print(f"{RED}Failed to initialize server{RESET}")
            return False

        # Run test cases
        all_passed = True
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n{YELLOW}Test {i}: {test_case['description']}{RESET}")
            print(f"  Crate: {test_case['crate']}")
            print(f"  Path: {test_case['path']}")

            # Call get_item_doc with incorrect path
            response = await client.call_tool(
                "get_item_doc",
                {
                    "crate_name": test_case["crate"],
                    "item_path": test_case["path"],
                },
            )

            # Check if we got an error with suggestions
            if "error" in response:
                error_content = response["error"].get("content", "")
                
                # Check if error contains suggestions
                if "Did you mean one of these?" in error_content:
                    print(f"{GREEN}  ✓ Got fuzzy suggestions in error message{RESET}")
                    
                    # Check if expected suggestion is present
                    if test_case["expected_suggestion"] in error_content:
                        print(
                            f"{GREEN}  ✓ Found expected suggestion: {test_case['expected_suggestion']}{RESET}"
                        )
                    else:
                        print(
                            f"{RED}  ✗ Expected suggestion not found: {test_case['expected_suggestion']}{RESET}"
                        )
                        print(f"  Error content: {error_content[:200]}...")
                        all_passed = False
                else:
                    print(f"{RED}  ✗ No fuzzy suggestions in error message{RESET}")
                    print(f"  Error: {error_content[:200]}...")
                    all_passed = False
            elif "result" in response:
                # If we got a result, the path was found (shouldn't happen for these tests)
                print(f"{YELLOW}  ⚠ Path was found (unexpected){RESET}")
            else:
                print(f"{RED}  ✗ Unexpected response format{RESET}")
                print(f"  Response: {response}")
                all_passed = False

        # Test a correct path to ensure normal functionality still works
        print(f"\n{YELLOW}Test: Correct path (should succeed){RESET}")
        print(f"  Crate: serde")
        print(f"  Path: serde::ser::Serialize")

        response = await client.call_tool(
            "get_item_doc",
            {"crate_name": "serde", "item_path": "serde::ser::Serialize"},
        )

        if "result" in response:
            result = response["result"]
            if result and "content" in result:
                content = result["content"]
                if "item_path" in content and "documentation" in content:
                    print(f"{GREEN}  ✓ Successfully retrieved documentation{RESET}")
                else:
                    print(f"{RED}  ✗ Unexpected result format{RESET}")
                    all_passed = False
            else:
                print(f"{RED}  ✗ Empty or invalid result{RESET}")
                all_passed = False
        else:
            print(f"{RED}  ✗ Failed to get documentation for correct path{RESET}")
            if "error" in response:
                print(f"  Error: {response['error']}")
            all_passed = False

        # Summary
        print(f"\n{YELLOW}=== Test Summary ==={RESET}")
        if all_passed:
            print(f"{GREEN}✓ All tests passed! Fuzzy resolution is working correctly.{RESET}")
        else:
            print(f"{RED}✗ Some tests failed. Check the output above for details.{RESET}")

        return all_passed

    except Exception as e:
        print(f"{RED}Test failed with exception: {e}{RESET}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        await client.stop()


async def main():
    """Main entry point."""
    print(f"{YELLOW}=== Fuzzy Path Resolution Fix Test ==={RESET}")
    print("This script tests the fix for the parameter mismatch bug")
    print("in get_fuzzy_suggestions_with_fallback function.\n")

    # Clear any existing cache first
    print(f"{YELLOW}Clearing local cache...{RESET}")
    subprocess.run(["rm", "-rf", "~/.cache/docsrs-mcp"], shell=True, check=False)

    # Run the tests
    success = await test_fuzzy_resolution()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())