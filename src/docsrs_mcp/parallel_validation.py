"""Parallel validation mode for comparing FastMCP and SDK implementations."""

import asyncio
import json
import logging
import subprocess
import sys
from typing import Any

logger = logging.getLogger(__name__)


class ParallelValidator:
    """Runs both FastMCP and SDK implementations for comparison."""

    def __init__(self):
        self.fastmcp_process: subprocess.Popen | None = None
        self.sdk_process: subprocess.Popen | None = None
        self.results_match = True
        self.comparison_results = []

    async def start_both_servers(self):
        """Start both MCP implementations in separate processes."""
        logger.info("Starting parallel validation mode...")

        # Start FastMCP server
        logger.info("Starting FastMCP server...")
        self.fastmcp_process = subprocess.Popen(
            [sys.executable, "-m", "docsrs_mcp.mcp_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Start SDK server
        logger.info("Starting MCP SDK server...")
        self.sdk_process = subprocess.Popen(
            [sys.executable, "-m", "docsrs_mcp.mcp_sdk_server"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        # Give servers time to start
        await asyncio.sleep(2)
        logger.info("Both servers started successfully")

    async def send_request(
        self, process: subprocess.Popen, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Send a request to a server and get the response."""
        try:
            # Send request as JSON-RPC
            request_str = json.dumps(request) + "\n"
            process.stdin.write(request_str.encode())
            process.stdin.flush()

            # Read response
            response_line = process.stdout.readline()
            if response_line:
                return json.loads(response_line.decode())
            return {"error": "No response received"}
        except Exception as e:
            logger.error(f"Error sending request: {e}")
            return {"error": str(e)}

    async def compare_responses(self, tool_name: str, params: dict[str, Any]):
        """Compare responses from both implementations."""
        logger.info(f"Testing tool: {tool_name}")

        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": f"tools/{tool_name}",
            "params": params,
        }

        # Send to both servers
        fastmcp_response = await self.send_request(self.fastmcp_process, request)
        sdk_response = await self.send_request(self.sdk_process, request)

        # Compare responses
        match = fastmcp_response == sdk_response
        self.results_match = self.results_match and match

        result = {
            "tool": tool_name,
            "params": params,
            "match": match,
            "fastmcp_response": fastmcp_response,
            "sdk_response": sdk_response,
        }

        self.comparison_results.append(result)

        if not match:
            logger.warning(f"Responses differ for {tool_name}:")
            logger.warning(f"  FastMCP: {json.dumps(fastmcp_response, indent=2)}")
            logger.warning(f"  SDK: {json.dumps(sdk_response, indent=2)}")
        else:
            logger.info(f"  ✓ Responses match for {tool_name}")

    async def run_validation_suite(self):
        """Run a suite of validation tests."""
        # Test cases for validation
        test_cases = [
            ("get_crate_summary", {"crate_name": "serde", "version": "latest"}),
            ("search_items", {"crate_name": "tokio", "query": "spawn task", "k": "5"}),
            (
                "get_item_doc",
                {"crate_name": "serde", "item_path": "serde::Deserialize"},
            ),
            ("get_module_tree", {"crate_name": "tokio"}),
            (
                "search_examples",
                {"crate_name": "tokio", "query": "async runtime", "k": "3"},
            ),
            ("list_versions", {"crate_name": "serde"}),
        ]

        for tool_name, params in test_cases:
            await self.compare_responses(tool_name, params)
            await asyncio.sleep(1)  # Small delay between tests

    async def stop_servers(self):
        """Stop both server processes."""
        logger.info("Stopping servers...")

        if self.fastmcp_process:
            self.fastmcp_process.terminate()
            await asyncio.sleep(1)
            if self.fastmcp_process.poll() is None:
                self.fastmcp_process.kill()

        if self.sdk_process:
            self.sdk_process.terminate()
            await asyncio.sleep(1)
            if self.sdk_process.poll() is None:
                self.sdk_process.kill()

        logger.info("Servers stopped")

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "=" * 60)
        print("PARALLEL VALIDATION SUMMARY")
        print("=" * 60)

        total_tests = len(self.comparison_results)
        passed_tests = sum(1 for r in self.comparison_results if r["match"])

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {(passed_tests / total_tests) * 100:.1f}%")

        if not self.results_match:
            print("\nFailed tests:")
            for result in self.comparison_results:
                if not result["match"]:
                    print(f"  - {result['tool']} with params {result['params']}")

        print("=" * 60)


def run_parallel_validation():
    """Main entry point for parallel validation mode."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        validator = ParallelValidator()

        try:
            # Start both servers
            await validator.start_both_servers()

            # Run validation tests
            await validator.run_validation_suite()

            # Print summary
            validator.print_summary()

        finally:
            # Clean up
            await validator.stop_servers()

        # Exit with appropriate code
        sys.exit(0 if validator.results_match else 1)

    asyncio.run(main())


if __name__ == "__main__":
    run_parallel_validation()
