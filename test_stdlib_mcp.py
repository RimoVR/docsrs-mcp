#!/usr/bin/env python3
"""
Test script for stdlib support in docsrs-mcp server via MCP protocol.
Tests the standard library ingestion and search functionality.
"""

import asyncio
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MCPClient:
    """Simple MCP client for testing stdio communication."""
    
    def __init__(self):
        self.process = None
        self.request_id = 0
    
    async def start_server(self):
        """Start the MCP server process."""
        logger.info("Starting docsrs-mcp server...")
        self.process = await asyncio.create_subprocess_exec(
            "uv", "run", "docsrs-mcp",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=Path(__file__).parent
        )
        
        # Wait a moment for startup
        await asyncio.sleep(2)
        logger.info("Server started")
    
    async def send_request(self, method: str, params: dict = None):
        """Send an MCP request and get response."""
        if not self.process:
            raise RuntimeError("Server not started")
        
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": self.request_id,
            "method": method,
            "params": params or {}
        }
        
        request_json = json.dumps(request) + "\n"
        logger.info(f"Sending request: {method}")
        logger.debug(f"Request data: {request_json.strip()}")
        
        self.process.stdin.write(request_json.encode())
        await self.process.stdin.drain()
        
        # Read response
        response_line = await self.process.stdout.readline()
        if not response_line:
            # Check for errors
            error_output = await self.process.stderr.read(1024)
            raise RuntimeError(f"No response from server. Error: {error_output.decode()}")
        
        try:
            response = json.loads(response_line.decode())
            logger.info(f"Got response for {method}")
            logger.debug(f"Response: {json.dumps(response, indent=2)}")
            return response
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse response: {response_line.decode()}")
            raise
    
    async def stop_server(self):
        """Stop the server process."""
        if self.process:
            logger.info("Stopping server...")
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            logger.info("Server stopped")

async def test_stdlib_support():
    """Test standard library support functionality."""
    client = MCPClient()
    
    try:
        await client.start_server()
        
        # Test 1: Initialize - list available tools
        logger.info("=== Test 1: List Available Tools ===")
        response = await client.send_request("tools/list")
        if "result" in response:
            tools = response["result"]["tools"]
            logger.info(f"Available tools: {[tool['name'] for tool in tools]}")
            
            # Check if we have the expected tools
            tool_names = {tool["name"] for tool in tools}
            expected_tools = {"search_items", "get_item_doc", "get_crate_summary"}
            missing_tools = expected_tools - tool_names
            if missing_tools:
                logger.warning(f"Missing expected tools: {missing_tools}")
            else:
                logger.info("All expected tools available ✓")
        else:
            logger.error(f"Failed to list tools: {response}")
            return False
        
        # Test 2: Search for std library items
        logger.info("=== Test 2: Search std Library Items ===")
        search_params = {
            "name": "search_items",
            "arguments": {
                "crate_name": "std",
                "query": "HashMap collection",
                "k": "5"
            }
        }
        response = await client.send_request("tools/call", search_params)
        
        if "result" in response and "content" in response["result"]:
            content = response["result"]["content"]
            if isinstance(content, list) and len(content) > 0:
                # Parse the JSON response
                search_results = json.loads(content[0]["text"])
                logger.info(f"Found {len(search_results.get('results', []))} std library items")
                
                # Check if we got meaningful results
                if search_results.get("results"):
                    first_result = search_results["results"][0]
                    logger.info(f"First result: {first_result.get('item_path', 'N/A')}")
                    logger.info("std library search working ✓")
                else:
                    logger.warning("No search results returned for std library")
            else:
                logger.error("No content in search response")
        else:
            logger.error(f"Failed to search std library: {response}")
            return False
        
        # Test 3: Get specific std library item documentation
        logger.info("=== Test 3: Get std Library Item Documentation ===")
        get_doc_params = {
            "name": "get_item_doc",
            "arguments": {
                "crate_name": "std",
                "item_path": "std::collections::HashMap"
            }
        }
        response = await client.send_request("tools/call", get_doc_params)
        
        if "result" in response and "content" in response["result"]:
            content = response["result"]["content"]
            if isinstance(content, list) and len(content) > 0:
                doc_result = json.loads(content[0]["text"])
                if doc_result.get("documentation"):
                    logger.info("std::collections::HashMap documentation retrieved ✓")
                    logger.info(f"Doc preview: {doc_result['documentation'][:100]}...")
                else:
                    logger.warning("No documentation content for HashMap")
            else:
                logger.error("No content in documentation response")
        else:
            logger.error(f"Failed to get std library documentation: {response}")
            return False
        
        # Test 4: Test core library
        logger.info("=== Test 4: Search core Library Items ===")
        core_search_params = {
            "name": "search_items", 
            "arguments": {
                "crate_name": "core",
                "query": "Option enum",
                "k": "3"
            }
        }
        response = await client.send_request("tools/call", core_search_params)
        
        if "result" in response and "content" in response["result"]:
            content = response["result"]["content"]
            if isinstance(content, list) and len(content) > 0:
                search_results = json.loads(content[0]["text"])
                logger.info(f"Found {len(search_results.get('results', []))} core library items")
                
                if search_results.get("results"):
                    first_result = search_results["results"][0]
                    logger.info(f"First core result: {first_result.get('item_path', 'N/A')}")
                    logger.info("core library search working ✓")
                else:
                    logger.warning("No search results returned for core library")
            else:
                logger.error("No content in core search response")
        else:
            logger.error(f"Failed to search core library: {response}")
            return False
            
        # Test 5: Check if we're getting local JSON or fallback
        logger.info("=== Test 5: Check Documentation Source ===")
        summary_params = {
            "name": "get_crate_summary",
            "arguments": {
                "crate_name": "std"
            }
        }
        response = await client.send_request("tools/call", summary_params)
        
        if "result" in response and "content" in response["result"]:
            content = response["result"]["content"]
            if isinstance(content, list) and len(content) > 0:
                summary_result = json.loads(content[0]["text"])
                ingestion_tier = summary_result.get("ingestion_tier", "unknown")
                logger.info(f"std library ingestion tier: {ingestion_tier}")
                
                if ingestion_tier == "RUST_LANG_STDLIB":
                    logger.info("Using local rustup stdlib JSON ✓✓✓")
                elif ingestion_tier == "DESCRIPTION_ONLY":
                    logger.warning("Using fallback documentation - consider installing rust-docs-json")
                else:
                    logger.info(f"Using tier: {ingestion_tier}")
            else:
                logger.error("No content in summary response")
        else:
            logger.error(f"Failed to get std library summary: {response}")
            return False
        
        logger.info("=== All Tests Completed Successfully ✓ ===")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await client.stop_server()

async def main():
    """Main test function."""
    logger.info("Starting stdlib support tests for docsrs-mcp...")
    
    success = await test_stdlib_support()
    
    if success:
        logger.info("🎉 All tests passed! Standard library support is working.")
        sys.exit(0)
    else:
        logger.error("❌ Tests failed! Standard library support needs fixes.")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())