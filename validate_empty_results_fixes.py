#!/usr/bin/env python3
"""
Validation script for empty results fixes in MCP tools.

This script tests the following tools that were reported as returning empty results:
1. get_code_intelligence
2. get_error_types  
3. get_unsafe_items
4. get_trait_implementors
5. get_type_traits
6. resolve_method
7. get_associated_items
8. compare_versions
9. suggest_migrations

Run with: python validate_empty_results_fixes.py
"""

import asyncio
import json
import time
import traceback
from pathlib import Path

# Import the MCP server functions directly
from src.docsrs_mcp.mcp_sdk_server import (
    get_code_intelligence,
    get_error_types,
    get_unsafe_items,
    get_trait_implementors,
    get_type_traits,
    resolve_method,
    get_associated_items,
    compare_versions,
    suggest_migrations_handler,
)
from src.docsrs_mcp.ingest import ingest_crate

class TestResult:
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.error = None
        self.result = None
        self.execution_time = 0
        self.has_data = False

async def test_tool(tool_func, tool_name: str, *args, **kwargs) -> TestResult:
    """Test a single MCP tool and return results."""
    result = TestResult(tool_name)
    start_time = time.time()
    
    try:
        print(f"Testing {tool_name}...")
        result.result = await tool_func(*args, **kwargs)
        result.success = True
        
        # Check if result contains meaningful data
        if isinstance(result.result, dict):
            # Check for common indicators of non-empty results
            if any(key in result.result for key in ['items', 'implementors', 'traits', 'methods', 'changes', 'error_types']):
                data_values = [v for k, v in result.result.items() if k in ['items', 'implementors', 'traits', 'methods', 'changes', 'error_types']]
                result.has_data = any(
                    (isinstance(v, (list, dict)) and len(v) > 0) or 
                    (isinstance(v, str) and v.strip()) or
                    (isinstance(v, (int, float)) and v > 0)
                    for v in data_values
                )
            elif 'error' not in result.result:
                result.has_data = True
        
        result.execution_time = time.time() - start_time
        print(f"✅ {tool_name} completed in {result.execution_time:.2f}s - {'Has data' if result.has_data else 'Empty/No data'}")
        
    except Exception as e:
        result.error = str(e)
        result.execution_time = time.time() - start_time
        print(f"❌ {tool_name} failed in {result.execution_time:.2f}s: {e}")
        traceback.print_exc()
    
    return result

async def main():
    print("🧪 Validating Empty Results Fixes")
    print("=" * 50)
    
    # Ensure test crate is ingested first
    print("Ensuring serde crate is ingested...")
    try:
        await ingest_crate("serde", "latest")
        print("✅ Serde crate ready")
    except Exception as e:
        print(f"❌ Failed to ingest serde: {e}")
        return
    
    results = []
    
    # Test Code Intelligence Tools
    print("\n📋 Testing Code Intelligence Tools")
    print("-" * 30)
    
    # Test with different path formats to validate the fix
    test_paths = [
        "serde::de::Deserialize",  # FQN format
        "de::Deserialize",         # Relative format  
        "Deserialize",             # Bare name format
    ]
    
    for path in test_paths:
        result = await test_tool(
            get_code_intelligence, 
            f"get_code_intelligence({path})",
            crate_name="serde", 
            item_path=path,
            version="latest"
        )
        results.append(result)
        if result.success and result.has_data:
            print(f"  📊 Found intelligence data for {path}")
            break
    
    results.append(await test_tool(
        get_error_types,
        "get_error_types",
        crate_name="serde",
        pattern="Error",
        version="latest"
    ))
    
    results.append(await test_tool(
        get_unsafe_items,
        "get_unsafe_items", 
        crate_name="serde",
        include_reasons="true",
        version="latest"
    ))
    
    # Test Trait Tools
    print("\n🔍 Testing Trait Tools")
    print("-" * 20)
    
    # Test common traits
    trait_paths = [
        "serde::de::Deserialize",
        "serde::ser::Serialize", 
        "std::fmt::Debug",
        "Clone",
    ]
    
    for trait_path in trait_paths:
        result = await test_tool(
            get_trait_implementors,
            f"get_trait_implementors({trait_path})",
            crate_name="serde",
            trait_path=trait_path,
            version="latest"
        )
        results.append(result)
        if result.success and result.has_data:
            print(f"  🎯 Found implementors for {trait_path}")
            break
    
    # Test type traits
    type_paths = [
        "serde::de::value::MapDeserializer", 
        "Value",
    ]
    
    for type_path in type_paths:
        result = await test_tool(
            get_type_traits,
            f"get_type_traits({type_path})",
            crate_name="serde",
            type_path=type_path,
            version="latest"
        )
        results.append(result)
        if result.success and result.has_data:
            print(f"  🧬 Found traits for {type_path}")
            break
    
    # Test method resolution
    results.append(await test_tool(
        resolve_method,
        "resolve_method",
        crate_name="serde",
        type_path="serde::de::value::MapDeserializer",
        method_name="deserialize",
        version="latest"
    ))
    
    results.append(await test_tool(
        get_associated_items,
        "get_associated_items",
        crate_name="serde",
        container_path="serde::de::Deserialize",
        version="latest"
    ))
    
    # Test Version Tools - this requires two different versions
    print("\n📚 Testing Version Tools")
    print("-" * 20)
    
    # Try to ingest an older version for comparison
    try:
        await ingest_crate("serde", "1.0.50")
        print("✅ Serde 1.0.50 ready for comparison")
        
        results.append(await test_tool(
            compare_versions,
            "compare_versions",
            crate_name="serde",
            version_a="1.0.50",
            version_b="latest",
            include_unchanged="true"
        ))
        
        results.append(await test_tool(
            suggest_migrations_handler,
            "suggest_migrations",
            crate_name="serde",
            from_version="1.0.50", 
            to_version="latest"
        ))
        
    except Exception as e:
        print(f"⚠️ Could not test version tools: {e}")
        results.append(TestResult("compare_versions (skipped - version unavailable)"))
        results.append(TestResult("suggest_migrations (skipped - version unavailable)"))
    
    # Print Summary
    print("\n📈 Test Results Summary")
    print("=" * 50)
    
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r.success)
    tests_with_data = sum(1 for r in results if r.success and r.has_data)
    failed_tests = sum(1 for r in results if not r.success)
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"With Data: {tests_with_data}")
    print(f"Failed: {failed_tests}")
    
    print(f"\nSuccess Rate: {successful_tests/total_tests*100:.1f}%")
    print(f"Data Coverage: {tests_with_data/total_tests*100:.1f}%")
    
    # Detailed results
    print(f"\n📋 Detailed Results")
    print("-" * 30)
    
    for result in results:
        status_icon = "✅" if result.success else "❌"
        data_icon = "📊" if result.has_data else "📝"
        print(f"{status_icon} {data_icon} {result.name} ({result.execution_time:.2f}s)")
        if result.error:
            print(f"    Error: {result.error}")
        elif result.success and result.result:
            # Print a summary of what we found
            if isinstance(result.result, dict):
                summary_items = []
                for key, value in result.result.items():
                    if isinstance(value, list):
                        summary_items.append(f"{key}: {len(value)} items")
                    elif isinstance(value, dict):
                        summary_items.append(f"{key}: {len(value)} keys")
                    elif isinstance(value, str) and len(value) > 50:
                        summary_items.append(f"{key}: {len(value)} chars")
                
                if summary_items:
                    print(f"    Data: {', '.join(summary_items)}")
    
    # Analysis and recommendations
    print(f"\n🎯 Analysis")
    print("-" * 10)
    
    if tests_with_data < total_tests * 0.5:
        print("⚠️  Many tools are still returning empty results")
        print("   Consider checking:")
        print("   - Database ingestion completeness")
        print("   - Path resolution logic")
        print("   - Table population during ingestion")
    elif tests_with_data >= total_tests * 0.8:
        print("🎉 Most tools are returning data - fixes appear successful!")
    else:
        print("📈 Good progress - some tools working, others may need investigation")
    
    if failed_tests > 0:
        print(f"\n🔧 Failed Tests Need Investigation:")
        for result in results:
            if not result.success:
                print(f"   - {result.name}: {result.error}")

if __name__ == "__main__":
    asyncio.run(main())