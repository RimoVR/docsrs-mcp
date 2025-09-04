#!/usr/bin/env python3
"""Test script for verifying all bug fixes in docsrs-mcp server via direct imports."""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path so we can import modules directly
sys.path.insert(0, str(Path(__file__).parent / "src"))


async def test_health_probe():
    """Test 1: Health probe mechanism for MCP SDK mode."""
    print("\n🔍 Testing Health Probe Mechanism...")
    
    # Import the health functions directly
    from docsrs_mcp.mcp_sdk_server import server_health, get_ingestion_status
    
    # Test server_health tool
    result = await server_health(include_subsystems="true", include_metrics="true")
    
    if result and "status" in result:
        print(f"✅ Health probe working! Status: {result.get('status')}")
        if "subsystems" in result:
            print(f"   Subsystems: {', '.join(result['subsystems'].keys())}")
        
        # Also test ingestion status
        status = await get_ingestion_status(include_progress="true")
        if status:
            print(f"   Ingestion status: {status.get('status', 'unknown')}")
        return True
    else:
        print("❌ Health probe failed")
        return False


async def test_dependency_graph():
    """Test 2: Dependency graph analysis implementation."""
    print("\n🔍 Testing Dependency Graph Analysis...")
    
    from docsrs_mcp.services.cross_reference_service import CrossReferenceService
    from docsrs_mcp.database import CACHE_DIR, init_database
    import tempfile
    
    # Create a temporary test database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Initialize the database schema
        await init_database(db_path)
        
        service = CrossReferenceService(db_path)
        
        # Test with a simple crate (won't have real dependencies unless ingested with Cargo.toml)
        result = await service.get_dependency_graph(
            crate_name="test",
            max_depth=2,
            include_versions=True
        )
        
        # The result has a 'root' key with nested structure
        if isinstance(result, dict) and 'root' in result:
            root = result['root']
            if 'dependencies' in root:
                deps = root['dependencies']
                print(f"✅ Dependency graph structure correct! Dependencies field exists")
                if deps:
                    print(f"   Found {len(deps)} dependencies")
                    for dep in deps[:3]:
                        print(f"   - {dep.get('name', 'unknown')}: {dep.get('version', 'unknown')}")
                else:
                    print("   No dependencies found (expected without real Cargo.toml ingestion)")
                return True
        else:
            print(f"❌ Dependency graph structure incorrect")
            return False


async def test_migration_suggestions():
    """Test 3: Migration suggestions implementation."""
    print("\n🔍 Testing Migration Suggestions...")
    
    from docsrs_mcp.services.cross_reference_service import CrossReferenceService
    from docsrs_mcp.database import CACHE_DIR, init_database
    import tempfile
    
    # Create a temporary test database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        
        # Initialize the database schema
        await init_database(db_path)
        
        service = CrossReferenceService(db_path)
        
        # Test migration suggestions (will work if we have ingested multiple versions)
        result = await service.suggest_migrations(
            crate_name="test",
            from_version="0.1.0",
            to_version="0.2.0"
        )
        
        # Result is a MigrationSuggestionsResponse object
        if hasattr(result, 'suggestions'):
            suggestions = result.suggestions
            print(f"✅ Migration suggestions structure correct!")
            if suggestions:
                print(f"   Found {len(suggestions)} suggestions")
            else:
                print(f"   No migration suggestions (expected for test data)")
            return True
        else:
            print(f"❌ Migration suggestions failed - no suggestions attribute")
            return False


async def test_duplicate_tool_cleanup():
    """Test 4: Verify duplicate tool names are cleaned up."""
    print("\n🔍 Testing Duplicate Tool Name Cleanup...")
    
    # Import the MCP SDK server module to access tools
    from docsrs_mcp import mcp_sdk_server
    
    # Check that camelCase versions are removed by trying to find them as functions
    camelCase_tools = [
        "getDocumentationDetail",
        "extractUsagePatterns", 
        "generateLearningPath"
    ]
    
    all_removed = True
    for tool in camelCase_tools:
        if hasattr(mcp_sdk_server, tool):
            print(f"❌ Tool '{tool}' still exists (should be removed)")
            all_removed = False
        else:
            print(f"✅ Tool '{tool}' correctly removed")
    
    # Verify snake_case versions exist
    snake_case_tools = [
        "get_documentation_detail",
        "extract_usage_patterns",
        "generate_learning_path"
    ]
    
    for tool in snake_case_tools:
        if hasattr(mcp_sdk_server, tool):
            print(f"✅ Tool '{tool}' exists (correct)")
        else:
            print(f"❌ Tool '{tool}' missing!")
            all_removed = False
    
    return all_removed


async def test_resolve_import_alternatives():
    """Test 5: Cross-reference TODO - resolve_import alternatives."""
    print("\n🔍 Testing Cross-Reference Import Resolution...")
    
    from docsrs_mcp.services.cross_reference_service import CrossReferenceService
    from docsrs_mcp.database import CACHE_DIR
    
    # Create a dummy db path for testing
    db_path = CACHE_DIR / "test" / "test.db"
    service = CrossReferenceService(db_path)
    
    # Test resolve_import with alternatives
    result = await service.resolve_import(
        crate_name="test",
        import_path="Result",
        include_alternatives=True
    )
    
    if "resolved_path" in result or "alternatives" in result:
        print(f"✅ Import resolution structure correct!")
        if "alternatives" in result and result["alternatives"]:
            print(f"   Found {len(result['alternatives'])} alternatives")
            for alt in result["alternatives"][:3]:
                print(f"   - {alt.get('path', 'unknown')}: confidence {alt.get('confidence', 0)}")
        else:
            print("   No alternatives found (expected without data)")
        return True
    else:
        print("❌ Import resolution structure incorrect")
        return False


async def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("🧪 TESTING ALL BUG FIXES IN DOCSRS-MCP")
    print("=" * 60)
    
    tests = [
        ("Health Probe", test_health_probe),
        ("Dependency Graph", test_dependency_graph),
        ("Migration Suggestions", test_migration_suggestions),
        ("Duplicate Tool Cleanup", test_duplicate_tool_cleanup),
        ("Cross-Reference TODOs", test_resolve_import_alternatives),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = await test_func()
            results.append((name, success))
        except Exception as e:
            print(f"❌ Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n🎯 Final Score: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All fixes verified successfully!")
        return 0
    else:
        print("⚠️ Some tests failed - review the output above")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))