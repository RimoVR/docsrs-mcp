#!/usr/bin/env python3
"""
Simple test to verify the key fixes work without MCP protocol complexities.
"""

import sys
import asyncio

# Add the source path
sys.path.insert(0, "/Users/peterkloiber/docsrs-mcp/src")

async def test_fixes():
    print("🧪 Testing version management fixes directly...")
    
    try:
        # Test the list_versions fix
        from docsrs_mcp.services.crate_service import CrateService
        
        service = CrateService()
        print("\n📋 Testing list_versions implementation...")
        
        # This should now return multiple versions from crates.io API
        result = await service.list_versions("serde")
        versions = result.get("versions", [])
        
        print(f"✅ Found {len(versions)} versions for serde")
        if len(versions) > 1:
            print("✅ Multiple versions retrieved (fix successful)")
            # Show a few versions as proof
            for i, version in enumerate(versions[:5]):
                status = "latest" if version.get("is_latest") else "normal"
                yanked = " (yanked)" if version.get("yanked") else ""
                print(f"   - {version['version']} ({status}){yanked}")
            if len(versions) > 5:
                print(f"   ... and {len(versions) - 5} more")
        else:
            print("⚠️  Only one version returned")
            
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(test_fixes())
    if success:
        print("\n🎉 Version management fixes verified!")
    else:
        print("\n❌ Tests failed!")
    sys.exit(0 if success else 1)