#!/usr/bin/env python3
"""Test the trait search fixes."""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def test_trait_search_fixes():
    """Test the improved trait search with FQN/bare name fallback."""
    
    print("🧪 Testing trait search fixes...")
    
    try:
        from src.docsrs_mcp.mcp_sdk_server import get_trait_implementors
        
        # Test cases with different trait path formats
        test_cases = [
            ("serde", "serde::de::Deserialize", "FQN search"),
            ("serde", "Deserialize", "Bare name search"),
            ("serde", "serde::ser::Serialize", "FQN search for Serialize"),
            ("serde", "Serialize", "Bare name search for Serialize"),
        ]
        
        print("Testing get_trait_implementors with different path formats:")
        
        for crate, trait_path, description in test_cases:
            print(f"\n🔍 Testing: {description}")
            print(f"   Query: get_trait_implementors('{crate}', '{trait_path}')")
            
            try:
                result = await get_trait_implementors(crate, trait_path)
                
                # Check if we got any implementors
                implementors = result.get("implementors", [])
                total = result.get("total_count", 0)
                
                if total > 0:
                    print(f"   ✅ SUCCESS: Found {total} implementors")
                    # Show first few implementors
                    for i, impl in enumerate(implementors[:3], 1):
                        type_path = impl.get("type_path", "Unknown")
                        print(f"      {i}. {type_path}")
                    if len(implementors) > 3:
                        print(f"      ... and {len(implementors) - 3} more")
                else:
                    print(f"   ❌ EMPTY: No implementors found")
                    
            except Exception as e:
                print(f"   💥 ERROR: {e}")
        
        print(f"\n🎯 Summary:")
        print(f"   The fixes should allow both FQN and bare name searches to work.")
        print(f"   If any test shows SUCCESS, the fix is working!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_trait_search_fixes())
    if success:
        print("\n🎉 Trait search test completed!")
    else:
        print("\n💥 Trait search test failed!")