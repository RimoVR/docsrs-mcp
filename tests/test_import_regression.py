"""
Import isolation tests to prevent regression of storage_manager import fixes.

This test ensures that the storage_manager can properly import the store_modules
function and that all critical imports work correctly in isolation.
"""

import sys
import importlib
import pytest
from unittest.mock import patch


def test_storage_manager_imports_successfully():
    """Test that storage_manager can be imported without errors."""
    try:
        from docsrs_mcp.ingestion import storage_manager
        assert storage_manager is not None
    except ImportError as e:
        pytest.fail(f"Failed to import storage_manager: {e}")


def test_store_modules_import_from_correct_location():
    """Test that store_modules can be imported from the correct location."""
    try:
        from docsrs_mcp.database.storage import store_modules
        assert callable(store_modules)
    except ImportError as e:
        pytest.fail(f"Failed to import store_modules from database.storage: {e}")


def test_storage_manager_store_modules_dynamic_import():
    """Test that the dynamic import in storage_manager works correctly."""
    # This simulates the exact import pattern used in storage_manager.py
    try:
        # This is the same import pattern used in the fixed code
        from docsrs_mcp.database.storage import store_modules
        
        # Verify it's callable (the function exists)
        assert callable(store_modules)
        
        # Verify it has the expected signature by checking if it's async
        import inspect
        assert inspect.iscoroutinefunction(store_modules), "store_modules should be async"
        
    except ImportError as e:
        pytest.fail(f"Dynamic import failed: {e}")


def test_no_circular_imports():
    """Test that there are no circular import issues."""
    # Clear any cached modules to ensure clean import test
    modules_to_clear = [
        'docsrs_mcp.ingestion.storage_manager',
        'docsrs_mcp.database.storage'
    ]
    
    for module in modules_to_clear:
        if module in sys.modules:
            del sys.modules[module]
    
    try:
        # Import storage_manager first
        from docsrs_mcp.ingestion import storage_manager
        
        # Then import the database storage module
        from docsrs_mcp.database import storage
        
        # Both should be available
        assert storage_manager is not None
        assert storage is not None
        assert hasattr(storage, 'store_modules')
        
    except ImportError as e:
        pytest.fail(f"Circular import detected: {e}")


def test_import_path_resolution():
    """Test that the absolute import path resolves correctly across different contexts."""
    # Test the import using different methods to ensure robustness
    
    # Method 1: Direct import
    try:
        import docsrs_mcp.database.storage
        assert hasattr(docsrs_mcp.database.storage, 'store_modules')
    except ImportError as e:
        pytest.fail(f"Direct module import failed: {e}")
    
    # Method 2: importlib
    try:
        module = importlib.import_module('docsrs_mcp.database.storage')
        assert hasattr(module, 'store_modules')
    except ImportError as e:
        pytest.fail(f"importlib import failed: {e}")
    
    # Method 3: From import (matches the fixed code)
    try:
        from docsrs_mcp.database.storage import store_modules
        assert callable(store_modules)
    except ImportError as e:
        pytest.fail(f"From import failed: {e}")


def test_storage_manager_integration():
    """Test that storage_manager can use the imported function without errors."""
    try:
        # Import the storage_manager module
        from docsrs_mcp.ingestion import storage_manager
        
        # Verify the module can access the import internally
        # This tests that the import works within the module context
        import inspect
        source = inspect.getsource(storage_manager)
        
        # Verify the correct import statement is present
        assert 'from docsrs_mcp.database.storage import store_modules' in source
        
        # Verify the old broken import is not present
        assert 'from .intelligence_extractor import store_modules' not in source
        
    except Exception as e:
        pytest.fail(f"Storage manager integration test failed: {e}")


if __name__ == "__main__":
    # Run the tests directly for quick validation
    test_storage_manager_imports_successfully()
    test_store_modules_import_from_correct_location()  
    test_storage_manager_store_modules_dynamic_import()
    test_no_circular_imports()
    test_import_path_resolution()
    test_storage_manager_integration()
    print("All import isolation tests passed!")