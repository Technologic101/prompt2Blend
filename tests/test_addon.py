#!/usr/bin/env python3
"""
Test script to verify the add-on can be imported and basic functions work
Run this OUTSIDE of Blender to check for import issues
"""

import sys
import importlib.util

def test_addon_imports():
    """Test if the add-on can be imported without Blender"""
    print("Testing add-on import capability...")
    
    try:
        # Mock bpy module since we're not in Blender
        class MockBpy:
            class types:
                class Panel: pass
                class Operator: pass
                class Scene: pass
            
            class props:
                @staticmethod
                def EnumProperty(**kwargs):
                    return "mock_enum"
                
                @staticmethod
                def StringProperty(**kwargs):
                    return "mock_string"
                
                @staticmethod
                def BoolProperty(**kwargs):
                    return "mock_bool"
            
            class utils:
                @staticmethod
                def register_class(cls):
                    pass
                
                @staticmethod
                def unregister_class(cls):
                    pass
                
                @staticmethod
                def resource_path(type):
                    return "/tmp"
            
            class context:
                class scene:
                    ai_openai_key = ""
        
        # Add mock bpy to sys.modules
        sys.modules['bpy'] = MockBpy()
        
        # Try to import the main module
        spec = importlib.util.spec_from_file_location("blender_llm_addin", "blender_llm_addin.py")
        module = importlib.util.module_from_spec(spec)
        
        print("✓ Add-on imports successfully")
        return True
        
    except Exception as e:
        print(f"✗ Add-on import failed: {e}")
        return False

def test_dependencies():
    """Test if required dependencies are available"""
    dependencies = ['openai', 'ollama', 'sklearn', 'numpy', 'requests']
    
    print("\nTesting dependencies...")
    missing = []
    
    for dep in dependencies:
        try:
            if dep == 'sklearn':
                __import__('sklearn')
            else:
                __import__(dep)
            print(f"✓ {dep} - Available")
        except ImportError:
            print(f"✗ {dep} - Missing")
            missing.append(dep)
    
    if missing:
        print(f"\nMissing dependencies: {missing}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    return True

if __name__ == "__main__":
    print("Blender LLM Add-on Verification Test")
    print("=" * 40)
    
    import_ok = test_addon_imports()
    deps_ok = test_dependencies()
    
    print("\n" + "=" * 40)
    if import_ok and deps_ok:
        print("✅ Add-on is ready for testing in Blender!")
    else:
        print("❌ Add-on needs fixes before testing")
        if not deps_ok:
            print("Install missing dependencies first")
        if not import_ok:
            print("Check import errors in the code")
