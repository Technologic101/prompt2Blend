# Blender add-on metadata
bl_info = {
    "name": "Prompt2Blend: AI-Powered 3D Model Generator",
    "author": "Anthony Chapman",
    "version": (0, 1, 0),  # Initial release version
    "blender": (4, 4, 1),
    "location": "View3D > Sidebar > Gen AI 3D Graphics Model",
    "description": "Generate 3D models using AI models (OpenAI GPT and Ollama)",
    "warning": "Requires OpenAI API key and/or Ollama installation",
    "doc_url": "https://github.com/Technologic101/prompt2blend",
    "tracker_url": "https://github.com/Technologic101/prompt2blend/issues",
    "category": "3D View",
}

def register():
    """Register all operators and panels"""
    # First unregister everything to handle reloads
    try:
        unregister()
        print("Previous registration cleared")
    except:
        print("No previous registration to clear")
    
    # Import the module
    from . import blender_llm_addin
    
    # Register the module
    try:
        print("Starting registration of Prompt2Blend add-on...")
        result = blender_llm_addin.register()
        print("✓ Prompt2Blend add-on registered successfully!")
        print("The panel should appear in the 3D Viewport sidebar under the 'Gen AI 3D' tab")
        print("Look for the tab in the right sidebar (press N if sidebar is hidden)")
        return result
    except Exception as e:
        print(f"✗ Error registering Prompt2Blend add-on: {e}")
        import traceback
        traceback.print_exc()
        return {'CANCELLED'}

def unregister():
    # Check if the module is loaded
    if 'blender_llm_addin' in globals():
        try:
            import importlib
            blender_llm_addin = importlib.import_module('.blender_llm_addin', package=__package__)
            if hasattr(blender_llm_addin, 'unregister'):
                blender_llm_addin.unregister()
            print("Prompt2Blend add-on unregistered successfully!")
        except Exception as e:
            print(f"Error during unregistration: {e}")
            import traceback
            traceback.print_exc()
    
    # Clear any remaining references
    if 'blender_llm_addin' in globals():
        del globals()['blender_llm_addin']
    
    # Force garbage collection
    import gc
    gc.collect()
    
    return {'FINISHED'}

# This allows you to run the script directly from Blender's text editor
if __name__ == "__main__":
    register()
