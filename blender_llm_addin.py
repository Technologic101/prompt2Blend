# title: Blender AI LLM Addon for Text to 3D Graphics Model
# version: 1.1.0
# date: 2025-6-7
# authors: Taewook Kang, Anthony Chapman
# email: laputa99999@gmail.com
# description: This addon is a Blender addon that allows you to generate 3D graphics models using AI models.

bl_info = {
    "name": "AI LLM 3D Graphics Model Generator",
    "author": "Taewook Kang, Anthony Chapman",
    "version": (1, 1, 0),
    "blender": (4, 4, 1),
    "location": "View3D > Sidebar > Gen AI 3D Graphics Model",
    "description": "Generate Blender Python code using AI models",
    "warning": "Requires OpenAI API key and/or Ollama installation",
    "doc_url": "",
    "category": "Object",
}

import bpy
import sys
import os

# Debug: Print to console that the addon is loading
print("=" * 50)
print("AI LLM Addon: Starting to load...")
print(f"Blender version: {bpy.app.version}")
print(f"Python version: {sys.version}")
print("=" * 50)

# Required dependencies
REQUIRED_PACKAGES = ['pandas', 'numpy', 'openai', 'ollama']

# Check for required dependencies with detailed feedback
DEPENDENCIES_STATUS = {}

def get_blender_python_path():
    """Get the path to Blender's Python executable"""
    return sys.executable

def check_dependency(module_name):
    """Check if a dependency is available and provide installation instructions if missing"""
    try:
        __import__(module_name)
        DEPENDENCIES_STATUS[module_name] = True
        print(f"✓ {module_name} - Available")
        return True
    except ImportError:
        DEPENDENCIES_STATUS[module_name] = False
        print(f"✗ {module_name} - Not available")
        
        # Get Blender's Python path
        blender_python = get_blender_python_path()
        
        # Provide clear installation instructions
        print("\nTo install missing packages:")
        print("1. Open Terminal/Command Prompt")
        print(f"2. Run this command: {blender_python} -m pip install {module_name}")
        print("\nIf you get a permission error:")
        print("On Windows: Run Command Prompt as Administrator")
        print("On Mac/Linux: Add 'sudo' before the command")
        return False

# Check all dependencies
print("\nChecking required packages...")
for package in REQUIRED_PACKAGES:
    check_dependency(package)

# Import available packages
if DEPENDENCIES_STATUS.get('pandas', False):
    import pandas as pd
if DEPENDENCIES_STATUS.get('numpy', False):
    import numpy as np
if DEPENDENCIES_STATUS.get('openai', False):
    from openai import OpenAI
if DEPENDENCIES_STATUS.get('ollama', False):
    from ollama import chat, ChatResponse

# Standard library imports
import json
import re
import textwrap
import ast
import random
import math

print("All imports completed.")

# Create Blender UI Panel
class AIMODEL_PT_MainPanel(bpy.types.Panel):
    bl_label = "AI Model Selector"
    bl_idname = "AIMODEL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gen AI 3D"  # Shorter category name
    bl_context = "objectmode"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Header
        layout.label(text="AI 3D Model Generator", icon='MESH_CUBE')
        layout.separator()
        
        # Model selection with refresh button
        box = layout.box()
        row = box.row()
        row.label(text="Select AI Model:")
        row.operator("aimodel.refresh_models", text="", icon='FILE_REFRESH')
        box.prop(scene, "ai_model_selection", text="")
        
        # Show dependency status
        selected_model = scene.ai_model_selection
        if selected_model == 'chatgpt' and not DEPENDENCIES_STATUS.get('openai', False):
            box.label(text="⚠ OpenAI library missing", icon='ERROR')
        elif selected_model != 'chatgpt' and not DEPENDENCIES_STATUS.get('ollama', False):
            box.label(text="⚠ Ollama library missing", icon='ERROR')
        
        # User prompt
        box = layout.box()
        box.label(text="Your Request:")
        box.prop(scene, "ai_user_prompt", text="")
        
        # Submit button
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("aimodel.submit_prompt", text="Generate 3D Model", icon='PLAY')
        
        # Debug info
        if scene.show_debug_info:
            layout.separator()
            debug_box = layout.box()
            debug_box.label(text="Debug Info:", icon='INFO')
            debug_box.label(text=f"OpenAI: {'✓' if DEPENDENCIES_STATUS.get('openai', False) else '✗'}")
            debug_box.label(text=f"Ollama: {'✓' if DEPENDENCIES_STATUS.get('ollama', False) else '✗'}")
            debug_box.label(text=f"Selected: {selected_model}")
        
        # Toggle debug info
        layout.prop(scene, "show_debug_info", text="Show Debug Info")

# Submit operator
class AIMODEL_OT_SubmitPrompt(bpy.types.Operator):
    bl_label = "Submit AI Prompt"
    bl_idname = "aimodel.submit_prompt"
    bl_description = "Generate 3D model using AI"
    
    def execute(self, context):
        print("Submit operator called!")
        
        scene = context.scene
        model = scene.ai_model_selection
        prompt = scene.ai_user_prompt
        
        print(f"Model: {model}")
        print(f"Prompt: {prompt}")
        
        # Validate input
        if not prompt.strip():
            self.report({'ERROR'}, "Please enter a prompt")
            return {'CANCELLED'}
        
        try:
            self.report({'INFO'}, f"Generating with {model}...")
            generate_3d_model(model, prompt)
            self.report({'INFO'}, "Code generation completed!")
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            print(f"Exception in execute: {error_msg}")
            self.report({'ERROR'}, error_msg)
            return {'CANCELLED'}
            
        return {'FINISHED'}

# Core AI functions
def get_openai_client():
    """Get OpenAI client with API key"""
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        raise Exception("Set OPENAI_API_KEY environment variable")
    return OpenAI(api_key=api_key)

def call_openai(prompt):
    """Call OpenAI API"""
    if not DEPENDENCIES_STATUS.get('openai', False):
        raise Exception("OpenAI library not available")
    
    client = get_openai_client()
    system_prompt = "You are a Blender Python expert. Generate clean, safe bpy code with comments. Only return Python code in ```python code blocks."
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Using mini for faster/cheaper responses
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Create Blender Python code for: {prompt}"}
        ],
        temperature=0.1,
        max_tokens=512
    )
    return response.choices[0].message.content

def call_ollama(model, prompt):
    """Call Ollama API"""
    if not DEPENDENCIES_STATUS.get('ollama', False):
        raise Exception("Ollama library not available")
    
    full_prompt = f"Create Blender Python code using bpy for: {prompt}. Return only code with comments."
    
    response = chat(
        model=model,
        messages=[{"role": "user", "content": full_prompt}]
    )
    
    if response and hasattr(response, 'message'):
        return response.message.content
    else:
        raise Exception("Invalid response from Ollama")

def extract_python_code(text):
    """Extract Python code from AI response"""
    # Look for code blocks
    pattern = r'```python\n(.*?)```'
    match = re.search(pattern, text, re.DOTALL)
    
    if match:
        code = match.group(1).strip()
    else:
        # Fallback: look for lines starting with bpy
        lines = text.split('\n')
        code_lines = [line for line in lines if line.strip().startswith(('bpy.', 'import bpy'))]
        code = '\n'.join(code_lines)
    
    if not code.strip():
        # Last resort: return the whole text if it looks like code
        if 'bpy.' in text:
            code = text
        else:
            raise Exception("No Python code found in AI response")
    
    return code.strip()

def validate_code_safety(code):
    """Basic safety validation"""
    dangerous_patterns = [
        'import os', 'import sys', 'import subprocess', 'exec(', 'eval(',
        'open(', '__import__', 'getattr', 'setattr'
    ]
    
    for pattern in dangerous_patterns:
        if pattern in code:
            raise Exception(f"Potentially unsafe code detected: {pattern}")
    
    # Try to parse the code
    try:
        ast.parse(code)
    except SyntaxError as e:
        raise Exception(f"Syntax error in generated code: {e}")
    
    return True

def generate_3d_model(model, prompt):
    """Main function to generate and execute 3D model code"""
    print(f"Generating 3D model with {model}: {prompt}")
    
    # Get AI response
    if model == 'chatgpt':
        ai_response = call_openai(prompt)
    else:
        ai_response = call_ollama(model, prompt)
    
    print(f"AI Response received: {len(ai_response)} characters")
    print(f"Response preview: {ai_response[:200]}...")
    
    # Extract and validate code
    code = extract_python_code(ai_response)
    print(f"Extracted code:\n{code}")
    
    validate_code_safety(code)
    
    # Execute the code
    try:
        # Create safe execution environment
        safe_globals = {
            'bpy': bpy,
            'math': math,
            'random': random,
        }
        exec(code, safe_globals)
        print("Code executed successfully!")
    except Exception as e:
        raise Exception(f"Code execution failed: {e}")

def get_ollama_models():
    """Fetch available models from Ollama"""
    try:
        if not DEPENDENCIES_STATUS.get('ollama', False):
            return []
        
        import requests
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [(model['name'], model['name'], f"Ollama {model['name']} model") 
                   for model in models]
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

# Properties
def register_properties():
    """Register addon properties"""
    print("Registering properties...")
    
    # Get available Ollama models
    ollama_models = get_ollama_models()
    
    # Base model items
    model_items = [
        ('chatgpt', "ChatGPT", "OpenAI ChatGPT (requires API key)"),
    ]
    
    # Add Ollama models if available
    model_items.extend(ollama_models)
    
    bpy.types.Scene.ai_model_selection = bpy.props.EnumProperty(
        name="AI Model",
        description="Select the AI model to use",
        items=model_items,
        default='chatgpt'
    )
    
    bpy.types.Scene.ai_user_prompt = bpy.props.StringProperty(
        name="User Prompt",
        description="Describe the 3D model you want to create",
        default="Create a red cube at position (2, 0, 1)",
        maxlen=2048
    )
    
    bpy.types.Scene.show_debug_info = bpy.props.BoolProperty(
        name="Debug Info",
        description="Show debug information",
        default=False
    )
    
    print("Properties registered successfully!")

def unregister_properties():
    """Unregister addon properties"""
    print("Unregistering properties...")
    try:
        del bpy.types.Scene.ai_model_selection
        del bpy.types.Scene.ai_user_prompt
        del bpy.types.Scene.show_debug_info
        print("Properties unregistered successfully!")
    except:
        print("Error unregistering properties (they may not exist)")

# Add refresh operator
class AIMODEL_OT_RefreshModels(bpy.types.Operator):
    bl_label = "Refresh Models"
    bl_idname = "aimodel.refresh_models"
    bl_description = "Refresh available Ollama models"
    
    def execute(self, context):
        # Get new model list
        ollama_models = get_ollama_models()
        
        # Update the enum property
        model_items = [
            ('chatgpt', "ChatGPT", "OpenAI ChatGPT (requires API key)"),
        ]
        model_items.extend(ollama_models)
        
        # Update the property
        bpy.types.Scene.ai_model_selection = bpy.props.EnumProperty(
            name="AI Model",
            description="Select the AI model to use",
            items=model_items,
            default='chatgpt'
        )
        
        self.report({'INFO'}, f"Found {len(ollama_models)} Ollama models")
        return {'FINISHED'}

# Registration
classes = (
    AIMODEL_PT_MainPanel,
    AIMODEL_OT_SubmitPrompt,
    AIMODEL_OT_RefreshModels,
)

def register():
    print("Starting addon registration...")
    
    # Register classes
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            print(f"✓ Registered class: {cls.__name__}")
        except Exception as e:
            print(f"✗ Failed to register class {cls.__name__}: {e}")
            raise
    
    # Register properties
    register_properties()
    
    print("🎉 AI LLM Addon registered successfully!")
    print("Look for 'Gen AI 3D' tab in the 3D Viewport sidebar (press N)")

def unregister():
    print("Starting addon unregistration...")
    
    # Unregister properties
    unregister_properties()
    
    # Unregister classes (in reverse order)
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
            print(f"✓ Unregistered class: {cls.__name__}")
        except Exception as e:
            print(f"✗ Failed to unregister class {cls.__name__}: {e}")
    
    print("AI LLM Addon unregistered successfully!")

# Test the addon when run directly
if __name__ == "__main__":
    print("Running addon directly for testing...")
    try:
        # Try to unregister first (in case it's already loaded)
        try:
            unregister()
        except:
            pass
        
        # Register the addon
        register()
        print("✅ Direct test successful!")
        
    except Exception as e:
        print(f"❌ Direct test failed: {e}")
        import traceback
        traceback.print_exc()