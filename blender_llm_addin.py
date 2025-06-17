# title: Blender AI LLM 3D Graphics Model Generator
# addon: prompt2blend
# version: 0.1.0
# date: 2025-6-16
# authors: Anthony Chapman
# email: adchap77@gmail.com
# description: This addon is a Blender addon that allows you to generate 3D graphics models using AI models.

# Import add-on metadata from __init__.py (single source of truth)
# bl_info is defined only in __init__.py

system_prompt = """
   You are an expert Blender Python developer. Write a complete, ready-to-run Python script using Blender's bpy API that matches the user's request with the following guidelines:

The script should not clear the scene unless explicitly instructed.

Only use version 4.4 of the Blender API.

The output should be a single Python script, formatted for direct execution in Blender's scripting workspace.

"""


import bpy
from datetime import datetime
from pathlib import Path
import json
import re
import ast
import random
import math
from typing import List
from pathlib import Path
import importlib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from openai import OpenAI
import ollama
try:
    from . import rag_agent
except ImportError:
    # Fallback for direct execution
    import rag_agent

# Create Blender UI Panel
class AIMODEL_PT_MainPanel(bpy.types.Panel):
    bl_label = "AI Model Selector"
    bl_idname = "AIMODEL_PT_main_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gen AI 3D"

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Header
        layout.label(text="AI 3D Model Generator", icon='MESH_CUBE')
        layout.separator()
        
        # Verify properties are registered
        if not hasattr(scene, "ai_user_prompt"):
            layout.label(text="Properties not registered correctly", icon='ERROR')
            layout.operator("aimodel.refresh_models", text="Try to fix", icon='FILE_REFRESH')
            return
        
        # API Key Management
        box = layout.box()
        box.label(text="OpenAI API Key:", icon='KEY')
        
        # Show API key status
        if scene.ai_openai_key:
            row = box.row()
            row.label(text="✓ API Key is configured", icon='CHECKMARK')
            row.operator("aimodel.manage_api_key", text="", icon='X').action = 'CLEAR'
        else:
            row = box.row()
            row.label(text="⚠ API Key not configured", icon='ERROR')
            op = row.operator("aimodel.manage_api_key", text="Set API Key", icon='KEYINGSET')
            op.action = 'SET'
            op.api_key = ""  # Clear any previous value
        
        # Model selection with refresh button
        box = layout.box()
        row = box.row()
        row.label(text="Select AI Model:")
        row.operator("aimodel.refresh_models", text="", icon='FILE_REFRESH')
        
        # Display the model selection dropdown
        box.prop(scene, "ai_model_selection", text="")
        
        # User prompt
        box = layout.box()
        box.label(text="Your Request:")
        box.prop(scene, "ai_user_prompt", text="")
        
        # Submit button
        layout.separator()
        row = layout.row()
        row.scale_y = 1.5
        row.operator("aimodel.submit_prompt", text="Generate 3D Model", icon='PLAY')
        

# Add a new status panel
class AIMODEL_PT_StatusPanel(bpy.types.Panel):
    bl_label = "Status & Messages"
    bl_idname = "AIMODEL_PT_status_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gen AI 3D"
    # Removed bl_context restriction to make panel visible in all modes
    bl_options = {'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls, context):
        return True

    def draw(self, context):
        layout = self.layout
        scene = context.scene
        
        # Show last operation status
        if hasattr(scene, "ai_last_status"):
            box = layout.box()
            box.label(text="Last Operation:", icon='INFO')
            box.label(text=scene.ai_last_status)
            
            # If the status contains a log file path, add a button to open it
            if "Error log saved to:" in scene.ai_last_status:
                row = box.row()
                row.operator("aimodel.open_log", text="Open Error Log", icon='TEXT')

# Submit operator
class AIMODEL_OT_SubmitPrompt(bpy.types.Operator):
    bl_label = "Submit AI Prompt"
    bl_idname = "aimodel.submit_prompt"
    bl_description = "Generate 3D model using AI"
    
    def execute(self, context):
        scene = context.scene
        model = scene.ai_model_selection
        prompt = scene.ai_user_prompt
        
        # Update status
        scene.ai_last_status = f"Starting generation with {model}..."
        self.report({'INFO'}, scene.ai_last_status)
        
        # Validate input
        if not prompt.strip():
            scene.ai_last_status = "Error: Please enter a prompt"
            self.report({'ERROR'}, scene.ai_last_status)
            return {'CANCELLED'}
        
        try:
            generate_3d_model(model, prompt)
            scene.ai_last_status = "Code generation completed successfully!"
            self.report({'INFO'}, scene.ai_last_status)
        except Exception as e:
            scene.ai_last_status = f"Error: {str(e)}"
            self.report({'ERROR'}, scene.ai_last_status)
            return {'CANCELLED'}
            
        return {'FINISHED'}

# Core AI functions
def get_openai_client():
    """Get OpenAI client with API key"""
    api_key = bpy.context.scene.ai_openai_key
    if not api_key:
        raise Exception("OpenAI API key not configured. Please set your API key in the addon settings.")
    return OpenAI(api_key=api_key)

def call_openai(prompt):
    """Call OpenAI API with RAG if available, using shared RAG logic"""
    client = get_openai_client()
    retriever = get_rag_retriever()
    if retriever:
        try:
            # Use shared RAG query logic
            return rag_agent.query_with_rag(
                prompt, retriever, provider='openai', model='gpt-4', openai_key=client.api_key
            )
        except Exception as e:
            print(f"RAG failed, falling back to direct prompt: {e}")
    # Fallback: direct prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model="gpt-4",
        messages=messages,  # type: ignore
        temperature=0.1,
        max_tokens=1512
    )
    return response.choices[0].message.content

def call_ollama(model, prompt):
    """Call Ollama API with RAG if available, using shared RAG logic"""
    retriever = get_rag_retriever()
    if retriever:
        try:
            # Use shared RAG query logic
            return rag_agent.query_with_rag(
                prompt, retriever, provider='ollama', model=model
            )
        except Exception as e:
            print(f"RAG failed, falling back to direct prompt: {e}")
    # Fallback: direct prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    response = ollama.chat(
        model=model,
        messages=messages  # type: ignore
    )
    try:
        if isinstance(response, dict) and 'message' in response:
            return response['message']['content']
        elif hasattr(response, 'message'):
            return response.message['content']
        else:
            return str(response)
    except Exception as e:
        print(f"Error parsing Ollama response: {e}")
        return str(response)

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

def get_log_path():
    """Get the path to the log directory"""
    log_dir = Path(bpy.utils.resource_path('USER')) / "scripts" / "addons" / "ai_llm_addon" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir

def save_error_log(prompt, model, generated_code, error):
    """Save error information to a log file"""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = get_log_path() / f"error_log_{timestamp}.txt"
        
        with open(log_file, 'w') as f:
            f.write("=" * 50 + "\n")
            f.write(f"Error Log - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Model: " + model + "\n")
            f.write("Prompt: " + prompt + "\n\n")
            
            f.write("Generated Code:\n")
            f.write("-" * 50 + "\n")
            f.write(generated_code + "\n\n")
            
            f.write("Error:\n")
            f.write("-" * 50 + "\n")
            f.write(str(error) + "\n")
        
        return str(log_file)
    except Exception as e:
        print(f"Failed to save error log: {e}")
        return None

def generate_3d_model(model, prompt):
    """Main function to generate and execute 3D model code using the selected model/provider"""
    # Determine provider
    if model == 'chatgpt':
        provider = 'openai'
        model_name = None
        openai_key = bpy.context.scene.ai_openai_key
    else:
        provider = 'ollama'
        model_name = model
        openai_key = None
    # Use RAG agent for retrieval and code generation
    retriever = get_rag_retriever()
    if retriever:
        ai_response = rag_agent.query_with_rag(prompt, retriever, provider=provider, model=model_name, openai_key=openai_key)
    else:
        # fallback to direct call
        if provider == 'openai':
            ai_response = call_openai(prompt)
        else:
            ai_response = call_ollama(model, prompt)
    if ai_response:
        print(f"Response preview: {ai_response[:200]}...")
    else:
        raise Exception("No response received from AI model")
    
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
        # Save error log
        log_path = save_error_log(prompt, model, code, e)
        if log_path:
            raise Exception(f"Code execution failed. Error log saved to: {log_path}\nError: {str(e)}")
        else:
            raise Exception(f"Code execution failed: {str(e)}")

def get_ollama_models():
    """Fetch available models from Ollama"""
    try:
        response = requests.get('http://localhost:11434/api/tags')
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [(model['name'], model['name'], f"Ollama {model['name']} model") 
                   for model in models]
        return []
    except Exception as e:
        print(f"Error fetching Ollama models: {e}")
        return []

def get_available_models(scene=None):
    """Return a list of (id, label, description) for available models only."""
    items = []
    
    # If scene is not provided or doesn't have the necessary attributes
    if scene is None:
        return items
        
    # OpenAI available if API key is set
    if hasattr(scene, 'ai_openai_key') and scene.ai_openai_key:
        items.append(('chatgpt', "ChatGPT", "OpenAI ChatGPT (requires API key)"))
        
    # Add Ollama models if available
    ollama_models = get_ollama_models()
    items.extend(ollama_models)
    return items

def get_model_items(self, context):
    """Get the current list of available models"""
    if not hasattr(context, 'scene'):
        return DEFAULT_MODEL_ITEMS
    
    models = get_available_models(context.scene)
    return models if models else DEFAULT_MODEL_ITEMS

# Default model items when no API key is set
DEFAULT_MODEL_ITEMS = [
    ('none', 'No models available', 'Please set up your API key first')
]

def update_model_list(context):
    """Update the model list when API key changes"""
    # During registration, context might not have scene attribute
    if not hasattr(context, 'scene'):
        # Use default models when scene is not available
        models = DEFAULT_MODEL_ITEMS
    else:
        models = get_available_models(context.scene)
        if not models:
            models = DEFAULT_MODEL_ITEMS
    
    # Update the property with new items
    bpy.types.Scene.ai_model_selection = bpy.props.EnumProperty(
        name="AI Model",
        description="Select the AI model to use",
        items=models,
        default=models[0][0] if models else 'none'
    )
    
    # Force UI update
    if context.screen:
        for area in context.screen.areas:
            if area.type == 'PROPERTIES' or area.type == 'VIEW_3D':
                area.tag_redraw()

# Properties
def register_properties():
    """Register addon properties"""
    print("Registering properties...")
    
    # Register properties that don't depend on others first
    bpy.types.Scene.ai_user_prompt = bpy.props.StringProperty(
        name="User Prompt",
        description="Describe the 3D model you want to create",
        default="Create a red cube at position (2, 0, 1)",
        maxlen=2048
    )
    
    bpy.types.Scene.ai_openai_key = bpy.props.StringProperty(
        name="OpenAI API Key",
        description="Your OpenAI API key",
        default="",
        subtype='PASSWORD',
        update=lambda self, context: update_model_list(context)
    )
    
    bpy.types.Scene.ai_last_status = bpy.props.StringProperty(
        name="Last Status",
        description="Last operation status",
        default="Ready"
    )
    
    bpy.types.Scene.show_debug_info = bpy.props.BoolProperty(
        name="Debug Info",
        description="Show debug information",
        default=False
    )
    
    # Register model selection last, after other properties are registered
    try:
        update_model_list(bpy.context)
    except Exception as e:
        print(f"Warning: Could not update model list during registration: {str(e)}")
        # Set up a default enum property
        bpy.types.Scene.ai_model_selection = bpy.props.EnumProperty(
            name="AI Model",
            description="Select the AI model to use",
            items=DEFAULT_MODEL_ITEMS,
            default=DEFAULT_MODEL_ITEMS[0][0]
        )
    
    print("Properties registered successfully!")

def unregister_properties():
    """Unregister addon properties"""
    print("Unregistering properties...")
    try:
        del bpy.types.Scene.ai_model_selection
        del bpy.types.Scene.ai_user_prompt
        del bpy.types.Scene.ai_openai_key
        del bpy.types.Scene.ai_last_status
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
        # Force refresh by clearing the Ollama models cache
        try:
            # This will force a fresh fetch of Ollama models
            ollama_models = get_ollama_models()
            # Update the model list
            update_model_list(context)
            
            # Update status
            context.scene.ai_last_status = f"Refreshed models: Found {len(ollama_models)} Ollama models"
            self.report({'INFO'}, context.scene.ai_last_status)
        except Exception as e:
            context.scene.ai_last_status = f"Error refreshing models: {str(e)}"
            self.report({'ERROR'}, context.scene.ai_last_status)
            
        return {'FINISHED'}

# Add API key management operator
class AIMODEL_OT_ManageAPIKey(bpy.types.Operator):
    bl_label = "Manage API Key"
    bl_idname = "aimodel.manage_api_key"
    bl_description = "Set or clear your OpenAI API key"
    
    action = bpy.props.EnumProperty(
        name="Action",
        items=[
            ('SET', "Set Key", "Set a new API key"),
            ('CLEAR', "Clear Key", "Remove the saved API key")
        ],
        default='SET'
    )
    
    api_key = bpy.props.StringProperty(
        name="API Key",
        description="Your OpenAI API key",
        default="",
        subtype='PASSWORD'
    )
    
    def execute(self, context):
        if self.action == 'SET':
            if not self.api_key:
                self.report({'ERROR'}, "Please enter an API key")
                return {'CANCELLED'}
            context.scene.ai_openai_key = self.api_key
            self.report({'INFO'}, "API key saved")
            context.scene.ai_last_status = "API key saved successfully"
        else:  # CLEAR
            context.scene.ai_openai_key = ""
            self.report({'INFO'}, "API key cleared")
            context.scene.ai_last_status = "API key cleared"
        return {'FINISHED'}

# Add operator to open log files
class AIMODEL_OT_OpenLog(bpy.types.Operator):
    bl_label = "Open Error Log"
    bl_idname = "aimodel.open_log"
    bl_description = "Open the error log file in Blender's text editor"
    
    def execute(self, context):
        # Extract log path from status message
        status = context.scene.ai_last_status
        if "Error log saved to:" in status:
            log_path = status.split("Error log saved to:")[1].split("\n")[0].strip()
            
            # Open the file in Blender's text editor
            try:
                bpy.ops.text.open(filepath=log_path)
                self.report({'INFO'}, f"Opened log file: {log_path}")
            except Exception as e:
                self.report({'ERROR'}, f"Failed to open log file: {e}")
        
        return {'FINISHED'}

# Add panel verification operator
class AIMODEL_OT_VerifyPanel(bpy.types.Operator):
    bl_idname = "aimodel.verify_panel"
    bl_label = "Debug Panels"
    bl_description = "Verify if panels are registered and check their visibility status"
    bl_options = {'INTERNAL'}

    def execute(self, context):
        # Check if our panels are registered
        print("\n--- Panel Verification ---")
        
        # Check our own panel registration
        panel_found = False
        
        for panel in bpy.types.Panel.__subclasses__():
            if panel.__name__.startswith("AIMODEL_PT_"):
                panel_found = True
                print(f"Found panel: {panel.__name__} in category '{getattr(panel, 'bl_category', 'None')}'")
                print(f"  - space_type: {getattr(panel, 'bl_space_type', 'None')}")
                print(f"  - region_type: {getattr(panel, 'bl_region_type', 'None')}")
                print(f"  - context: {getattr(panel, 'bl_context', 'None')}")
                
        if not panel_found:
            print("No AIMODEL_PT panels found in registered panel classes!")
        
        print("\nTo make the panel visible:")
        print("1. Press N in the 3D view to show the sidebar")
        print("2. Look for the 'Gen AI 3D' tab in the sidebar")
        print("3. If not visible, try switching modes (Object Mode/Edit Mode)")
        
        self.report({'INFO'}, "Panel verification completed. Check Blender console for details.")
        return {'FINISHED'}


# Store classes to unregister
def get_classes():
    return [
        AIMODEL_PT_MainPanel,
        AIMODEL_PT_StatusPanel,
        AIMODEL_OT_SubmitPrompt,
        AIMODEL_OT_RefreshModels,
        AIMODEL_OT_ManageAPIKey,
        AIMODEL_OT_OpenLog,
        AIMODEL_OT_VerifyPanel,
    ]

def register():
    # First, register the properties to ensure they exist when panels need them
    try:
        print("Registering properties first...")
        register_properties()
        print("✓ Properties registered successfully")
        print("✓ The panel should appear in one of two places:")
        print("  1. In the 'Gen AI 3D' tab in the 3D View sidebar (press N to show)")
        print("  2. Or in the 'Tool' tab (as a fallback)")
    except Exception as e:
        print(f"✗ Error registering properties: {e}")
        import traceback
        traceback.print_exc()
        raise

    classes = get_classes()
    print("Classes to register:", [cls.__name__ for cls in classes])
    print(f"Total classes: {len(classes)}")
    print("Registering UI classes...")
    registered_classes = []
    for cls in classes:
        try:
            bpy.utils.register_class(cls)
            registered_classes.append(cls)
            print(f"✓ Registered class: {cls.__name__}")
        except Exception as e:
            print(f"✗ Failed to register class {cls.__name__}: {e}")
            # Unregister any classes that were registered before the error
            for cls in reversed(registered_classes):
                try:
                    bpy.utils.unregister_class(cls)
                except:
                    pass
            raise
    
    print("AI LLM Addon registered successfully!")
    return {'FINISHED'}

def unregister():
    print("Starting addon unregistration...")
    
    # Unregister properties first
    unregister_properties()
    
    # Unregister classes in reverse order
    for cls in reversed(get_classes()):
        try:
            if hasattr(bpy.types, cls.__name__):
                bpy.utils.unregister_class(cls)
                print(f"✓ Unregistered class: {cls.__name__}")
        except Exception as e:
            print(f"✗ Failed to unregister class {cls.__name__}: {e}")
    
    # Force garbage collection
    import gc
    gc.collect()
    
    print("AI LLM Addon unregistered successfully!")
    return {'FINISHED'}

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

def get_rag_retriever():
    """Get the RAG retriever instance from ChromaDB using rag_agent.py logic."""
    try:
        # If we're running as a module (installed addon)
        if hasattr(rag_agent, 'get_retriever'):
            return rag_agent.get_retriever()
        # If we're running directly
        else:
            # Initialize the retriever with default settings
            from rag_agent import ChromaDBRetriever
            return ChromaDBRetriever()
    except Exception as e:
        print(f"Error initializing RAG retriever: {e}")
        return None

