# title: Blender AI LLM 3D Graphics Model Generator
# addon: prompt2blend
# version: 0.1.0
# date: 2025-6-16
# authors: Anthony Chapman
# email: adchap77@gmail.com
# description: This addon is a Blender addon that allows you to generate 3D graphics models using AI models.

import bpy
from bpy.props import StringProperty, EnumProperty
from datetime import datetime
from pathlib import Path
import re
import ast
import random
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from openai import OpenAI
import ollama
from . import rag_agent
from . import prompts

# Use the system prompt from the prompts module
system_prompt = prompts.BLENDER_EXPERT_PROMPT

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
        row = box.row()
        row.label(text="OpenAI API Key", icon='KEYINGSET')
        
        # Show API key status and input
        if scene.ai_openai_key:
            row = box.row(align=True)
            row.label(text="Status: ", icon='CHECKMARK')
            row.label(text="API Key is configured")
            row.operator("aimodel.manage_api_key", text="", icon='X', emboss=False).action = 'CLEAR'
        else:
            row = box.row(align=True)
            row.label(text="Status: ", icon='ERROR')
            row.label(text="API Key not configured")
            
        # Always show the set key button, but disable it if key is already set
        row = box.row()
        op = row.operator("aimodel.manage_api_key", 
                         text="Change Key" if scene.ai_openai_key else "Set API Key", 
                         icon='KEYINGSET')
        op.action = 'SET'
        op.api_key = scene.ai_openai_key if scene.ai_openai_key else ""
        op = row.operator("aimodel.manage_api_key", 
                         text="Clear" if scene.ai_openai_key else "", 
                         icon='X',
                         emboss=bool(scene.ai_openai_key))
        op.action = 'CLEAR'
        
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
    if not hasattr(bpy.context, 'scene') or not hasattr(bpy.context.scene, 'ai_openai_key'):
        raise Exception("Blender scene or API key property not properly initialized.")
        
    api_key = (bpy.context.scene.ai_openai_key or '').strip()
    
    if not api_key:
        raise Exception("OpenAI API key is empty. Please set your API key in the addon settings.")
        
    if not api_key.startswith('sk-'):
        raise Exception("Invalid OpenAI API key format. It should start with 'sk-'.")
    
    try:
        # Initialize the client with the API key
        client = OpenAI(api_key=api_key)
        
        # Test the client with a simple request to validate the key
        client.models.list()
        
        return client
    except Exception as e:
        error_msg = str(e).lower()
        if "invalid api key" in error_msg or "401" in error_msg:
            raise Exception("The provided OpenAI API key is invalid or unauthorized. Please check your key and try again.")
        elif "rate limit" in error_msg:
            raise Exception("OpenAI API rate limit exceeded. Please try again later.")
        else:
            raise Exception(f"Failed to initialize OpenAI client: {str(e)}")

def call_openai(prompt):
    """Call OpenAI API with the provided prompt"""
    client = get_openai_client()
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
    """Call Ollama API with the provided prompt"""
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

def generate_3d_model(model, prompt, max_retries=2):
    """
    Main function to generate and execute 3D model code using the selected model/provider.
    Will attempt to fix errors in the generated code by asking the AI to correct them.
    
    Args:
        model: The AI model to use
        prompt: The user's prompt
        max_retries: Maximum number of attempts to fix code errors (default: 2)
    """
    # Determine provider
    if model == 'chatgpt':
        provider = 'openai'
        model_name = None
        openai_key = bpy.context.scene.ai_openai_key
    else:
        provider = 'ollama'
        model_name = model
        openai_key = None
    
    # Track attempt number for logging
    attempt = 1
    last_error = None
    original_prompt = prompt
    
    while attempt <= max_retries:
        bpy.context.scene.ai_last_status = f"Generating code (attempt {attempt}/{max_retries})..."
        print(f"Attempt {attempt}/{max_retries} to generate code")
        
        # Use RAG agent for retrieval and code generation
        retriever = get_rag_retriever()
        try:
            if retriever:
                ai_response = rag_agent.query_with_rag(prompt, retriever, provider=provider, model=model_name, openai_key=openai_key)
            else:
                # Fallback to direct call if retriever not available
                if provider == 'openai':
                    ai_response = call_openai(prompt)
                else:
                    ai_response = call_ollama(model, prompt)
                
            if not ai_response:
                raise Exception("No response received from AI model")
                
            print(f"Response preview: {ai_response[:200]}...")
            
            # Extract and validate code
            code = extract_python_code(ai_response)
            if not code:
                raise Exception("No Python code found in the response")
            
            validate_code_safety(code)
            
            # Execute the code
            # Create safe execution environment
            safe_globals = {
                'bpy': bpy,
                'math': math,
                'random': random,
                'np': np,
            }

            print(f"Executing code:\n{code}")
            exec(code, safe_globals)
            print("Code executed successfully!")
            return  # Success - exit the function
            
        except Exception as e:
            last_error = str(e)
            print(f"Error on attempt {attempt}: {last_error}")
            
            if attempt < max_retries:
                # Create error correction prompt
                error_correction_prompt = f"""
The previous code generated an error. Please fix the code and ensure it runs correctly.

Original request: {original_prompt}

Error message: {last_error}

Previous code:
```python
{code}
```

Please provide a corrected version of the code that fixes this error and fulfills the original request.
Make sure the code is complete and executable.
"""
                # Update prompt for next attempt
                prompt = error_correction_prompt
                attempt += 1
            else:
                # Last attempt failed, save error log
                log_path = save_error_log(original_prompt, model, code, last_error)
                if log_path:
                    raise Exception(f"Code execution failed after {max_retries} attempts. Error log saved to: {log_path}\nFinal error: {last_error}")
                else:
                    raise Exception(f"Code execution failed after {max_retries} attempts: {last_error}")

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
    bl_options = {'REGISTER'}
    
    action: bpy.props.EnumProperty(
        name="Action",
        items=[
            ('SET', "Set Key", "Set a new API key"),
            ('CLEAR', "Clear Key", "Remove the saved API key")
        ],
        default='SET'
    ) # type: ignore
    
    api_key: bpy.props.StringProperty(
        name="API Key",
        description="Your OpenAI API key (starts with 'sk-')",
        default="",
        subtype='PASSWORD',
        options={'SKIP_SAVE'}
    ) # type: ignore
    
    def invoke(self, context, event):
        if self.action == 'SET':
            # If we're setting a key, show the popup
            self.api_key = context.scene.ai_openai_key  # Pre-fill with current key if exists
            return context.window_manager.invoke_props_dialog(self, width=400)
        else:
            # For clear action, just execute directly
            return self.execute(context)
    
    def draw(self, context):
        layout = self.layout
        if self.action == 'SET':
            row = layout.row()
            row.prop(self, "api_key", text="")
            layout.label(text="Your API key will be stored in Blender's configuration.", icon='INFO')
    
    def execute(self, context):
        if self.action == 'SET':
            if not self.api_key or not self.api_key.startswith('sk-'):
                self.report({'ERROR'}, "Please enter a valid OpenAI API key (starts with 'sk-')")
                return {'CANCELLED'}
            context.scene.ai_openai_key = self.api_key
            self.report({'INFO'}, "API key saved")
            context.scene.ai_last_status = "API key saved successfully"
            # Update model list when key is set
            update_model_list(context)
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

# Store classes to unregister
def get_classes():
    return [
        AIMODEL_PT_MainPanel,
        AIMODEL_PT_StatusPanel,
        AIMODEL_OT_SubmitPrompt,
        AIMODEL_OT_RefreshModels,
        AIMODEL_OT_ManageAPIKey,
        AIMODEL_OT_OpenLog,
    ]

def register():
    # First, register the properties to ensure they exist when panels need them
    try:
        register_properties()
        print("✓ Properties registered successfully")
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
    """Get the RAG retriever instance"""
    try:
        # If we're running as a module (installed addon)
        if hasattr(rag_agent, 'get_retriever'):
            return rag_agent.get_retriever()
    except Exception as e:
        print(f"Error initializing RAG retriever: {e}")
        return None

