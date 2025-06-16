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

system_prompt = """
   You are an expert Blender Python developer. Write a complete, ready-to-run Python script using Blender's bpy API that matches the user's request with the following guidelines:

The script should not clear the scene unless explicitly instructed.

Do not include any code outside the Blender Python API 4.4 (bpy).

The output should be a single Python script, formatted for direct execution in Blender's scripting workspace.

"""


import bpy
import sys
import os
from datetime import datetime
from pathlib import Path

# Required dependencies
REQUIRED_PACKAGES = ['openai', 'ollama', 'scikit-learn', 'requests']

# Check for required dependencies with detailed feedback
DEPENDENCIES_STATUS = {}

def get_blender_python_path():
    """Get the path to Blender's Python executable"""
    return sys.executable

def check_dependency(module_name):
    """Check if a dependency is available and provide installation instructions if missing"""
    try:
        if module_name == 'openai':
            return OPENAI_AVAILABLE
        elif module_name == 'ollama':
            return OLLAMA_AVAILABLE
        elif module_name == 'scikit-learn':
            return SKLEARN_AVAILABLE
        else:
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


import json
import re
import textwrap
import ast
import random
import math
from typing import List, Dict
from pathlib import Path
try:
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("scikit-learn not available - RAG features will be disabled")

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI library not available")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama library not available")

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

# Add a new status panel
class AIMODEL_PT_StatusPanel(bpy.types.Panel):
    bl_label = "Status & Messages"
    bl_idname = "AIMODEL_PT_status_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Gen AI 3D"
    bl_context = "objectmode"
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
        
        # Show RAG status
        box = layout.box()
        box.label(text="RAG Status:", icon='FILE_TEXT')
        retriever = get_rag_retriever()
        if retriever:
            box.label(text="✓ RAG enabled with Blender API context", icon='CHECKMARK')
        else:
            box.label(text="⚠ RAG not available", icon='ERROR')
            box.label(text="Add blender_api_embeddings.json to addon directory")

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
    """Call OpenAI API with RAG if available"""
    if not OPENAI_AVAILABLE:
        raise Exception("OpenAI library not available")
    
    client = get_openai_client()
    
    # Try to use RAG if available
    retriever = get_rag_retriever()
    if retriever:
        try:
            # Get embeddings for the prompt
            query_embedding = embed_texts([prompt])[0]
            
            # Retrieve relevant context
            top_chunks = retriever.retrieve(query_embedding)
            context = "\n\n".join(top_chunks)
            
            # Create messages with context - using proper typing
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Relevant Blender API context:\n{context}\n\nYour request: {prompt}"}
            ]
        except Exception as e:
            print(f"RAG failed, falling back to direct prompt: {e}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
    else:
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
    """Call Ollama API with RAG if available"""
    if not OLLAMA_AVAILABLE:
        raise Exception("Ollama library not available")
    
    # Try to use RAG if available
    retriever = get_rag_retriever()
    if retriever:
        try:
            # Get embeddings for the prompt
            query_embedding = embed_texts([prompt])[0]
            
            # Retrieve relevant context
            top_chunks = retriever.retrieve(query_embedding)
            context = "\n\n".join(top_chunks)
            
            # Create messages with context
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Relevant Blender API context:\n{context}\n\nYour request: {prompt}"}
            ]
        except Exception as e:
            print(f"RAG failed, falling back to direct prompt: {e}")
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
    else:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    
    response = ollama.chat(
        model=model,
        messages=messages  # type: ignore
    )
    
    # Handle different response formats from ollama
    try:
        if isinstance(response, dict) and 'message' in response:
            return response['message']['content']
        else:
            # Try accessing as object attributes
            return str(getattr(getattr(response, 'message', {}), 'content', ''))
    except (AttributeError, KeyError, TypeError) as e:
        raise Exception(f"Failed to parse Ollama response: {e}")

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
    """Main function to generate and execute 3D model code"""
    # Get AI response
    if model == 'chatgpt':
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
    
    bpy.types.Scene.ai_openai_key = bpy.props.StringProperty(
        name="OpenAI API Key",
        description="Your OpenAI API key",
        default="",
        subtype='PASSWORD'
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
        
        # Update status
        context.scene.ai_last_status = f"Refreshed models: Found {len(ollama_models)} Ollama models"
        self.report({'INFO'}, context.scene.ai_last_status)
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

# Registration
classes = (
    AIMODEL_PT_MainPanel,
    AIMODEL_PT_StatusPanel,
    AIMODEL_OT_SubmitPrompt,
    AIMODEL_OT_RefreshModels,
    AIMODEL_OT_ManageAPIKey,
    AIMODEL_OT_OpenLog,
)

def register():
    
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

class RAGDocChunk:
    def __init__(self, content: str, embedding: List[float]):
        self.content = content
        self.embedding = embedding

class RAGRetriever:
    def __init__(self, chunks: List[RAGDocChunk]):
        self.chunks = chunks

    @classmethod
    def from_json(cls, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        chunks = [RAGDocChunk(d['content'], d['embedding']) for d in data]
        return cls(chunks)

    def retrieve(self, query_embedding: List[float], top_k=4) -> List[str]:
        if not SKLEARN_AVAILABLE:
            # Fallback: return first few chunks if sklearn not available
            return [chunk.content for chunk in self.chunks[:top_k]]
        
        vectors = np.array([chunk.embedding for chunk in self.chunks])
        query_vector = np.array([query_embedding])
        similarities = cosine_similarity(query_vector, vectors)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        return [self.chunks[i].content for i in top_indices]

def get_rag_retriever():
    """Get the RAG retriever instance"""
    try:
        # Look for the embeddings file in the addon directory
        addon_path = Path(__file__).parent
        embeddings_path = addon_path / "blender_api_embeddings.json"
        
        if not embeddings_path.exists():
            return None
            
        return RAGRetriever.from_json(str(embeddings_path))
    except Exception as e:
        print(f"Failed to load RAG retriever: {e}")
        return None

def embed_texts(texts: List[str]) -> List[List[float]]:
    """Get embeddings for texts using OpenAI API"""
    try:
        client = get_openai_client()
        response = client.embeddings.create(
            input=texts,
            model="text-embedding-3-small"
        )
        return [item.embedding for item in response.data]
    except Exception as e:
        print(f"Failed to get embeddings: {e}")
        return []