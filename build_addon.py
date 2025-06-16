#!/usr/bin/env python3
"""
Build script for creating a Blender add-on zip file.
"""
import os
import shutil
import sys
import zipfile
from pathlib import Path

def get_bl_info():
    """Get the bl_info dictionary from __init__.py (single source of truth)."""

    import ast
    with open('__init__.py', 'r', encoding='utf-8') as f:
        content = f.read()
        
    # Parse the AST to find bl_info
    tree = ast.parse(content)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            if isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'bl_info':
                # Convert the AST back to a dictionary
                bl_info = {}
                for key, value in zip(node.value.keys, node.value.values):
                    key_name = key.value
                    if isinstance(value, ast.Str):
                        bl_info[key_name] = value.s
                    elif isinstance(value, (ast.Tuple, ast.List)):
                        bl_info[key_name] = ast.literal_eval(value)
                    else:
                        bl_info[key_name] = ast.literal_eval(ast.unparse(value))
                return bl_info

def create_zip_addon():
    """Create a zip file of the add-on for Blender installation."""
    # Define paths
    root_dir = Path(__file__).parent
    dist_dir = root_dir / 'dist'
    addon_name = 'prompt2blend'
    zip_path = dist_dir / f'{addon_name}.zip'
    
    # Get bl_info by parsing the file directly
    try:
        bl_info = get_bl_info()
    except Exception as e:
        print(f"Warning: Could not get bl_info: {e}")
        bl_info = {
            'name': 'Prompt2Blend',
            'description': 'AI-Powered 3D Model Generator for Blender',
        }
    
    # Create dist directory if it doesn't exist
    dist_dir.mkdir(exist_ok=True)
    
    # Remove existing zip if it exists
    if zip_path.exists():
        print(f"Removing existing {zip_path}")
        zip_path.unlink()
    
    # Files and directories to exclude
    exclude_dirs = {
        '__pycache__',
        '.pytest_cache',
        '.git',
        '.github',
        '.idea',
        '.vscode',
        'venv',
        '.venv',
        'env',
        'dist',
        'build',
        '*.egg-info',
        'tests',
        'examples',
        'src',
    }
    
    exclude_files = {
        'build_addon.py',
        'pyproject.toml',
        'MANIFEST.in',
        'requirements.txt',
        'requirements-dev.txt',
        '.gitignore',
        'README.md',
        'ASSESSMENT_REPORT.md',
        'prompt2Blend.code-workspace',
    }
    
    exclude_extensions = {
        '.pyc',
        '.pyo',
        '.pyd',
        '.DS_Store',
        '.blend1',
        '.blend2',
        '.log',
    }
    
    print(f"Creating {zip_path}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Add all files from the root directory
        for root, dirs, files in os.walk(root_dir):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if d not in exclude_dirs]
            
            for file in files:
                # Skip excluded files by name and extension
                if (file in exclude_files or 
                    any(file.endswith(ext) for ext in exclude_extensions)):
                    continue
                
                file_path = Path(root) / file
                # Skip the zip file if it exists
                if file_path == zip_path:
                    continue
                
                # Get the relative path for the zip file
                rel_path = file_path.relative_to(root_dir)
                # Determine the path in the zip file
                if rel_path.parts[0] == 'prompt2blend':
                    # If file is already in prompt2blend/, keep that structure
                    zip_path_in_zip = str(rel_path)
                elif rel_path.parts[0] == 'chroma_db':
                    # Put chroma_db inside prompt2blend/
                    zip_path_in_zip = str(Path('prompt2blend') / rel_path)
                elif rel_path.parts[0] in ['__init__.py', 'blender_llm_addin.py', 'rag_agent.py']:
                    # Move root Python files into prompt2blend/
                    zip_path_in_zip = str(Path('prompt2blend') / rel_path)
                else:
                    # Skip other files in the root directory
                    continue
                
                print(f"  Adding: {zip_path_in_zip}")
                zipf.write(file_path, zip_path_in_zip)
    
    print(f"\n✅ Successfully created {zip_path}")

def main():
    """Entry point for console script."""
    create_zip_addon()

if __name__ == "__main__":
    main()
