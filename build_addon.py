#!/usr/bin/env python3
"""
Build script for creating a Blender add-on zip file.
"""
import os
import zipfile
from pathlib import Path

def create_zip_addon():
    """Create a zip file of the add-on for Blender installation."""
    # Define paths
    root_dir = Path(__file__).parent
    dist_dir = root_dir / 'dist'
    addon_name = 'prompt2blend'
    zip_path = dist_dir / f'{addon_name}.zip'
    
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
        # Add all top-level .py files from the addon directory
        py_files = [f for f in root_dir.iterdir() if f.is_file() and f.suffix == '.py']

        for py_file in py_files:
            arcname = f'{addon_name}/{py_file.name}'
            print(f"  Adding: {arcname}")
            zipf.write(py_file, arcname)

        # Add other resources (e.g., chroma_db directory and its contents)
        chroma_db_dir = root_dir / 'chroma_db'
        if chroma_db_dir.exists():
            for root, dirs, files in os.walk(chroma_db_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = f'{addon_name}/chroma_db/{file_path.relative_to(root_dir).as_posix()}'
                    print(f"  Adding: {arcname}")
                    zipf.write(file_path, arcname)
    
    print(f"\n✅ Successfully created {zip_path}")

def main():
    """Entry point for console script."""
    create_zip_addon()

if __name__ == "__main__":
    main()
