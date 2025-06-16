#!/usr/bin/env python3

import os
import sys
import shutil
from setuptools import setup, find_packages
from setuptools.command.install import install
from pathlib import Path

# Read the contents of README.md for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the contents of the requirements file
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = f.read().splitlines()

# Get the version from the add-on file
version = '1.0.0'
with open('blender_llm_addin.py', 'r', encoding='utf-8') as f:
    for line in f:
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip("'\"")
            break

class InstallCommand(install):
    """Custom install command to handle Blender add-on installation."""
    user_options = install.user_options + [
        ('blender-addons-dir=', None, 'Path to Blender add-ons directory'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.blender_addons_dir = None

    def finalize_options(self):
        install.finalize_options(self)
        if self.blender_addons_dir is None:
            # Default Blender add-ons directory
            if sys.platform == 'darwin':  # macOS
                self.blender_addons_dir = os.path.expanduser('~/Library/Application Support/Blender/4.4/scripts/addons')
            elif sys.platform == 'win32':  # Windows
                self.blender_addons_dir = os.path.expanduser('~/AppData/Roaming/Blender Foundation/Blender/4.4/scripts/addons')
            else:  # Linux and others
                self.blender_addons_dir = os.path.expanduser('~/.config/blender/4.4/scripts/addons')

    def run(self):
        # Run the standard install
        install.run(self)
        
        # Copy the package to Blender's add-ons directory
        target_dir = os.path.join(self.blender_addons_dir, 'prompt2blend')
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy all files in the package
        data_files = []
        for item in os.listdir('.'):
            if os.path.isdir(item) and item != '__pycache__' and item != '.git' and item != '.github':
                for dirpath, dirnames, filenames in os.walk(item):
                    if '__pycache__' in dirnames:
                        dirnames.remove('__pycache__')
                    if '.git' in dirnames:
                        dirnames.remove('.git')
                    if '.github' in dirnames:
                        dirnames.remove('.github')
                    for filename in filenames:
                        if not (filename.endswith('.pyc') or filename == '.DS_Store'):
                            data_files.append(os.path.join(dirpath, filename))
            elif os.path.isfile(item) and not (item.endswith('.pyc') or item == '.DS_Store' or item.endswith('.py')):
                data_files.append(item)
        
        print(f"\nSuccessfully installed prompt2blend to {target_dir}")
        print("To complete the installation, please enable the add-on in Blender's preferences:")
        print("1. Open Blender")
        print("2. Go to Edit > Preferences > Add-ons")
        print("3. Search for 'prompt2blend'")
        print("4. Enable the checkbox next to the add-on")

setup(
    name="prompt2blend",
    version=version,
    author="Anthony Chapman",
    author_email="adchap77@gmail.com",
    description="Blender add-on for AI-assisted 3D modeling using LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Technologic101/prompt2blend",
    data_files=[('', [f for f in data_files if os.path.isfile(f)])],
    python_requires='>=3.7',
    install_requires=[
        'numpy>=1.24.0,<2.0.0',
        'requests>=2.31.0',
        'scikit-learn>=1.3.0',
        'chromadb>=0.4.24',
        'openai>=1.0.0',
        'ollama>=0.1.5',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Multimedia :: Graphics :: 3D :: Modeling",
    ],
    cmdclass={
        'install': InstallCommand,
    },
    entry_points={
        'console_scripts': [
            'prompt2blend-install=setup:main',
        ],
    },
)
