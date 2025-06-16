# This makes the directory a Python package for Blender add-on
# The main add-on code is in blender_llm_addin.py

from . import blender_llm_addin

def register():
    blender_llm_addin.register()

def unregister():
    blender_llm_addin.unregister()

if __name__ == "__main__":
    register()
