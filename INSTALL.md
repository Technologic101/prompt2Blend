# Blender LLM Add-on Installation Guide

## Prerequisites

1. **Blender 4.4+** installed
2. **Python dependencies** installed in Blender's Python environment

## Installation Steps

### Step 1: Install Python Dependencies

Open a terminal/command prompt and navigate to your Blender installation:

**Windows:**
```bash
# Navigate to Blender's Python directory
cd "C:\Program Files\Blender Foundation\Blender 4.4\4.4\python\bin"

# Install dependencies
python.exe -m pip install openai ollama scikit-learn numpy requests
```

**macOS:**
```bash
# Navigate to Blender.app's Python directory
cd /Applications/Blender.app/Contents/Resources/4.4/python/bin

# Install dependencies
./python3.11 -m pip install openai ollama scikit-learn numpy requests
```

**Linux:**
```bash
# Navigate to Blender's Python directory (adjust path as needed)
cd /opt/blender/4.4/python/bin

# Install dependencies
./python3.11 -m pip install openai ollama scikit-learn numpy requests
```

### Step 2: Install the Add-on

1. Download or clone this repository
2. In Blender, go to **Edit > Preferences > Add-ons**
3. Click **Install...** and select the entire `blender-llm-addin` folder
4. Enable the add-on "AI LLM 3D Graphics Model Generator"

### Step 3: Configure API Keys

1. In Blender's 3D Viewport, look for the **Gen AI 3D** panel in the sidebar (press `N` if not visible)
2. Click **Set API Key** to configure your OpenAI API key (if using ChatGPT)
3. For Ollama models, ensure Ollama is running locally on port 11434

## Usage

1. Select an AI model from the dropdown
2. Enter your prompt (e.g., "Create 10 red cubes in a circle")
3. Click **Generate 3D Model**
4. The generated code will execute automatically in Blender

## Troubleshooting

- If dependencies fail to install, try running Blender as administrator/root
- Ensure Ollama is running if using local models
- Check the Status panel for error messages and logs
