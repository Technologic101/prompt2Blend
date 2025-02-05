# Text-to-3D Model addin for Blender using Ollama and OpenAI GPT

This project demonstrates how to integrate AI-powered large language models (LLMs) with Blender to create a text-to-graphics modeling workflow. The system supports both open-source LLMs (e.g., Ollama-supported models) and OpenAI GPT, enabling users to generate 3D models in Blender by simply entering text prompts.
</br>
The below explains how to operate this in detail.</br>
- [Using Open-Source Models with Blender for AI-Assisted 3D Modeling: Comparative Study with OpenAI GPT](https://medium.com/@laputa99999/using-open-source-models-with-blender-for-ai-assisted-3d-modeling-comparative-study-with-openai-9848209f93b8)
- In addition, LLM media art tool and demo(https://github.com/mac999/llm-media-art-demo) link explain what is good tool how to visualize data.</br>
<img src='https://miro.medium.com/v2/resize:fit:720/format:webp/1*afjizeWcUVYJuFNx99ZKKA.png'><br>Prompt: Create 100 cubes along circle line with radius 30. The each cube has random color and size.</br></img></br>
<img src='https://miro.medium.com/v2/resize:fit:720/format:webp/1*afjizeWcUVYJuFNx99ZKKA.png'><br>Prompt: Create 100 cubes along circle line with radius 30. The each cube has random color and size.</br></img>

## Features
- **Custom Blender UI Panel**: Allows users to select an AI model and input text prompts directly in Blender's 3D Viewport.
- **AI Model Support**: Compatible with both open-source models (e.g., `Gemma`, `Phi`) and OpenAI GPT.
- **Dynamic 3D Modeling**: Generates Blender Python scripts based on user prompts to create 3D objects like grids, sinusoidal patterns, and more.
- **Error Handling**: Automatically fixes and retries failed AI-generated code.

## Getting Started

### Prerequisites
1. **Blender**: Install Blender (version 2.9x or higher) from [Blender.org](https://www.blender.org/).
2. **Python Dependencies**: Install the required Python libraries in Blender's Python environment.

### Installation

1. Navigate to Blender's Python interpreter directory:
   ```bash
   cd "C:/Program Files/Blender Foundation/Blender <version>/python/bin"
   ```
2. Install the required libraries:
   ```bash
   ./python.exe -m ensurepip
   ./python.exe -m pip install pandas numpy openai ollama
   ```

### Clone the Repository
Clone this GitHub repository:
```bash
git clone https://github.com/mac999/blender-llm-addin.git
cd blender-llm-addin
```

### Setting Up API Keys
For OpenAI GPT integration, replace `<input your OpenAI API key>` in the script with your OpenAI API key.

## Usage
1. Open Blender and go to the **Scripting** editor.
2. Copy and paste the script into a new file.
3. Run the script to load the custom UI panel.
4. In Blender's 3D Viewport:
   - Select an AI model from the dropdown.
   - Enter your text prompt.
   - Click **Submit** to generate and execute the Blender Python script.
5. View the 3D model generated in the Blender scene.
<img src="https://github.com/mac999/blender-llm-addin/blob/main/doc/img1.PNG"></img>

## Example Prompts
- "Create a grid of cubes with random colors and sizes."
- "Generate cubes following a cosine wave pattern along the y-axis."
- "Make a large yellow box at (6, -3) with a size of 5 units."

## Supported Models
- **OpenAI GPT** (e.g., `gpt-4`, `gpt-4o`)
- **Open Source Models via Ollama**:
  - `Gemma`
  - `Phi`
  - `CodeLlama`
  - `Qwen2.5`

## Results
Example outputs include:
- Grids of cubes with varying colors and sizes.
- Sinusoidal placement of objects.
- Dynamic scaling and positioning based on user-defined properties.

## Contributing
Contributions are welcome! If you have ideas for improvements or want to add support for additional models, feel free to submit a pull request.

## License
This project is licensed under the MIT License.

## Author
**Taewook Kang**  
If you have questions or suggestions, feel free to reach out at [laputa99999@gmail.com](mailto:laputa99999@gmail.com).
