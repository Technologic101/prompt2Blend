# Prompt2Blend: AI-Powered 3D Model Generator for Blender

![Blender Add-on](https://img.shields.io/badge/Blender-4.4+-orange?logo=blender)
![License](https://img.shields.io/badge/License-MIT-blue)

Transform natural language prompts into 3D models directly in Blender using state-of-the-art AI models. This add-on integrates with OpenAI's GPT models and local Ollama models to generate and manipulate 3D content through simple text prompts.

## 🌟 Features

- **AI-Powered 3D Generation**: Create complex 3D models using natural language prompts
- **Multiple AI Backends**: Supports both OpenAI's API and local Ollama models
- **RAG Integration**: Enhanced context awareness with Retrieval-Augmented Generation
- **Blender Integration**: Seamless workflow within Blender's interface
- **Customizable**: Fine-tune parameters and model selection

## 🚀 Installation

1. **Download the latest release** from the [releases page](https://github.com/Technologic101/prompt2blend/releases)
2. **Install in Blender**:
   - Open Blender
   - Go to `Edit > Preferences > Add-ons`
   - Click "Install..." and select the downloaded `.zip` file
   - Enable the add-on by checking the box next to "Prompt2Blend"

## 🔑 Prerequisites

- Blender 4.4 or later
- Python 3.10+
- For OpenAI integration: Valid OpenAI API key
- For local models: [Ollama](https://ollama.ai/) installed and running

## 🛠️ Setup

1. **Configure API Keys**:
   - Open Blender and go to the 3D View
   - Find the "Gen AI 3D" tab in the sidebar (press 'N' if not visible)
   - Enter your OpenAI API key in the settings

2. **Using Local Models**:
   - Install and run Ollama on your system
   - Pull the desired model (e.g., `ollama pull mistral`)
   - Select the model from the dropdown in the add-on panel

## 🎮 Usage

1. Open Blender and navigate to the 3D View
2. Find the "Gen AI 3D" tab in the sidebar (press 'N' if not visible)
3. Select your preferred AI model from the dropdown
4. Enter your prompt in the text box (e.g., "Create a low-poly tree")
5. Click "Generate 3D Model"
6. Watch as your model comes to life!

## 🧩 Features in Detail

### AI Model Selection

- Choose between various OpenAI models (GPT-4, GPT-3.5-turbo, etc.)
- Support for local models through Ollama
- Model refresh button to update available models

### Advanced Options

- Temperature control for generation creativity
- Token limits for response length
- System prompt customization

### RAG Integration

- Enhanced context awareness using document retrieval
- Pre-loaded with Blender 4.4 documentation
- Custom knowledge base support

## 🛠 Development

### Building the Add-on

```bash
python build_addon.py
```

### Project Structure

- `blender_llm_addon.py` - Main add-on implementation
- `ui_panels.py` - Blender UI components
- `rag_agent.py` - RAG implementation for enhanced context
- `build_addon.py` - Build script for packaging

## 🤝 Contributing

Contributions are welcome! Please read our [contributing guidelines](CONTRIBUTING.md) before submitting pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Blender Foundation for the amazing 3D creation suite
- OpenAI for their powerful language models
- The Ollama team for making local AI models accessible

---

💡 **Tip**: For best results, be specific in your prompts. Instead of "a car," try "a low-poly sports car with 4 wheels and a spoiler."
