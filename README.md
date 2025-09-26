# CodeDocAI: An NLP-Powered Code Documentation Generator

## Project Overview
This project aims to help programmers write documentation for their code, using a fine-tuned T5 model to generate readable documentation from code chunks. The training is done on the command line and the documentation is generated on a web interface through Streamlit.

## ✨ New Features (v2.0)

- **🔧 Advanced Configuration System**: Centralized configuration management with environment variable support
- **⚙️ Flexible Model Parameters**: Configurable temperature, beam search, and token limits
- **🎛️ Interactive Web Controls**: Real-time parameter adjustment in the web interface
- **📁 File Upload Support**: Upload code files directly with validation
- **🛠️ CLI Configuration Tools**: Command-line utilities for configuration management
- **📊 Enhanced Evaluation**: Multiple metrics and better performance tracking
- **🐳 Production Ready**: Environment-specific configurations and Docker support

## Features

- Documentation generation for code chunks with configurable AI parameters
- Interactive Web Interface made using Streamlit with real-time controls
- Model fine-tuned on code-documentation pairs
- Comprehensive configuration management system
- Support for multiple programming languages
- File upload and download capabilities

## Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/Nebu0528/CodeDocAI
```

### 2. Install Dependencies
Run the command below to install all the dependencies needed:
```bash
pip install -r requirements.txt
```

### 3. Configure the Application (Optional)
The application comes with sensible defaults, but you can customize it:

**Option A: Using Environment Variables**
```bash
export MODEL_NAME=t5-base
export TEMPERATURE=1.2
export NUM_BEAMS=6
export WEB_PORT=8502
```

**Option B: Using Configuration Files**
```bash
# Copy and edit the example configuration
cp config.json my_config.json
# Edit my_config.json with your preferred settings

# Or use the CLI tool
python3 config_manager.py show     # View current config
python3 config_manager.py validate # Validate settings
```

**Option C: Using Environment File**
```bash
cp .env.example .env
# Edit .env with your settings
```
### 4. Train the Model 
This step is necessary to teach the model how to interpret code and generate corresponding documentation based on that training. The model is being trained on this:
```bash
https://huggingface.co/datasets/jtatman/python-code-dataset-500k
```
- 1. Train the model using:
```bash
python src/model.py
```
- 2. After training, the model will be saved in the models/codetext_t5/ directory.

### 5. Run the Streamlit App
Once the model is trained, run the app:
```bash
streamlit run interface/website.py
```

### 6. Access App
Once the app is running, open the URL provided by Streamlit 
```bash 
http://localhost:8501
```

## Configuration Management

The application now includes a powerful configuration system. Use the CLI tool to manage settings:

```bash
# View all current settings
python3 config_manager.py show

# Validate your configuration
python3 config_manager.py validate

# Export current config to file
python3 config_manager.py export my_settings.json

# Import config from file
python3 config_manager.py import production_settings.json

# Create environment template
python3 config_manager.py template
```

## Advanced Usage

### Customizable Parameters

The web interface now provides real-time controls for:
- **Temperature**: Controls creativity vs focus (0.1-2.0)
- **Number of Beams**: Quality vs speed trade-off (1-10)
- **Max Tokens**: Length of generated documentation (50-500)

### File Upload Support

- Upload code files directly (supports .py, .js, .java, .cpp, .c, .go, .rs)
- Automatic file validation and size limits
- Download generated documentation as markdown files

### Environment-Specific Configurations

```bash
# Development
export LOG_LEVEL=DEBUG
export BATCH_SIZE=4

# Production  
export MODEL_PATH=/opt/models/production_model
export WEB_HOST=0.0.0.0
export WEB_PORT=80
```

## Usage
1. **Web Interface**: Paste a code snippet into the text area or upload a code file
2. **Adjust Parameters**: Use the sidebar to customize generation settings
3. **Generate**: Click "Generate Documentation" to create documentation
4. **Download**: Save the generated documentation as a markdown file

### Example Code Snippet
```python
def multiply(a, b):
    """This function returns the product of variables a and b"""
    return a * b
```

**Generated Documentation Example:**
```markdown
Function: multiply

Description:
Calculates the product of two input parameters and returns the result.

Parameters:
- a: First numeric value for multiplication
- b: Second numeric value for multiplication  

Returns:
The mathematical product of parameters a and b
```

## Docker Support

Build and run with Docker:
```bash
# Build the image
docker build -t codedocai .

# Run the container
docker run -p 8501:8501 codedocai

# Run with custom config
docker run -p 8501:8501 -v $(pwd)/my_config.json:/app/config.json codedocai
```

## Project Structure

```
CodeDocAI/
├── src/                          # Core application code
│   ├── config.py                # Configuration management
│   ├── model.py                 # T5 model training
│   ├── inference.py             # Documentation generation
│   ├── evaluate.py              # Model evaluation
│   └── data_loader.py           # Data loading utilities
├── interface/
│   └── website.py               # Streamlit web interface
├── models/                      # Trained model storage
├── data/                        # Training data
├── logs/                        # Application logs
├── config.json                  # Example configuration
├── .env.example                 # Environment template
├── config_manager.py            # CLI configuration tool
├── test_system.py               # System tests
├── requirements.txt             # Python dependencies
├── CONFIGURATION.md             # Configuration documentation
└── README.md                    # This file
```

## Future Improvements
1. ✅ Advanced configuration system (implemented)
2. ✅ File upload support (implemented)
3. ✅ Download generated documentation (implemented)
4. Handle different programming languages beyond Python
5. Integrate additional models (BART, CodeBERT) for enhanced generation
6. Support for batch processing multiple files
7. API endpoint for programmatic access
8. Integration with IDEs and code editors
9. Train on larger datasets like [The Stack](https://huggingface.co/datasets/bigcode/the-stack)
10. Add support for different documentation formats (JSDoc, Sphinx, etc.)

## Evaluate the Model

To evaluate the model's performance using BLEU scores, run the following command:

```bash
python src/evaluate.py
```

This will calculate the average BLEU score for the model's generated documentation compared to the reference documentation.

## Troubleshooting

### Common Issues

**Configuration Errors:**
```bash
# Check configuration validity
python3 config_manager.py validate

# View current settings
python3 config_manager.py show
```

**Import Errors:**
- Ensure you're running commands from the project root directory
- Verify all dependencies are installed: `pip install -r requirements.txt`

**Model Not Found:**
- Train the model first: `python src/model.py`  
- Check model path in configuration: `echo $MODEL_PATH`

**Port Already in Use:**
```bash
# Use different port
export WEB_PORT=8502
streamlit run interface/website.py
```

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test: `python3 test_system.py`
4. Validate configuration: `python3 config_manager.py validate`
5. Submit a pull request

## License

This project is open source and available under the [MIT License](LICENSE).

---

For detailed configuration options, see [CONFIGURATION.md](CONFIGURATION.md)
