# Configuration Management

The CodeDocAI project now includes a comprehensive configuration management system that allows you to easily customize model behavior, training parameters, and web interface settings.

## Configuration Structure

The configuration is organized into five main sections:

### Model Configuration
- `model_name`: Base model to use (default: "t5-base")
- `model_path`: Path to save/load the trained model
- `max_length`: Maximum input sequence length
- `max_new_tokens`: Maximum tokens to generate
- `num_beams`: Number of beams for beam search
- `temperature`: Sampling temperature for generation
- `top_k`, `top_p`: Sampling parameters

### Training Configuration
- `epochs`: Number of training epochs
- `batch_size`: Training batch size
- `learning_rate`: Learning rate for training
- `warmup_steps`: Number of warmup steps
- `output_dir`: Directory to save training outputs
- `evaluation_strategy`: When to run evaluation

### Data Configuration
- `dataset_name`: HuggingFace dataset name
- `training_data_path`: Local training data path
- `test_split`: Fraction of data for testing
- `max_samples`: Maximum samples to use (optional)

### Web Configuration
- `host`: Web interface host
- `port`: Web interface port
- `title`: Page title
- `max_file_size`: Maximum upload file size
- `allowed_extensions`: Supported file extensions

### Logging Configuration
- `level`: Logging level (INFO, DEBUG, etc.)
- `log_dir`: Directory for log files
- `format`: Log message format

## Usage

### 1. Environment Variables
Set environment variables to override defaults:

```bash
export MODEL_NAME=t5-base
export EPOCHS=10
export BATCH_SIZE=16
export WEB_PORT=8502
```

### 2. Configuration Files
Create a `config.json` file:

```json
{
  "model_config": {
    "temperature": 1.2,
    "num_beams": 6
  },
  "training_config": {
    "epochs": 10,
    "batch_size": 16
  }
}
```

Load with:
```python
from src.config import AppConfig
config = AppConfig.load_from_file('config.json')
```

### 3. Programmatic Usage
```python
from src.config import config

# Access configuration
print(f"Model path: {config.model_config.model_path}")

# Modify configuration
config.model_config.temperature = 1.5
config.training_config.epochs = 10
```

## Configuration Management CLI

Use the `config_manager.py` script to manage configurations:

```bash
# Show current configuration
python3 config_manager.py show

# Validate configuration
python3 config_manager.py validate

# Export configuration to file
python3 config_manager.py export my_config.json

# Import configuration from file
python3 config_manager.py import my_config.json

# Create environment template
python3 config_manager.py template
```

## Environment File

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

Then load environment variables:
```bash
source .env  # or use python-dotenv
```

## Benefits

1. **Centralized Settings**: All configuration in one place
2. **Environment Flexibility**: Easy deployment across different environments
3. **Validation**: Built-in configuration validation
4. **Documentation**: Self-documenting configuration structure
5. **Backward Compatibility**: Sensible defaults maintain existing behavior

## Best Practices

1. Use environment variables for deployment-specific settings
2. Use configuration files for experiment-specific settings
3. Validate configuration before training/inference
4. Keep sensitive information in environment variables, not config files
5. Use the CLI tools for configuration management

## Example Workflows

### Development
```bash
# Set development-specific variables
export LOG_LEVEL=DEBUG
export BATCH_SIZE=4  # Smaller batch for development
python3 src/train.py
```

### Production
```bash
# Use production config file
python3 config_manager.py import production_config.json
streamlit run interface/website.py
```

### Experimentation
```bash
# Create experiment config
python3 config_manager.py export experiment_baseline.json
# Modify experiment_baseline.json for different experiments
python3 config_manager.py import experiment_variant1.json
```

This configuration system makes CodeDocAI more maintainable, flexible, and production-ready!