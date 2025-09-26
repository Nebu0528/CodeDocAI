"""
Configuration management utility for CodeDocAI.
This script provides command-line tools to manage configuration settings.
"""
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from config import AppConfig

def show_config():
    """Display current configuration."""
    config = AppConfig.load_from_env()
    
    print("=== CodeDocAI Configuration ===\n")
    
    print("Model Configuration:")
    print(f"  Model Name: {config.model_config.model_name}")
    print(f"  Model Path: {config.model_config.model_path}")
    print(f"  Max Length: {config.model_config.max_length}")
    print(f"  Max New Tokens: {config.model_config.max_new_tokens}")
    print(f"  Temperature: {config.model_config.temperature}")
    print(f"  Num Beams: {config.model_config.num_beams}")
    
    print("\nTraining Configuration:")
    print(f"  Epochs: {config.training_config.epochs}")
    print(f"  Batch Size: {config.training_config.batch_size}")
    print(f"  Learning Rate: {config.training_config.learning_rate}")
    print(f"  Output Directory: {config.training_config.output_dir}")
    
    print("\nData Configuration:")
    print(f"  Dataset Name: {config.data_config.dataset_name}")
    print(f"  Training Data Path: {config.data_config.training_data_path}")
    print(f"  Test Split: {config.data_config.test_split}")
    print(f"  Max Samples: {config.data_config.max_samples}")
    
    print("\nWeb Configuration:")
    print(f"  Host: {config.web_config.host}")
    print(f"  Port: {config.web_config.port}")
    print(f"  Title: {config.web_config.title}")
    
    print("\nLogging Configuration:")
    print(f"  Level: {config.logging_config.level}")
    print(f"  Log Directory: {config.logging_config.log_dir}")

def validate_config():
    """Validate current configuration."""
    config = AppConfig.load_from_env()
    
    print("Validating configuration...")
    
    if config.validate():
        print("✅ Configuration is valid!")
        return True
    else:
        print("❌ Configuration validation failed!")
        return False

def export_config(filepath):
    """Export current configuration to a file."""
    config = AppConfig.load_from_env()
    config.save_to_file(filepath)
    print(f"✅ Configuration exported to {filepath}")

def import_config(filepath):
    """Import configuration from a file."""
    if not os.path.exists(filepath):
        print(f"❌ Configuration file {filepath} not found!")
        return
    
    config = AppConfig.load_from_file(filepath)
    if config.validate():
        print(f"✅ Configuration loaded from {filepath}")
        print("Note: To apply this configuration permanently, set the corresponding environment variables.")
    else:
        print(f"❌ Configuration file {filepath} contains invalid settings!")

def create_env_template():
    """Create a template .env file."""
    template_content = """# CodeDocAI Environment Configuration
# Copy this file to .env and modify as needed

# Model Configuration
MODEL_NAME=t5-base
MODEL_PATH=models/codetext_t5
MAX_LENGTH=512
MAX_NEW_TOKENS=200
NUM_BEAMS=4
TEMPERATURE=1.0

# Training Configuration
EPOCHS=5
BATCH_SIZE=8
LEARNING_RATE=5e-5
OUTPUT_DIR=./results

# Data Configuration
DATASET_NAME=jtatman/python-code-dataset-500k
TRAINING_DATA_PATH=data/training_data.txt
TEST_SPLIT=0.2
# MAX_SAMPLES=10000  # Uncomment to limit dataset size

# Web Configuration
WEB_HOST=localhost
WEB_PORT=8501
WEB_TITLE="CodeDocAI - Generate Documentation for Code"

# Logging Configuration
LOG_LEVEL=INFO
LOG_DIR=logs
"""
    
    with open('.env.template', 'w') as f:
        f.write(template_content)
    
    print("✅ Created .env.template file")
    print("Copy this to .env and modify as needed:")
    print("  cp .env.template .env")

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="CodeDocAI Configuration Management")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Show command
    subparsers.add_parser('show', help='Show current configuration')
    
    # Validate command
    subparsers.add_parser('validate', help='Validate current configuration')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export configuration to file')
    export_parser.add_argument('filepath', help='Output file path')
    
    # Import command
    import_parser = subparsers.add_parser('import', help='Import configuration from file')
    import_parser.add_argument('filepath', help='Input file path')
    
    # Template command
    subparsers.add_parser('template', help='Create .env template file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'show':
            show_config()
        elif args.command == 'validate':
            valid = validate_config()
            sys.exit(0 if valid else 1)
        elif args.command == 'export':
            export_config(args.filepath)
        elif args.command == 'import':
            import_config(args.filepath)
        elif args.command == 'template':
            create_env_template()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()