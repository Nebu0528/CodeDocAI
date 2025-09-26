"""
Configuration settings for the CodeDocAI project.
"""
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class ModelConfig:
    """Model configuration settings."""
    model_name: str = "t5-base"
    model_path: str = "models/codetext_t5"
    max_length: int = 512
    max_new_tokens: int = 200
    num_beams: int = 4
    length_penalty: float = 2.0
    early_stopping: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    
    # API Configuration
    use_api: bool = False
    api_provider: str = "openai"  # openai, anthropic, gemini
    api_key: Optional[str] = None
    api_model: Optional[str] = None  # Provider-specific: gpt-3.5-turbo, claude-3-haiku-20240307, gemini-2.0-flash
    api_base_url: Optional[str] = None

@dataclass
class TrainingConfig:
    """Training configuration settings."""
    output_dir: str = "./results"
    epochs: int = 5
    batch_size: int = 8
    learning_rate: float = 5e-5
    warmup_steps: int = 500
    weight_decay: float = 0.01
    logging_dir: str = "./logs"
    logging_steps: int = 100
    save_strategy: str = "epoch"
    evaluation_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

@dataclass
class DataConfig:
    """Data configuration settings."""
    dataset_name: str = "jtatman/python-code-dataset-500k"
    training_data_path: str = "data/training_data.txt"
    max_samples: Optional[int] = None
    test_split: float = 0.2
    validation_split: float = 0.1
    min_code_length: int = 50
    max_code_length: int = 2048
    min_doc_length: int = 10
    max_doc_length: int = 512

@dataclass
class WebConfig:
    """Web interface configuration settings."""
    host: str = "localhost"
    port: int = 8501
    title: str = "CodeDocAI - Generate Documentation for Code"
    description: str = "Transform your code into comprehensive documentation using AI"
    max_file_size: int = 1024 * 1024  # 1MB
    allowed_extensions: tuple = (".py", ".js", ".java", ".cpp", ".c", ".go", ".rs")

@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_dir: str = "logs"
    max_log_files: int = 5
    log_file_size: int = 10 * 1024 * 1024  # 10MB

@dataclass
class AppConfig:
    """Main application configuration container."""
    model_config: ModelConfig
    training_config: TrainingConfig
    data_config: DataConfig
    web_config: WebConfig
    logging_config: LoggingConfig
    
    def __init__(self):
        self.model_config = ModelConfig()
        self.training_config = TrainingConfig()
        self.data_config = DataConfig()
        self.web_config = WebConfig()
        self.logging_config = LoggingConfig()
    
    @classmethod
    def load_from_env(cls):
        """Load configuration from environment variables."""
        config = cls()
        
        # Model configuration from environment
        config.model_config.model_name = os.getenv('MODEL_NAME', config.model_config.model_name)
        config.model_config.model_path = os.getenv('MODEL_PATH', config.model_config.model_path)
        config.model_config.max_length = int(os.getenv('MAX_LENGTH', str(config.model_config.max_length)))
        config.model_config.max_new_tokens = int(os.getenv('MAX_NEW_TOKENS', str(config.model_config.max_new_tokens)))
        config.model_config.num_beams = int(os.getenv('NUM_BEAMS', str(config.model_config.num_beams)))
        config.model_config.temperature = float(os.getenv('TEMPERATURE', str(config.model_config.temperature)))
        
        # API configuration from environment
        config.model_config.use_api = os.getenv('USE_API', '').lower() in ('true', '1', 'yes')
        config.model_config.api_provider = os.getenv('API_PROVIDER', config.model_config.api_provider)
        config.model_config.api_key = os.getenv('API_KEY', config.model_config.api_key)
        config.model_config.api_model = os.getenv('API_MODEL', config.model_config.api_model)
        config.model_config.api_base_url = os.getenv('API_BASE_URL', config.model_config.api_base_url)
        
        # Training configuration from environment
        config.training_config.epochs = int(os.getenv('EPOCHS', str(config.training_config.epochs)))
        config.training_config.batch_size = int(os.getenv('BATCH_SIZE', str(config.training_config.batch_size)))
        config.training_config.learning_rate = float(os.getenv('LEARNING_RATE', str(config.training_config.learning_rate)))
        config.training_config.output_dir = os.getenv('OUTPUT_DIR', config.training_config.output_dir)
        
        # Data configuration from environment
        config.data_config.dataset_name = os.getenv('DATASET_NAME', config.data_config.dataset_name)
        config.data_config.training_data_path = os.getenv('TRAINING_DATA_PATH', config.data_config.training_data_path)
        config.data_config.test_split = float(os.getenv('TEST_SPLIT', str(config.data_config.test_split)))
        if os.getenv('MAX_SAMPLES'):
            config.data_config.max_samples = int(os.getenv('MAX_SAMPLES'))
        
        # Web configuration from environment
        config.web_config.host = os.getenv('WEB_HOST', config.web_config.host)
        config.web_config.port = int(os.getenv('WEB_PORT', str(config.web_config.port)))
        config.web_config.title = os.getenv('WEB_TITLE', config.web_config.title)
        
        # Logging configuration from environment
        config.logging_config.level = os.getenv('LOG_LEVEL', config.logging_config.level)
        config.logging_config.log_dir = os.getenv('LOG_DIR', config.logging_config.log_dir)
        
        return config
    
    def save_to_file(self, filepath: str):
        """Save current configuration to a file."""
        import json
        
        config_dict = {
            'model_config': self.model_config.__dict__,
            'training_config': self.training_config.__dict__,
            'data_config': self.data_config.__dict__,
            'web_config': self.web_config.__dict__,
            'logging_config': self.logging_config.__dict__
        }
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str):
        """Load configuration from a JSON file."""
        import json
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        config = cls()
        
        # Update configurations from file
        if 'model_config' in config_dict:
            for key, value in config_dict['model_config'].items():
                if hasattr(config.model_config, key):
                    setattr(config.model_config, key, value)
        
        if 'training_config' in config_dict:
            for key, value in config_dict['training_config'].items():
                if hasattr(config.training_config, key):
                    setattr(config.training_config, key, value)
        
        if 'data_config' in config_dict:
            for key, value in config_dict['data_config'].items():
                if hasattr(config.data_config, key):
                    setattr(config.data_config, key, value)
        
        if 'web_config' in config_dict:
            for key, value in config_dict['web_config'].items():
                if hasattr(config.web_config, key):
                    setattr(config.web_config, key, value)
        
        if 'logging_config' in config_dict:
            for key, value in config_dict['logging_config'].items():
                if hasattr(config.logging_config, key):
                    setattr(config.logging_config, key, value)
        
        return config
    
    def validate(self) -> bool:
        """Validate configuration values."""
        errors = []
        
        # Validate model config
        if self.model_config.max_length <= 0:
            errors.append("max_length must be positive")
        
        if self.model_config.max_new_tokens <= 0:
            errors.append("max_new_tokens must be positive")
        
        if self.model_config.num_beams <= 0:
            errors.append("num_beams must be positive")
        
        # Validate training config
        if self.training_config.epochs <= 0:
            errors.append("epochs must be positive")
        
        if self.training_config.batch_size <= 0:
            errors.append("batch_size must be positive")
        
        if self.training_config.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        # Validate data config
        if not (0 < self.data_config.test_split < 1):
            errors.append("test_split must be between 0 and 1")
        
        if not (0 < self.data_config.validation_split < 1):
            errors.append("validation_split must be between 0 and 1")
        
        # Validate web config
        if not (1024 <= self.web_config.port <= 65535):
            errors.append("port must be between 1024 and 65535")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True

# Global configuration instance
config = AppConfig.load_from_env()

# Validate configuration on import
if not config.validate():
    print("Warning: Configuration validation failed. Using default values where possible.")