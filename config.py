# config.py
import os
import yaml
from pathlib import Path

class Config:
    """Base configuration class"""
    
    def __init__(self):
        self.load_config()
    
    def load_config(self):
        """Load configuration from YAML file"""
        config_path = Path("config.yml")
        if not config_path.exists():
            raise FileNotFoundError("config.yml not found")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Load API settings
        self.APP_TITLE = config['api']['title']
        self.HOST = config['api']['host']
        self.PORT = config['api']['port']
        
        # Load detection settings
        self.CONF_THRESHOLD = config['detection']['conf_threshold']
        self.IOU_THRESHOLD = config['detection']['iou_threshold']
        self.IMG_SIZE = config['detection']['img_size']
        
        # Load model settings
        self.models = config['models']
        self.current_model = 'default'
        self.MODEL_PATH = Path(self.models[self.current_model]['path'])
        
        # Load temporary directory settings
        self.TEMP_DIR_PREFIX = config['temp_dir_prefix']
        
        # Class Names Fallback (if not found in model)
        self.DEFAULT_CLASS_NAMES = {i: f"Class {i}" for i in range(100)}
    
    def switch_model(self, model_key):
        """Switch to a different model configuration"""
        if model_key not in self.models:
            raise ValueError(f"Model {model_key} not found in configuration")
        
        self.current_model = model_key
        self.MODEL_PATH = Path(self.models[model_key]['path'])
        return self.MODEL_PATH
    
    def validate(self):
        """Validate configuration settings"""
        if not self.MODEL_PATH.exists():
            raise ValueError(f"Model path {self.MODEL_PATH} does not exist")
        if not 0 <= self.CONF_THRESHOLD <= 1:
            raise ValueError(f"Confidence threshold must be between 0 and 1, got {self.CONF_THRESHOLD}")
        if not 0 <= self.IOU_THRESHOLD <= 1:
            raise ValueError(f"IoU threshold must be between 0 and 1, got {self.IOU_THRESHOLD}")
        if self.IMG_SIZE <= 0:
            raise ValueError(f"Image size must be positive, got {self.IMG_SIZE}")

class DevelopmentConfig(Config):
    """Development-specific configuration"""
    DEBUG = True

class ProductionConfig(Config):
    """Production-specific configuration"""
    DEBUG = False

# Select configuration based on environment variable
env = os.getenv("ENV", "development").lower()
if env == "production":
    config = ProductionConfig()
else:
    config = DevelopmentConfig()

# Validate the selected configuration
config.validate()

# Export the config object
CONFIG = config