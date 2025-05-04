"""Configuration management for Doc2Deck."""
import os
from typing import Dict, Any, Type

class BaseConfig:
    """Base configuration settings."""
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY')
    API_KEY_ENV_VAR = 'LLM_API_KEY'
    DEFAULT_MAX_TOKENS = 4096
    DEFAULT_TEMPERATURE = 0.7
    ALLOWED_EXTENSIONS = ('.docx', '.pptx', '.pdf')
    MAX_CONTENT_LENGTH = 20 * 1024 * 1024  # 20MB max upload

class DevConfig(BaseConfig):
    """Development configuration."""
    DEBUG = True
    TESTING = False
    SECRET_KEY = SECRET_KEY or os.urandom(24)
    
class TestConfig(BaseConfig):
    """Testing configuration."""
    DEBUG = False
    TESTING = True
    SECRET_KEY = "test_secret_key"
    
class ProdConfig(BaseConfig):
    """Production configuration."""
    DEBUG = False
    TESTING = False
    
    def __init__(self):
        if not self.SECRET_KEY:
            raise ValueError("FLASK_SECRET_KEY must be set in production")

# Configuration selector
CONFIG_CLASSES: Dict[str, Type[BaseConfig]] = {
    'development': DevConfig,
    'testing': TestConfig,
    'production': ProdConfig
}

def get_config() -> BaseConfig:
    """Get configuration based on environment."""
    env = os.environ.get('FLASK_ENV', 'development')
    config_class = CONFIG_CLASSES.get(env, DevConfig)
    return config_class()