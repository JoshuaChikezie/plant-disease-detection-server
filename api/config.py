"""
Configuration settings for the Plant Disease Detection API

This module contains all configuration settings including:
- API settings
- Database configuration
- AI model paths
- Security settings
- Environment variables
"""

import os
from typing import List
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # CORS Settings
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8080",
        "https://plant-disease-detection.gh"
    ]
    
    # Database Settings
    DATABASE_URL: str = "sqlite:///./data/plant_disease.db"
    DATABASE_ECHO: bool = False
    
    # AI Model Settings
    MODEL_PATH: str = "models/disease_detection_model.h5"
    VOICE_MODEL_PATH: str = "models/voice_models/"
    KNOWLEDGE_BASE_PATH: str = "data/knowledge_base.db"
    
    # File Upload Settings
    UPLOAD_DIR: str = "uploads/"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_IMAGE_TYPES: List[str] = ["image/jpeg", "image/png", "image/jpg"]
    ALLOWED_AUDIO_TYPES: List[str] = ["audio/wav", "audio/mp3", "audio/m4a"]
    
    # Security Settings
    SECRET_KEY: str = "your-secret-key-here"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # External API Settings
    GOOGLE_TRANSLATE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    
    # Logging Settings
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/api.log"
    
    # Cache Settings
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL: int = 3600  # 1 hour
    
    # Email Settings
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    
    # Mobile App Settings
    MOBILE_APP_VERSION: str = "1.0.0"
    MIN_APP_VERSION: str = "1.0.0"
    
    # Web Scraping Settings
    SCRAPING_DELAY: float = 1.0  # seconds
    MAX_RETRIES: int = 3
    USER_AGENT: str = "Plant-Disease-Detection-Bot/1.0"
    
    # Voice Processing Settings
    VOICE_SAMPLE_RATE: int = 16000
    VOICE_CHUNK_SIZE: int = 1024
    VOICE_TIMEOUT: float = 5.0
    
    # Disease Detection Settings
    CONFIDENCE_THRESHOLD: float = 0.7
    MAX_PREDICTIONS: int = 3
    SUPPORTED_CROPS: List[str] = [
        "cassava", "maize", "cocoa", "yam", "plantain",
        "rice", "millet", "sorghum", "groundnut", "cowpea"
    ]
    
    # Language Settings
    SUPPORTED_LANGUAGES: List[str] = ["en", "tw", "ga", "ha", "ee"]
    DEFAULT_LANGUAGE: str = "en"
    
    # Knowledge Base Settings
    KNOWLEDGE_UPDATE_FREQUENCY: int = 7  # days
    MAX_FEEDBACK_AGE: int = 365  # days
    
    # Performance Settings
    MAX_CONCURRENT_REQUESTS: int = 100
    REQUEST_TIMEOUT: float = 30.0
    
    # Monitoring Settings
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create settings instance
settings = Settings()


# Environment-specific configurations
class DevelopmentSettings(Settings):
    """Development environment settings."""
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    DATABASE_ECHO: bool = True


class ProductionSettings(Settings):
    """Production environment settings."""
    DEBUG: bool = False
    LOG_LEVEL: str = "WARNING"
    DATABASE_ECHO: bool = False


class TestingSettings(Settings):
    """Testing environment settings."""
    DEBUG: bool = True
    DATABASE_URL: str = "sqlite:///./test.db"
    LOG_LEVEL: str = "DEBUG"


# Get environment-specific settings
def get_settings() -> Settings:
    """Get environment-specific settings."""
    environment = os.getenv("ENVIRONMENT", "development").lower()
    
    if environment == "production":
        return ProductionSettings()
    elif environment == "testing":
        return TestingSettings()
    else:
        return DevelopmentSettings()


# Export settings
settings = get_settings() 