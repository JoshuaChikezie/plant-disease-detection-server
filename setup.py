#!/usr/bin/env python3
"""
Setup Script for Plant Disease Detection System

This script helps users set up the system by:
- Installing dependencies
- Creating necessary directories
- Initializing the database
- Setting up configuration
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import json


def print_banner():
    """Print setup banner."""
    print("""
🌱 Plant Disease Detection System Setup
=======================================

This script will help you set up the AI-powered plant disease detection
system for Ghanaian farmers.

Features:
- Image-based disease detection
- Multilingual voice interface
- Agricultural knowledge base
- Local language support
- Farmer feedback integration
    """)


def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 9):
        print("❌ Error: Python 3.9 or higher is required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    
    print(f"✅ Python version: {sys.version}")


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    directories = [
        "data",
        "models",
        "uploads",
        "logs",
        "static",
        "exports",
        "config",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"  ✅ Created: {directory}/")


def install_dependencies():
    """Install Python dependencies."""
    print("\n📦 Installing dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        return False
    
    return True


def create_env_file():
    """Create .env file from template."""
    print("\n⚙️ Setting up environment configuration...")
    
    env_example = Path(".env.example")
    env_file = Path(".env")
    
    if env_example.exists():
        if not env_file.exists():
            shutil.copy(env_example, env_file)
            print("✅ Created .env file from template")
            print("  📝 Please edit .env file with your configuration")
        else:
            print("✅ .env file already exists")
    else:
        print("⚠️  .env.example not found, creating basic .env file")
        create_basic_env_file()


def create_basic_env_file():
    """Create a basic .env file."""
    env_content = """# Plant Disease Detection System - Environment Configuration

# Environment
ENVIRONMENT=development

# API Settings
HOST=0.0.0.0
PORT=8000
DEBUG=true

# Database Settings
DATABASE_URL=sqlite:///./data/plant_disease.db
DATABASE_ECHO=false

# Security Settings
SECRET_KEY=your-super-secret-key-change-this-in-production
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Logging Settings
LOG_LEVEL=INFO
LOG_FILE=logs/api.log

# File Upload Settings
UPLOAD_DIR=uploads/
MAX_FILE_SIZE=10485760

# Disease Detection Settings
CONFIDENCE_THRESHOLD=0.7
MAX_PREDICTIONS=3

# Language Settings
DEFAULT_LANGUAGE=en
SUPPORTED_LANGUAGES=en,tw,ga,ha,ee

# Supported Crops
SUPPORTED_CROPS=cassava,maize,cocoa,yam,plantain,rice,millet,sorghum,groundnut,cowpea
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("✅ Created basic .env file")


def initialize_database():
    """Initialize the database with sample data."""
    print("\n🗄️ Initializing database...")
    
    try:
        subprocess.check_call([sys.executable, "scripts/init_database.py"])
        print("✅ Database initialized successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error initializing database: {e}")
        return False
    
    return True


def create_sample_config():
    """Create sample configuration files."""
    print("\n📋 Creating sample configuration files...")
    
    # AI Model Configuration
    model_config = {
        "model_architecture": "efficientnet_b0",
        "input_shape": [224, 224, 3],
        "confidence_threshold": 0.7,
        "max_predictions": 3,
        "supported_crops": [
            "cassava", "maize", "cocoa", "yam", "plantain",
            "rice", "millet", "sorghum", "groundnut", "cowpea"
        ],
        "disease_classes": {
            "cassava": [
                "healthy", "cassava_mosaic_disease", 
                "cassava_brown_streak", "cassava_bacterial_blight"
            ],
            "maize": [
                "healthy", "maize_rust", "maize_smut", "maize_leaf_blight"
            ],
            "cocoa": [
                "healthy", "cocoa_black_pod", "cocoa_swollen_shoot", 
                "cocoa_mirid_bug_damage"
            ]
        }
    }
    
    with open("config/model_config.json", "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Voice Configuration
    voice_config = {
        "supported_languages": {
            "en": "English",
            "tw": "Twi",
            "ga": "Ga",
            "ha": "Hausa",
            "ee": "Ewe"
        },
        "voice_settings": {
            "sample_rate": 16000,
            "chunk_size": 1024,
            "timeout": 5.0
        },
        "speech_recognition": {
            "model_size": "base",
            "confidence_threshold": 0.7
        }
    }
    
    with open("config/voice_config.json", "w") as f:
        json.dump(voice_config, f, indent=2)
    
    # Knowledge Base Configuration
    kb_config = {
        "data_sources": {
            "agricultural_databases": [
                "https://www.fao.org/agriculture/crops/",
                "https://www.cabi.org/",
                "https://www.plantwise.org/"
            ],
            "research_institutions": [
                "https://www.csir.org.gh/",
                "https://www.mofa.gov.gh/",
                "https://www.ug.edu.gh/"
            ]
        },
        "update_frequency": {
            "disease_data": 7,
            "treatment_info": 14,
            "research_papers": 30
        }
    }
    
    with open("config/knowledge_base_config.json", "w") as f:
        json.dump(kb_config, f, indent=2)
    
    print("✅ Configuration files created")


def run_tests():
    """Run basic tests to verify setup."""
    print("\n🧪 Running basic tests...")
    
    try:
        # Test imports
        import tensorflow as tf
        import torch
        import cv2
        import fastapi
        print("✅ All major dependencies imported successfully")
        
        # Test database connection
        import sqlite3
        conn = sqlite3.connect("data/knowledge_base.db")
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM diseases")
        disease_count = cursor.fetchone()[0]
        conn.close()
        print(f"✅ Database connection successful ({disease_count} diseases loaded)")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    return True


def print_next_steps():
    """Print next steps for the user."""
    print("""
🎉 Setup completed successfully!

Next steps:
1. Edit the .env file with your specific configuration
2. Start the API server: python api/main.py
3. Access the API documentation: http://localhost:8000/docs
4. Test the system with sample images

For development:
- Run tests: python -m pytest tests/
- Check logs: tail -f logs/api.log
- Monitor performance: http://localhost:8000/health

For production:
- Set ENVIRONMENT=production in .env
- Use a proper database (PostgreSQL/MySQL)
- Set up SSL certificates
- Configure monitoring and logging

Support:
- Email: support@plant-disease-detection.gh
- Documentation: https://plant-disease-detection.gh/docs
- Community: https://github.com/plant-disease-detection

🌱 Empowering Ghanaian farmers with AI-driven solutions!
    """)


def main():
    """Main setup function."""
    print_banner()
    
    # Check Python version
    check_python_version()
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        sys.exit(1)
    
    # Create environment file
    create_env_file()
    
    # Create sample configurations
    create_sample_config()
    
    # Initialize database
    if not initialize_database():
        print("❌ Setup failed at database initialization")
        sys.exit(1)
    
    # Run tests
    if not run_tests():
        print("❌ Setup failed at testing")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main() 