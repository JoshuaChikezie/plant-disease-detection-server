"""
API Module for Plant Disease Detection System

This module provides REST API services for:
- Disease detection endpoints
- Voice processing endpoints
- Knowledge base queries
- User management and feedback
- Mobile app integration
"""

from .main import app
from .routes import disease_routes, voice_routes, knowledge_routes, user_routes

__all__ = [
    "app",
    "disease_routes",
    "voice_routes", 
    "knowledge_routes",
    "user_routes"
] 