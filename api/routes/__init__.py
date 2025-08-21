"""
API Routes Package

This package contains all API route definitions for the Plant Disease Detection System.
Each route module handles specific functionality:
- disease_routes: Disease detection and analysis endpoints
- voice_routes: Voice processing and synthesis endpoints  
- knowledge_routes: Knowledge base and information endpoints
- user_routes: User management and feedback endpoints
"""

from . import disease_routes, voice_routes, knowledge_routes, user_routes

__all__ = [
    "disease_routes",
    "voice_routes", 
    "knowledge_routes",
    "user_routes"
]
