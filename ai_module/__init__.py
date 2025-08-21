"""
AI Module for Plant Disease Detection System

This module contains the core AI components for:
- Computer vision and disease detection
- Natural language processing
- Voice recognition and synthesis
- Knowledge base management
- Web scraping and data collection
"""

__version__ = "1.0.0"
__author__ = "Plant Disease Detection Team"
__email__ = "support@plant-disease-detection.gh"

from .computer_vision import DiseaseDetector
from .nlp import LanguageProcessor
from .voice import VoiceInterface
from .knowledge_base import KnowledgeBase

__all__ = [
    "DiseaseDetector",
    "LanguageProcessor", 
    "VoiceInterface",
    "KnowledgeBase"
] 