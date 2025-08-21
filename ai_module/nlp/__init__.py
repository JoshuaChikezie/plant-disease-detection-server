"""
Natural Language Processing Module for Plant Disease Detection

This module handles:
- Multilingual text processing (English, Twi, Ga, Hausa, Ewe)
- Text translation and localization
- Disease description processing
- Query understanding and intent recognition
- Response generation
"""

from .language_processor import LanguageProcessor
from .translator import Translator
from .intent_recognizer import IntentRecognizer
from .response_generator import ResponseGenerator

__all__ = [
    "LanguageProcessor",
    "Translator", 
    "IntentRecognizer",
    "ResponseGenerator"
] 