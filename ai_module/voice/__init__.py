"""
Voice Interface Module for Plant Disease Detection

This module handles:
- Multilingual speech recognition (English, Twi, Ga, Hausa, Ewe)
- Text-to-speech synthesis in local languages
- Voice command processing
- Audio preprocessing and enhancement
- Voice-based disease reporting
"""

from .voice_interface import VoiceInterface
from .speech_recognition import SpeechRecognizer
from .speech_synthesis import SpeechSynthesizer
from .audio_processor import AudioProcessor

__all__ = [
    "VoiceInterface",
    "SpeechRecognizer",
    "SpeechSynthesizer", 
    "AudioProcessor"
]