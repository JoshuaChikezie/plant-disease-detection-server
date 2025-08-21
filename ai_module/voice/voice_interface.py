"""
Voice Interface Module

This module provides speech recognition and text-to-speech functionality
with support for local Ghanaian languages (Twi, Ga, Hausa, Ewe) and English.
"""

import os
import logging
import tempfile
from typing import Dict, Any, Optional
import speech_recognition as sr
import pyttsx3
from gtts import gTTS
import pygame
from pydub import AudioSegment
import whisper
import numpy as np
import time

# Configure logging
logger = logging.getLogger(__name__)


class VoiceInterface:
    """
    Voice processing interface for speech recognition and synthesis.
    
    Features:
    - Multi-language speech recognition using Whisper
    - Text-to-speech synthesis with local language support
    - Audio preprocessing and enhancement
    - Voice command processing for agricultural queries
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize voice interface.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
        """
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Initialize text-to-speech engine
        self.tts_engine = pyttsx3.init()
        self._configure_tts_engine()
        
        # Load Whisper model for speech recognition
        try:
            self.whisper_model = whisper.load_model(model_size)
            logger.info(f"Whisper model '{model_size}' loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            self.whisper_model = None
        
        # Language mappings for local languages
        self.language_codes = {
            "en": "english",
            "tw": "twi",      # Twi (Akan)
            "ga": "ga",       # Ga
            "ha": "hausa",    # Hausa
            "ee": "ewe"       # Ewe
        }
        
        # Initialize pygame for audio playback
        try:
            pygame.mixer.init()
        except Exception as e:
            logger.warning(f"Failed to initialize pygame mixer: {e}")
    
    def _configure_tts_engine(self):
        """Configure the TTS engine with appropriate settings."""
        try:
            # Set speech rate (words per minute)
            self.tts_engine.setProperty('rate', 150)
            
            # Set volume (0.0 to 1.0)
            self.tts_engine.setProperty('volume', 0.9)
            
            # Get available voices
            voices = self.tts_engine.getProperty('voices')
            if voices:
                # Prefer female voice if available
                for voice in voices:
                    if 'female' in voice.name.lower():
                        self.tts_engine.setProperty('voice', voice.id)
                        break
                else:
                    # Use first available voice
                    self.tts_engine.setProperty('voice', voices[0].id)
            
            logger.info("TTS engine configured successfully")
            
        except Exception as e:
            logger.error(f"Failed to configure TTS engine: {e}")
    
    def speech_to_text(
        self, 
        audio_path: str, 
        language: str = "en", 
        enhance_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Convert speech audio to text using Whisper.
        
        Args:
            audio_path: Path to audio file
            language: Language code for recognition
            enhance_audio: Whether to enhance audio quality
            
        Returns:
            Dictionary containing transcription results
        """
        try:
            start_time = time.time()
            
            # Enhance audio quality if requested
            if enhance_audio:
                audio_path = self._enhance_audio(audio_path)
            
            # Use Whisper for transcription
            if self.whisper_model:
                result = self.whisper_model.transcribe(
                    audio_path,
                    language=self.language_codes.get(language, "english")
                )
                
                transcription = result["text"].strip()
                confidence = self._calculate_confidence(result)
                
            else:
                # Fallback to speech_recognition library
                transcription, confidence = self._fallback_speech_recognition(audio_path)
            
            processing_time = time.time() - start_time
            
            return {
                "success": True,
                "text": transcription,
                "confidence": confidence,
                "language": language,
                "processing_time": processing_time,
                "audio_duration": self._get_audio_duration(audio_path)
            }
            
        except Exception as e:
            logger.error(f"Speech-to-text conversion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "text": "",
                "confidence": 0.0
            }
    
    def text_to_speech(
        self,
        text: str,
        language: str = "en",
        voice_type: str = "female",
        speed: float = 1.0,
        output_path: str = None
    ) -> Dict[str, Any]:
        """
        Convert text to speech audio.
        
        Args:
            text: Text to convert to speech
            language: Language code for synthesis
            voice_type: Voice type (male, female)
            speed: Speech speed multiplier
            output_path: Path to save audio file
            
        Returns:
            Dictionary containing synthesis results
        """
        try:
            if not output_path:
                output_path = tempfile.mktemp(suffix=".wav")
            
            # Use gTTS for better language support
            if language in ["en", "tw", "ga", "ha", "ee"]:
                success = self._synthesize_with_gtts(text, language, output_path)
            else:
                # Fallback to pyttsx3
                success = self._synthesize_with_pyttsx3(text, speed, output_path)
            
            if success and os.path.exists(output_path):
                return {
                    "success": True,
                    "output_path": output_path,
                    "language": language,
                    "voice_type": voice_type,
                    "speed": speed,
                    "duration": self._get_audio_duration(output_path)
                }
            else:
                return {
                    "success": False,
                    "error": "Failed to generate speech audio"
                }
                
        except Exception as e:
            logger.error(f"Text-to-speech conversion failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _enhance_audio(self, audio_path: str) -> str:
        """
        Enhance audio quality for better speech recognition.
        
        Args:
            audio_path: Path to original audio file
            
        Returns:
            Path to enhanced audio file
        """
        try:
            # Load audio file
            audio = AudioSegment.from_file(audio_path)
            
            # Apply audio enhancements
            # Normalize volume
            audio = audio.normalize()
            
            # Apply noise reduction (simple high-pass filter)
            audio = audio.high_pass_filter(300)
            
            # Convert to mono if stereo
            if audio.channels > 1:
                audio = audio.set_channels(1)
            
            # Set sample rate to 16kHz (optimal for speech recognition)
            audio = audio.set_frame_rate(16000)
            
            # Save enhanced audio
            enhanced_path = audio_path.replace(".wav", "_enhanced.wav")
            audio.export(enhanced_path, format="wav")
            
            return enhanced_path
            
        except Exception as e:
            logger.warning(f"Audio enhancement failed, using original: {e}")
            return audio_path
    
    def _calculate_confidence(self, whisper_result: Dict) -> float:
        """
        Calculate confidence score from Whisper result.
        
        Args:
            whisper_result: Whisper transcription result
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        try:
            # Extract segments and calculate average confidence
            segments = whisper_result.get("segments", [])
            if not segments:
                return 0.5  # Default confidence
            
            confidences = []
            for segment in segments:
                # Whisper doesn't provide direct confidence, estimate from other metrics
                no_speech_prob = segment.get("no_speech_prob", 0.5)
                confidence = 1.0 - no_speech_prob
                confidences.append(confidence)
            
            return sum(confidences) / len(confidences)
            
        except Exception as e:
            logger.warning(f"Failed to calculate confidence: {e}")
            return 0.5
    
    def _fallback_speech_recognition(self, audio_path: str) -> tuple:
        """
        Fallback speech recognition using speech_recognition library.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Tuple of (transcription, confidence)
        """
        try:
            with sr.AudioFile(audio_path) as source:
                audio = self.recognizer.record(source)
                
            # Use Google Speech Recognition as fallback
            transcription = self.recognizer.recognize_google(audio)
            confidence = 0.8  # Estimated confidence for Google API
            
            return transcription, confidence
            
        except sr.UnknownValueError:
            logger.warning("Speech recognition could not understand audio")
            return "", 0.0
        except sr.RequestError as e:
            logger.error(f"Speech recognition service error: {e}")
            return "", 0.0
    
    def _synthesize_with_gtts(self, text: str, language: str, output_path: str) -> bool:
        """
        Synthesize speech using Google Text-to-Speech.
        
        Args:
            text: Text to synthesize
            language: Language code
            output_path: Output file path
            
        Returns:
            True if synthesis successful
        """
        try:
            # Map language codes to gTTS codes
            gtts_lang_map = {
                "en": "en",
                "tw": "en",  # gTTS doesn't support Twi, use English
                "ga": "en",  # gTTS doesn't support Ga, use English
                "ha": "ha",  # Hausa is supported
                "ee": "en"   # gTTS doesn't support Ewe, use English
            }
            
            gtts_lang = gtts_lang_map.get(language, "en")
            
            # Create gTTS object
            tts = gTTS(text=text, lang=gtts_lang, slow=False)
            
            # Save to file
            tts.save(output_path)
            
            return True
            
        except Exception as e:
            logger.error(f"gTTS synthesis failed: {e}")
            return False
    
    def _synthesize_with_pyttsx3(self, text: str, speed: float, output_path: str) -> bool:
        """
        Synthesize speech using pyttsx3 engine.
        
        Args:
            text: Text to synthesize
            speed: Speech speed multiplier
            output_path: Output file path
            
        Returns:
            True if synthesis successful
        """
        try:
            # Set speech rate
            rate = int(150 * speed)  # Base rate of 150 WPM
            self.tts_engine.setProperty('rate', rate)
            
            # Save to file
            self.tts_engine.save_to_file(text, output_path)
            self.tts_engine.runAndWait()
            
            return True
            
        except Exception as e:
            logger.error(f"pyttsx3 synthesis failed: {e}")
            return False
    
    def _get_audio_duration(self, audio_path: str) -> float:
        """
        Get duration of audio file in seconds.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Duration in seconds
        """
        try:
            audio = AudioSegment.from_file(audio_path)
            return len(audio) / 1000.0  # Convert milliseconds to seconds
        except Exception as e:
            logger.warning(f"Failed to get audio duration: {e}")
            return 0.0
    
    def get_supported_languages(self) -> Dict[str, str]:
        """
        Get list of supported languages.
        
        Returns:
            Dictionary mapping language codes to language names
        """
        return {
            "en": "English",
            "tw": "Twi (Akan)",
            "ga": "Ga",
            "ha": "Hausa", 
            "ee": "Ewe"
        }
    
    def test_microphone(self) -> Dict[str, Any]:
        """
        Test microphone functionality.
        
        Returns:
            Microphone test results
        """
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source)
            
            return {
                "success": True,
                "message": "Microphone is working properly",
                "ambient_noise_level": "adjusted"
            }
            
        except Exception as e:
            logger.error(f"Microphone test failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "message": "Microphone test failed"
            }