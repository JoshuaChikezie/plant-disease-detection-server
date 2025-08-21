"""
Voice Processing API Routes

This module provides REST API endpoints for voice processing functionality.
Includes speech-to-text, text-to-speech, and multilingual voice interactions.
"""

import os
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime

from ..config import settings
from ...ai_module.voice.voice_interface import VoiceInterface

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize voice interface
voice_interface = VoiceInterface()


# Pydantic models for request/response validation
class SpeechToTextRequest(BaseModel):
    """Request model for speech-to-text conversion"""
    language: Optional[str] = "en"
    enhance_audio: Optional[bool] = True


class TextToSpeechRequest(BaseModel):
    """Request model for text-to-speech conversion"""
    text: str
    language: Optional[str] = "en"
    voice_type: Optional[str] = "female"
    speed: Optional[float] = 1.0


class VoiceQueryRequest(BaseModel):
    """Request model for voice-based disease queries"""
    language: Optional[str] = "en"
    crop_type: Optional[str] = None


@router.post("/speech-to-text")
async def convert_speech_to_text(
    audio: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    enhance_audio: Optional[bool] = Form(True)
):
    """
    Convert speech audio to text.
    
    Args:
        audio: Audio file (WAV, MP3, M4A)
        language: Language code (en, tw, ga, ha, ee)
        enhance_audio: Whether to enhance audio quality before processing
        
    Returns:
        Transcribed text and confidence score
    """
    try:
        # Validate file type
        if audio.content_type not in settings.ALLOWED_AUDIO_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid audio type. Allowed types: {settings.ALLOWED_AUDIO_TYPES}"
            )
        
        # Save uploaded audio file temporarily
        audio_id = str(uuid.uuid4())
        upload_dir = os.path.join(settings.UPLOAD_DIR, "audio")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{audio_id}_{audio.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await audio.read()
            buffer.write(content)
        
        # Process speech to text
        result = voice_interface.speech_to_text(
            audio_path=file_path,
            language=language,
            enhance_audio=enhance_audio
        )
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return {
            "success": True,
            "audio_id": audio_id,
            "timestamp": datetime.now().isoformat(),
            "language": language,
            "transcription": result.get("text", ""),
            "confidence": result.get("confidence", 0.0),
            "processing_time": result.get("processing_time", 0.0)
        }
        
    except Exception as e:
        logger.error(f"Speech-to-text conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech conversion failed: {str(e)}")


@router.post("/text-to-speech")
async def convert_text_to_speech(request: TextToSpeechRequest):
    """
    Convert text to speech audio.
    
    Args:
        request: Text-to-speech parameters
        
    Returns:
        Audio file with synthesized speech
    """
    try:
        # Validate language support
        if request.language not in settings.SUPPORTED_LANGUAGES:
            raise HTTPException(
                status_code=400,
                detail=f"Language '{request.language}' not supported"
            )
        
        # Generate speech audio
        audio_id = str(uuid.uuid4())
        output_dir = os.path.join(settings.UPLOAD_DIR, "generated_audio")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{audio_id}.wav")
        
        result = voice_interface.text_to_speech(
            text=request.text,
            language=request.language,
            voice_type=request.voice_type,
            speed=request.speed,
            output_path=output_path
        )
        
        if result.get("success"):
            return FileResponse(
                path=output_path,
                media_type="audio/wav",
                filename=f"speech_{audio_id}.wav"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to generate speech")
        
    except Exception as e:
        logger.error(f"Text-to-speech conversion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Speech synthesis failed: {str(e)}")


@router.post("/voice-query")
async def process_voice_query(
    audio: UploadFile = File(...),
    language: Optional[str] = Form("en"),
    crop_type: Optional[str] = Form(None)
):
    """
    Process a voice query about plant diseases.
    
    Args:
        audio: Audio file containing the query
        language: Language of the audio query
        crop_type: Optional crop type context
        
    Returns:
        Text response and audio response for the query
    """
    try:
        # Convert speech to text
        speech_result = await convert_speech_to_text(audio, language)
        
        if not speech_result.get("success"):
            raise HTTPException(status_code=400, detail="Failed to process voice query")
        
        query_text = speech_result.get("transcription", "")
        
        # Process the query (this would integrate with knowledge base)
        response_text = await _process_disease_query(query_text, language, crop_type)
        
        # Convert response back to speech
        tts_request = TextToSpeechRequest(
            text=response_text,
            language=language,
            voice_type="female",
            speed=1.0
        )
        
        # Generate audio response
        audio_response = await convert_text_to_speech(tts_request)
        
        return {
            "success": True,
            "query_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "original_query": query_text,
            "response_text": response_text,
            "language": language,
            "audio_response_available": True
        }
        
    except Exception as e:
        logger.error(f"Voice query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Voice query failed: {str(e)}")


@router.get("/supported-languages")
async def get_supported_languages():
    """
    Get list of supported languages for voice processing.
    
    Returns:
        List of supported languages with their codes and names
    """
    return {
        "success": True,
        "languages": [
            {"code": "en", "name": "English", "native_name": "English"},
            {"code": "tw", "name": "Twi", "native_name": "Twi"},
            {"code": "ga", "name": "Ga", "native_name": "Ga"},
            {"code": "ha", "name": "Hausa", "native_name": "Hausa"},
            {"code": "ee", "name": "Ewe", "native_name": "Eʋegbe"}
        ]
    }


@router.get("/voice-settings")
async def get_voice_settings():
    """
    Get current voice processing settings and capabilities.
    
    Returns:
        Voice processing configuration and supported features
    """
    return {
        "success": True,
        "settings": {
            "sample_rate": settings.VOICE_SAMPLE_RATE,
            "chunk_size": settings.VOICE_CHUNK_SIZE,
            "timeout": settings.VOICE_TIMEOUT,
            "supported_formats": settings.ALLOWED_AUDIO_TYPES,
            "max_file_size": settings.MAX_FILE_SIZE,
            "voice_types": ["male", "female"],
            "speed_range": {"min": 0.5, "max": 2.0}
        }
    }


# Helper functions
async def _process_disease_query(query_text: str, language: str, crop_type: Optional[str]) -> str:
    """
    Process a natural language query about plant diseases.
    
    Args:
        query_text: The transcribed query text
        language: Language of the query
        crop_type: Optional crop type context
        
    Returns:
        Response text in the same language
    """
    # This would typically integrate with NLP module and knowledge base
    # For now, return a basic response
    
    if language == "en":
        return f"Thank you for your question about {crop_type or 'plant'} diseases. Based on your query '{query_text}', I recommend consulting with a local agricultural extension officer for specific advice. You can also upload an image of the affected plant for automated disease detection."
    elif language == "tw":
        return f"Meda wo ase wɔ wo bisabisa ho {crop_type or 'afifide'} yareɛ. Sɛ wohwɛ wo bisabisa '{query_text}' a, mekamfo wo sɛ wo ne kuayɛfoɔ adwumayɛfoɔ nkasa na woanya akwankyerɛ pɔtee. Wobɛtumi nso de afifide a ɛyare no mfonini aba ha na yɛahwɛ."
    else:
        return f"Thank you for your question. Please consult with local agricultural experts for specific advice about {crop_type or 'plant'} diseases."
