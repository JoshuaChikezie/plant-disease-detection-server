"""
Disease Detection API Routes

This module provides REST API endpoints for plant disease detection functionality.
Includes image upload, disease analysis, batch processing, and result visualization.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
import uuid
from datetime import datetime

from ..config import settings
from ...ai_module.computer_vision.disease_detector import DiseaseDetector
from ...ai_module.computer_vision.image_processor import ImageProcessor

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize AI components
disease_detector = DiseaseDetector()
image_processor = ImageProcessor()


# Pydantic models for request/response validation
class DiseaseDetectionRequest(BaseModel):
    """Request model for disease detection"""
    crop_type: Optional[str] = None
    confidence_threshold: Optional[float] = 0.7
    max_predictions: Optional[int] = 3
    include_visualization: Optional[bool] = False


class DiseaseDetectionResponse(BaseModel):
    """Response model for disease detection results"""
    success: bool
    detection_id: str
    timestamp: str
    image_info: Dict[str, Any]
    predictions: List[Dict[str, Any]]
    crop_type: str
    confidence_threshold: float
    processing_time: float
    recommendations: List[Dict[str, Any]]


class BatchDetectionRequest(BaseModel):
    """Request model for batch disease detection"""
    crop_type: Optional[str] = None
    confidence_threshold: Optional[float] = 0.7


@router.post("/detect", response_model=DiseaseDetectionResponse)
async def detect_disease(
    image: UploadFile = File(...),
    crop_type: Optional[str] = Form(None),
    confidence_threshold: Optional[float] = Form(0.7),
    max_predictions: Optional[int] = Form(3),
    include_visualization: Optional[bool] = Form(False)
):
    """
    Detect diseases in a plant image.
    
    Args:
        image: Uploaded image file (JPEG, PNG)
        crop_type: Type of crop (optional, will auto-detect if not provided)
        confidence_threshold: Minimum confidence for predictions (0.0-1.0)
        max_predictions: Maximum number of predictions to return
        include_visualization: Whether to generate visualization image
        
    Returns:
        Disease detection results with predictions and recommendations
    """
    start_time = datetime.now()
    detection_id = str(uuid.uuid4())
    
    try:
        # Validate file type
        if image.content_type not in settings.ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Allowed types: {settings.ALLOWED_IMAGE_TYPES}"
            )
        
        # Validate file size
        if image.size > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE} bytes"
            )
        
        # Save uploaded file temporarily
        upload_dir = os.path.join(settings.UPLOAD_DIR, "temp")
        os.makedirs(upload_dir, exist_ok=True)
        
        file_path = os.path.join(upload_dir, f"{detection_id}_{image.filename}")
        
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Perform disease detection
        results = disease_detector.detect_disease(
            image_path=file_path,
            crop_type=crop_type
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Get image information
        image_info = image_processor.get_image_info(file_path)
        
        # Generate recommendations based on predictions
        recommendations = []
        if results.get('success') and results.get('predictions'):
            for prediction in results['predictions'][:max_predictions]:
                if prediction['confidence'] >= confidence_threshold:
                    recommendation = await _generate_recommendations(
                        prediction['label'], 
                        crop_type or results.get('crop_type', 'unknown')
                    )
                    recommendations.append(recommendation)
        
        # Clean up temporary file
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return DiseaseDetectionResponse(
            success=results.get('success', False),
            detection_id=detection_id,
            timestamp=datetime.now().isoformat(),
            image_info=image_info,
            predictions=results.get('predictions', [])[:max_predictions],
            crop_type=crop_type or results.get('crop_type', 'unknown'),
            confidence_threshold=confidence_threshold,
            processing_time=processing_time,
            recommendations=recommendations
        )
        
    except Exception as e:
        logger.error(f"Disease detection failed for {detection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.post("/batch-detect")
async def batch_detect_diseases(
    images: List[UploadFile] = File(...),
    request: BatchDetectionRequest = Depends()
):
    """
    Detect diseases in multiple plant images.
    
    Args:
        images: List of uploaded image files
        request: Batch detection parameters
        
    Returns:
        List of detection results for each image
    """
    try:
        if len(images) > 10:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 images allowed per batch request"
            )
        
        results = []
        
        for image in images:
            try:
                # Process each image individually
                result = await detect_disease(
                    image=image,
                    crop_type=request.crop_type,
                    confidence_threshold=request.confidence_threshold
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process image {image.filename}: {str(e)}")
                results.append({
                    "success": False,
                    "filename": image.filename,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "batch_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "total_images": len(images),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch detection failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@router.get("/supported-crops")
async def get_supported_crops():
    """
    Get list of supported crop types for disease detection.
    
    Returns:
        List of supported crops with their scientific names
    """
    return {
        "success": True,
        "crops": [
            {"name": "cassava", "scientific_name": "Manihot esculenta", "local_names": ["bankye", "rogo"]},
            {"name": "maize", "scientific_name": "Zea mays", "local_names": ["aburo", "masara"]},
            {"name": "cocoa", "scientific_name": "Theobroma cacao", "local_names": ["koko", "kwakuo"]},
            {"name": "yam", "scientific_name": "Dioscorea spp.", "local_names": ["bayere", "ewa"]},
            {"name": "plantain", "scientific_name": "Musa spp.", "local_names": ["borode", "ayaba"]},
            {"name": "rice", "scientific_name": "Oryza sativa", "local_names": ["emo", "shinkafa"]},
            {"name": "millet", "scientific_name": "Pennisetum glaucum", "local_names": ["awa", "gero"]},
            {"name": "sorghum", "scientific_name": "Sorghum bicolor", "local_names": ["aburoo", "dawa"]},
            {"name": "groundnut", "scientific_name": "Arachis hypogaea", "local_names": ["nkatee", "gyedu"]},
            {"name": "cowpea", "scientific_name": "Vigna unguiculata", "local_names": ["adua", "wake"]}
        ]
    }


@router.get("/diseases/{crop_type}")
async def get_crop_diseases(crop_type: str):
    """
    Get list of diseases for a specific crop type.
    
    Args:
        crop_type: Type of crop to get diseases for
        
    Returns:
        List of diseases that affect the specified crop
    """
    try:
        if crop_type not in settings.SUPPORTED_CROPS:
            raise HTTPException(
                status_code=404,
                detail=f"Crop type '{crop_type}' not supported"
            )
        
        # Get disease information from detector
        diseases = disease_detector.get_crop_diseases(crop_type)
        
        return {
            "success": True,
            "crop_type": crop_type,
            "diseases": diseases,
            "total_diseases": len(diseases)
        }
        
    except Exception as e:
        logger.error(f"Failed to get diseases for {crop_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/disease-info/{disease_name}")
async def get_disease_information(
    disease_name: str,
    crop_type: str,
    language: Optional[str] = "en"
):
    """
    Get detailed information about a specific disease.
    
    Args:
        disease_name: Name of the disease
        crop_type: Type of crop affected
        language: Language for response (en, tw, ga, ha, ee)
        
    Returns:
        Detailed disease information including symptoms, treatments, prevention
    """
    try:
        # Get disease information
        disease_info = disease_detector.get_disease_info(disease_name, crop_type)
        
        if not disease_info:
            raise HTTPException(
                status_code=404,
                detail=f"Disease '{disease_name}' not found for crop '{crop_type}'"
            )
        
        return {
            "success": True,
            "disease_name": disease_name,
            "crop_type": crop_type,
            "language": language,
            "information": disease_info
        }
        
    except Exception as e:
        logger.error(f"Failed to get disease info for {disease_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/detection-history/{user_id}")
async def get_detection_history(
    user_id: str,
    limit: Optional[int] = 20,
    offset: Optional[int] = 0
):
    """
    Get detection history for a specific user.
    
    Args:
        user_id: User identifier
        limit: Maximum number of results to return
        offset: Number of results to skip
        
    Returns:
        List of previous detection results for the user
    """
    try:
        # This would typically fetch from a database
        # For now, return a placeholder response
        return {
            "success": True,
            "user_id": user_id,
            "total_detections": 0,
            "history": [],
            "pagination": {
                "limit": limit,
                "offset": offset,
                "has_more": False
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get detection history for {user_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/detection/{detection_id}")
async def delete_detection(detection_id: str):
    """
    Delete a specific detection record.
    
    Args:
        detection_id: Unique identifier for the detection
        
    Returns:
        Confirmation of deletion
    """
    try:
        # This would typically delete from a database
        # For now, return a placeholder response
        return {
            "success": True,
            "detection_id": detection_id,
            "message": "Detection record deleted successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete detection {detection_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Helper functions
async def _generate_recommendations(disease_name: str, crop_type: str) -> Dict[str, Any]:
    """
    Generate treatment and prevention recommendations for a detected disease.
    
    Args:
        disease_name: Name of the detected disease
        crop_type: Type of crop affected
        
    Returns:
        Dictionary containing recommendations
    """
    # This would typically fetch from knowledge base
    # For now, return basic recommendations structure
    return {
        "disease": disease_name,
        "crop": crop_type,
        "severity": "moderate",
        "immediate_actions": [
            "Remove affected leaves/parts",
            "Isolate infected plants",
            "Improve drainage if applicable"
        ],
        "treatments": {
            "organic": [
                "Neem oil spray",
                "Baking soda solution",
                "Compost tea application"
            ],
            "chemical": [
                "Copper-based fungicide",
                "Systemic fungicide (as per local availability)"
            ]
        },
        "prevention": [
            "Proper plant spacing",
            "Regular field inspection",
            "Crop rotation practices",
            "Soil health management"
        ],
        "local_suppliers": [
            "Contact local agricultural extension office",
            "Visit nearest agro-chemical shop"
        ]
    }
