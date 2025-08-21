#!/usr/bin/env python3
"""
Simple Plant Disease Detection Server (No TensorFlow Required)
This is a basic version to test server functionality without AI dependencies
"""

import os
import logging
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from typing import Optional, List, Dict, Any
import json
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Simple Plant Disease Detection API",
    description="Basic API for plant disease detection (AI features disabled)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mock disease detector class
class MockDiseaseDetector:
    def __init__(self):
        self.disease_classes = {
            'maize': {
                'healthy': 'Healthy Maize',
                'maize_rust': 'Maize Rust (Puccinia sorghi)',
                'maize_smut': 'Maize Smut (Ustilago maydis)',
                'maize_leaf_blight': 'Northern Corn Leaf Blight (Exserohilum turcicum)',
                'maize_gray_leaf_spot': 'Gray Leaf Spot (Cercospora zeae-maydis)',
                'maize_common_rust': 'Common Rust (Puccinia sorghi)',
                'maize_southern_leaf_blight': 'Southern Leaf Blight (Bipolaris maydis)',
                'maize_anthracnose': 'Anthracnose Leaf Blight (Colletotrichum graminicola)',
                'maize_eyespot': 'Eyespot (Kabatiella zeae)',
                'maize_stewart_wilt': 'Stewart\'s Wilt (Pantoea stewartii)'
            }
        }
    
    def detect_disease(self, image_path: str, crop_type: str = "maize", **kwargs):
        """Mock disease detection - returns simulated results"""
        import random
        
        # Simulate processing time
        import time
        time.sleep(0.5)
        
        # Generate mock predictions
        diseases = list(self.disease_classes.get(crop_type, {}).keys())
        selected_disease = random.choice(diseases)
        
        return {
            "success": True,
            "detection_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "image_info": {
                "width": 800,
                "height": 600,
                "file_size": 1024000,
                "format": "JPEG"
            },
            "predictions": [
                {
                    "disease": selected_disease,
                    "label": self.disease_classes[crop_type][selected_disease],
                    "confidence": round(random.uniform(0.7, 0.95), 3),
                    "rank": 1,
                    "severity_level": random.choice(['mild', 'moderate', 'severe']),
                    "treatment_urgency": random.choice(['low', 'moderate', 'urgent'])
                }
            ],
            "crop_type": crop_type,
            "confidence_threshold": 0.7,
            "processing_time": 0.5,
            "recommendations": [
                {
                    "disease": selected_disease,
                    "crop": crop_type,
                    "severity": "moderate",
                    "immediate_actions": ["Remove affected leaves", "Improve air circulation"],
                    "treatments": {
                        "organic": ["Neem oil application", "Copper-based fungicide"],
                        "chemical": ["Fungicide treatment", "Systemic protection"]
                    },
                    "prevention": ["Crop rotation", "Proper spacing", "Regular monitoring"],
                    "local_suppliers": ["Local agricultural store", "Online suppliers"]
                }
            ],
            "severity_assessment": {
                "overall_severity": "moderate",
                "average_confidence": 0.85,
                "max_confidence": 0.92,
                "recommendations": ["Monitor closely", "Apply preventive measures"]
            },
            "model_architecture": "mock_cnn",
            "processing_timestamp": datetime.now().isoformat()
        }

# Initialize mock detector
disease_detector = MockDiseaseDetector()

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Simple Plant Disease Detection API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "ai_available": False,
        "message": "Server is running (AI features disabled)"
    }

@app.post("/api/v1/disease/detect")
async def detect_disease(
    image: UploadFile = File(...),
    crop_type: Optional[str] = Form("maize"),
    confidence_threshold: Optional[float] = Form(0.7),
    max_predictions: Optional[int] = Form(3),
    include_visualization: Optional[bool] = Form(False)
):
    """Detect plant disease from uploaded image"""
    try:
        # Validate file type
        if not image.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Create upload directory if it doesn't exist
        upload_dir = "uploads"
        os.makedirs(upload_dir, exist_ok=True)
        
        # Save uploaded image
        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{image.filename}")
        with open(file_path, "wb") as buffer:
            content = await image.read()
            buffer.write(content)
        
        # Mock disease detection
        result = disease_detector.detect_disease(
            file_path, 
            crop_type=crop_type,
            confidence_threshold=confidence_threshold,
            max_predictions=max_predictions
        )
        
        # Clean up uploaded file
        try:
            os.remove(file_path)
        except:
            pass
        
        return result
        
    except Exception as e:
        logger.error(f"Error in disease detection: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/api/v1/disease/supported-crops")
async def get_supported_crops():
    """Get list of supported crops"""
    return {
        "crops": list(disease_detector.disease_classes.keys()),
        "maize_diseases": list(disease_detector.disease_classes.get("maize", {}).keys())
    }

@app.get("/api/v1/disease/model-info")
async def get_model_info():
    """Get model information"""
    return {
        "model_name": "Mock Disease Detection Model",
        "version": "1.0.0",
        "architecture": "mock_cnn",
        "supported_crops": list(disease_detector.disease_classes.keys()),
        "maize_specific": True,
        "ai_available": False,
        "message": "This is a mock model for testing purposes"
    }

@app.get("/api/v1/disease/diseases")
async def get_diseases(crop_type: str = "maize"):
    """Get diseases for a specific crop"""
    if crop_type not in disease_detector.disease_classes:
        raise HTTPException(status_code=400, detail=f"Crop type '{crop_type}' not supported")
    
    return {
        "crop_type": crop_type,
        "diseases": disease_detector.disease_classes[crop_type]
    }

if __name__ == "__main__":
    logger.info("Starting Simple Plant Disease Detection Server...")
    logger.info("Note: AI features are disabled - using mock detection")
    
    uvicorn.run(
        "simple_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
