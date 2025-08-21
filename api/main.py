"""
Main FastAPI Application for Plant Disease Detection System

This module provides the main API server with endpoints for:
- Disease detection and analysis
- Voice processing and synthesis
- Knowledge base queries
- User management and feedback
"""

import os
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import uvicorn
from contextlib import asynccontextmanager
from datetime import datetime

from .routes import disease_routes, voice_routes, knowledge_routes, user_routes
from .middleware import auth_middleware, logging_middleware
from .config import settings

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(settings.LOG_FILE) if settings.LOG_FILE else logging.StreamHandler(),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown tasks."""
    # Startup
    logger.info("Starting Plant Disease Detection API Server...")
    
    try:
        # Create necessary directories
        os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)
        os.makedirs("static", exist_ok=True)
        
        # Initialize AI components
        from ai_module.computer_vision.disease_detector import DiseaseDetector
        from ai_module.voice.voice_interface import VoiceInterface
        from ai_module.knowledge_base.knowledge_base import KnowledgeBase
        
        # Store components in app state for access across routes
        app.state.disease_detector = DiseaseDetector()
        app.state.voice_interface = VoiceInterface()
        app.state.knowledge_base = KnowledgeBase()
        
        logger.info("AI components initialized successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize AI components: {e}")
        # Continue startup even if AI components fail to load
    
    yield
    
    # Shutdown
    logger.info("Shutting down Plant Disease Detection API Server...")
    
    # Close database connections
    if hasattr(app.state, 'knowledge_base'):
        app.state.knowledge_base.close()


# Create FastAPI application
app = FastAPI(
    title="Plant Disease Detection API",
    description="AI-powered plant disease detection system for Ghanaian farmers",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
    contact={
        "name": "Plant Disease Detection Team",
        "email": "support@plant-disease-detection.gh",
    },
    license_info={
        "name": "MIT License",
        "url": "https://opensource.org/licenses/MIT",
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Add custom middleware
app.add_middleware(logging_middleware.LoggingMiddleware)
app.add_middleware(auth_middleware.AuthMiddleware)

# Mount static files (create directory if it doesn't exist)
static_dir = "static"
if not os.path.exists(static_dir):
    os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include API routers with proper prefixes and tags
app.include_router(
    disease_routes.router, 
    prefix="/api/v1/disease", 
    tags=["Disease Detection"]
)
app.include_router(
    voice_routes.router, 
    prefix="/api/v1/voice", 
    tags=["Voice Processing"]
)
app.include_router(
    knowledge_routes.router, 
    prefix="/api/v1/knowledge", 
    tags=["Knowledge Base"]
)
app.include_router(
    user_routes.router, 
    prefix="/api/v1/user", 
    tags=["User Management"]
)


@app.get("/")
async def root():
    """Root endpoint providing API information."""
    return {
        "message": "Plant Disease Detection API",
        "version": "1.0.0",
        "status": "running",
        "description": "AI-powered plant disease detection for Ghanaian farmers",
        "documentation": "/docs",
        "health_check": "/health",
        "api_info": "/api/v1/info"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring service status."""
    try:
        # Check AI components status
        services_status = {
            "api": "operational",
            "database": "operational"
        }
        
        # Check if AI components are loaded
        if hasattr(app.state, 'disease_detector'):
            services_status["disease_detection"] = "operational"
        else:
            services_status["disease_detection"] = "unavailable"
            
        if hasattr(app.state, 'voice_interface'):
            services_status["voice_processing"] = "operational"
        else:
            services_status["voice_processing"] = "unavailable"
            
        if hasattr(app.state, 'knowledge_base'):
            services_status["knowledge_base"] = "operational"
        else:
            services_status["knowledge_base"] = "unavailable"
        
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "services": services_status,
            "uptime": "running",
            "version": "1.0.0"
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )


@app.get("/api/v1/info")
async def api_info():
    """Comprehensive API information endpoint."""
    return {
        "name": "Plant Disease Detection API",
        "version": "1.0.0",
        "description": "Comprehensive AI system for plant disease detection",
        "features": [
            "Image-based disease detection using CNN models",
            "Multilingual voice interface (English, Twi, Ga, Hausa, Ewe)",
            "Comprehensive agricultural knowledge base",
            "Local language support for Ghanaian farmers",
            "Farmer feedback integration for continuous learning",
            "Treatment and prevention recommendations"
        ],
        "supported_languages": [
            {"code": "en", "name": "English"},
            {"code": "tw", "name": "Twi"},
            {"code": "ga", "name": "Ga"},
            {"code": "ha", "name": "Hausa"},
            {"code": "ee", "name": "Ewe"}
        ],
        "supported_crops": [
            "Cassava", "Maize", "Cocoa", "Yam", "Plantain",
            "Rice", "Millet", "Sorghum", "Groundnut", "Cowpea"
        ],
        "api_endpoints": {
            "disease_detection": "/api/v1/disease/",
            "voice_processing": "/api/v1/voice/",
            "knowledge_base": "/api/v1/knowledge/",
            "user_management": "/api/v1/user/"
        },
        "contact": {
            "email": "support@plant-disease-detection.gh",
            "website": "https://plant-disease-detection.gh"
        },
        "documentation": "/docs",
        "health_check": "/health"
    }


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Global HTTP exception handler with detailed error information."""
    logger.warning(f"HTTP {exc.status_code} error on {request.method} {request.url.path}: {exc.detail}")
    
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled errors."""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "status_code": 500,
            "path": str(request.url.path),
            "method": request.method,
            "timestamp": datetime.now().isoformat(),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )


if __name__ == "__main__":
    # Run the application with proper configuration
    uvicorn.run(
        "api.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )