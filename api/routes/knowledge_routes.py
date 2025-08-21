"""
Knowledge Base API Routes

This module provides REST API endpoints for accessing agricultural knowledge base.
Includes disease information, treatment guides, prevention methods, and farmer resources.
"""

import logging
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime

from ..config import settings
from ...ai_module.knowledge_base.knowledge_base import KnowledgeBase

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Initialize knowledge base
knowledge_base = KnowledgeBase()


# Pydantic models for request/response validation
class SearchRequest(BaseModel):
    """Request model for knowledge base search"""
    query: str
    language: Optional[str] = "en"
    crop_type: Optional[str] = None
    category: Optional[str] = None  # symptoms, treatments, prevention, etc.


class ArticleRequest(BaseModel):
    """Request model for specific article retrieval"""
    article_id: str
    language: Optional[str] = "en"


@router.get("/search")
async def search_knowledge_base(
    query: str = Query(..., description="Search query"),
    language: Optional[str] = Query("en", description="Language code"),
    crop_type: Optional[str] = Query(None, description="Filter by crop type"),
    category: Optional[str] = Query(None, description="Filter by category"),
    limit: Optional[int] = Query(10, description="Maximum results to return")
):
    """
    Search the agricultural knowledge base.
    
    Args:
        query: Search query string
        language: Language for results (en, tw, ga, ha, ee)
        crop_type: Filter results by crop type
        category: Filter by category (symptoms, treatments, prevention, etc.)
        limit: Maximum number of results to return
        
    Returns:
        Search results with relevant articles and information
    """
    try:
        # Perform knowledge base search
        results = knowledge_base.search(
            query=query,
            language=language,
            crop_type=crop_type,
            category=category,
            limit=limit
        )
        
        return {
            "success": True,
            "query": query,
            "language": language,
            "filters": {
                "crop_type": crop_type,
                "category": category
            },
            "total_results": len(results),
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Knowledge base search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/diseases")
async def get_all_diseases(
    crop_type: Optional[str] = Query(None, description="Filter by crop type"),
    language: Optional[str] = Query("en", description="Language code")
):
    """
    Get comprehensive list of plant diseases.
    
    Args:
        crop_type: Filter diseases by crop type
        language: Language for disease names and descriptions
        
    Returns:
        List of diseases with basic information
    """
    try:
        diseases = knowledge_base.get_diseases(crop_type=crop_type, language=language)
        
        return {
            "success": True,
            "crop_type": crop_type,
            "language": language,
            "total_diseases": len(diseases),
            "diseases": diseases
        }
        
    except Exception as e:
        logger.error(f"Failed to get diseases: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/treatments/{disease_name}")
async def get_disease_treatments(
    disease_name: str,
    crop_type: str = Query(..., description="Crop type affected by disease"),
    language: Optional[str] = Query("en", description="Language code"),
    treatment_type: Optional[str] = Query(None, description="Filter by treatment type (organic, chemical, cultural)")
):
    """
    Get treatment options for a specific disease.
    
    Args:
        disease_name: Name of the disease
        crop_type: Type of crop affected
        language: Language for treatment descriptions
        treatment_type: Filter by treatment type
        
    Returns:
        Detailed treatment information and recommendations
    """
    try:
        treatments = knowledge_base.get_treatments(
            disease_name=disease_name,
            crop_type=crop_type,
            language=language,
            treatment_type=treatment_type
        )
        
        return {
            "success": True,
            "disease_name": disease_name,
            "crop_type": crop_type,
            "language": language,
            "treatment_type": treatment_type,
            "treatments": treatments
        }
        
    except Exception as e:
        logger.error(f"Failed to get treatments for {disease_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/prevention/{crop_type}")
async def get_prevention_methods(
    crop_type: str,
    language: Optional[str] = Query("en", description="Language code"),
    season: Optional[str] = Query(None, description="Growing season (dry, wet)")
):
    """
    Get disease prevention methods for a specific crop.
    
    Args:
        crop_type: Type of crop
        language: Language for prevention descriptions
        season: Growing season context
        
    Returns:
        Prevention methods and best practices
    """
    try:
        prevention_methods = knowledge_base.get_prevention_methods(
            crop_type=crop_type,
            language=language,
            season=season
        )
        
        return {
            "success": True,
            "crop_type": crop_type,
            "language": language,
            "season": season,
            "prevention_methods": prevention_methods
        }
        
    except Exception as e:
        logger.error(f"Failed to get prevention methods for {crop_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/article/{article_id}")
async def get_article(
    article_id: str,
    language: Optional[str] = Query("en", description="Language code")
):
    """
    Get a specific knowledge base article.
    
    Args:
        article_id: Unique identifier for the article
        language: Language for article content
        
    Returns:
        Full article content with metadata
    """
    try:
        article = knowledge_base.get_article(article_id=article_id, language=language)
        
        if not article:
            raise HTTPException(
                status_code=404,
                detail=f"Article '{article_id}' not found"
            )
        
        return {
            "success": True,
            "article_id": article_id,
            "language": language,
            "article": article
        }
        
    except Exception as e:
        logger.error(f"Failed to get article {article_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/categories")
async def get_knowledge_categories(
    language: Optional[str] = Query("en", description="Language code")
):
    """
    Get available knowledge base categories.
    
    Args:
        language: Language for category names
        
    Returns:
        List of available categories with descriptions
    """
    return {
        "success": True,
        "language": language,
        "categories": [
            {
                "id": "symptoms",
                "name": "Disease Symptoms" if language == "en" else "Yareɛ ho nsɛnkyerɛnne",
                "description": "Visual and physical symptoms of plant diseases"
            },
            {
                "id": "treatments",
                "name": "Treatments" if language == "en" else "Aduro",
                "description": "Treatment options including organic and chemical methods"
            },
            {
                "id": "prevention",
                "name": "Prevention" if language == "en" else "Ammabubu",
                "description": "Preventive measures and best practices"
            },
            {
                "id": "farming_practices",
                "name": "Farming Practices" if language == "en" else "Kuayɛ akwan",
                "description": "General agricultural practices and techniques"
            },
            {
                "id": "pest_management",
                "name": "Pest Management" if language == "en" else "Mmoawa ho dwumadi",
                "description": "Integrated pest management strategies"
            }
        ]
    }


@router.get("/recent-updates")
async def get_recent_updates(
    days: Optional[int] = Query(7, description="Number of days to look back"),
    language: Optional[str] = Query("en", description="Language code")
):
    """
    Get recently updated knowledge base content.
    
    Args:
        days: Number of days to look back for updates
        language: Language for content
        
    Returns:
        List of recently updated articles and information
    """
    try:
        updates = knowledge_base.get_recent_updates(days=days, language=language)
        
        return {
            "success": True,
            "days_back": days,
            "language": language,
            "total_updates": len(updates),
            "updates": updates,
            "last_checked": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get recent updates: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_knowledge_feedback(
    article_id: str,
    feedback_type: str,  # helpful, not_helpful, incorrect, outdated
    comments: Optional[str] = None,
    user_id: Optional[str] = None
):
    """
    Submit feedback on knowledge base content.
    
    Args:
        article_id: ID of the article being reviewed
        feedback_type: Type of feedback (helpful, not_helpful, incorrect, outdated)
        comments: Optional additional comments
        user_id: Optional user identifier
        
    Returns:
        Confirmation of feedback submission
    """
    try:
        feedback_id = knowledge_base.submit_feedback(
            article_id=article_id,
            feedback_type=feedback_type,
            comments=comments,
            user_id=user_id
        )
        
        return {
            "success": True,
            "feedback_id": feedback_id,
            "article_id": article_id,
            "message": "Feedback submitted successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
