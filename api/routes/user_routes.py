"""
User Management API Routes

This module provides REST API endpoints for user management functionality.
Includes user registration, authentication, profile management, and feedback collection.
"""

import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
import uuid
from datetime import datetime, timedelta

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Create router instance
router = APIRouter()

# Security scheme
security = HTTPBearer()


# Pydantic models for request/response validation
class UserRegistration(BaseModel):
    """User registration request model"""
    username: str
    email: EmailStr
    password: str
    full_name: str
    phone_number: Optional[str] = None
    location: Optional[str] = None
    farm_size: Optional[float] = None
    primary_crops: Optional[list] = []
    preferred_language: Optional[str] = "en"


class UserLogin(BaseModel):
    """User login request model"""
    username: str
    password: str


class UserProfile(BaseModel):
    """User profile update model"""
    full_name: Optional[str] = None
    phone_number: Optional[str] = None
    location: Optional[str] = None
    farm_size: Optional[float] = None
    primary_crops: Optional[list] = None
    preferred_language: Optional[str] = None


class FeedbackSubmission(BaseModel):
    """User feedback submission model"""
    detection_id: str
    feedback_type: str  # correct, incorrect, partially_correct
    correct_disease: Optional[str] = None
    comments: Optional[str] = None
    rating: Optional[int] = None  # 1-5 stars


@router.post("/register")
async def register_user(user_data: UserRegistration):
    """
    Register a new user account.
    
    Args:
        user_data: User registration information
        
    Returns:
        User account creation confirmation with user ID
    """
    try:
        # Check if username or email already exists
        # This would typically check against a database
        
        user_id = str(uuid.uuid4())
        
        # Create user record (placeholder implementation)
        user_record = {
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "full_name": user_data.full_name,
            "phone_number": user_data.phone_number,
            "location": user_data.location,
            "farm_size": user_data.farm_size,
            "primary_crops": user_data.primary_crops,
            "preferred_language": user_data.preferred_language,
            "created_at": datetime.now().isoformat(),
            "is_active": True,
            "email_verified": False
        }
        
        return {
            "success": True,
            "message": "User registered successfully",
            "user_id": user_id,
            "username": user_data.username,
            "email": user_data.email,
            "next_steps": [
                "Verify your email address",
                "Complete your profile setup",
                "Start using disease detection features"
            ]
        }
        
    except Exception as e:
        logger.error(f"User registration failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Registration failed: {str(e)}")


@router.post("/login")
async def login_user(login_data: UserLogin):
    """
    Authenticate user login.
    
    Args:
        login_data: User login credentials
        
    Returns:
        Authentication token and user information
    """
    try:
        # Validate credentials (placeholder implementation)
        # This would typically verify against a database
        
        # Generate access token (placeholder)
        access_token = str(uuid.uuid4())
        
        return {
            "success": True,
            "message": "Login successful",
            "access_token": access_token,
            "token_type": "bearer",
            "expires_in": settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
            "user_info": {
                "user_id": "placeholder_user_id",
                "username": login_data.username,
                "preferred_language": "en"
            }
        }
        
    except Exception as e:
        logger.error(f"User login failed: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid credentials")


@router.get("/profile/{user_id}")
async def get_user_profile(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get user profile information.
    
    Args:
        user_id: User identifier
        credentials: Authentication credentials
        
    Returns:
        User profile data
    """
    try:
        # Verify token and get user data (placeholder implementation)
        
        profile_data = {
            "user_id": user_id,
            "username": "farmer_user",
            "email": "farmer@example.com",
            "full_name": "John Farmer",
            "phone_number": "+233123456789",
            "location": "Kumasi, Ghana",
            "farm_size": 2.5,
            "primary_crops": ["cassava", "maize", "cocoa"],
            "preferred_language": "en",
            "created_at": "2024-01-01T00:00:00Z",
            "last_login": datetime.now().isoformat(),
            "detection_count": 15,
            "feedback_count": 3
        }
        
        return {
            "success": True,
            "profile": profile_data
        }
        
    except Exception as e:
        logger.error(f"Failed to get user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/profile/{user_id}")
async def update_user_profile(
    user_id: str,
    profile_data: UserProfile,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Update user profile information.
    
    Args:
        user_id: User identifier
        profile_data: Updated profile information
        credentials: Authentication credentials
        
    Returns:
        Updated profile confirmation
    """
    try:
        # Verify token and update user data (placeholder implementation)
        
        return {
            "success": True,
            "message": "Profile updated successfully",
            "user_id": user_id,
            "updated_fields": [
                field for field, value in profile_data.dict().items() 
                if value is not None
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to update user profile: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/feedback")
async def submit_user_feedback(
    feedback_data: FeedbackSubmission,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Submit user feedback on disease detection results.
    
    Args:
        feedback_data: Feedback information
        credentials: Authentication credentials
        
    Returns:
        Feedback submission confirmation
    """
    try:
        feedback_id = str(uuid.uuid4())
        
        # Store feedback (placeholder implementation)
        feedback_record = {
            "feedback_id": feedback_id,
            "detection_id": feedback_data.detection_id,
            "feedback_type": feedback_data.feedback_type,
            "correct_disease": feedback_data.correct_disease,
            "comments": feedback_data.comments,
            "rating": feedback_data.rating,
            "submitted_at": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "message": "Feedback submitted successfully",
            "feedback_id": feedback_id,
            "points_earned": 10,  # Reward system for feedback
            "thank_you_message": "Thank you for helping improve our disease detection system!"
        }
        
    except Exception as e:
        logger.error(f"Failed to submit feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/statistics/{user_id}")
async def get_user_statistics(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Get user usage statistics and achievements.
    
    Args:
        user_id: User identifier
        credentials: Authentication credentials
        
    Returns:
        User statistics and achievements
    """
    try:
        # Get user statistics (placeholder implementation)
        stats = {
            "user_id": user_id,
            "total_detections": 25,
            "successful_detections": 22,
            "accuracy_rate": 0.88,
            "crops_analyzed": ["cassava", "maize", "cocoa"],
            "diseases_detected": [
                "cassava_mosaic_disease",
                "maize_rust", 
                "cocoa_black_pod"
            ],
            "feedback_submitted": 5,
            "points_earned": 150,
            "achievements": [
                "First Detection",
                "Disease Expert",
                "Feedback Champion"
            ],
            "monthly_usage": {
                "current_month": 8,
                "last_month": 12,
                "average": 10
            }
        }
        
        return {
            "success": True,
            "statistics": stats,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get user statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/account/{user_id}")
async def delete_user_account(
    user_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """
    Delete user account and associated data.
    
    Args:
        user_id: User identifier
        credentials: Authentication credentials
        
    Returns:
        Account deletion confirmation
    """
    try:
        # Delete user account and data (placeholder implementation)
        
        return {
            "success": True,
            "message": "Account deleted successfully",
            "user_id": user_id,
            "deleted_at": datetime.now().isoformat(),
            "data_retention": "Anonymized usage data may be retained for system improvement"
        }
        
    except Exception as e:
        logger.error(f"Failed to delete user account: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout")
async def logout_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Logout user and invalidate token.
    
    Args:
        credentials: Authentication credentials
        
    Returns:
        Logout confirmation
    """
    try:
        # Invalidate token (placeholder implementation)
        
        return {
            "success": True,
            "message": "Logged out successfully",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Logout failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
