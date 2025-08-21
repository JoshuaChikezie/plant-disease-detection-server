"""
Authentication Middleware

This module provides authentication middleware for the Plant Disease Detection API.
Handles JWT token validation, user authentication, and request authorization.
"""

import logging
import jwt
from typing import Optional
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)

# Security scheme
security = HTTPBearer()


class AuthMiddleware(BaseHTTPMiddleware):
    """
    Custom authentication middleware for JWT token validation.
    
    This middleware:
    - Validates JWT tokens on protected endpoints
    - Adds user information to request state
    - Handles authentication errors gracefully
    - Logs authentication attempts
    """
    
    def __init__(self, app, secret_key: str = None):
        """
        Initialize authentication middleware.
        
        Args:
            app: FastAPI application instance
            secret_key: JWT secret key for token validation
        """
        super().__init__(app)
        self.secret_key = secret_key or settings.SECRET_KEY
        self.algorithm = settings.ALGORITHM
        
        # Define public endpoints that don't require authentication
        self.public_endpoints = {
            "/",
            "/health",
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/v1/info",
            "/api/v1/disease/supported-crops",
            "/api/v1/voice/supported-languages",
            "/api/v1/knowledge/categories",
            "/api/v1/user/register",
            "/api/v1/user/login"
        }
    
    async def dispatch(self, request: Request, call_next):
        """
        Process incoming requests and validate authentication.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response
        """
        start_time = time.time()
        
        try:
            # Check if endpoint requires authentication
            if self._is_public_endpoint(request.url.path):
                # Skip authentication for public endpoints
                response = await call_next(request)
                return response
            
            # Extract and validate JWT token
            token = self._extract_token(request)
            if not token:
                return self._create_auth_error("Missing authentication token")
            
            # Decode and validate token
            user_data = self._validate_token(token)
            if not user_data:
                return self._create_auth_error("Invalid or expired token")
            
            # Add user information to request state
            request.state.user = user_data
            request.state.user_id = user_data.get("user_id")
            request.state.username = user_data.get("username")
            
            # Log successful authentication
            logger.info(f"Authenticated user {user_data.get('username')} for {request.method} {request.url.path}")
            
            # Continue to next middleware/endpoint
            response = await call_next(request)
            
            # Log request completion
            process_time = time.time() - start_time
            logger.info(f"Request completed in {process_time:.3f}s for user {user_data.get('username')}")
            
            return response
            
        except HTTPException as e:
            logger.warning(f"Authentication failed for {request.method} {request.url.path}: {e.detail}")
            return self._create_auth_error(e.detail, e.status_code)
            
        except Exception as e:
            logger.error(f"Authentication middleware error: {str(e)}")
            return self._create_auth_error("Authentication service unavailable", 503)
    
    def _is_public_endpoint(self, path: str) -> bool:
        """
        Check if an endpoint is public (doesn't require authentication).
        
        Args:
            path: Request URL path
            
        Returns:
            True if endpoint is public, False otherwise
        """
        # Check exact matches
        if path in self.public_endpoints:
            return True
        
        # Check path prefixes for static files and docs
        public_prefixes = ["/static", "/docs", "/redoc"]
        return any(path.startswith(prefix) for prefix in public_prefixes)
    
    def _extract_token(self, request: Request) -> Optional[str]:
        """
        Extract JWT token from request headers.
        
        Args:
            request: HTTP request object
            
        Returns:
            JWT token string or None if not found
        """
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            return auth_header.split(" ")[1]
        
        # Check query parameter (for WebSocket connections)
        token = request.query_params.get("token")
        if token:
            return token
        
        return None
    
    def _validate_token(self, token: str) -> Optional[dict]:
        """
        Validate JWT token and extract user data.
        
        Args:
            token: JWT token string
            
        Returns:
            User data dictionary or None if invalid
        """
        try:
            # Decode JWT token
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm]
            )
            
            # Check token expiration
            exp_timestamp = payload.get("exp")
            if exp_timestamp and time.time() > exp_timestamp:
                logger.warning("Token has expired")
                return None
            
            # Extract user information
            user_data = {
                "user_id": payload.get("user_id"),
                "username": payload.get("username"),
                "email": payload.get("email"),
                "role": payload.get("role", "user"),
                "exp": exp_timestamp
            }
            
            return user_data
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token has expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Token validation error: {str(e)}")
            return None
    
    def _create_auth_error(self, message: str, status_code: int = 401) -> JSONResponse:
        """
        Create standardized authentication error response.
        
        Args:
            message: Error message
            status_code: HTTP status code
            
        Returns:
            JSON error response
        """
        return JSONResponse(
            status_code=status_code,
            content={
                "error": "Authentication failed",
                "message": message,
                "status_code": status_code,
                "timestamp": time.time()
            }
        )


def create_jwt_token(user_data: dict, expires_delta: Optional[int] = None) -> str:
    """
    Create a JWT token for user authentication.
    
    Args:
        user_data: User information to encode in token
        expires_delta: Token expiration time in minutes
        
    Returns:
        Encoded JWT token string
    """
    try:
        # Set expiration time
        expire_minutes = expires_delta or settings.ACCESS_TOKEN_EXPIRE_MINUTES
        expire_timestamp = time.time() + (expire_minutes * 60)
        
        # Create token payload
        payload = {
            "user_id": user_data.get("user_id"),
            "username": user_data.get("username"),
            "email": user_data.get("email"),
            "role": user_data.get("role", "user"),
            "exp": expire_timestamp,
            "iat": time.time()
        }
        
        # Encode token
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        
        logger.info(f"JWT token created for user {user_data.get('username')}")
        return token
        
    except Exception as e:
        logger.error(f"Failed to create JWT token: {str(e)}")
        raise HTTPException(status_code=500, detail="Token creation failed")


def verify_token(token: str) -> Optional[dict]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded user data or None if invalid
    """
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM]
        )
        return payload
        
    except jwt.ExpiredSignatureError:
        logger.warning("Token has expired")
        return None
    except jwt.InvalidTokenError:
        logger.warning("Invalid token")
        return None
