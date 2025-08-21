"""
Logging Middleware

This module provides comprehensive logging middleware for the Plant Disease Detection API.
Handles request/response logging, performance monitoring, and error tracking.
"""

import logging
import time
import json
from typing import Dict, Any
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime
import uuid

from ..config import settings

# Configure logging
logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Custom logging middleware for comprehensive request/response tracking.
    
    This middleware:
    - Logs all incoming requests with details
    - Tracks response times and status codes
    - Monitors API usage patterns
    - Records errors and exceptions
    - Provides structured logging for analysis
    """
    
    def __init__(self, app):
        """
        Initialize logging middleware.
        
        Args:
            app: FastAPI application instance
        """
        super().__init__(app)
        self.sensitive_headers = {
            "authorization", "cookie", "x-api-key", 
            "x-auth-token", "x-access-token"
        }
        
    async def dispatch(self, request: Request, call_next):
        """
        Process incoming requests and log comprehensive information.
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware or endpoint handler
            
        Returns:
            HTTP response with logging
        """
        # Generate unique request ID
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        # Extract request information
        request_info = await self._extract_request_info(request, request_id)
        
        # Log incoming request
        logger.info(f"[{request_id}] Incoming request: {request_info['method']} {request_info['path']}")
        logger.debug(f"[{request_id}] Request details: {json.dumps(request_info, indent=2)}")
        
        try:
            # Add request ID to request state
            request.state.request_id = request_id
            request.state.start_time = start_time
            
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Extract response information
            response_info = self._extract_response_info(response, process_time)
            
            # Log response
            self._log_response(request_id, request_info, response_info)
            
            # Add custom headers
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}s"
            
            return response
            
        except Exception as e:
            # Calculate processing time for error case
            process_time = time.time() - start_time
            
            # Log error
            logger.error(f"[{request_id}] Request failed after {process_time:.3f}s: {str(e)}")
            logger.error(f"[{request_id}] Error details: {type(e).__name__}: {str(e)}")
            
            # Re-raise the exception
            raise
    
    async def _extract_request_info(self, request: Request, request_id: str) -> Dict[str, Any]:
        """
        Extract comprehensive information from incoming request.
        
        Args:
            request: HTTP request object
            request_id: Unique request identifier
            
        Returns:
            Dictionary containing request information
        """
        # Get client information
        client_host = request.client.host if request.client else "unknown"
        client_port = request.client.port if request.client else "unknown"
        
        # Get headers (filter sensitive ones)
        headers = {}
        for name, value in request.headers.items():
            if name.lower() not in self.sensitive_headers:
                headers[name] = value
            else:
                headers[name] = "[REDACTED]"
        
        # Get query parameters
        query_params = dict(request.query_params)
        
        # Get basic request info
        request_info = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "method": request.method,
            "path": request.url.path,
            "full_url": str(request.url),
            "client": {
                "host": client_host,
                "port": client_port,
                "user_agent": headers.get("user-agent", "unknown")
            },
            "headers": headers,
            "query_params": query_params,
            "path_params": dict(request.path_params),
            "content_type": headers.get("content-type", "unknown"),
            "content_length": headers.get("content-length", "unknown")
        }
        
        # Add user information if available
        if hasattr(request.state, "user"):
            request_info["user"] = {
                "user_id": request.state.user.get("user_id"),
                "username": request.state.user.get("username")
            }
        
        return request_info
    
    def _extract_response_info(self, response: Response, process_time: float) -> Dict[str, Any]:
        """
        Extract information from outgoing response.
        
        Args:
            response: HTTP response object
            process_time: Request processing time in seconds
            
        Returns:
            Dictionary containing response information
        """
        return {
            "status_code": response.status_code,
            "process_time": process_time,
            "headers": dict(response.headers),
            "content_type": response.headers.get("content-type", "unknown"),
            "content_length": response.headers.get("content-length", "unknown")
        }
    
    def _log_response(self, request_id: str, request_info: Dict[str, Any], response_info: Dict[str, Any]):
        """
        Log response information with appropriate log level.
        
        Args:
            request_id: Unique request identifier
            request_info: Request information dictionary
            response_info: Response information dictionary
        """
        status_code = response_info["status_code"]
        process_time = response_info["process_time"]
        method = request_info["method"]
        path = request_info["path"]
        
        # Determine log level based on status code
        if status_code < 400:
            log_level = logging.INFO
            log_message = f"[{request_id}] {method} {path} - {status_code} - {process_time:.3f}s"
        elif status_code < 500:
            log_level = logging.WARNING
            log_message = f"[{request_id}] {method} {path} - {status_code} (Client Error) - {process_time:.3f}s"
        else:
            log_level = logging.ERROR
            log_message = f"[{request_id}] {method} {path} - {status_code} (Server Error) - {process_time:.3f}s"
        
        # Log with appropriate level
        logger.log(log_level, log_message)
        
        # Log detailed information for debugging (only in debug mode)
        if settings.DEBUG:
            logger.debug(f"[{request_id}] Response details: {json.dumps(response_info, indent=2)}")
        
        # Log performance warnings for slow requests
        if process_time > 5.0:  # Requests taking more than 5 seconds
            logger.warning(f"[{request_id}] Slow request detected: {process_time:.3f}s for {method} {path}")
        
        # Log usage statistics
        self._log_usage_stats(request_info, response_info)
    
    def _log_usage_stats(self, request_info: Dict[str, Any], response_info: Dict[str, Any]):
        """
        Log usage statistics for monitoring and analytics.
        
        Args:
            request_info: Request information dictionary
            response_info: Response information dictionary
        """
        try:
            # Create usage log entry
            usage_entry = {
                "timestamp": request_info["timestamp"],
                "endpoint": request_info["path"],
                "method": request_info["method"],
                "status_code": response_info["status_code"],
                "process_time": response_info["process_time"],
                "user_agent": request_info["client"]["user_agent"],
                "client_ip": request_info["client"]["host"]
            }
            
            # Add user information if available
            if "user" in request_info:
                usage_entry["user_id"] = request_info["user"]["user_id"]
                usage_entry["username"] = request_info["user"]["username"]
            
            # Log usage statistics (this could be sent to analytics service)
            logger.info(f"USAGE_STATS: {json.dumps(usage_entry)}")
            
        except Exception as e:
            logger.error(f"Failed to log usage statistics: {str(e)}")


class PerformanceMonitor:
    """
    Performance monitoring utility for tracking API metrics.
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.request_counts = {}
        self.response_times = {}
        self.error_counts = {}
    
    def record_request(self, endpoint: str, method: str, process_time: float, status_code: int):
        """
        Record request metrics for performance monitoring.
        
        Args:
            endpoint: API endpoint path
            method: HTTP method
            process_time: Request processing time
            status_code: HTTP status code
        """
        key = f"{method} {endpoint}"
        
        # Count requests
        self.request_counts[key] = self.request_counts.get(key, 0) + 1
        
        # Track response times
        if key not in self.response_times:
            self.response_times[key] = []
        self.response_times[key].append(process_time)
        
        # Count errors
        if status_code >= 400:
            self.error_counts[key] = self.error_counts.get(key, 0) + 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.
        
        Returns:
            Dictionary containing performance metrics
        """
        metrics = {
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "endpoints": {}
        }
        
        for endpoint, count in self.request_counts.items():
            times = self.response_times.get(endpoint, [])
            errors = self.error_counts.get(endpoint, 0)
            
            metrics["endpoints"][endpoint] = {
                "request_count": count,
                "error_count": errors,
                "error_rate": errors / count if count > 0 else 0,
                "avg_response_time": sum(times) / len(times) if times else 0,
                "max_response_time": max(times) if times else 0,
                "min_response_time": min(times) if times else 0
            }
        
        return metrics


# Global performance monitor instance
performance_monitor = PerformanceMonitor()
