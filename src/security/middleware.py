"""
Security Middleware
Handles request authentication, authorization, rate limiting, and security checks
"""

import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
from flask import request, jsonify, g
from functools import wraps
from collections import defaultdict, deque
import logging

from .auth import AuthenticationSystem
from .authorization import AuthorizationSystem, ResourceType, Action
from .monitoring import SecurityMonitor, SecurityEventType
from .encryption import EncryptionManager

logger = logging.getLogger(__name__)

class RateLimiter:
    """Rate limiting implementation"""
    
    def __init__(self):
        self.requests = defaultdict(lambda: deque())
        self.windows = {
            'minute': 60,
            'hour': 3600,
            'day': 86400
        }
        
        # Default rate limits
        self.limits = {
            'default': {'minute': 60, 'hour': 1000, 'day': 10000},
            'admin': {'minute': 200, 'hour': 5000, 'day': 50000},
            'premium_user': {'minute': 120, 'hour': 2000, 'day': 20000},
            'user': {'minute': 60, 'hour': 1000, 'day': 10000},
            'guest': {'minute': 20, 'hour': 200, 'day': 1000}
        }
    
    def is_allowed(self, key: str, role: str = 'default') -> tuple[bool, Dict[str, Any]]:
        """Check if request is allowed under rate limits"""
        now = time.time()
        user_requests = self.requests[key]
        
        # Clean old requests
        while user_requests and user_requests[0] < now - self.windows['day']:
            user_requests.popleft()
        
        # Count requests in each window
        counts = {}
        for window, duration in self.windows.items():
            cutoff = now - duration
            counts[window] = sum(1 for req_time in user_requests if req_time >= cutoff)
        
        # Check limits
        role_limits = self.limits.get(role, self.limits['default'])
        for window, count in counts.items():
            if window in role_limits and count >= role_limits[window]:
                return False, {
                    'error': 'Rate limit exceeded',
                    'window': window,
                    'limit': role_limits[window],
                    'current': count,
                    'reset_at': now + self.windows[window]
                }
        
        # Record request
        user_requests.append(now)
        
        return True, {'status': 'allowed', 'counts': counts}

class SecurityMiddleware:
    """Comprehensive security middleware for Flask applications"""
    
    def __init__(self, auth_system: AuthenticationSystem, 
                 authorization_system: AuthorizationSystem,
                 security_monitor: SecurityMonitor,
                 encryption_manager: EncryptionManager):
        self.auth = auth_system
        self.authz = authorization_system
        self.monitor = security_monitor
        self.encryption = encryption_manager
        self.rate_limiter = RateLimiter()
        
        # Security headers
        self.security_headers = {
            'X-Content-Type-Options': 'nosniff',
            'X-Frame-Options': 'DENY',
            'X-XSS-Protection': '1; mode=block',
            'Strict-Transport-Security': 'max-age=31536000; includeSubDomains',
            'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
            'Referrer-Policy': 'strict-origin-when-cross-origin'
        }
        
        # Public endpoints that don't require authentication
        self.public_endpoints = {
            '/',
            '/health',
            '/api/status',
            '/cognitive'  # Main interface
        }
        
        # Rate limiting exempt endpoints
        self.rate_limit_exempt = {
            '/health'
        }
    
    def init_app(self, app):
        """Initialize middleware with Flask app"""
        app.before_request(self.before_request)
        app.after_request(self.after_request)
        
        # Add security headers to all responses
        @app.after_request
        def add_security_headers(response):
            for header, value in self.security_headers.items():
                response.headers[header] = value
            return response
    
    def before_request(self):
        """Process request before handling"""
        # Get request info
        client_ip = self._get_client_ip()
        user_agent = request.headers.get('User-Agent', '')
        endpoint = request.endpoint or request.path
        
        # Check if IP is blocked
        if self.monitor.is_ip_blocked(client_ip):
            self.monitor.log_security_event(
                SecurityEventType.SECURITY_VIOLATION,
                ip_address=client_ip,
                user_agent=user_agent,
                details={'reason': 'blocked_ip_access', 'endpoint': endpoint}
            )
            return jsonify({'error': 'Access denied'}), 403
        
        # Rate limiting (except for exempt endpoints)
        if endpoint not in self.rate_limit_exempt:
            rate_key = f"ip:{client_ip}"
            user_role = 'guest'  # Default role for unauthenticated requests
            
            # Try to get user role from token if present
            auth_header = request.headers.get('Authorization')
            if auth_header and auth_header.startswith('Bearer '):
                token = auth_header[7:]
                is_valid, payload = self.auth.verify_token(token)
                if is_valid and payload:
                    user_role = payload.get('roles', ['guest'])[0] if payload.get('roles') else 'guest'
                    rate_key = f"user:{payload.get('user_id')}"
            
            allowed, rate_info = self.rate_limiter.is_allowed(rate_key, user_role)
            if not allowed:
                self.monitor.log_security_event(
                    SecurityEventType.SECURITY_VIOLATION,
                    ip_address=client_ip,
                    user_agent=user_agent,
                    details={'reason': 'rate_limit_exceeded', 'rate_info': rate_info}
                )
                return jsonify(rate_info), 429
        
        # Log API access
        self.monitor.log_security_event(
            SecurityEventType.API_ACCESS,
            ip_address=client_ip,
            user_agent=user_agent,
            details={
                'method': request.method,
                'endpoint': endpoint,
                'query_params': dict(request.args)
            }
        )
        
        # Store request info for later use
        g.client_ip = client_ip
        g.user_agent = user_agent
        g.request_start_time = time.time()
    
    def after_request(self, response):
        """Process response after handling"""
        # Calculate request duration
        if hasattr(g, 'request_start_time'):
            duration = time.time() - g.request_start_time
            response.headers['X-Response-Time'] = f"{duration:.3f}s"
        
        return response
    
    def require_auth(self, required_permissions: List[str] = None, 
                    required_roles: List[str] = None,
                    resource_type: Optional[ResourceType] = None,
                    action: Optional[Action] = None):
        """Decorator for endpoints requiring authentication and authorization"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Check authentication
                auth_result = self._authenticate_request()
                if not auth_result['success']:
                    return jsonify({'error': auth_result['error']}), 401
                
                user_id = auth_result['user_id']
                user_roles = auth_result['roles']
                
                # Check authorization
                authz_result = self._authorize_request(
                    user_id, user_roles, required_permissions, required_roles, 
                    resource_type, action, kwargs
                )
                if not authz_result['success']:
                    # Log permission denied
                    self.monitor.log_security_event(
                        SecurityEventType.PERMISSION_DENIED,
                        user_id=user_id,
                        ip_address=g.client_ip,
                        user_agent=g.user_agent,
                        details={
                            'endpoint': request.endpoint,
                            'required_permissions': required_permissions,
                            'required_roles': required_roles,
                            'error': authz_result['error']
                        }
                    )
                    return jsonify({'error': authz_result['error']}), 403
                
                # Store user info in request context
                g.current_user_id = user_id
                g.current_user_roles = user_roles
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def require_role(self, *roles):
        """Decorator for endpoints requiring specific roles"""
        return self.require_auth(required_roles=list(roles))
    
    def require_permission(self, *permissions):
        """Decorator for endpoints requiring specific permissions"""
        return self.require_auth(required_permissions=list(permissions))
    
    def require_admin(self):
        """Decorator for admin-only endpoints"""
        return self.require_role('admin')
    
    def optional_auth(self):
        """Decorator for endpoints with optional authentication"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                auth_result = self._authenticate_request()
                if auth_result['success']:
                    g.current_user_id = auth_result['user_id']
                    g.current_user_roles = auth_result['roles']
                else:
                    g.current_user_id = None
                    g.current_user_roles = ['guest']
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _authenticate_request(self) -> Dict[str, Any]:
        """Authenticate the current request"""
        # Check for API key first
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return self._authenticate_api_key(api_key)
        
        # Check for Bearer token
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]
            return self._authenticate_token(token)
        
        # Check for session-based auth (cookie)
        session_token = request.cookies.get('session_token')
        if session_token:
            return self._authenticate_token(session_token)
        
        return {'success': False, 'error': 'No authentication provided'}
    
    def _authenticate_token(self, token: str) -> Dict[str, Any]:
        """Authenticate JWT token"""
        try:
            is_valid, payload = self.auth.verify_token(token)
            if not is_valid:
                return {'success': False, 'error': 'Invalid or expired token'}
            
            user_id = payload['user_id']
            user = self.auth.get_user(user_id)
            if not user or not user.is_active:
                return {'success': False, 'error': 'User account inactive'}
            
            # Log successful authentication
            self.monitor.log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=user_id,
                ip_address=g.client_ip,
                user_agent=g.user_agent,
                details={'auth_method': 'token'}
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'username': user.username,
                'roles': payload.get('roles', [])
            }
            
        except Exception as e:
            logger.error(f"Token authentication error: {e}")
            return {'success': False, 'error': 'Authentication failed'}
    
    def _authenticate_api_key(self, api_key: str) -> Dict[str, Any]:
        """Authenticate API key"""
        try:
            # Parse API key format: key_id:key_secret
            if ':' not in api_key:
                return {'success': False, 'error': 'Invalid API key format'}
            
            key_id, key_secret = api_key.split(':', 1)
            key_info = self.encryption.validate_api_key(key_id, key_secret)
            
            if not key_info or not key_info.get('valid'):
                return {'success': False, 'error': 'Invalid API key'}
            
            user_id = key_info['user_id']
            scopes = key_info.get('scopes', [])
            
            # Log API key authentication
            self.monitor.log_security_event(
                SecurityEventType.LOGIN_SUCCESS,
                user_id=user_id,
                ip_address=g.client_ip,
                user_agent=g.user_agent,
                details={'auth_method': 'api_key', 'key_id': key_id}
            )
            
            return {
                'success': True,
                'user_id': user_id,
                'username': f'api_user_{key_id}',
                'roles': ['api_user'],
                'scopes': scopes
            }
            
        except Exception as e:
            logger.error(f"API key authentication error: {e}")
            return {'success': False, 'error': 'Authentication failed'}
    
    def _authorize_request(self, user_id: str, user_roles: List[str], 
                          required_permissions: List[str] = None,
                          required_roles: List[str] = None,
                          resource_type: Optional[ResourceType] = None,
                          action: Optional[Action] = None,
                          route_kwargs: Dict[str, Any] = None) -> Dict[str, Any]:
        """Authorize the current request"""
        
        # Check required roles
        if required_roles:
            user_role_set = set(user_roles)
            if not any(role in user_role_set for role in required_roles):
                return {'success': False, 'error': 'Insufficient role privileges'}
        
        # Check required permissions
        if required_permissions:
            for permission in required_permissions:
                resource_id = None
                if route_kwargs:
                    # Try to extract resource ID from route parameters
                    resource_id = route_kwargs.get('session_id') or route_kwargs.get('resource_id')
                
                if not self.authz.has_permission(user_id, permission, resource_id):
                    return {'success': False, 'error': f'Missing permission: {permission}'}
        
        # Check resource-level access
        if resource_type and action:
            resource_id = None
            if route_kwargs:
                resource_id = route_kwargs.get('session_id') or route_kwargs.get('resource_id')
            
            if not self.authz.can_access_resource(user_id, resource_type, action, resource_id):
                return {'success': False, 'error': 'Access denied to resource'}
        
        return {'success': True}
    
    def _get_client_ip(self) -> str:
        """Get client IP address, handling proxies"""
        # Check for forwarded IP headers
        forwarded_headers = [
            'X-Forwarded-For',
            'X-Real-IP',
            'X-Client-IP',
            'CF-Connecting-IP'  # Cloudflare
        ]
        
        for header in forwarded_headers:
            ip = request.headers.get(header)
            if ip:
                # Handle comma-separated IPs (take first one)
                return ip.split(',')[0].strip()
        
        return request.remote_addr or 'unknown'
    
    def create_session_token(self, user_id: str) -> Optional[str]:
        """Create a session token for a user"""
        user = self.auth.get_user(user_id)
        if not user:
            return None
        
        auth_token = self.auth.generate_token(user)
        return auth_token.token
    
    def logout_user(self, token: str) -> bool:
        """Logout user by revoking token"""
        try:
            # Verify token to get user info
            is_valid, payload = self.auth.verify_token(token)
            if is_valid and payload:
                user_id = payload['user_id']
                
                # Revoke token
                self.auth.revoke_token(token)
                
                # Log logout
                self.monitor.log_security_event(
                    SecurityEventType.LOGOUT,
                    user_id=user_id,
                    ip_address=g.client_ip if hasattr(g, 'client_ip') else None,
                    user_agent=g.user_agent if hasattr(g, 'user_agent') else None
                )
                
                return True
            
        except Exception as e:
            logger.error(f"Logout error: {e}")
        
        return False
    
    def get_current_user(self) -> Optional[Dict[str, Any]]:
        """Get current authenticated user info"""
        if hasattr(g, 'current_user_id') and g.current_user_id:
            user = self.auth.get_user(g.current_user_id)
            if user:
                return user.to_dict()
        return None
    
    def has_permission(self, permission: str, resource_id: Optional[str] = None) -> bool:
        """Check if current user has permission"""
        if not hasattr(g, 'current_user_id') or not g.current_user_id:
            return False
        
        return self.authz.has_permission(g.current_user_id, permission, resource_id)
    
    def has_role(self, role: str) -> bool:
        """Check if current user has role"""
        if not hasattr(g, 'current_user_roles'):
            return False
        
        return role in g.current_user_roles
    
    def enforce_https(self):
        """Decorator to enforce HTTPS"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if not request.is_secure and not request.headers.get('X-Forwarded-Proto') == 'https':
                    return jsonify({'error': 'HTTPS required'}), 400
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def validate_content_type(self, content_type: str):
        """Decorator to validate request content type"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                if request.content_type != content_type:
                    return jsonify({'error': f'Content-Type must be {content_type}'}), 400
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def sanitize_input(self):
        """Decorator to sanitize request input"""
        def decorator(f):
            @wraps(f)
            def decorated_function(*args, **kwargs):
                # Basic input sanitization
                if request.is_json:
                    try:
                        data = request.get_json()
                        if data:
                            # Recursively sanitize strings in the data
                            sanitized_data = self._sanitize_dict(data)
                            request._cached_json = (sanitized_data, True)
                    except Exception as e:
                        logger.warning(f"Input sanitization error: {e}")
                
                return f(*args, **kwargs)
            return decorated_function
        return decorator
    
    def _sanitize_dict(self, data: Any) -> Any:
        """Recursively sanitize dictionary data"""
        if isinstance(data, dict):
            return {key: self._sanitize_dict(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_dict(item) for item in data]
        elif isinstance(data, str):
            # Basic string sanitization
            return data.strip()
        else:
            return data