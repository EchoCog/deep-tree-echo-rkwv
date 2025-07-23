"""
Enhanced RESTful API Server
Comprehensive API endpoints with versioning, rate limiting, and documentation
"""

from flask import Flask, request, jsonify, Blueprint, current_app
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from functools import wraps
import time
import logging
from typing import Dict, Any, Optional
import json
from datetime import datetime, timedelta
import uuid

# Setup logging
logger = logging.getLogger(__name__)

class APIVersioning:
    """Handle API versioning"""
    
    @staticmethod
    def get_api_version(request):
        """Extract API version from request"""
        # Check header first
        version = request.headers.get('API-Version')
        if version:
            return version
        
        # Check URL path
        if request.path.startswith('/api/v'):
            try:
                return request.path.split('/')[2]  # /api/v1/...
            except IndexError:
                pass
        
        # Default to v1
        return 'v1'
    
    @staticmethod
    def version_compatible(required_version: str, provided_version: str) -> bool:
        """Check if API versions are compatible"""
        # Simple version compatibility check
        req_major = int(required_version.replace('v', '').split('.')[0])
        prov_major = int(provided_version.replace('v', '').split('.')[0])
        
        return req_major <= prov_major

def require_api_version(min_version: str = 'v1'):
    """Decorator to require minimum API version"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_version = APIVersioning.get_api_version(request)
            
            if not APIVersioning.version_compatible(min_version, client_version):
                return jsonify({
                    'error': 'API version not supported',
                    'required_minimum': min_version,
                    'provided': client_version,
                    'supported_versions': ['v1', 'v2']
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def api_response(data: Any = None, error: str = None, status_code: int = 200, 
                version: str = None, meta: Dict = None) -> tuple:
    """Standardized API response format"""
    
    response = {
        'timestamp': datetime.now().isoformat(),
        'version': version or APIVersioning.get_api_version(request),
        'success': error is None
    }
    
    if data is not None:
        response['data'] = data
    
    if error:
        response['error'] = error
    
    if meta:
        response['meta'] = meta
    
    return jsonify(response), status_code

class RateLimiter:
    """Enhanced rate limiting with quotas"""
    
    def __init__(self):
        self.usage_tracking = {}  # In production, use Redis
        self.quota_limits = {
            'free': {'requests_per_hour': 100, 'requests_per_day': 1000},
            'basic': {'requests_per_hour': 1000, 'requests_per_day': 10000},
            'premium': {'requests_per_hour': 10000, 'requests_per_day': 100000}
        }
    
    def get_user_tier(self, api_key: str) -> str:
        """Get user tier from API key - mock implementation"""
        # In production, this would query a database
        if not api_key:
            return 'free'
        if api_key.startswith('basic_'):
            return 'basic'
        if api_key.startswith('premium_'):
            return 'premium'
        return 'free'
    
    def check_quota(self, identifier: str, tier: str) -> Dict[str, Any]:
        """Check if user has quota remaining"""
        now = datetime.now()
        hour_key = f"{identifier}_{now.hour}"
        day_key = f"{identifier}_{now.date()}"
        
        if identifier not in self.usage_tracking:
            self.usage_tracking[identifier] = {}
        
        user_usage = self.usage_tracking[identifier]
        
        # Get current usage
        hour_usage = user_usage.get(hour_key, 0)
        day_usage = user_usage.get(day_key, 0)
        
        # Get limits for tier
        limits = self.quota_limits.get(tier, self.quota_limits['free'])
        
        return {
            'allowed': (
                hour_usage < limits['requests_per_hour'] and 
                day_usage < limits['requests_per_day']
            ),
            'hour_usage': hour_usage,
            'hour_limit': limits['requests_per_hour'],
            'day_usage': day_usage,
            'day_limit': limits['requests_per_day'],
            'tier': tier
        }
    
    def increment_usage(self, identifier: str):
        """Increment usage counters"""
        now = datetime.now()
        hour_key = f"{identifier}_{now.hour}"
        day_key = f"{identifier}_{now.date()}"
        
        if identifier not in self.usage_tracking:
            self.usage_tracking[identifier] = {}
        
        user_usage = self.usage_tracking[identifier]
        user_usage[hour_key] = user_usage.get(hour_key, 0) + 1
        user_usage[day_key] = user_usage.get(day_key, 0) + 1

# Global rate limiter instance
rate_limiter = RateLimiter()

def require_api_key(f):
    """Decorator to require API key authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key') or request.args.get('api_key')
        
        if not api_key:
            return api_response(error='API key required', status_code=401)
        
        # In production, validate API key against database
        if not api_key.startswith(('free_', 'basic_', 'premium_')):
            return api_response(error='Invalid API key', status_code=401)
        
        # Add API key to request context
        request.api_key = api_key
        return f(*args, **kwargs)
    return decorated_function

def check_rate_limit(f):
    """Decorator to check rate limits"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get identifier (API key or IP)
        identifier = getattr(request, 'api_key', get_remote_address())
        tier = rate_limiter.get_user_tier(getattr(request, 'api_key', ''))
        
        # Check quota
        quota_check = rate_limiter.check_quota(identifier, tier)
        
        if not quota_check['allowed']:
            return api_response(
                error='Rate limit exceeded',
                status_code=429,
                meta={
                    'quota_info': quota_check,
                    'retry_after': 3600  # 1 hour
                }
            )
        
        # Increment usage
        rate_limiter.increment_usage(identifier)
        
        # Add quota info to response headers
        response = f(*args, **kwargs)
        if isinstance(response, tuple) and len(response) == 2:
            json_response, status_code = response
            if hasattr(json_response, 'headers'):
                json_response.headers['X-RateLimit-Tier'] = quota_check['tier']
                json_response.headers['X-RateLimit-Hour-Remaining'] = str(
                    quota_check['hour_limit'] - quota_check['hour_usage']
                )
                json_response.headers['X-RateLimit-Day-Remaining'] = str(
                    quota_check['day_limit'] - quota_check['day_usage']
                )
        
        return response
    return decorated_function

def create_enhanced_api_blueprint(echo_system=None) -> Blueprint:
    """Create enhanced API blueprint with comprehensive endpoints"""
    
    api_v1 = Blueprint('api_v1', __name__, url_prefix='/api/v1')
    api_v2 = Blueprint('api_v2', __name__, url_prefix='/api/v2')
    
    # ============================================================================
    # V1 API Endpoints (Enhanced versions of existing endpoints)
    # ============================================================================
    
    @api_v1.route('/status')
    @require_api_version('v1')
    @check_rate_limit
    def get_system_status():
        """Enhanced system status endpoint"""
        try:
            status_data = {
                'system': {
                    'status': 'online',
                    'version': '1.0.0',
                    'uptime': time.time(),
                    'echo_system_initialized': echo_system is not None
                },
                'services': {
                    'cognitive_processing': True,
                    'memory_system': True,
                    'api_server': True
                },
                'performance': {
                    'response_time_ms': 45,
                    'throughput_rpm': 2500,
                    'cache_hit_rate': 0.78
                }
            }
            
            return api_response(data=status_data)
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return api_response(error=str(e), status_code=500)
    
    @api_v1.route('/cognitive/process', methods=['POST'])
    @require_api_version('v1')
    @require_api_key
    @check_rate_limit
    def process_cognitive_input():
        """Enhanced cognitive processing endpoint"""
        try:
            data = request.get_json()
            if not data or 'input' not in data:
                return api_response(error='Input is required', status_code=400)
            
            input_text = data['input']
            options = data.get('options', {})
            
            if not echo_system:
                # Mock response for demonstration
                result = {
                    'input': input_text,
                    'output': f"Enhanced cognitive processing result for: {input_text}",
                    'processing_time': 0.045,
                    'confidence': 0.92,
                    'membranes_activated': ['memory', 'reasoning', 'grammar'],
                    'session_id': str(uuid.uuid4())
                }
            else:
                result = echo_system.process_cognitive_input(input_text)
            
            return api_response(data=result)
            
        except Exception as e:
            logger.error(f"Error processing cognitive input: {e}")
            return api_response(error=str(e), status_code=500)
    
    @api_v1.route('/memory/search', methods=['POST'])
    @require_api_version('v1')
    @require_api_key
    @check_rate_limit
    def search_memory():
        """Enhanced memory search endpoint"""
        try:
            data = request.get_json()
            if not data or 'query' not in data:
                return api_response(error='Query is required', status_code=400)
            
            query = data['query']
            filters = data.get('filters', {})
            limit = data.get('limit', 10)
            
            # Mock implementation
            results = {
                'query': query,
                'results': [
                    {
                        'id': str(uuid.uuid4()),
                        'content': f"Memory result {i+1} for query: {query}",
                        'relevance_score': 0.9 - (i * 0.1),
                        'timestamp': datetime.now().isoformat(),
                        'type': 'declarative' if i % 2 == 0 else 'procedural'
                    }
                    for i in range(min(limit, 5))
                ],
                'total_results': min(limit, 5),
                'processing_time': 0.023
            }
            
            return api_response(data=results)
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return api_response(error=str(e), status_code=500)
    
    @api_v1.route('/sessions', methods=['POST'])
    @require_api_version('v1')
    @require_api_key
    @check_rate_limit
    def create_session():
        """Create a new cognitive session"""
        try:
            data = request.get_json() or {}
            
            session_data = {
                'session_id': str(uuid.uuid4()),
                'created_at': datetime.now().isoformat(),
                'status': 'active',
                'configuration': {
                    'temperature': data.get('temperature', 0.8),
                    'max_context_length': data.get('max_context_length', 2048),
                    'memory_persistence': data.get('memory_persistence', True)
                },
                'metadata': data.get('metadata', {})
            }
            
            return api_response(data=session_data, status_code=201)
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            return api_response(error=str(e), status_code=500)
    
    @api_v1.route('/sessions/<session_id>', methods=['GET'])
    @require_api_version('v1')
    @require_api_key
    @check_rate_limit
    def get_session(session_id):
        """Get session information"""
        try:
            # Mock session data
            session_data = {
                'session_id': session_id,
                'status': 'active',
                'created_at': '2024-01-01T00:00:00',
                'last_activity': datetime.now().isoformat(),
                'message_count': 42,
                'total_tokens_processed': 15432,
                'cognitive_state_summary': {
                    'active_memories': 156,
                    'reasoning_chains': 23,
                    'grammar_patterns': 78
                }
            }
            
            return api_response(data=session_data)
            
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return api_response(error=str(e), status_code=500)
    
    # ============================================================================
    # V2 API Endpoints (Advanced features)
    # ============================================================================
    
    @api_v2.route('/cognitive/advanced-process', methods=['POST'])
    @require_api_version('v2')
    @require_api_key
    @check_rate_limit
    def advanced_cognitive_process():
        """Advanced cognitive processing with multi-modal support"""
        try:
            data = request.get_json()
            if not data:
                return api_response(error='Request data is required', status_code=400)
            
            # Enhanced processing with multiple input types
            inputs = data.get('inputs', [])
            processing_modes = data.get('processing_modes', ['cognitive'])
            
            result = {
                'request_id': str(uuid.uuid4()),
                'inputs_processed': len(inputs),
                'processing_modes': processing_modes,
                'outputs': [
                    {
                        'type': 'cognitive_response',
                        'content': f"Advanced processing result for input {i+1}",
                        'confidence': 0.95,
                        'processing_time': 0.032
                    }
                    for i in range(len(inputs))
                ],
                'total_processing_time': 0.078,
                'api_version': 'v2'
            }
            
            return api_response(data=result)
            
        except Exception as e:
            logger.error(f"Error in advanced cognitive processing: {e}")
            return api_response(error=str(e), status_code=500)
    
    @api_v2.route('/analytics/usage', methods=['GET'])
    @require_api_version('v2')
    @require_api_key
    @check_rate_limit
    def get_usage_analytics():
        """Get usage analytics for API key"""
        try:
            api_key = request.api_key
            tier = rate_limiter.get_user_tier(api_key)
            
            # Mock analytics data
            analytics = {
                'api_key': api_key[:10] + '...',  # Partial key for security
                'tier': tier,
                'usage_summary': {
                    'total_requests': 1247,
                    'successful_requests': 1189,
                    'error_requests': 58,
                    'average_response_time': 0.045
                },
                'endpoint_usage': {
                    '/api/v1/cognitive/process': 856,
                    '/api/v1/memory/search': 234,
                    '/api/v1/sessions': 157
                },
                'quota_status': rate_limiter.check_quota(api_key, tier),
                'period': 'last_30_days'
            }
            
            return api_response(data=analytics)
            
        except Exception as e:
            logger.error(f"Error getting usage analytics: {e}")
            return api_response(error=str(e), status_code=500)
    
    return api_v1, api_v2

def setup_flask_limiter(app: Flask) -> Limiter:
    """Setup Flask-Limiter for basic rate limiting"""
    limiter = Limiter(
        app,
        key_func=get_remote_address,
        default_limits=["200 per day", "50 per hour"],
        storage_uri="memory://"
    )
    
    return limiter