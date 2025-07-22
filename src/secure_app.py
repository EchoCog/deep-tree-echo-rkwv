"""
Secure Deep Tree Echo Application
Main application with integrated security framework
"""

from flask import Flask, render_template, request, jsonify, g, make_response
from flask_cors import CORS
import os
import json
import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid

# Import existing components
from persistent_memory import PersistentMemorySystem, MemoryQuery

# Import security framework
from security import (
    AuthenticationSystem, 
    AuthorizationSystem, 
    EncryptionManager, 
    SecurityMonitor, 
    SecurityMiddleware
)
from security.authorization import ResourceType, Action
from security.monitoring import SecurityEventType
from security_config import SecurityConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configuration
CONFIG = {
    'webvm_mode': os.environ.get('ECHO_WEBVM_MODE', 'true').lower() == 'true',
    'memory_limit_mb': int(os.environ.get('ECHO_MEMORY_LIMIT', '600')),
    'max_sessions': 10,
    'session_timeout_minutes': 30,
    'enable_persistence': True,
    'data_dir': '/tmp/echo_data',
    'security_enabled': True
}

# Global systems
cognitive_sessions = {}
system_metrics = {
    'total_sessions': 0,
    'active_sessions': 0,
    'total_requests': 0,
    'start_time': datetime.now(),
    'last_activity': None
}
persistent_memory = None

# Security systems
auth_system = None
authz_system = None
encryption_manager = None
security_monitor = None
security_middleware = None

def initialize_security():
    """Initialize security systems"""
    global auth_system, authz_system, encryption_manager, security_monitor, security_middleware
    
    logger.info("Initializing security framework...")
    
    # Validate security configuration
    warnings = SecurityConfig.validate_config()
    for warning in warnings:
        logger.warning(warning)
    
    # Initialize security components
    auth_system = AuthenticationSystem(
        secret_key=SecurityConfig.SECRET_KEY,
        token_expiry_hours=SecurityConfig.TOKEN_EXPIRY_HOURS
    )
    
    authz_system = AuthorizationSystem()
    encryption_manager = EncryptionManager()
    security_monitor = SecurityMonitor(SecurityConfig.ALERT_RETENTION_DAYS)
    
    security_middleware = SecurityMiddleware(
        auth_system, authz_system, security_monitor, encryption_manager
    )
    
    # Initialize middleware with app
    security_middleware.init_app(app)
    
    # Create default admin user if it doesn't exist
    _create_default_admin()
    
    logger.info("✅ Security framework initialized successfully")

def _create_default_admin():
    """Create default admin user for initial setup"""
    try:
        success, message, admin_user = auth_system.register_user(
            username=SecurityConfig.DEFAULT_ADMIN_USERNAME,
            email=SecurityConfig.DEFAULT_ADMIN_EMAIL,
            password=SecurityConfig.DEFAULT_ADMIN_PASSWORD,
            roles=['admin']
        )
        
        if success:
            authz_system.assign_role_to_user(admin_user.user_id, 'admin')
            logger.info(f"✅ Default admin user created: {admin_user.username}")
            logger.warning("⚠️  Please change the default admin password immediately!")
        else:
            logger.debug(f"Default admin user already exists or creation failed: {message}")
            
    except Exception as e:
        logger.error(f"Failed to create default admin user: {e}")

def initialize_app():
    """Initialize the application"""
    global persistent_memory
    
    logger.info("Initializing Deep Tree Echo application...")
    
    # Initialize security first
    if CONFIG['security_enabled']:
        initialize_security()
    
    # Configure CORS
    if SecurityConfig.CORS_ENABLED:
        CORS(app, origins=SecurityConfig.CORS_ORIGINS)
    
    # Initialize persistent memory
    if CONFIG['enable_persistence']:
        try:
            persistent_memory = PersistentMemorySystem(CONFIG['data_dir'])
            logger.info("✅ Persistent memory system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize persistent memory: {e}")
    
    logger.info("✅ Application initialized successfully")

# Cognitive Session Class (simplified version)
class CognitiveSession:
    """Simplified cognitive session for security integration"""
    
    def __init__(self, session_id: str, user_id: str = None):
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.conversation_history = []
        self.memory_state = {'episodic': [], 'semantic': {}}
        self.processing_stats = {
            'total_inputs': 0,
            'avg_processing_time': 0
        }
    
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """Process cognitive input with security context"""
        start_time = time.time()
        
        # Update activity
        self.last_activity = datetime.now()
        self.processing_stats['total_inputs'] += 1
        
        # Log data access
        if security_monitor:
            security_monitor.log_security_event(
                SecurityEventType.DATA_ACCESS,
                user_id=self.user_id,
                details={
                    'action': 'cognitive_processing',
                    'session_id': self.session_id,
                    'input_length': len(input_text)
                }
            )
        
        # Mock processing (simplified)
        response = f"Processed: {input_text} (Session: {self.session_id[:8]}...)"
        
        processing_time = time.time() - start_time
        
        # Create conversation entry
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'response': response,
            'processing_time': processing_time,
            'user_id': self.user_id
        }
        
        self.conversation_history.append(conversation_entry)
        
        return conversation_entry
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get session state summary"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'conversation_count': len(self.conversation_history),
            'processing_stats': self.processing_stats
        }

# Utility functions
def create_cognitive_session(user_id: str = None) -> str:
    """Create new cognitive session"""
    session_id = str(uuid.uuid4())
    session = CognitiveSession(session_id, user_id)
    
    cognitive_sessions[session_id] = session
    system_metrics['total_sessions'] += 1
    system_metrics['active_sessions'] = len(cognitive_sessions)
    
    # Set resource ownership
    if user_id and authz_system:
        authz_system.set_resource_owner(session_id, user_id)
    
    logger.info(f"Created cognitive session {session_id} for user {user_id}")
    return session_id

def get_cognitive_session(session_id: str) -> Optional[CognitiveSession]:
    """Get cognitive session by ID"""
    return cognitive_sessions.get(session_id)

# Routes

@app.route('/')
def index():
    """Main application interface"""
    return render_template('index.html') if os.path.exists('templates/index.html') else jsonify({
        'message': 'Deep Tree Echo - Secure Cognitive Architecture',
        'version': '2.0.0-security',
        'security_enabled': CONFIG['security_enabled'],
        'endpoints': {
            'authentication': '/api/auth/*',
            'sessions': '/api/session/*',
            'cognitive': '/api/process',
            'status': '/api/status'
        }
    })

# Authentication Routes
@app.route('/api/auth/register', methods=['POST'])
def register():
    """User registration"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['username', 'email', 'password']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Register user
        success, message, user = auth_system.register_user(
            username=data['username'],
            email=data['email'],
            password=data['password'],
            roles=['user']  # Default role
        )
        
        if not success:
            return jsonify({'error': message}), 400
        
        # Assign default role in authorization system
        authz_system.assign_role_to_user(user.user_id, 'user')
        
        # Log registration
        security_monitor.log_security_event(
            SecurityEventType.LOGIN_SUCCESS,
            user_id=user.user_id,
            ip_address=g.client_ip,
            user_agent=g.user_agent,
            details={'action': 'user_registration'}
        )
        
        return jsonify({
            'message': 'User registered successfully',
            'user_id': user.user_id
        })
        
    except Exception as e:
        logger.error(f"Registration error: {e}")
        return jsonify({'error': 'Registration failed'}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    """User authentication"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    try:
        data = request.get_json()
        
        # Validate input
        if not data or 'username' not in data or 'password' not in data:
            return jsonify({'error': 'Username and password required'}), 400
        
        # Authenticate user
        success, message, token = auth_system.authenticate_user(
            username=data['username'],
            password=data['password'],
            mfa_token=data.get('mfa_token')
        )
        
        if not success:
            # Log failed login
            security_monitor.log_security_event(
                SecurityEventType.LOGIN_FAILED,
                ip_address=g.client_ip,
                user_agent=g.user_agent,
                details={'username': data['username'], 'reason': message}
            )
            return jsonify({'error': message}), 401
        
        # Create response with token
        response = make_response(jsonify({
            'message': 'Login successful',
            'token': token.token,
            'expires_at': token.expires_at.isoformat()
        }))
        
        # Set secure cookie
        response.set_cookie(
            'session_token',
            token.token,
            max_age=SecurityConfig.TOKEN_EXPIRY_HOURS * 3600,
            secure=SecurityConfig.SESSION_COOKIE_SECURE,
            httponly=SecurityConfig.SESSION_COOKIE_HTTPONLY,
            samesite=SecurityConfig.SESSION_COOKIE_SAMESITE
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Login failed'}), 500

@app.route('/api/auth/logout', methods=['POST'])
def logout():
    """User logout"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    try:
        # Get token from header or cookie
        token = None
        auth_header = request.headers.get('Authorization')
        if auth_header and auth_header.startswith('Bearer '):
            token = auth_header[7:]
        else:
            token = request.cookies.get('session_token')
        
        if token:
            security_middleware.logout_user(token)
        
        # Create response
        response = make_response(jsonify({'message': 'Logout successful'}))
        response.set_cookie('session_token', '', expires=0)
        
        return response
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        return jsonify({'error': 'Logout failed'}), 500

# Session Management Routes
@app.route('/api/session', methods=['POST'])
def create_session():
    """Create new cognitive session"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    # Check authentication manually
    auth_result = security_middleware._authenticate_request() if security_middleware else {'success': False}
    if not auth_result['success']:
        return jsonify({'error': 'Authentication required'}), 401
        
    try:
        user_id = auth_result['user_id']
        
        # Check permission
        if not authz_system.has_permission(user_id, 'session.create'):
            return jsonify({'error': 'Permission denied'}), 403
            
        session_id = create_cognitive_session(user_id)
        session = get_cognitive_session(session_id)
        
        return jsonify({
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'status': 'active'
        })
        
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id: str):
    """Get session information"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    # Check authentication manually
    auth_result = security_middleware._authenticate_request() if security_middleware else {'success': False}
    if not auth_result['success']:
        return jsonify({'error': 'Authentication required'}), 401
        
    try:
        user_id = auth_result['user_id']
        
        # Check if user can access this session
        if not authz_system.can_access_resource(user_id, ResourceType.SESSION, Action.READ, session_id):
            return jsonify({'error': 'Access denied'}), 403
        
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(session.get_state_summary())
        
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return jsonify({'error': str(e)}), 500

# Cognitive Processing Routes
@app.route('/api/process', methods=['POST'])
def process_cognitive_input():
    """Process cognitive input"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    # Check authentication manually
    auth_result = security_middleware._authenticate_request() if security_middleware else {'success': False}
    if not auth_result['success']:
        return jsonify({'error': 'Authentication required'}), 401
        
    try:
        user_id = auth_result['user_id']
        
        # Check permission
        if not authz_system.has_permission(user_id, 'api.cognitive_process'):
            return jsonify({'error': 'Permission denied'}), 403
            
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'No input provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        # Check session access
        if not authz_system.can_access_resource(user_id, ResourceType.SESSION, Action.UPDATE, session_id):
            return jsonify({'error': 'Access denied'}), 403
        
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Process input
        input_text = data['input']
        result = session.process_input(input_text)
        
        system_metrics['total_requests'] += 1
        system_metrics['last_activity'] = datetime.now()
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing cognitive input: {e}")
        return jsonify({'error': str(e)}), 500

# Status and Monitoring Routes
@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'operational',
        'timestamp': datetime.now().isoformat(),
        'security_enabled': CONFIG['security_enabled'],
        'active_sessions': len(cognitive_sessions),
        'uptime_seconds': (datetime.now() - system_metrics['start_time']).total_seconds()
    })

@app.route('/api/security/summary')
def get_security_summary():
    """Get security summary (admin only)"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    # Check authentication manually
    auth_result = security_middleware._authenticate_request() if security_middleware else {'success': False}
    if not auth_result['success']:
        return jsonify({'error': 'Authentication required'}), 401
        
    user_roles = auth_result.get('roles', [])
    if not any(role in ['admin', 'moderator'] for role in user_roles):
        return jsonify({'error': 'Admin privileges required'}), 403
        
    try:
        summary = security_monitor.get_security_summary(24)
        return jsonify(summary)
    except Exception as e:
        logger.error(f"Error getting security summary: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/security/alerts')
def get_security_alerts():
    """Get active security alerts (admin only)"""
    if not security_middleware:
        return jsonify({'error': 'Security system not initialized'}), 500
        
    # Check authentication manually
    auth_result = security_middleware._authenticate_request() if security_middleware else {'success': False}
    if not auth_result['success']:
        return jsonify({'error': 'Authentication required'}), 401
        
    user_roles = auth_result.get('roles', [])
    if not any(role in ['admin', 'moderator'] for role in user_roles):
        return jsonify({'error': 'Admin privileges required'}), 403
        
    try:
        alerts = security_monitor.get_active_alerts()
        alert_data = []
        for alert in alerts:
            alert_data.append({
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'level': alert.alert_level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            })
        return jsonify({'alerts': alert_data})
    except Exception as e:
        logger.error(f"Error getting security alerts: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'security': 'enabled' if CONFIG['security_enabled'] else 'disabled'
    })

# Error handlers
@app.errorhandler(401)
def unauthorized(error):
    return jsonify({'error': 'Authentication required'}), 401

@app.errorhandler(403)
def forbidden(error):
    return jsonify({'error': 'Access forbidden'}), 403

@app.errorhandler(429)
def rate_limit_exceeded(error):
    return jsonify({'error': 'Rate limit exceeded'}), 429

if __name__ == '__main__':
    # Initialize application
    initialize_app()
    
    # Log startup information
    logger.info("=" * 60)
    logger.info("🚀 DEEP TREE ECHO - SECURE COGNITIVE ARCHITECTURE")
    logger.info("=" * 60)
    logger.info(f"Security Framework: {'✅ ENABLED' if CONFIG['security_enabled'] else '❌ DISABLED'}")
    logger.info(f"Persistent Memory: {'✅ ENABLED' if CONFIG['enable_persistence'] else '❌ DISABLED'}")
    logger.info(f"WebVM Mode: {'✅ ENABLED' if CONFIG['webvm_mode'] else '❌ DISABLED'}")
    logger.info("=" * 60)
    
    # Start application
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV', 'production') != 'production'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug,
        threaded=True
    )