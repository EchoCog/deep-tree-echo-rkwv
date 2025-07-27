"""
Authentication Middleware for Deep Tree Echo
Implements JWT-based authentication with rate limiting and security features
"""

import os
import json
import time
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from dataclasses import dataclass, asdict
import jwt
from security_config import SecurityConfig

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User model"""
    id: str
    username: str
    email: str
    password_hash: str
    role: str
    created_at: str
    last_login: str
    failed_login_attempts: int
    locked_until: Optional[str]
    is_active: bool
    metadata: Dict[str, Any]

@dataclass
class Session:
    """Session model"""
    id: str
    user_id: str
    token: str
    refresh_token: str
    created_at: str
    expires_at: str
    last_activity: str
    ip_address: str
    user_agent: str
    is_active: bool

class SimpleUserStore:
    """Simple in-memory user store for demonstration"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.rate_limits: Dict[str, List[float]] = {}
        self._init_default_users()
    
    def _init_default_users(self):
        """Initialize with default admin user"""
        admin_id = "admin-" + secrets.token_hex(8)
        admin_user = User(
            id=admin_id,
            username="admin",
            email="admin@deepecho.ai",
            password_hash=self._hash_password("admin123"),
            role="admin",
            created_at=datetime.now().isoformat(),
            last_login=datetime.now().isoformat(),
            failed_login_attempts=0,
            locked_until=None,
            is_active=True,
            metadata={"created_by": "system"}
        )
        self.users[admin_id] = admin_user
        logger.info("Default admin user created (username: admin, password: admin123)")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(16)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def _verify_password(self, password: str, password_hash: str) -> bool:
        """Verify password against hash"""
        try:
            salt, hash_hex = password_hash.split(':')
            password_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return password_check.hex() == hash_hex
        except:
            return False
    
    def create_user(self, username: str, email: str, password: str, role: str = "user") -> Optional[str]:
        """Create new user"""
        # Check if user exists
        for user in self.users.values():
            if user.username == username or user.email == email:
                return None
        
        user_id = f"user-{secrets.token_hex(8)}"
        user = User(
            id=user_id,
            username=username,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
            created_at=datetime.now().isoformat(),
            last_login="",
            failed_login_attempts=0,
            locked_until=None,
            is_active=True,
            metadata={}
        )
        
        self.users[user_id] = user
        logger.info(f"Created user: {username}")
        return user_id
    
    def authenticate_user(self, username: str, password: str, ip_address: str) -> Optional[User]:
        """Authenticate user with rate limiting and lockout"""
        # Check rate limiting
        if not self._check_rate_limit(ip_address):
            logger.warning(f"Rate limit exceeded for IP: {ip_address}")
            return None
        
        # Find user
        user = None
        for u in self.users.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            self._track_request(ip_address)
            return None
        
        # Check if account is locked
        if user.locked_until:
            locked_until = datetime.fromisoformat(user.locked_until)
            if datetime.now() < locked_until:
                logger.warning(f"Account locked for user: {username}")
                return None
            else:
                # Unlock account
                user.locked_until = None
                user.failed_login_attempts = 0
        
        # Verify password
        if self._verify_password(password, user.password_hash):
            # Successful login
            user.last_login = datetime.now().isoformat()
            user.failed_login_attempts = 0
            self._track_request(ip_address)
            logger.info(f"Successful login for user: {username}")
            return user
        else:
            # Failed login
            user.failed_login_attempts += 1
            if user.failed_login_attempts >= SecurityConfig.MAX_FAILED_LOGIN_ATTEMPTS:
                # Lock account
                lockout_duration = timedelta(minutes=SecurityConfig.LOCKOUT_DURATION_MINUTES)
                user.locked_until = (datetime.now() + lockout_duration).isoformat()
                logger.warning(f"Account locked for user: {username} after {user.failed_login_attempts} failed attempts")
            
            self._track_request(ip_address)
            return None
    
    def _check_rate_limit(self, ip_address: str) -> bool:
        """Check if IP address is within rate limits"""
        if not SecurityConfig.RATE_LIMIT_ENABLED:
            return True
        
        current_time = time.time()
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        
        # Clean old requests
        self.rate_limits[ip_address] = [
            req_time for req_time in self.rate_limits[ip_address]
            if current_time - req_time < 60  # Last minute
        ]
        
        # Check limit
        limit = SecurityConfig.RATE_LIMIT_GUEST_PER_MINUTE
        return len(self.rate_limits[ip_address]) < limit
    
    def _track_request(self, ip_address: str):
        """Track request for rate limiting"""
        current_time = time.time()
        if ip_address not in self.rate_limits:
            self.rate_limits[ip_address] = []
        self.rate_limits[ip_address].append(current_time)

class AuthenticationManager:
    """Main authentication manager"""
    
    def __init__(self, user_store: SimpleUserStore = None):
        self.user_store = user_store or SimpleUserStore()
        self.secret_key = SecurityConfig.SECRET_KEY
        
    def login(self, username: str, password: str, ip_address: str, user_agent: str) -> Optional[Dict[str, Any]]:
        """Login user and create session"""
        user = self.user_store.authenticate_user(username, password, ip_address)
        if not user:
            return None
        
        # Create session
        session_id = f"session-{secrets.token_hex(16)}"
        
        # Create JWT token
        token_payload = {
            'user_id': user.id,
            'username': user.username,
            'role': user.role,
            'session_id': session_id,
            'exp': datetime.now() + timedelta(hours=SecurityConfig.TOKEN_EXPIRY_HOURS),
            'iat': datetime.now()
        }
        
        token = jwt.encode(token_payload, self.secret_key, algorithm='HS256')
        refresh_token = secrets.token_urlsafe(32)
        
        # Store session
        session = Session(
            id=session_id,
            user_id=user.id,
            token=token,
            refresh_token=refresh_token,
            created_at=datetime.now().isoformat(),
            expires_at=(datetime.now() + timedelta(hours=SecurityConfig.TOKEN_EXPIRY_HOURS)).isoformat(),
            last_activity=datetime.now().isoformat(),
            ip_address=ip_address,
            user_agent=user_agent,
            is_active=True
        )
        
        self.user_store.sessions[session_id] = session
        
        return {
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'role': user.role
            },
            'token': token,
            'refresh_token': refresh_token,
            'expires_at': session.expires_at
        }
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return user info"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if session is still active
            session_id = payload.get('session_id')
            if session_id in self.user_store.sessions:
                session = self.user_store.sessions[session_id]
                if session.is_active:
                    # Update last activity
                    session.last_activity = datetime.now().isoformat()
                    return payload
            
            return None
            
        except jwt.ExpiredSignatureError:
            logger.warning("Token expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def logout(self, session_id: str) -> bool:
        """Logout user and invalidate session"""
        if session_id in self.user_store.sessions:
            self.user_store.sessions[session_id].is_active = False
            logger.info(f"Session {session_id} logged out")
            return True
        return False
    
    def refresh_token(self, refresh_token: str) -> Optional[Dict[str, Any]]:
        """Refresh JWT token using refresh token"""
        for session in self.user_store.sessions.values():
            if session.refresh_token == refresh_token and session.is_active:
                # Create new token
                user = self.user_store.users.get(session.user_id)
                if user:
                    return self.login(user.username, "", session.ip_address, session.user_agent)
        return None

# Flask decorators and middleware
def require_auth(required_role: str = None):
    """Decorator to require authentication"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            from flask import request, jsonify, g
            
            auth_header = request.headers.get('Authorization')
            if not auth_header or not auth_header.startswith('Bearer '):
                return jsonify({'error': 'Authentication required'}), 401
            
            token = auth_header.split(' ')[1]
            auth_manager = getattr(g, 'auth_manager', None)
            if not auth_manager:
                return jsonify({'error': 'Authentication system not available'}), 500
            
            user_info = auth_manager.verify_token(token)
            if not user_info:
                return jsonify({'error': 'Invalid or expired token'}), 401
            
            # Check role if required
            if required_role and user_info.get('role') != required_role:
                return jsonify({'error': 'Insufficient permissions'}), 403
            
            # Add user info to request context
            g.current_user = user_info
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def init_auth_middleware(app):
    """Initialize authentication middleware for Flask app"""
    auth_manager = AuthenticationManager()
    
    @app.before_request
    def before_request():
        from flask import g
        g.auth_manager = auth_manager
    
    @app.route('/api/auth/login', methods=['POST'])
    def login():
        from flask import request, jsonify
        
        data = request.get_json()
        if not data or not data.get('username') or not data.get('password'):
            return jsonify({'error': 'Username and password required'}), 400
        
        result = auth_manager.login(
            data['username'],
            data['password'],
            request.remote_addr,
            request.headers.get('User-Agent', '')
        )
        
        if result:
            return jsonify(result)
        else:
            return jsonify({'error': 'Invalid credentials'}), 401
    
    @app.route('/api/auth/logout', methods=['POST'])
    @require_auth()
    def logout():
        from flask import g, jsonify
        
        session_id = g.current_user.get('session_id')
        if auth_manager.logout(session_id):
            return jsonify({'message': 'Logged out successfully'})
        else:
            return jsonify({'error': 'Logout failed'}), 500
    
    @app.route('/api/auth/me', methods=['GET'])
    @require_auth()
    def get_current_user():
        from flask import g, jsonify
        return jsonify(g.current_user)
    
    @app.route('/api/auth/register', methods=['POST'])
    def register():
        from flask import request, jsonify
        
        data = request.get_json()
        required_fields = ['username', 'email', 'password']
        
        if not data or not all(field in data for field in required_fields):
            return jsonify({'error': 'Username, email, and password required'}), 400
        
        user_id = auth_manager.user_store.create_user(
            data['username'],
            data['email'],
            data['password'],
            data.get('role', 'user')
        )
        
        if user_id:
            return jsonify({'message': 'User created successfully', 'user_id': user_id}), 201
        else:
            return jsonify({'error': 'User already exists'}), 409
    
    logger.info("Authentication middleware initialized")
    return auth_manager