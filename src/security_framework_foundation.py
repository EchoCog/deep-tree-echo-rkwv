"""
P0-003: Security Framework Foundation
This module provides the foundation for enterprise-grade security features.
"""

import os
import json
import hashlib
import secrets
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
import threading
from enum import Enum

logger = logging.getLogger(__name__)

class UserRole(Enum):
    """User roles for authorization"""
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"
    API_USER = "api_user"

class SecurityEvent(Enum):
    """Security event types for logging"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    ACCESS_DENIED = "access_denied"
    PASSWORD_CHANGE = "password_change"
    SESSION_EXPIRED = "session_expired"
    API_KEY_USED = "api_key_used"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"

@dataclass
class User:
    """User account information"""
    user_id: str
    username: str
    email: str
    password_hash: str
    role: UserRole
    created_at: float
    last_login: Optional[float] = None
    is_active: bool = True
    failed_login_attempts: int = 0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class Session:
    """User session information"""
    session_id: str
    user_id: str
    created_at: float
    last_accessed: float
    expires_at: float
    ip_address: str
    user_agent: str
    is_active: bool = True
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class APIKey:
    """API key for programmatic access"""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    permissions: List[str]
    created_at: float
    expires_at: Optional[float] = None
    last_used: Optional[float] = None
    is_active: bool = True
    rate_limit: int = 1000  # requests per hour
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class SecurityStorage(ABC):
    """Abstract storage interface for security data"""
    
    @abstractmethod
    def store_user(self, user: User) -> bool:
        """Store user account"""
        pass
    
    @abstractmethod
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        pass
    
    @abstractmethod
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        pass
    
    @abstractmethod
    def store_session(self, session: Session) -> bool:
        """Store user session"""
        pass
    
    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session by ID"""
        pass
    
    @abstractmethod
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        pass
    
    @abstractmethod
    def store_api_key(self, api_key: APIKey) -> bool:
        """Store API key"""
        pass
    
    @abstractmethod
    def get_api_key(self, key_hash: str) -> Optional[APIKey]:
        """Get API key by hash"""
        pass

class InMemorySecurityStorage(SecurityStorage):
    """In-memory storage for development and testing"""
    
    def __init__(self):
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, Session] = {}
        self.api_keys: Dict[str, APIKey] = {}
        self._lock = threading.Lock()
        
        # Create default admin user
        self._create_default_admin()
        logger.info("In-memory security storage initialized")
    
    def _create_default_admin(self):
        """Create default admin user for initial setup"""
        admin_password = "admin123"  # In production, this should be randomly generated
        admin_user = User(
            user_id="admin",
            username="admin",
            email="admin@localhost",
            password_hash=self._hash_password(admin_password),
            role=UserRole.ADMIN,
            created_at=time.time()
        )
        self.users["admin"] = admin_user
        logger.info("Default admin user created (username: admin, password: admin123)")
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    def store_user(self, user: User) -> bool:
        with self._lock:
            self.users[user.username] = user
            return True
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        return self.users.get(username)
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        for user in self.users.values():
            if user.user_id == user_id:
                return user
        return None
    
    def store_session(self, session: Session) -> bool:
        with self._lock:
            self.sessions[session.session_id] = session
            return True
    
    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                return True
            return False
    
    def store_api_key(self, api_key: APIKey) -> bool:
        with self._lock:
            self.api_keys[api_key.key_hash] = api_key
            return True
    
    def get_api_key(self, key_hash: str) -> Optional[APIKey]:
        return self.api_keys.get(key_hash)

class PasswordManager:
    """Password hashing and validation"""
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"{salt}:{password_hash.hex()}"
    
    @staticmethod
    def verify_password(password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        try:
            salt, stored_hash = password_hash.split(':')
            password_hash_check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
            return stored_hash == password_hash_check.hex()
        except (ValueError, AttributeError):
            return False
    
    @staticmethod
    def generate_secure_password(length: int = 16) -> str:
        """Generate a secure random password"""
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        return ''.join(secrets.choice(alphabet) for _ in range(length))

class SessionManager:
    """Manage user sessions"""
    
    def __init__(self, storage: SecurityStorage, session_timeout: int = 3600):
        self.storage = storage
        self.session_timeout = session_timeout  # seconds
        self._cleanup_thread = None
        self._start_cleanup_thread()
        logger.info(f"Session manager initialized with {session_timeout}s timeout")
    
    def create_session(self, user_id: str, ip_address: str, user_agent: str) -> str:
        """Create a new user session"""
        session_id = secrets.token_urlsafe(32)
        now = time.time()
        
        session = Session(
            session_id=session_id,
            user_id=user_id,
            created_at=now,
            last_accessed=now,
            expires_at=now + self.session_timeout,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        if self.storage.store_session(session):
            logger.info(f"Created session {session_id} for user {user_id}")
            return session_id
        else:
            raise Exception("Failed to create session")
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[Session]:
        """Validate and refresh a session"""
        session = self.storage.get_session(session_id)
        
        if not session:
            return None
        
        now = time.time()
        
        # Check if expired
        if not session.is_active or now > session.expires_at:
            self.storage.delete_session(session_id)
            return None
        
        # Check IP address if provided
        if ip_address and session.ip_address != ip_address:
            logger.warning(f"IP address mismatch for session {session_id}: {ip_address} vs {session.ip_address}")
            # Could implement IP change handling here
        
        # Refresh session
        session.last_accessed = now
        session.expires_at = now + self.session_timeout
        self.storage.store_session(session)
        
        return session
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session (logout)"""
        result = self.storage.delete_session(session_id)
        if result:
            logger.info(f"Deleted session {session_id}")
        return result
    
    def _start_cleanup_thread(self):
        """Start background thread to cleanup expired sessions"""
        def cleanup():
            while True:
                try:
                    time.sleep(300)  # Check every 5 minutes
                    # In a real implementation, this would query storage for expired sessions
                    # For now, this is just a placeholder
                except Exception as e:
                    logger.error(f"Session cleanup error: {e}")
        
        self._cleanup_thread = threading.Thread(target=cleanup, daemon=True)
        self._cleanup_thread.start()

class APIKeyManager:
    """Manage API keys for programmatic access"""
    
    def __init__(self, storage: SecurityStorage):
        self.storage = storage
        logger.info("API key manager initialized")
    
    def create_api_key(self, user_id: str, name: str, permissions: List[str] = None,
                      expires_in_days: int = None) -> Tuple[str, str]:
        """Create a new API key"""
        if permissions is None:
            permissions = ["read"]
        
        # Generate key
        key = secrets.token_urlsafe(32)
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        now = time.time()
        expires_at = None
        if expires_in_days:
            expires_at = now + (expires_in_days * 24 * 3600)
        
        api_key = APIKey(
            key_id=secrets.token_hex(8),
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            permissions=permissions,
            created_at=now,
            expires_at=expires_at
        )
        
        if self.storage.store_api_key(api_key):
            logger.info(f"Created API key {api_key.key_id} for user {user_id}")
            return key, api_key.key_id
        else:
            raise Exception("Failed to create API key")
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate an API key"""
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        api_key = self.storage.get_api_key(key_hash)
        
        if not api_key:
            return None
        
        now = time.time()
        
        # Check if expired or inactive
        if not api_key.is_active or (api_key.expires_at and now > api_key.expires_at):
            return None
        
        # Update last used
        api_key.last_used = now
        self.storage.store_api_key(api_key)
        
        return api_key

class SecurityAuditLogger:
    """Log security events for auditing"""
    
    def __init__(self, log_file: str = "/tmp/echo_security.log"):
        self.log_file = log_file
        self._setup_logger()
        logger.info(f"Security audit logger initialized: {log_file}")
    
    def _setup_logger(self):
        """Setup dedicated security logger"""
        self.security_logger = logging.getLogger("security_audit")
        self.security_logger.setLevel(logging.INFO)
        
        # File handler
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        handler.setFormatter(formatter)
        self.security_logger.addHandler(handler)
    
    def log_event(self, event: SecurityEvent, user_id: str = None, 
                  session_id: str = None, ip_address: str = None,
                  additional_data: Dict[str, Any] = None):
        """Log a security event"""
        event_data = {
            "event": event.value,
            "timestamp": time.time(),
            "user_id": user_id,
            "session_id": session_id,
            "ip_address": ip_address
        }
        
        if additional_data:
            event_data.update(additional_data)
        
        self.security_logger.info(json.dumps(event_data))

class SecurityFramework:
    """Main security framework coordinating all security components"""
    
    def __init__(self, storage: SecurityStorage = None):
        if storage is None:
            storage = InMemorySecurityStorage()
        
        self.storage = storage
        self.password_manager = PasswordManager()
        self.session_manager = SessionManager(storage)
        self.api_key_manager = APIKeyManager(storage)
        self.audit_logger = SecurityAuditLogger()
        
        # Rate limiting (simple implementation)
        self.rate_limits = {}
        self._rate_limit_window = 3600  # 1 hour
        
        logger.info("Security framework initialized")
    
    def authenticate_user(self, username: str, password: str, 
                         ip_address: str, user_agent: str) -> Optional[str]:
        """Authenticate user and create session"""
        user = self.storage.get_user_by_username(username)
        
        if not user or not user.is_active:
            self.audit_logger.log_event(
                SecurityEvent.LOGIN_FAILURE, 
                user_id=user.user_id if user else None,
                ip_address=ip_address,
                additional_data={"reason": "user_not_found_or_inactive"}
            )
            return None
        
        # Check password
        if not self.password_manager.verify_password(password, user.password_hash):
            user.failed_login_attempts += 1
            self.storage.store_user(user)
            
            self.audit_logger.log_event(
                SecurityEvent.LOGIN_FAILURE,
                user_id=user.user_id,
                ip_address=ip_address,
                additional_data={"reason": "invalid_password", "attempts": user.failed_login_attempts}
            )
            return None
        
        # Reset failed attempts on successful login
        user.failed_login_attempts = 0
        user.last_login = time.time()
        self.storage.store_user(user)
        
        # Create session
        session_id = self.session_manager.create_session(user.user_id, ip_address, user_agent)
        
        self.audit_logger.log_event(
            SecurityEvent.LOGIN_SUCCESS,
            user_id=user.user_id,
            session_id=session_id,
            ip_address=ip_address
        )
        
        return session_id
    
    def validate_session(self, session_id: str, ip_address: str = None) -> Optional[User]:
        """Validate session and return user"""
        session = self.session_manager.validate_session(session_id, ip_address)
        
        if not session:
            return None
        
        user = self.storage.get_user_by_id(session.user_id)
        return user if user and user.is_active else None
    
    def validate_api_key(self, api_key: str, required_permissions: List[str] = None) -> Optional[User]:
        """Validate API key and return user"""
        api_key_obj = self.api_key_manager.validate_api_key(api_key)
        
        if not api_key_obj:
            return None
        
        # Check permissions
        if required_permissions:
            if not all(perm in api_key_obj.permissions for perm in required_permissions):
                self.audit_logger.log_event(
                    SecurityEvent.ACCESS_DENIED,
                    user_id=api_key_obj.user_id,
                    additional_data={"reason": "insufficient_permissions", "required": required_permissions}
                )
                return None
        
        # Rate limiting check
        if not self._check_rate_limit(api_key_obj.key_hash, api_key_obj.rate_limit):
            self.audit_logger.log_event(
                SecurityEvent.ACCESS_DENIED,
                user_id=api_key_obj.user_id,
                additional_data={"reason": "rate_limit_exceeded"}
            )
            return None
        
        self.audit_logger.log_event(
            SecurityEvent.API_KEY_USED,
            user_id=api_key_obj.user_id,
            additional_data={"key_id": api_key_obj.key_id}
        )
        
        user = self.storage.get_user_by_id(api_key_obj.user_id)
        return user if user and user.is_active else None
    
    def logout_user(self, session_id: str):
        """Logout user and delete session"""
        session = self.storage.get_session(session_id)
        if session:
            self.session_manager.delete_session(session_id)
            self.audit_logger.log_event(
                SecurityEvent.LOGOUT,
                user_id=session.user_id,
                session_id=session_id
            )
    
    def create_user(self, username: str, email: str, password: str, 
                   role: UserRole = UserRole.USER) -> str:
        """Create a new user account"""
        # Check if user already exists
        if self.storage.get_user_by_username(username):
            raise Exception("User already exists")
        
        user_id = secrets.token_hex(16)
        password_hash = self.password_manager.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            role=role,
            created_at=time.time()
        )
        
        if self.storage.store_user(user):
            logger.info(f"Created user {username} with role {role.value}")
            return user_id
        else:
            raise Exception("Failed to create user")
    
    def _check_rate_limit(self, identifier: str, limit: int) -> bool:
        """Simple rate limiting implementation"""
        now = time.time()
        window_start = now - self._rate_limit_window
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Clean old entries
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if timestamp > window_start
        ]
        
        # Check limit
        if len(self.rate_limits[identifier]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security system status"""
        return {
            "storage_type": self.storage.__class__.__name__,
            "session_timeout": self.session_manager.session_timeout,
            "rate_limit_window": self._rate_limit_window,
            "audit_log_file": getattr(self.audit_logger, 'log_file', 'unknown')
        }

# Global security framework instance
_global_security_framework = None

def get_security_framework(storage: SecurityStorage = None) -> SecurityFramework:
    """Get or create global security framework"""
    global _global_security_framework
    if _global_security_framework is None:
        _global_security_framework = SecurityFramework(storage)
    return _global_security_framework