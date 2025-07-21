"""
Authentication System
Handles user authentication, JWT tokens, MFA, and session management
"""

import jwt
import hashlib
import secrets
import pyotp
import qrcode
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import json
import os
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import logging

logger = logging.getLogger(__name__)

@dataclass
class User:
    """User data model"""
    user_id: str
    username: str
    email: str
    password_hash: str
    salt: str
    mfa_secret: Optional[str] = None
    mfa_enabled: bool = False
    roles: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    is_active: bool = True
    profile: Dict[str, Any] = field(default_factory=dict)
    failed_login_attempts: int = 0
    locked_until: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert user to dictionary"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'roles': self.roles,
            'created_at': self.created_at.isoformat(),
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'is_active': self.is_active,
            'profile': self.profile,
            'mfa_enabled': self.mfa_enabled
        }

@dataclass
class AuthToken:
    """Authentication token data model"""
    token: str
    user_id: str
    expires_at: datetime
    token_type: str = 'access'  # access, refresh, reset
    scopes: List[str] = field(default_factory=list)

class AuthenticationSystem:
    """Comprehensive authentication system with JWT, MFA, and session management"""
    
    def __init__(self, secret_key: str, token_expiry_hours: int = 24):
        self.secret_key = secret_key
        self.token_expiry_hours = token_expiry_hours
        self.users: Dict[str, User] = {}
        self.active_tokens: Dict[str, AuthToken] = {}
        self.blacklisted_tokens: set = set()
        self.password_policy = {
            'min_length': 8,
            'require_uppercase': True,
            'require_lowercase': True,
            'require_numbers': True,
            'require_special': True
        }
        self.lockout_policy = {
            'max_attempts': 5,
            'lockout_duration_minutes': 30
        }
        
    def hash_password(self, password: str) -> tuple[str, str]:
        """Hash password with salt"""
        salt = secrets.token_hex(32)
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        return base64.b64encode(password_hash).decode('utf-8'), salt
    
    def verify_password(self, password: str, password_hash: str, salt: str) -> bool:
        """Verify password against hash"""
        try:
            computed_hash = hashlib.pbkdf2_hmac(
                'sha256',
                password.encode('utf-8'),
                salt.encode('utf-8'),
                100000
            )
            return base64.b64encode(computed_hash).decode('utf-8') == password_hash
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    def validate_password_policy(self, password: str) -> tuple[bool, List[str]]:
        """Validate password against policy"""
        errors = []
        
        if len(password) < self.password_policy['min_length']:
            errors.append(f"Password must be at least {self.password_policy['min_length']} characters")
        
        if self.password_policy['require_uppercase'] and not any(c.isupper() for c in password):
            errors.append("Password must contain at least one uppercase letter")
        
        if self.password_policy['require_lowercase'] and not any(c.islower() for c in password):
            errors.append("Password must contain at least one lowercase letter")
        
        if self.password_policy['require_numbers'] and not any(c.isdigit() for c in password):
            errors.append("Password must contain at least one number")
        
        if self.password_policy['require_special'] and not any(c in "!@#$%^&*()_+-=" for c in password):
            errors.append("Password must contain at least one special character")
        
        return len(errors) == 0, errors
    
    def register_user(self, username: str, email: str, password: str, roles: List[str] = None) -> tuple[bool, str, Optional[User]]:
        """Register new user"""
        try:
            # Validate inputs
            if not username or not email or not password:
                return False, "Username, email, and password are required", None
            
            # Check if user exists
            if any(u.username == username or u.email == email for u in self.users.values()):
                return False, "Username or email already exists", None
            
            # Validate password
            is_valid, password_errors = self.validate_password_policy(password)
            if not is_valid:
                return False, "; ".join(password_errors), None
            
            # Create user
            user_id = secrets.token_urlsafe(16)
            password_hash, salt = self.hash_password(password)
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                password_hash=password_hash,
                salt=salt,
                roles=roles or ['user']
            )
            
            self.users[user_id] = user
            logger.info(f"User registered: {username} ({user_id})")
            
            return True, "User registered successfully", user
            
        except Exception as e:
            logger.error(f"User registration error: {e}")
            return False, f"Registration failed: {str(e)}", None
    
    def authenticate_user(self, username: str, password: str, mfa_token: Optional[str] = None) -> tuple[bool, str, Optional[AuthToken]]:
        """Authenticate user with optional MFA"""
        try:
            # Find user
            user = None
            for u in self.users.values():
                if u.username == username or u.email == username:
                    user = u
                    break
            
            if not user:
                return False, "Invalid credentials", None
            
            # Check if account is locked
            if user.locked_until and datetime.now() < user.locked_until:
                return False, "Account temporarily locked", None
            
            # Verify password
            if not self.verify_password(password, user.password_hash, user.salt):
                # Increment failed attempts
                user.failed_login_attempts += 1
                if user.failed_login_attempts >= self.lockout_policy['max_attempts']:
                    user.locked_until = datetime.now() + timedelta(minutes=self.lockout_policy['lockout_duration_minutes'])
                    logger.warning(f"Account locked for user {username}")
                return False, "Invalid credentials", None
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_token:
                    return False, "MFA token required", None
                
                if not self.verify_mfa_token(user, mfa_token):
                    return False, "Invalid MFA token", None
            
            # Reset failed attempts on successful login
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.now()
            
            # Generate auth token
            token = self.generate_token(user)
            
            logger.info(f"User authenticated: {username}")
            return True, "Authentication successful", token
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            return False, f"Authentication failed: {str(e)}", None
    
    def generate_token(self, user: User, token_type: str = 'access', scopes: List[str] = None) -> AuthToken:
        """Generate JWT token"""
        expires_at = datetime.now() + timedelta(hours=self.token_expiry_hours)
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'roles': user.roles,
            'scopes': scopes or [],
            'token_type': token_type,
            'exp': expires_at.timestamp(),
            'iat': datetime.now().timestamp()
        }
        
        token_string = jwt.encode(payload, self.secret_key, algorithm='HS256')
        
        auth_token = AuthToken(
            token=token_string,
            user_id=user.user_id,
            expires_at=expires_at,
            token_type=token_type,
            scopes=scopes or []
        )
        
        self.active_tokens[token_string] = auth_token
        return auth_token
    
    def verify_token(self, token: str) -> tuple[bool, Optional[Dict[str, Any]]]:
        """Verify JWT token"""
        try:
            # Check if token is blacklisted
            if token in self.blacklisted_tokens:
                return False, None
            
            # Decode token
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            
            # Check if token is in active tokens
            if token not in self.active_tokens:
                return False, None
            
            # Check expiration
            if datetime.now().timestamp() > payload['exp']:
                self.revoke_token(token)
                return False, None
            
            return True, payload
            
        except jwt.ExpiredSignatureError:
            self.revoke_token(token)
            return False, None
        except jwt.InvalidTokenError:
            return False, None
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            return False, None
    
    def revoke_token(self, token: str):
        """Revoke token"""
        self.blacklisted_tokens.add(token)
        if token in self.active_tokens:
            del self.active_tokens[token]
    
    def setup_mfa(self, user_id: str) -> tuple[bool, str, Optional[str]]:
        """Setup MFA for user"""
        try:
            user = self.users.get(user_id)
            if not user:
                return False, "User not found", None
            
            # Generate MFA secret
            secret = pyotp.random_base32()
            user.mfa_secret = secret
            
            # Generate QR code URL
            totp = pyotp.TOTP(secret)
            qr_url = totp.provisioning_uri(
                name=user.email,
                issuer_name="Deep Tree Echo"
            )
            
            logger.info(f"MFA setup for user {user.username}")
            return True, "MFA setup successful", qr_url
            
        except Exception as e:
            logger.error(f"MFA setup error: {e}")
            return False, f"MFA setup failed: {str(e)}", None
    
    def enable_mfa(self, user_id: str, verification_token: str) -> tuple[bool, str]:
        """Enable MFA after verification"""
        try:
            user = self.users.get(user_id)
            if not user or not user.mfa_secret:
                return False, "MFA not set up"
            
            if self.verify_mfa_token(user, verification_token):
                user.mfa_enabled = True
                logger.info(f"MFA enabled for user {user.username}")
                return True, "MFA enabled successfully"
            else:
                return False, "Invalid verification token"
                
        except Exception as e:
            logger.error(f"MFA enable error: {e}")
            return False, f"MFA enable failed: {str(e)}"
    
    def verify_mfa_token(self, user: User, token: str) -> bool:
        """Verify MFA token"""
        try:
            if not user.mfa_secret:
                return False
            
            totp = pyotp.TOTP(user.mfa_secret)
            return totp.verify(token)
            
        except Exception as e:
            logger.error(f"MFA verification error: {e}")
            return False
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        return self.users.get(user_id)
    
    def update_user_profile(self, user_id: str, profile_data: Dict[str, Any]) -> tuple[bool, str]:
        """Update user profile"""
        try:
            user = self.users.get(user_id)
            if not user:
                return False, "User not found"
            
            # Update allowed profile fields
            allowed_fields = {'display_name', 'phone', 'timezone', 'preferences'}
            for key, value in profile_data.items():
                if key in allowed_fields:
                    user.profile[key] = value
            
            logger.info(f"Profile updated for user {user.username}")
            return True, "Profile updated successfully"
            
        except Exception as e:
            logger.error(f"Profile update error: {e}")
            return False, f"Profile update failed: {str(e)}"
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> tuple[bool, str]:
        """Change user password"""
        try:
            user = self.users.get(user_id)
            if not user:
                return False, "User not found"
            
            # Verify current password
            if not self.verify_password(current_password, user.password_hash, user.salt):
                return False, "Current password is incorrect"
            
            # Validate new password
            is_valid, password_errors = self.validate_password_policy(new_password)
            if not is_valid:
                return False, "; ".join(password_errors)
            
            # Update password
            user.password_hash, user.salt = self.hash_password(new_password)
            
            logger.info(f"Password changed for user {user.username}")
            return True, "Password changed successfully"
            
        except Exception as e:
            logger.error(f"Password change error: {e}")
            return False, f"Password change failed: {str(e)}"
    
    def cleanup_expired_tokens(self):
        """Clean up expired tokens"""
        now = datetime.now()
        expired_tokens = [
            token for token, auth_token in self.active_tokens.items()
            if auth_token.expires_at < now
        ]
        
        for token in expired_tokens:
            self.revoke_token(token)
        
        logger.debug(f"Cleaned up {len(expired_tokens)} expired tokens")