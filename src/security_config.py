"""
Security Configuration
Configuration settings for the Deep Tree Echo security framework
"""

import os
from datetime import timedelta

class SecurityConfig:
    """Security configuration settings"""
    
    # Authentication settings
    SECRET_KEY = os.environ.get('SECURITY_SECRET_KEY', 'deep_tree_echo_default_secret_key_change_in_production')
    TOKEN_EXPIRY_HOURS = int(os.environ.get('TOKEN_EXPIRY_HOURS', '24'))
    REFRESH_TOKEN_EXPIRY_DAYS = int(os.environ.get('REFRESH_TOKEN_EXPIRY_DAYS', '30'))
    
    # Password policy
    PASSWORD_MIN_LENGTH = int(os.environ.get('PASSWORD_MIN_LENGTH', '8'))
    PASSWORD_REQUIRE_UPPERCASE = os.environ.get('PASSWORD_REQUIRE_UPPERCASE', 'true').lower() == 'true'
    PASSWORD_REQUIRE_LOWERCASE = os.environ.get('PASSWORD_REQUIRE_LOWERCASE', 'true').lower() == 'true'
    PASSWORD_REQUIRE_NUMBERS = os.environ.get('PASSWORD_REQUIRE_NUMBERS', 'true').lower() == 'true'
    PASSWORD_REQUIRE_SPECIAL = os.environ.get('PASSWORD_REQUIRE_SPECIAL', 'true').lower() == 'true'
    
    # Account lockout policy
    MAX_FAILED_LOGIN_ATTEMPTS = int(os.environ.get('MAX_FAILED_LOGIN_ATTEMPTS', '5'))
    LOCKOUT_DURATION_MINUTES = int(os.environ.get('LOCKOUT_DURATION_MINUTES', '30'))
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_GUEST_PER_MINUTE = int(os.environ.get('RATE_LIMIT_GUEST_PER_MINUTE', '20'))
    RATE_LIMIT_USER_PER_MINUTE = int(os.environ.get('RATE_LIMIT_USER_PER_MINUTE', '60'))
    RATE_LIMIT_PREMIUM_PER_MINUTE = int(os.environ.get('RATE_LIMIT_PREMIUM_PER_MINUTE', '120'))
    RATE_LIMIT_ADMIN_PER_MINUTE = int(os.environ.get('RATE_LIMIT_ADMIN_PER_MINUTE', '200'))
    
    # Security monitoring
    SECURITY_MONITORING_ENABLED = os.environ.get('SECURITY_MONITORING_ENABLED', 'true').lower() == 'true'
    ALERT_RETENTION_DAYS = int(os.environ.get('ALERT_RETENTION_DAYS', '90'))
    AUTO_BLOCK_ENABLED = os.environ.get('AUTO_BLOCK_ENABLED', 'true').lower() == 'true'
    
    # Encryption
    ENCRYPTION_ENABLED = os.environ.get('ENCRYPTION_ENABLED', 'true').lower() == 'true'
    KEY_ROTATION_INTERVAL_DAYS = int(os.environ.get('KEY_ROTATION_INTERVAL_DAYS', '30'))
    
    # HTTPS enforcement
    ENFORCE_HTTPS = os.environ.get('ENFORCE_HTTPS', 'false').lower() == 'true'
    
    # CORS settings
    CORS_ENABLED = os.environ.get('CORS_ENABLED', 'true').lower() == 'true'
    CORS_ORIGINS = os.environ.get('CORS_ORIGINS', '*').split(',')
    
    # Session settings
    SESSION_COOKIE_SECURE = os.environ.get('SESSION_COOKIE_SECURE', 'false').lower() == 'true'
    SESSION_COOKIE_HTTPONLY = os.environ.get('SESSION_COOKIE_HTTPONLY', 'true').lower() == 'true'
    SESSION_COOKIE_SAMESITE = os.environ.get('SESSION_COOKIE_SAMESITE', 'Lax')
    
    # Default admin user (for initial setup)
    DEFAULT_ADMIN_USERNAME = os.environ.get('DEFAULT_ADMIN_USERNAME', 'admin')
    DEFAULT_ADMIN_EMAIL = os.environ.get('DEFAULT_ADMIN_EMAIL', 'admin@deeptreeecho.com')
    DEFAULT_ADMIN_PASSWORD = os.environ.get('DEFAULT_ADMIN_PASSWORD', 'ChangeMe123!')
    
    # Security headers
    SECURITY_HEADERS = {
        'X-Content-Type-Options': 'nosniff',
        'X-Frame-Options': 'DENY',
        'X-XSS-Protection': '1; mode=block',
        'Strict-Transport-Security': 'max-age=31536000; includeSubDomains' if ENFORCE_HTTPS else None,
        'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'",
        'Referrer-Policy': 'strict-origin-when-cross-origin'
    }
    
    @classmethod
    def get_config_dict(cls) -> dict:
        """Get configuration as dictionary"""
        return {
            'secret_key': cls.SECRET_KEY,
            'token_expiry_hours': cls.TOKEN_EXPIRY_HOURS,
            'password_policy': {
                'min_length': cls.PASSWORD_MIN_LENGTH,
                'require_uppercase': cls.PASSWORD_REQUIRE_UPPERCASE,
                'require_lowercase': cls.PASSWORD_REQUIRE_LOWERCASE,
                'require_numbers': cls.PASSWORD_REQUIRE_NUMBERS,
                'require_special': cls.PASSWORD_REQUIRE_SPECIAL
            },
            'lockout_policy': {
                'max_attempts': cls.MAX_FAILED_LOGIN_ATTEMPTS,
                'lockout_duration_minutes': cls.LOCKOUT_DURATION_MINUTES
            },
            'rate_limiting': {
                'enabled': cls.RATE_LIMIT_ENABLED,
                'limits': {
                    'guest': {'minute': cls.RATE_LIMIT_GUEST_PER_MINUTE},
                    'user': {'minute': cls.RATE_LIMIT_USER_PER_MINUTE},
                    'premium_user': {'minute': cls.RATE_LIMIT_PREMIUM_PER_MINUTE},
                    'admin': {'minute': cls.RATE_LIMIT_ADMIN_PER_MINUTE}
                }
            },
            'monitoring': {
                'enabled': cls.SECURITY_MONITORING_ENABLED,
                'alert_retention_days': cls.ALERT_RETENTION_DAYS,
                'auto_block_enabled': cls.AUTO_BLOCK_ENABLED
            },
            'encryption': {
                'enabled': cls.ENCRYPTION_ENABLED,
                'key_rotation_interval_days': cls.KEY_ROTATION_INTERVAL_DAYS
            },
            'https': {
                'enforce': cls.ENFORCE_HTTPS
            },
            'cors': {
                'enabled': cls.CORS_ENABLED,
                'origins': cls.CORS_ORIGINS
            }
        }
    
    @classmethod
    def validate_config(cls) -> list:
        """Validate configuration and return list of warnings/errors"""
        warnings = []
        
        # Check secret key
        if cls.SECRET_KEY == 'deep_tree_echo_default_secret_key_change_in_production':
            warnings.append("⚠️  Using default secret key - change in production!")
        
        if len(cls.SECRET_KEY) < 32:
            warnings.append("⚠️  Secret key should be at least 32 characters long")
        
        # Check default admin password
        if cls.DEFAULT_ADMIN_PASSWORD == 'ChangeMe123!':
            warnings.append("⚠️  Using default admin password - change immediately!")
        
        # Check HTTPS enforcement in production
        if not cls.ENFORCE_HTTPS and os.environ.get('FLASK_ENV') == 'production':
            warnings.append("⚠️  HTTPS not enforced in production environment")
        
        # Check CORS origins
        if '*' in cls.CORS_ORIGINS and os.environ.get('FLASK_ENV') == 'production':
            warnings.append("⚠️  CORS allows all origins in production - security risk")
        
        return warnings