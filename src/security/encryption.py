"""
Encryption Manager
Handles data encryption, key management, and secure communication
"""

import os
import base64
import json
import hashlib
import secrets
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.backends import default_backend
import logging

logger = logging.getLogger(__name__)

class EncryptionManager:
    """Comprehensive encryption and key management system"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.backend = default_backend()
        self.master_key = master_key or self._generate_master_key()
        self.encryption_keys: Dict[str, bytes] = {}
        self.key_metadata: Dict[str, Dict[str, Any]] = {}
        self.key_rotation_interval_days = 30
        
        # Initialize default encryption contexts
        self._initialize_encryption_contexts()
    
    def _generate_master_key(self) -> str:
        """Generate a master key for the encryption system"""
        return base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8')
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password and salt"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=self.backend
        )
        return kdf.derive(password.encode('utf-8'))
    
    def _initialize_encryption_contexts(self):
        """Initialize encryption contexts for different data types"""
        contexts = [
            'user_data',      # User profile and authentication data
            'session_data',   # Session information
            'memory_data',    # Cognitive memory storage
            'system_data',    # System configuration and logs
            'temp_data'       # Temporary data encryption
        ]
        
        for context in contexts:
            self.create_encryption_context(context)
    
    def create_encryption_context(self, context_name: str) -> bool:
        """Create a new encryption context with its own key"""
        try:
            # Generate unique salt for this context
            salt = secrets.token_bytes(32)
            
            # Derive key from master key and context-specific salt
            context_key = self._derive_key(self.master_key, salt)
            
            # Store key and metadata
            self.encryption_keys[context_name] = context_key
            self.key_metadata[context_name] = {
                'created_at': datetime.now().isoformat(),
                'salt': base64.b64encode(salt).decode('utf-8'),
                'rotation_due': (datetime.now() + timedelta(days=self.key_rotation_interval_days)).isoformat(),
                'usage_count': 0
            }
            
            logger.debug(f"Created encryption context: {context_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create encryption context {context_name}: {e}")
            return False
    
    def encrypt_data(self, data: str, context: str = 'temp_data') -> Optional[str]:
        """Encrypt data using specified context"""
        try:
            if context not in self.encryption_keys:
                if not self.create_encryption_context(context):
                    return None
            
            # Get encryption key for context
            key = self.encryption_keys[context]
            
            # Create Fernet cipher
            fernet = Fernet(base64.urlsafe_b64encode(key))
            
            # Encrypt data
            encrypted_data = fernet.encrypt(data.encode('utf-8'))
            
            # Update usage count
            self.key_metadata[context]['usage_count'] += 1
            
            # Return base64 encoded encrypted data
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"Encryption failed for context {context}: {e}")
            return None
    
    def decrypt_data(self, encrypted_data: str, context: str = 'temp_data') -> Optional[str]:
        """Decrypt data using specified context"""
        try:
            if context not in self.encryption_keys:
                logger.error(f"Encryption context not found: {context}")
                return None
            
            # Get encryption key for context
            key = self.encryption_keys[context]
            
            # Create Fernet cipher
            fernet = Fernet(base64.urlsafe_b64encode(key))
            
            # Decode and decrypt data
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = fernet.decrypt(encrypted_bytes)
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"Decryption failed for context {context}: {e}")
            return None
    
    def encrypt_json(self, data: Dict[str, Any], context: str = 'temp_data') -> Optional[str]:
        """Encrypt JSON data"""
        try:
            json_str = json.dumps(data, default=str)
            return self.encrypt_data(json_str, context)
        except Exception as e:
            logger.error(f"JSON encryption failed: {e}")
            return None
    
    def decrypt_json(self, encrypted_data: str, context: str = 'temp_data') -> Optional[Dict[str, Any]]:
        """Decrypt JSON data"""
        try:
            decrypted_str = self.decrypt_data(encrypted_data, context)
            if decrypted_str:
                return json.loads(decrypted_str)
            return None
        except Exception as e:
            logger.error(f"JSON decryption failed: {e}")
            return None
    
    def hash_data(self, data: str, salt: Optional[str] = None) -> Tuple[str, str]:
        """Create secure hash of data with salt"""
        if salt is None:
            salt = secrets.token_hex(32)
        
        hash_object = hashlib.pbkdf2_hmac(
            'sha256',
            data.encode('utf-8'),
            salt.encode('utf-8'),
            100000
        )
        
        return base64.b64encode(hash_object).decode('utf-8'), salt
    
    def verify_hash(self, data: str, hash_value: str, salt: str) -> bool:
        """Verify data against hash"""
        try:
            computed_hash, _ = self.hash_data(data, salt)
            return computed_hash == hash_value
        except Exception as e:
            logger.error(f"Hash verification failed: {e}")
            return False
    
    def generate_rsa_keypair(self) -> Tuple[bytes, bytes]:
        """Generate RSA public/private key pair"""
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=self.backend
        )
        
        public_key = private_key.public_key()
        
        # Serialize keys
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        )
        
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )
        
        return private_pem, public_pem
    
    def encrypt_with_public_key(self, data: str, public_key_pem: bytes) -> Optional[str]:
        """Encrypt data using RSA public key"""
        try:
            public_key = serialization.load_pem_public_key(
                public_key_pem,
                backend=self.backend
            )
            
            encrypted_data = public_key.encrypt(
                data.encode('utf-8'),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return base64.b64encode(encrypted_data).decode('utf-8')
            
        except Exception as e:
            logger.error(f"RSA encryption failed: {e}")
            return None
    
    def decrypt_with_private_key(self, encrypted_data: str, private_key_pem: bytes) -> Optional[str]:
        """Decrypt data using RSA private key"""
        try:
            private_key = serialization.load_pem_private_key(
                private_key_pem,
                password=None,
                backend=self.backend
            )
            
            encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
            decrypted_data = private_key.decrypt(
                encrypted_bytes,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            
            return decrypted_data.decode('utf-8')
            
        except Exception as e:
            logger.error(f"RSA decryption failed: {e}")
            return None
    
    def rotate_context_key(self, context: str) -> bool:
        """Rotate encryption key for a context"""
        try:
            if context not in self.encryption_keys:
                return False
            
            # Store old key for data migration
            old_key = self.encryption_keys[context]
            
            # Generate new key
            salt = secrets.token_bytes(32)
            new_key = self._derive_key(self.master_key, salt)
            
            # Update key and metadata
            self.encryption_keys[context] = new_key
            self.key_metadata[context].update({
                'rotated_at': datetime.now().isoformat(),
                'salt': base64.b64encode(salt).decode('utf-8'),
                'rotation_due': (datetime.now() + timedelta(days=self.key_rotation_interval_days)).isoformat(),
                'usage_count': 0
            })
            
            logger.info(f"Rotated encryption key for context: {context}")
            return True
            
        except Exception as e:
            logger.error(f"Key rotation failed for context {context}: {e}")
            return False
    
    def check_key_rotation_needed(self) -> List[str]:
        """Check which contexts need key rotation"""
        contexts_needing_rotation = []
        now = datetime.now()
        
        for context, metadata in self.key_metadata.items():
            rotation_due = datetime.fromisoformat(metadata['rotation_due'])
            if now >= rotation_due:
                contexts_needing_rotation.append(context)
        
        return contexts_needing_rotation
    
    def anonymize_data(self, data: str, anonymization_level: str = 'basic') -> str:
        """Anonymize sensitive data"""
        if anonymization_level == 'basic':
            # Replace email patterns
            import re
            data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', data)
            # Replace phone patterns
            data = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', data)
            # Replace potential IDs
            data = re.sub(r'\b[A-Za-z0-9]{8,32}\b', '[ID]', data)
            
        elif anonymization_level == 'aggressive':
            # More aggressive anonymization
            import re
            data = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', data)
            data = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', data)
            data = re.sub(r'\b[A-Za-z0-9]{8,32}\b', '[ID]', data)
            data = re.sub(r'\b\d{4,}\b', '[NUMBER]', data)
            # Replace names (common patterns)
            data = re.sub(r'\b[A-Z][a-z]+ [A-Z][a-z]+\b', '[NAME]', data)
        
        return data
    
    def create_secure_token(self, length: int = 32) -> str:
        """Create a cryptographically secure random token"""
        return secrets.token_urlsafe(length)
    
    def create_api_key(self, user_id: str, scopes: List[str] = None) -> Tuple[str, str]:
        """Create an API key for a user"""
        # Generate key ID and secret
        key_id = f"dtecho_{secrets.token_urlsafe(8)}"
        key_secret = secrets.token_urlsafe(32)
        
        # Create key metadata
        key_data = {
            'user_id': user_id,
            'scopes': scopes or [],
            'created_at': datetime.now().isoformat(),
            'last_used': None,
            'usage_count': 0
        }
        
        # Encrypt key metadata
        encrypted_metadata = self.encrypt_json(key_data, 'system_data')
        
        return key_id, key_secret
    
    def validate_api_key(self, key_id: str, key_secret: str) -> Optional[Dict[str, Any]]:
        """Validate API key and return metadata"""
        # This would typically check against a database
        # For now, return basic validation
        if key_id.startswith('dtecho_') and len(key_secret) >= 32:
            return {
                'valid': True,
                'user_id': 'api_user',
                'scopes': ['api.cognitive_process'],
                'last_used': datetime.now().isoformat()
            }
        return None
    
    def get_encryption_stats(self) -> Dict[str, Any]:
        """Get encryption system statistics"""
        return {
            'contexts': list(self.encryption_keys.keys()),
            'total_contexts': len(self.encryption_keys),
            'contexts_needing_rotation': self.check_key_rotation_needed(),
            'master_key_set': bool(self.master_key),
            'backend': str(self.backend)
        }
    
    def export_encrypted_backup(self, data: Dict[str, Any]) -> Optional[str]:
        """Create encrypted backup of sensitive data"""
        try:
            # Add timestamp to backup
            backup_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            
            # Encrypt backup
            encrypted_backup = self.encrypt_json(backup_data, 'system_data')
            
            logger.info("Created encrypted backup")
            return encrypted_backup
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None
    
    def import_encrypted_backup(self, encrypted_backup: str) -> Optional[Dict[str, Any]]:
        """Import encrypted backup data"""
        try:
            backup_data = self.decrypt_json(encrypted_backup, 'system_data')
            if backup_data and 'data' in backup_data:
                logger.info("Successfully imported encrypted backup")
                return backup_data['data']
            return None
            
        except Exception as e:
            logger.error(f"Backup import failed: {e}")
            return None