"""
Security Framework Integration Test
Tests the comprehensive security system including authentication, authorization, encryption, and monitoring
"""

import pytest
import json
from datetime import datetime, timedelta
from security import (
    AuthenticationSystem, 
    AuthorizationSystem, 
    EncryptionManager, 
    SecurityMonitor, 
    SecurityMiddleware
)
from security.monitoring import SecurityEventType, AlertLevel
from security.authorization import ResourceType, Action

def test_authentication_system():
    """Test authentication system functionality"""
    print("Testing Authentication System...")
    
    # Initialize auth system
    auth = AuthenticationSystem("test_secret_key_123456789")
    
    # Test user registration
    success, message, user = auth.register_user(
        username="testuser",
        email="test@example.com", 
        password="SecurePass123!",
        roles=["user"]
    )
    assert success, f"User registration failed: {message}"
    assert user is not None
    print(f"✅ User registered: {user.username}")
    
    # Test user authentication
    success, message, token = auth.authenticate_user("testuser", "SecurePass123!")
    assert success, f"Authentication failed: {message}"
    assert token is not None
    print(f"✅ User authenticated, token: {token.token[:20]}...")
    
    # Test token verification
    is_valid, payload = auth.verify_token(token.token)
    assert is_valid, "Token verification failed"
    assert payload['username'] == "testuser"
    print("✅ Token verification successful")
    
    # Test MFA setup
    success, message, qr_url = auth.setup_mfa(user.user_id)
    assert success, f"MFA setup failed: {message}"
    print("✅ MFA setup successful")
    
    print("Authentication System: ✅ PASSED\n")

def test_authorization_system():
    """Test authorization system functionality"""
    print("Testing Authorization System...")
    
    # Initialize authorization system
    authz = AuthorizationSystem()
    
    # Test role assignment
    user_id = "test_user_123"
    success = authz.assign_role_to_user(user_id, "user")
    assert success, "Role assignment failed"
    print("✅ Role assigned to user")
    
    # Test permission checking
    has_perm = authz.has_permission(user_id, "session.create")
    assert has_perm, "User should have session.create permission"
    print("✅ Permission check successful")
    
    # Test resource ownership
    resource_id = "session_123"
    authz.set_resource_owner(resource_id, user_id)
    can_access = authz.can_access_resource(user_id, ResourceType.SESSION, Action.READ, resource_id)
    assert can_access, "User should be able to access owned resource"
    print("✅ Resource ownership check successful")
    
    # Test admin privileges
    admin_user = "admin_user_123"
    authz.assign_role_to_user(admin_user, "admin")
    can_admin_access = authz.can_access_resource(admin_user, ResourceType.SESSION, Action.DELETE, resource_id)
    assert can_admin_access, "Admin should be able to access any resource"
    print("✅ Admin privileges check successful")
    
    print("Authorization System: ✅ PASSED\n")

def test_encryption_manager():
    """Test encryption manager functionality"""
    print("Testing Encryption Manager...")
    
    # Initialize encryption manager
    encryption = EncryptionManager()
    
    # Test data encryption/decryption
    test_data = "This is sensitive data that needs encryption"
    encrypted = encryption.encrypt_data(test_data, "user_data")
    assert encrypted is not None, "Data encryption failed"
    print("✅ Data encryption successful")
    
    decrypted = encryption.decrypt_data(encrypted, "user_data")
    assert decrypted == test_data, "Data decryption failed"
    print("✅ Data decryption successful")
    
    # Test JSON encryption/decryption
    test_json = {"user_id": "123", "sensitive": "data", "timestamp": datetime.now().isoformat()}
    encrypted_json = encryption.encrypt_json(test_json, "system_data")
    assert encrypted_json is not None, "JSON encryption failed"
    print("✅ JSON encryption successful")
    
    decrypted_json = encryption.decrypt_json(encrypted_json, "system_data")
    assert decrypted_json["user_id"] == test_json["user_id"], "JSON decryption failed"
    print("✅ JSON decryption successful")
    
    # Test password hashing
    password = "TestPassword123!"
    hash_value, salt = encryption.hash_data(password)
    is_valid = encryption.verify_hash(password, hash_value, salt)
    assert is_valid, "Password hash verification failed"
    print("✅ Password hashing successful")
    
    # Test API key generation
    key_id, key_secret = encryption.create_api_key("user_123", ["api.cognitive_process"])
    assert key_id.startswith("dtecho_"), "API key generation failed"
    print(f"✅ API key generated: {key_id}")
    
    print("Encryption Manager: ✅ PASSED\n")

def test_security_monitor():
    """Test security monitoring functionality"""
    print("Testing Security Monitor...")
    
    # Initialize security monitor
    monitor = SecurityMonitor()
    
    # Test security event logging
    event_id = monitor.log_security_event(
        SecurityEventType.LOGIN_SUCCESS,
        user_id="test_user",
        ip_address="192.168.1.100",
        user_agent="Test Browser",
        details={"method": "password"}
    )
    assert event_id is not None, "Security event logging failed"
    print("✅ Security event logged")
    
    # Test failed login detection
    for i in range(6):  # Trigger brute force detection
        monitor.log_security_event(
            SecurityEventType.LOGIN_FAILED,
            user_id="test_user",
            ip_address="192.168.1.100",
            details={"attempt": i+1}
        )
    
    # Check if alert was created
    alerts = monitor.get_active_alerts(AlertLevel.MEDIUM)
    assert len(alerts) > 0, "Brute force alert not created"
    print("✅ Brute force detection successful")
    
    # Test IP blocking
    monitor.block_ip("192.168.1.200", "Manual test block")
    is_blocked = monitor.is_ip_blocked("192.168.1.200")
    assert is_blocked, "IP blocking failed"
    print("✅ IP blocking successful")
    
    # Test security summary
    summary = monitor.get_security_summary(24)
    assert summary["total_events"] > 0, "Security summary generation failed"
    print("✅ Security summary generated")
    
    print("Security Monitor: ✅ PASSED\n")

def test_integrated_security_workflow():
    """Test integrated security workflow"""
    print("Testing Integrated Security Workflow...")
    
    # Initialize all systems
    auth = AuthenticationSystem("integrated_test_key")
    authz = AuthorizationSystem()
    encryption = EncryptionManager()
    monitor = SecurityMonitor()
    
    # Simulate user registration and authentication workflow
    print("1. User Registration...")
    success, message, user = auth.register_user(
        username="integrationuser",
        email="integration@test.com",
        password="SecureIntegration123!",
        roles=["user"]
    )
    assert success
    
    # Assign role in authorization system
    authz.assign_role_to_user(user.user_id, "user")
    
    print("2. User Authentication...")
    success, message, token = auth.authenticate_user("integrationuser", "SecureIntegration123!")
    assert success
    
    # Log authentication event
    monitor.log_security_event(
        SecurityEventType.LOGIN_SUCCESS,
        user_id=user.user_id,
        ip_address="192.168.1.50",
        details={"auth_method": "password"}
    )
    
    print("3. Resource Access Control...")
    # Create a resource and set ownership
    resource_id = "session_integration_test"
    authz.set_resource_owner(resource_id, user.user_id)
    
    # Test access control
    can_access = authz.can_access_resource(user.user_id, ResourceType.SESSION, Action.READ, resource_id)
    assert can_access, "User should be able to access owned resource"
    
    print("4. Sensitive Data Encryption...")
    # Encrypt sensitive user data
    sensitive_data = {
        "user_id": user.user_id,
        "session_data": {"memory": "sensitive cognitive data"},
        "timestamp": datetime.now().isoformat()
    }
    encrypted_data = encryption.encrypt_json(sensitive_data, "session_data")
    assert encrypted_data is not None
    
    print("5. Security Audit...")
    # Generate audit report
    audit_report = monitor.generate_audit_report(user.user_id, days=1)
    assert audit_report["total_events"] > 0
    assert user.user_id in str(audit_report)
    
    print("✅ Integrated Security Workflow: PASSED\n")

def main():
    """Run all security tests"""
    print("=" * 60)
    print("DEEP TREE ECHO SECURITY FRAMEWORK TEST")
    print("=" * 60)
    print()
    
    try:
        test_authentication_system()
        test_authorization_system() 
        test_encryption_manager()
        test_security_monitor()
        test_integrated_security_workflow()
        
        print("=" * 60)
        print("🎉 ALL SECURITY TESTS PASSED!")
        print("✅ Authentication system working")
        print("✅ Authorization system working") 
        print("✅ Encryption manager working")
        print("✅ Security monitoring working")
        print("✅ Integrated security workflow working")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("=" * 60)
        raise

if __name__ == "__main__":
    main()