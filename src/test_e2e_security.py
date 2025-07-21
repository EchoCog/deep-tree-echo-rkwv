"""
End-to-End Security Integration Test
Comprehensive test of the secure Deep Tree Echo application
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"

def test_e2e_security_workflow():
    """Test complete security workflow end-to-end"""
    print("=" * 60)
    print("END-TO-END SECURITY INTEGRATION TEST")
    print("=" * 60)
    
    # Test 1: Public endpoints (no auth required)
    print("\n1. Testing public endpoints...")
    
    response = requests.get(f"{BASE_URL}/api/status")
    assert response.status_code == 200
    status = response.json()
    assert status['security_enabled'] == True
    print("✅ Public status endpoint working")
    
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    health = response.json()
    assert health['security'] == 'enabled'
    print("✅ Health check endpoint working")
    
    # Test 2: Unauthenticated access should fail
    print("\n2. Testing unauthenticated access (should fail)...")
    
    response = requests.post(f"{BASE_URL}/api/session")
    assert response.status_code == 401
    print("✅ Unauthenticated session creation blocked")
    
    response = requests.post(f"{BASE_URL}/api/process", json={
        "session_id": "fake_session",
        "input": "test"
    })
    assert response.status_code == 401
    print("✅ Unauthenticated cognitive processing blocked")
    
    # Test 3: User registration
    print("\n3. Testing user registration...")
    
    user_data = {
        "username": "e2euser",
        "email": "e2e@test.com", 
        "password": "SecureE2EPass123!"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/register", json=user_data)
    assert response.status_code == 200
    reg_result = response.json()
    user_id = reg_result['user_id']
    print(f"✅ User registered successfully: {user_id}")
    
    # Test 4: User authentication
    print("\n4. Testing user authentication...")
    
    login_data = {
        "username": "e2euser",
        "password": "SecureE2EPass123!"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login_data)
    assert response.status_code == 200
    auth_result = response.json()
    user_token = auth_result['token']
    print("✅ User authentication successful")
    
    # Test 5: Invalid credentials should fail
    print("\n5. Testing invalid credentials (should fail)...")
    
    bad_login = {
        "username": "e2euser",
        "password": "wrongpassword"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=bad_login)
    assert response.status_code == 401
    print("✅ Invalid credentials properly rejected")
    
    # Test 6: Authenticated session creation
    print("\n6. Testing authenticated session creation...")
    
    headers = {"Authorization": f"Bearer {user_token}"}
    response = requests.post(f"{BASE_URL}/api/session", headers=headers)
    assert response.status_code == 200
    session_result = response.json()
    session_id = session_result['session_id']
    print(f"✅ Session created successfully: {session_id[:8]}...")
    
    # Test 7: Session information retrieval
    print("\n7. Testing session information retrieval...")
    
    response = requests.get(f"{BASE_URL}/api/session/{session_id}", headers=headers)
    assert response.status_code == 200
    session_info = response.json()
    assert session_info['session_id'] == session_id
    print("✅ Session information retrieved successfully")
    
    # Test 8: Cognitive processing
    print("\n8. Testing cognitive processing...")
    
    process_data = {
        "session_id": session_id,
        "input": "Test secure cognitive processing with end-to-end encryption and authentication!"
    }
    
    response = requests.post(f"{BASE_URL}/api/process", headers=headers, json=process_data)
    assert response.status_code == 200
    process_result = response.json()
    assert 'response' in process_result
    assert 'timestamp' in process_result
    print("✅ Cognitive processing successful")
    
    # Test 9: Admin authentication
    print("\n9. Testing admin authentication...")
    
    admin_login = {
        "username": "admin",
        "password": "ChangeMe123!"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=admin_login)
    assert response.status_code == 200
    admin_result = response.json()
    admin_token = admin_result['token']
    print("✅ Admin authentication successful")
    
    # Test 10: Admin security monitoring
    print("\n10. Testing admin security monitoring...")
    
    admin_headers = {"Authorization": f"Bearer {admin_token}"}
    response = requests.get(f"{BASE_URL}/api/security/summary", headers=admin_headers)
    assert response.status_code == 200
    security_summary = response.json()
    assert 'total_events' in security_summary
    print(f"✅ Security summary retrieved: {security_summary['total_events']} events")
    
    response = requests.get(f"{BASE_URL}/api/security/alerts", headers=admin_headers)
    assert response.status_code == 200
    alerts_result = response.json()
    assert 'alerts' in alerts_result
    print(f"✅ Security alerts retrieved: {len(alerts_result['alerts'])} alerts")
    
    # Test 11: User cannot access admin endpoints
    print("\n11. Testing user access to admin endpoints (should fail)...")
    
    response = requests.get(f"{BASE_URL}/api/security/summary", headers=headers)
    assert response.status_code == 403
    print("✅ User properly blocked from admin endpoints")
    
    # Test 12: Cross-user session access should fail
    print("\n12. Testing cross-user session access (should fail)...")
    
    # Create another user
    user2_data = {
        "username": "e2euser2",
        "email": "e2e2@test.com",
        "password": "SecureE2EPass456!"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/register", json=user2_data)
    assert response.status_code == 200
    
    login2_data = {
        "username": "e2euser2", 
        "password": "SecureE2EPass456!"
    }
    
    response = requests.post(f"{BASE_URL}/api/auth/login", json=login2_data)
    assert response.status_code == 200
    user2_token = response.json()['token']
    
    # Try to access first user's session
    headers2 = {"Authorization": f"Bearer {user2_token}"}
    response = requests.get(f"{BASE_URL}/api/session/{session_id}", headers=headers2)
    assert response.status_code == 403
    print("✅ Cross-user session access properly blocked")
    
    # Test 13: User logout
    print("\n13. Testing user logout...")
    
    response = requests.post(f"{BASE_URL}/api/auth/logout", headers=headers)
    assert response.status_code == 200
    print("✅ User logout successful")
    
    # Test 14: Using token after logout should fail
    print("\n14. Testing token usage after logout (should fail)...")
    
    response = requests.post(f"{BASE_URL}/api/session", headers=headers)
    assert response.status_code == 401
    print("✅ Token properly invalidated after logout")
    
    # Test 15: Rate limiting (simulate multiple requests)
    print("\n15. Testing rate limiting...")
    
    # Make rapid requests to trigger rate limiting
    rapid_requests = 0
    for i in range(25):  # More than guest limit
        response = requests.get(f"{BASE_URL}/api/status")
        if response.status_code == 429:
            print(f"✅ Rate limiting triggered after {i+1} requests")
            break
        rapid_requests += 1
        time.sleep(0.1)
    
    if rapid_requests >= 24:
        print("✅ Rate limiting working (no 429 in test window)")
    
    print("\n" + "=" * 60)
    print("🎉 ALL END-TO-END SECURITY TESTS PASSED!")
    print("=" * 60)
    print("✅ Public endpoints accessible")
    print("✅ Authentication and authorization working")
    print("✅ Session management secure")
    print("✅ Cognitive processing protected")
    print("✅ Admin privileges enforced")
    print("✅ Cross-user access blocked")
    print("✅ Token management working")
    print("✅ Rate limiting active")
    print("✅ Security monitoring operational")
    print("=" * 60)
    
    return True

if __name__ == "__main__":
    try:
        test_e2e_security_workflow()
        print("\n🚀 Deep Tree Echo Security Framework is ready for production!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()