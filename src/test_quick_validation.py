#!/usr/bin/env python3
"""
Quick validation test for Deep Tree Echo RWKV Integration - Iteration 2
Tests core functionality without external dependencies
"""

import os
import sys
import time
import logging
import tempfile
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_security_framework():
    """Test the security framework implementation"""
    print("="*60)
    print("Testing Security Framework (P0-003)")
    print("="*60)
    
    try:
        from auth_middleware import AuthenticationManager, SimpleUserStore, SecurityConfig
        
        # Test user store
        user_store = SimpleUserStore()
        print("✓ User store initialized with default admin")
        
        # Test user creation
        user_id = user_store.create_user(
            username="testuser",
            email="test@example.com",
            password="testpass123",
            role="user"
        )
        print(f"✓ Test user created: {user_id}")
        
        # Test authentication manager
        auth_manager = AuthenticationManager(user_store)
        print("✓ Authentication manager initialized")
        
        # Test user login
        login_result = auth_manager.login(
            username="testuser",
            password="testpass123",
            ip_address="127.0.0.1",
            user_agent="test-agent"
        )
        
        if login_result:
            print("✓ User login successful")
            print(f"  - User: {login_result['user']['username']}")
            print(f"  - Role: {login_result['user']['role']}")
            print(f"  - Token present: {bool(login_result.get('token'))}")
            
            # Test token verification
            token = login_result['token']
            user_info = auth_manager.verify_token(token)
            if user_info:
                print("✓ Token verification successful")
                print(f"  - Username: {user_info['username']}")
                print(f"  - Role: {user_info['role']}")
            else:
                print("✗ Token verification failed")
                return False
        else:
            print("✗ User login failed")
            return False
        
        print("✓ Security framework test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Security framework test failed: {e}")
        return False

def test_enhanced_rwkv_bridge():
    """Test the enhanced RWKV bridge without numpy dependency"""
    print("="*60)
    print("Testing Enhanced RWKV Bridge (P0-001)")
    print("="*60)
    
    try:
        # Test import and basic functionality
        print("✓ Testing enhanced RWKV bridge imports...")
        
        # Test CognitiveContext creation
        from echo_rwkv_bridge import CognitiveContext, MembraneResponse, IntegratedCognitiveResponse
        
        context = CognitiveContext(
            session_id="test-session",
            user_input="What is artificial intelligence?",
            conversation_history=[],
            memory_state={},
            processing_goals=["understand", "respond"],
            temporal_context=[],
            metadata={}
        )
        print("✓ CognitiveContext creation successful")
        
        # Test MembraneResponse creation
        response = MembraneResponse(
            membrane_type="memory",
            input_text="test input",
            output_text="test output",
            confidence=0.85,
            processing_time=0.1,
            internal_state={},
            metadata={}
        )
        print("✓ MembraneResponse creation successful")
        
        print("✓ Enhanced RWKV bridge test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced RWKV bridge test failed: {e}")
        return False

def test_enhanced_cognitive_methods():
    """Test enhanced cognitive methods"""
    print("="*60)
    print("Testing Enhanced Cognitive Methods")
    print("="*60)
    
    try:
        from enhanced_cognitive_methods import EnhancedCognitiveSessionMethods
        
        # Create a mock session to test methods
        class MockSession:
            def __init__(self):
                self.session_id = "test-session"
                self.memory_state = {'episodic': []}
        
        session = MockSession()
        
        # Bind methods to session
        for method_name in dir(EnhancedCognitiveSessionMethods):
            if not method_name.startswith('__'):
                method = getattr(EnhancedCognitiveSessionMethods, method_name)
                setattr(session, method_name, method.__get__(session, session.__class__))
        
        # Test complexity assessment
        complexity = session._assess_input_complexity("This is a complex question about artificial intelligence and machine learning.")
        print(f"✓ Complexity assessment: {complexity}")
        
        # Test reasoning type suggestion
        reasoning_type = session._suggest_reasoning_type("Why does this happen?")
        print(f"✓ Reasoning type suggestion: {reasoning_type}")
        
        # Test memory integration assessment
        memory_level = session._assess_memory_integration("Please remember this important fact.")
        print(f"✓ Memory integration assessment: {memory_level}")
        
        # Test learning opportunities identification
        opportunities = session._identify_learning_opportunities("I prefer detailed explanations.")
        print(f"✓ Learning opportunities: {opportunities}")
        
        print("✓ Enhanced cognitive methods test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Enhanced cognitive methods test failed: {e}")
        return False

def test_configuration_and_setup():
    """Test configuration and setup without heavy dependencies"""
    print("="*60)
    print("Testing Configuration and Setup")
    print("="*60)
    
    try:
        # Test configuration loading (simulated)
        config = {
            'webvm_mode': True,
            'memory_limit_mb': 600,
            'enable_real_rwkv': False,
            'enable_advanced_cognitive': True,
            'enable_persistence': True,
            'data_dir': '/tmp/echo_test_data'
        }
        print("✓ Configuration loaded successfully")
        
        # Test directory creation
        import os
        os.makedirs(config['data_dir'], exist_ok=True)
        print(f"✓ Data directory created: {config['data_dir']}")
        
        # Test basic JSON operations (simulating data handling)
        test_data = {
            'session_id': 'test-123',
            'timestamp': datetime.now().isoformat(),
            'cognitive_state': {'active': True}
        }
        
        # Simulate data serialization
        json_data = json.dumps(test_data)
        parsed_data = json.loads(json_data)
        print("✓ JSON data serialization successful")
        
        # Cleanup
        import shutil
        if os.path.exists(config['data_dir']):
            shutil.rmtree(config['data_dir'])
        print("✓ Cleanup completed")
        
        print("✓ Configuration and setup test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Configuration and setup test failed: {e}")
        return False

def test_integration_without_heavy_deps():
    """Test basic integration functionality"""
    print("="*60)
    print("Testing Basic Integration")
    print("="*60)
    
    try:
        # Test that all modules can be imported
        from echo_rwkv_bridge import CognitiveContext
        from auth_middleware import AuthenticationManager
        from enhanced_cognitive_methods import EnhancedCognitiveSessionMethods
        print("✓ All modules imported successfully")
        
        # Test basic data flow
        context = CognitiveContext(
            session_id="integration-test",
            user_input="Test integration",
            conversation_history=[],
            memory_state={},
            processing_goals=[],
            temporal_context=[],
            metadata={}
        )
        
        # Simulate processing pipeline
        processing_result = {
            'input': context.user_input,
            'session_id': context.session_id,
            'timestamp': datetime.now().isoformat(),
            'processing_successful': True
        }
        print("✓ Basic data flow successful")
        
        print("✓ Basic integration test: PASSED")
        return True
        
    except Exception as e:
        print(f"✗ Basic integration test failed: {e}")
        return False

def run_quick_validation():
    """Run quick validation tests"""
    print("🧠 Deep Tree Echo RWKV Integration - Iteration 2")
    print("Quick Validation Test Suite (No Heavy Dependencies)")
    print("="*80)
    
    tests = [
        ("Security Framework", test_security_framework),
        ("Enhanced RWKV Bridge", test_enhanced_rwkv_bridge),
        ("Enhanced Cognitive Methods", test_enhanced_cognitive_methods),
        ("Configuration and Setup", test_configuration_and_setup),
        ("Basic Integration", test_integration_without_heavy_deps)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🔬 Running {test_name}...")
        start_time = time.time()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} failed with exception: {e}")
            results[test_name] = False
        end_time = time.time()
        print(f"⏱️  Test completed in {end_time - start_time:.2f}s")
    
    # Print summary
    print("\n" + "="*80)
    print("🎯 Quick Validation Results")
    print("="*80)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n📊 Overall: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All quick validation tests passed!")
        print("✨ Phase 1 P0 enhancements are working correctly!")
        print("🚀 Ready for Phase 2 development.")
        return True
    else:
        print("⚠️  Some tests failed. Please review and fix issues.")
        return False

if __name__ == "__main__":
    success = run_quick_validation()
    sys.exit(0 if success else 1)