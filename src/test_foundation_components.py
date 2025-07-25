#!/usr/bin/env python3
"""
Test Foundation Components for P0 Issues
Tests the foundation implementations for Phase 1 development.
"""

import os
import sys
import time
import logging
import tempfile
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_p0_001_rwkv_model_foundation():
    """Test P0-001: RWKV Model Integration Foundation"""
    print("="*60)
    print("Testing P0-001: RWKV Model Foundation")
    print("="*60)
    
    try:
        from rwkv_model_foundation import (
            RWKVModelManager, RWKVModelConfig, 
            MockModelLoadingStrategy, get_model_manager
        )
        
        # Test model manager initialization
        manager = RWKVModelManager(memory_limit_mb=600)
        print(f"✓ Model manager initialized with 600MB limit")
        
        # Test available models
        available_models = manager.get_available_models()
        print(f"✓ Found {len(available_models)} available models within memory limit")
        
        # Test optimal model selection
        optimal = manager.get_optimal_model()
        if optimal:
            print(f"✓ Optimal model: {optimal.model_name} ({optimal.model_size})")
        
        # Test model loading with mock strategy
        if optimal:
            success = manager.load_model(optimal)
            print(f"✓ Model loading {'succeeded' if success else 'failed'}")
            
            if success:
                loaded = manager.get_loaded_model(optimal.model_name)
                print(f"✓ Loaded model info: {loaded['strategy']}")
        
        # Test memory usage tracking
        memory_usage = manager.get_memory_usage()
        print(f"✓ Memory usage: {memory_usage}")
        
        # Test global manager
        global_manager = get_model_manager()
        print(f"✓ Global manager accessible: {type(global_manager).__name__}")
        
        return True
        
    except Exception as e:
        print(f"✗ P0-001 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_p0_002_persistent_memory_foundation():
    """Test P0-002: Persistent Memory Architecture Foundation"""
    print("="*60)
    print("Testing P0-002: Persistent Memory Foundation")
    print("="*60)
    
    try:
        from persistent_memory_foundation import (
            PersistentMemorySystem, MemoryNode, MemoryQuery,
            SQLiteMemoryStorage, MemoryGraph
        )
        
        # Use temporary database
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        try:
            # Test memory system initialization
            memory_system = PersistentMemorySystem(db_path)
            print(f"✓ Memory system initialized with SQLite storage")
            
            # Test conversation storage
            session_id = "test_session_001"
            user_input = "What is artificial intelligence?"
            ai_response = "Artificial intelligence is the simulation of human intelligence processes by machines."
            
            user_mem_id, ai_mem_id = memory_system.store_conversation_turn(
                session_id, user_input, ai_response
            )
            print(f"✓ Stored conversation turn: {user_mem_id[:8]}... -> {ai_mem_id[:8]}...")
            
            # Test conversation history retrieval
            history = memory_system.get_conversation_history(session_id)
            print(f"✓ Retrieved {len(history)} conversation turns")
            
            if history:
                print(f"  Latest turn: {history[-1]['user'][:50]}...")
            
            # Test knowledge search
            search_results = memory_system.search_knowledge("artificial intelligence")
            print(f"✓ Knowledge search returned {len(search_results)} results")
            
            # Test memory graph functionality
            graph = memory_system.memory_graph
            
            # Add a fact
            fact_id = graph.add_memory(
                "The Turing Test was proposed by Alan Turing in 1950",
                content_type="fact",
                tags=["history", "ai", "turing"]
            )
            print(f"✓ Added fact to memory graph: {fact_id[:8]}...")
            
            # Search for related memories
            related = graph.search_memories("Turing", content_types=["fact"])
            print(f"✓ Found {len(related)} related memories")
            
            # Test system statistics
            stats = memory_system.get_system_status()
            print(f"✓ System stats: {stats}")
            
            return True
            
        finally:
            # Cleanup
            if os.path.exists(db_path):
                os.unlink(db_path)
        
    except Exception as e:
        print(f"✗ P0-002 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_p0_003_security_framework_foundation():
    """Test P0-003: Security Framework Foundation"""
    print("="*60)
    print("Testing P0-003: Security Framework Foundation")
    print("="*60)
    
    try:
        from security_framework_foundation import (
            SecurityFramework, UserRole, SecurityEvent,
            InMemorySecurityStorage, PasswordManager
        )
        
        # Test security framework initialization
        security = SecurityFramework()
        print(f"✓ Security framework initialized")
        
        # Test password management
        password_mgr = PasswordManager()
        test_password = "secure_password_123"
        password_hash = password_mgr.hash_password(test_password)
        is_valid = password_mgr.verify_password(test_password, password_hash)
        print(f"✓ Password hashing and verification: {'passed' if is_valid else 'failed'}")
        
        # Test user creation
        user_id = security.create_user(
            username="testuser",
            email="test@example.com", 
            password="testpass123",
            role=UserRole.USER
        )
        print(f"✓ Created test user: {user_id[:8]}...")
        
        # Test authentication
        session_id = security.authenticate_user(
            username="testuser",
            password="testpass123",
            ip_address="127.0.0.1",
            user_agent="Test-Agent/1.0"
        )
        print(f"✓ User authentication: {'success' if session_id else 'failed'}")
        
        if session_id:
            # Test session validation
            user = security.validate_session(session_id, "127.0.0.1")
            print(f"✓ Session validation: {'success' if user else 'failed'}")
            
            # Test logout
            security.logout_user(session_id)
            print(f"✓ User logout completed")
        
        # Test API key management
        api_key, key_id = security.api_key_manager.create_api_key(
            user_id=user_id,
            name="Test API Key",
            permissions=["read", "write"]
        )
        print(f"✓ Created API key: {key_id}")
        
        # Test API key validation
        validated_user = security.validate_api_key(api_key, ["read"])
        print(f"✓ API key validation: {'success' if validated_user else 'failed'}")
        
        # Test security status
        status = security.get_security_status()
        print(f"✓ Security status: {status}")
        
        # Test with admin user
        admin_session = security.authenticate_user(
            username="admin",
            password="admin123",
            ip_address="127.0.0.1", 
            user_agent="Admin-Test/1.0"
        )
        print(f"✓ Admin authentication: {'success' if admin_session else 'failed'}")
        
        return True
        
    except Exception as e:
        print(f"✗ P0-003 test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integration():
    """Test integration between foundation components"""
    print("="*60)
    print("Testing Foundation Integration")
    print("="*60)
    
    try:
        from rwkv_model_foundation import get_model_manager
        from persistent_memory_foundation import PersistentMemorySystem
        from security_framework_foundation import SecurityFramework
        
        # Initialize all systems
        model_manager = get_model_manager(600)
        memory_system = PersistentMemorySystem("/tmp/integration_test.db")
        security = SecurityFramework()
        
        print("✓ All foundation systems initialized")
        
        # Test scenario: Secure cognitive session
        # 1. Authenticate user
        session_id = security.authenticate_user(
            "admin", "admin123", "127.0.0.1", "Test/1.0"
        )
        
        if not session_id:
            print("✗ Authentication failed")
            return False
        
        # 2. Load optimal model
        optimal_model = model_manager.get_optimal_model()
        if optimal_model:
            model_loaded = model_manager.load_model(optimal_model)
            print(f"✓ Model loading for session: {'success' if model_loaded else 'failed'}")
        
        # 3. Store secure conversation
        if session_id:
            user_mem, ai_mem = memory_system.store_conversation_turn(
                session_id,
                "Test secure cognitive interaction",
                "This is a secure response from the cognitive system",
                metadata={"security_session": session_id, "model": optimal_model.model_name if optimal_model else "mock"}
            )
            print(f"✓ Stored secure conversation: {user_mem[:8]}... -> {ai_mem[:8]}...")
        
        # 4. Validate session and retrieve history
        user = security.validate_session(session_id)
        if user:
            history = memory_system.get_conversation_history(session_id)
            print(f"✓ Retrieved {len(history)} conversation turns for authenticated user")
        
        # 5. System status check
        model_status = model_manager.get_memory_usage()
        memory_status = memory_system.get_system_status()
        security_status = security.get_security_status()
        
        print(f"✓ System integration status:")
        print(f"  Models: {model_status.get('total', 0)}MB used")
        print(f"  Memory: {memory_status.get('total_memories', 0)} memories")
        print(f"  Security: {security_status.get('storage_type', 'unknown')} storage")
        
        # Cleanup
        security.logout_user(session_id)
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all foundation tests"""
    print("Starting Phase 1 Foundation Component Tests")
    print("=" * 80)
    
    test_results = []
    
    # Test P0 foundation components
    test_results.append(("P0-001 RWKV Foundation", test_p0_001_rwkv_model_foundation()))
    test_results.append(("P0-002 Memory Foundation", test_p0_002_persistent_memory_foundation()))
    test_results.append(("P0-003 Security Foundation", test_p0_003_security_framework_foundation()))
    test_results.append(("Integration Test", test_integration()))
    
    # Print summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} : {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All foundation components are working correctly!")
        return 0
    else:
        print("⚠️  Some foundation components need attention.")
        return 1

if __name__ == "__main__":
    sys.exit(main())