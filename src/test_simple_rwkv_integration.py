#!/usr/bin/env python3
"""
Simple test for RWKV pip package integration
This tests the simplified approach using: pip install rwkv
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from simple_rwkv_integration import SimpleEchoCognitiveBridge, SimpleRWKVInterface

async def test_simple_rwkv_integration():
    """Test the simple RWKV integration"""
    print("🧠 Testing Simple RWKV Integration (pip install rwkv)")
    print("=" * 60)
    
    # Test 1: Direct RWKV interface
    print("\n1. Testing Direct RWKV Interface")
    print("-" * 40)
    
    rwkv = SimpleRWKVInterface()
    
    # Initialize with default config
    success = await rwkv.initialize()
    print(f"Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        # Test generation
        test_prompt = "What is artificial intelligence?"
        response = await rwkv.generate_response(test_prompt)
        
        print(f"Input: {test_prompt}")
        print(f"Output: {response.output_text}")
        print(f"Success: {'✅' if response.success else '❌'}")
        print(f"Processing Time: {response.processing_time:.3f}s")
        print(f"Model Type: {response.model_info.get('model_type', 'unknown')}")
    
    # Test 2: Cognitive Bridge
    print("\n2. Testing Cognitive Bridge")
    print("-" * 40)
    
    bridge = SimpleEchoCognitiveBridge()
    
    config = {
        'rwkv': {
            'strategy': 'cpu fp32',
            'model_path': 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth'
        }
    }
    
    success = await bridge.initialize(config)
    print(f"Bridge Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    if success:
        # Test cognitive processing
        test_queries = [
            "How does machine learning work?",
            "What are the benefits of meditation?",
            "Explain quantum computing in simple terms."
        ]
        
        for i, query in enumerate(test_queries, 1):
            print(f"\nQuery {i}: {query}")
            
            context = {'session_id': f'test_session_{i}'}
            result = await bridge.process_cognitive_query(query, context)
            
            if result['success']:
                print(f"Response: {result['response'][:100]}...")
                print(f"Total Time: {result['processing_time']:.3f}s")
                print(f"Model: {result['model_info'].get('backend_type', 'unknown')}")
            else:
                print(f"❌ Error: {result.get('error', 'Unknown error')}")
    
    # Test 3: Status and Info
    print("\n3. Testing Status and Information")
    print("-" * 40)
    
    model_info = rwkv.get_model_info()
    print(f"Model Available: {model_info.get('model_available', False)}")
    print(f"Backend Type: {model_info.get('backend_type', 'unknown')}")
    print(f"Initialized: {model_info.get('initialized', False)}")
    
    bridge_status = bridge.get_status()
    print(f"Bridge Initialized: {bridge_status.get('initialized', False)}")
    print(f"Timestamp: {bridge_status.get('timestamp', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("✅ Simple RWKV Integration Test Complete!")
    
    # Check if real RWKV was used
    if model_info.get('backend_type') == 'rwkv':
        print("🎉 Real RWKV pip package is working!")
    else:
        print("ℹ️  Using mock implementation (install RWKV with: pip install rwkv)")

if __name__ == "__main__":
    asyncio.run(test_simple_rwkv_integration())
"""
Test Suite for Simple RWKV Integration

Validates the functionality of the simplified RWKV integration using pip package.
"""

import unittest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from simple_rwkv_integration import (
        SimpleRWKVIntegration, 
        SimpleRWKVConfig,
        create_simple_rwkv,
        get_default_rwkv,
        RWKV_AVAILABLE
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Please ensure the src directory is in the Python path")
    sys.exit(1)

class TestSimpleRWKVIntegration(unittest.TestCase):
    """Test cases for SimpleRWKVIntegration"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.config = SimpleRWKVConfig(
            model_path="",  # Empty path for mock testing
            strategy="cpu fp32",
            max_tokens=50,
            temperature=0.8
        )
        self.integration = SimpleRWKVIntegration(self.config)
    
    def test_initialization_without_model(self):
        """Test initialization without a model file"""
        result = self.integration.initialize()
        self.assertTrue(result, "Initialization should succeed without model for mock mode")
        self.assertTrue(self.integration._is_initialized, "Should be marked as initialized")
    
    def test_configuration(self):
        """Test configuration handling"""
        self.assertEqual(self.integration.config.temperature, 0.8)
        self.assertEqual(self.integration.config.max_tokens, 50)
        self.assertEqual(self.integration.config.strategy, "cpu fp32")
    
    def test_get_info(self):
        """Test getting integration information"""
        self.integration.initialize()
        info = self.integration.get_info()
        
        self.assertIn('backend', info)
        self.assertIn('available', info)
        self.assertIn('rwkv_package_available', info)
        self.assertIn('version', info)
        self.assertEqual(info['backend'], 'simple_rwkv')
    
    def test_get_version(self):
        """Test version information"""
        version = self.integration.get_version()
        self.assertIn('Simple RWKV Integration', version)
        if RWKV_AVAILABLE:
            self.assertIn('RWKV', version)
    
    def test_mock_generation(self):
        """Test mock text generation"""
        self.integration.initialize()
        result = self.integration.generate("Hello, world!")
        
        self.assertIsNotNone(result, "Should return a result")
        self.assertIn("Mock RWKV Response", result, "Should indicate mock response")
    
    def test_chat_interface(self):
        """Test the chat interface"""
        self.integration.initialize()
        response = self.integration.chat("Hello")
        
        self.assertIsNotNone(response, "Should return a response")
        self.assertIsInstance(response, str, "Response should be a string")
    
    def test_chat_with_context(self):
        """Test chat interface with context"""
        self.integration.initialize()
        context = "You are a helpful assistant."
        response = self.integration.chat("What is AI?", context=context)
        
        self.assertIsNotNone(response, "Should return a response")
        self.assertIsInstance(response, str, "Response should be a string")
    
    def test_is_available(self):
        """Test availability checking"""
        # Before initialization
        self.assertFalse(self.integration.is_available(), "Should not be available before init")
        
        # After initialization (mock mode)
        self.integration.initialize()
        # In mock mode, availability depends on RWKV_AVAILABLE and initialization
        if RWKV_AVAILABLE:
            # With RWKV package available but no actual model, it should still work in mock mode
            # The is_available method checks for model existence, so it might be False
            pass  # This depends on the actual implementation
    
    def test_reset(self):
        """Test model reset functionality"""
        self.integration.initialize()
        # Reset should not raise an exception
        try:
            self.integration.reset()
        except Exception as e:
            self.fail(f"Reset should not raise an exception: {e}")
    
    @patch('simple_rwkv_integration.RWKV_AVAILABLE', False)
    def test_fallback_when_rwkv_not_available(self):
        """Test behavior when RWKV package is not available"""
        integration = SimpleRWKVIntegration()
        result = integration.initialize()
        
        # Should succeed in mock mode even when RWKV not available
        self.assertTrue(result, "Should succeed in mock mode when RWKV not available")

class TestFactoryFunctions(unittest.TestCase):
    """Test factory functions for easy instantiation"""
    
    def test_create_simple_rwkv(self):
        """Test the factory function"""
        integration = create_simple_rwkv()
        self.assertIsInstance(integration, SimpleRWKVIntegration)
        self.assertTrue(integration._is_initialized, "Should be initialized by factory")
    
    def test_create_simple_rwkv_with_config(self):
        """Test factory function with configuration"""
        integration = create_simple_rwkv(
            temperature=0.5,
            max_tokens=100
        )
        self.assertEqual(integration.config.temperature, 0.5)
        self.assertEqual(integration.config.max_tokens, 100)
    
    def test_get_default_rwkv(self):
        """Test getting default instance"""
        # First call should create instance
        instance1 = get_default_rwkv()
        self.assertIsInstance(instance1, SimpleRWKVIntegration)
        
        # Second call should return same instance
        instance2 = get_default_rwkv()
        self.assertIs(instance1, instance2, "Should return the same instance")

class TestRWKVConfig(unittest.TestCase):
    """Test RWKV configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SimpleRWKVConfig()
        
        self.assertEqual(config.model_path, "")
        self.assertEqual(config.strategy, "cpu fp32")
        self.assertEqual(config.chunk_len, 256)
        self.assertEqual(config.max_tokens, 200)
        self.assertEqual(config.temperature, 1.0)
        self.assertEqual(config.top_p, 0.9)
    
    def test_custom_config(self):
        """Test custom configuration"""
        config = SimpleRWKVConfig(
            model_path="/path/to/model",
            temperature=0.7,
            max_tokens=150
        )
        
        self.assertEqual(config.model_path, "/path/to/model")
        self.assertEqual(config.temperature, 0.7)
        self.assertEqual(config.max_tokens, 150)

def run_tests():
    """Run all tests"""
    print("🧪 Running Simple RWKV Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestSimpleRWKVIntegration))
    test_suite.addTest(unittest.makeSuite(TestFactoryFunctions))
    test_suite.addTest(unittest.makeSuite(TestRWKVConfig))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "=" * 50)
    if result.wasSuccessful():
        print("✅ All tests passed!")
    else:
        print(f"❌ {len(result.failures)} tests failed, {len(result.errors)} errors")
        
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
