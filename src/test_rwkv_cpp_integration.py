"""
Test RWKV.cpp Integration with Deep Tree Echo Framework
"""

import os
import sys
import asyncio
import unittest
from unittest.mock import Mock, patch
import tempfile
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '.'))

try:
    from rwkv_cpp_integration import (
        RWKVConfig, 
        RWKVCognitiveState, 
        RWKVCppCognitiveEngine,
        RWKVCppMembraneProcessor,
        create_rwkv_processor,
        RWKV_CPP_AVAILABLE
    )
    from echo_rwkv_bridge import (
        RealRWKVInterface,
        CognitiveContext,
        RWKV_CPP_INTEGRATION_AVAILABLE
    )
    IMPORTS_SUCCESSFUL = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_SUCCESSFUL = False

class TestRWKVCppIntegration(unittest.TestCase):
    """Test RWKV.cpp integration with Deep Tree Echo"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_model_path = "/tmp/test_rwkv_model.bin"
        
    def test_imports_available(self):
        """Test that all required modules can be imported"""
        self.assertTrue(IMPORTS_SUCCESSFUL, "Failed to import required modules")
    
    def test_rwkv_config_creation(self):
        """Test RWKV configuration creation"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        config = RWKVConfig(
            model_path=self.test_model_path,
            thread_count=2,
            temperature=0.7
        )
        
        self.assertEqual(config.model_path, self.test_model_path)
        self.assertEqual(config.thread_count, 2)
        self.assertEqual(config.temperature, 0.7)
    
    def test_cognitive_state_creation(self):
        """Test cognitive state initialization"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        state = RWKVCognitiveState()
        
        self.assertIsNone(state.model_state)
        self.assertIsNone(state.logits)
        self.assertIsInstance(state.conversation_context, list)
        self.assertIsInstance(state.memory_state, dict)
        self.assertIsInstance(state.processing_metadata, dict)
    
    def test_cognitive_engine_creation(self):
        """Test cognitive engine initialization"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        config = RWKVConfig(model_path=self.test_model_path)
        engine = RWKVCppCognitiveEngine(config)
        
        self.assertIsNotNone(engine)
        self.assertEqual(engine.config.model_path, self.test_model_path)
        self.assertIsInstance(engine.state, RWKVCognitiveState)
        
        # Should work even without actual model file (falls back to mock)
        self.assertIsNotNone(engine.is_available)
    
    def test_membrane_processor_creation(self):
        """Test membrane processor creation"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        processor = create_rwkv_processor(self.test_model_path)
        
        self.assertIsNotNone(processor)
        self.assertIsInstance(processor, RWKVCppMembraneProcessor)
        self.assertEqual(processor.config.model_path, self.test_model_path)
    
    def test_membrane_processing(self):
        """Test membrane processing functionality"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        processor = create_rwkv_processor(self.test_model_path)
        
        input_data = {
            "text": "What is the meaning of life?",
            "context": {"test": True}
        }
        
        # Test each membrane type
        memory_result = processor.process_memory_membrane(input_data)
        reasoning_result = processor.process_reasoning_membrane(input_data)
        grammar_result = processor.process_grammar_membrane(input_data)
        
        for result in [memory_result, reasoning_result, grammar_result]:
            self.assertIn("membrane_type", result)
            self.assertIn("output", result)
            self.assertIn("confidence", result)
            self.assertIn("metadata", result)
            self.assertIsInstance(result["confidence"], (int, float))
    
    def test_processor_status(self):
        """Test processor status reporting"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        processor = create_rwkv_processor(self.test_model_path)
        status = processor.get_status()
        
        self.assertIn("rwkv_cpp_available", status)
        self.assertIn("model_loaded", status)
        self.assertIn("model_path", status)
        self.assertIn("config", status)
        self.assertEqual(status["model_path"], self.test_model_path)

class TestEchoRWKVBridge(unittest.TestCase):
    """Test the updated Echo RWKV bridge with RWKV.cpp support"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_model_path = "/tmp/test_rwkv_model.bin"
    
    async def async_setUp(self):
        """Async setup for tests"""
        if IMPORTS_SUCCESSFUL and RWKV_CPP_INTEGRATION_AVAILABLE:
            self.interface = RealRWKVInterface()
            self.config = {"model_path": self.test_model_path}
    
    def test_real_rwkv_interface_creation(self):
        """Test RealRWKVInterface creation"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        interface = RealRWKVInterface()
        self.assertFalse(interface.initialized)
        self.assertIsInstance(interface.model_config, dict)
        self.assertIsInstance(interface.memory_store, list)
        self.assertIsInstance(interface.conversation_context, list)
    
    def test_integration_availability_flags(self):
        """Test availability flags"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        # These should be defined
        self.assertIsInstance(RWKV_CPP_INTEGRATION_AVAILABLE, bool)
        self.assertIsInstance(RWKV_CPP_AVAILABLE, bool)
    
    def test_async_initialization(self):
        """Test async initialization"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        async def run_test():
            interface = RealRWKVInterface()
            config = {"model_path": self.test_model_path}
            
            # Should work even without real model (fallback to mock)
            result = await interface.initialize(config)
            self.assertIsInstance(result, bool)
            
            if result:
                self.assertTrue(interface.initialized)
        
        asyncio.run(run_test())
    
    def test_model_state_reporting(self):
        """Test model state reporting"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        async def run_test():
            interface = RealRWKVInterface()
            config = {"model_path": self.test_model_path}
            
            await interface.initialize(config)
            state = interface.get_model_state()
            
            self.assertIn("initialized", state)
            self.assertIn("model_config", state)
            self.assertIn("rwkv_cpp_integration_available", state)
            self.assertIn("rwkv_cpp_available", state)
            self.assertIn("model_type", state)
        
        asyncio.run(run_test())
    
    def test_cognitive_context_processing(self):
        """Test cognitive context processing"""
        if not IMPORTS_SUCCESSFUL:
            self.skipTest("Imports not available")
            
        async def run_test():
            interface = RealRWKVInterface()
            config = {"model_path": self.test_model_path}
            
            await interface.initialize(config)
            
            context = CognitiveContext(
                session_id="test_session",
                user_input="Hello, how are you?",
                conversation_history=[],
                memory_state={},
                processing_goals=["respond"],
                temporal_context=[],
                metadata={}
            )
            
            # Should work even with mock/fallback
            response = await interface.generate_response("Test prompt", context)
            self.assertIsInstance(response, str)
            self.assertGreater(len(response), 0)
        
        asyncio.run(run_test())

class TestIntegrationStatus(unittest.TestCase):
    """Test integration status and availability"""
    
    def test_integration_status(self):
        """Test overall integration status"""
        print(f"\n=== RWKV.cpp Integration Status ===")
        print(f"Imports successful: {IMPORTS_SUCCESSFUL}")
        
        if IMPORTS_SUCCESSFUL:
            print(f"RWKV.cpp integration available: {RWKV_CPP_INTEGRATION_AVAILABLE}")
            print(f"RWKV.cpp backend available: {RWKV_CPP_AVAILABLE}")
            
            # Test processor creation
            try:
                processor = create_rwkv_processor("/tmp/test_model.bin")
                print(f"Processor creation: SUCCESS")
                print(f"Processor status: {processor.get_status()}")
            except Exception as e:
                print(f"Processor creation failed: {e}")
        
        print("=" * 40)

def run_tests():
    """Run all tests"""
    print("Running RWKV.cpp Integration Tests...")
    
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestRWKVCppIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEchoRWKVBridge))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationStatus))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    if success:
        print("\n✅ All tests passed!")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)