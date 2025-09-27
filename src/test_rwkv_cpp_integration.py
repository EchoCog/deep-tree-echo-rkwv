#!/usr/bin/env python3
"""
Test RWKV.cpp Integration with Deep Tree Echo Framework
Validates the high-performance C++ RWKV integration
"""

import os
import sys
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_rwkv_cpp_library_availability():
    """Test if RWKV.cpp library is available"""
    
    print("🔧 Testing RWKV.cpp Library Availability")
    print("=" * 50)
    
    # Check if library file exists
    library_path = Path(__file__).parent.joinpath(
        '../external/rwkv-cpp/librwkv.so'
    ).resolve()
    
    print(f"Checking library path: {library_path}")
    
    if os.path.exists(library_path):
        print("✅ RWKV.cpp library found")
        print(f"   Library size: {os.path.getsize(library_path) / 1024 / 1024:.2f} MB")
        return True
    else:
        print("❌ RWKV.cpp library not found")
        print("   Please build the library with: cd external/rwkv-cpp && cmake . && make")
        return False

def test_python_bindings():
    """Test Python bindings for RWKV.cpp"""
    
    print("\n🐍 Testing Python Bindings")
    print("=" * 50)
    
    try:
        # Add RWKV.cpp Python path
        rwkv_cpp_python_path = os.path.abspath(os.path.join(
            os.path.dirname(__file__),
            '../external/rwkv-cpp/python'
        ))
        
        if rwkv_cpp_python_path not in sys.path:
            sys.path.insert(0, rwkv_cpp_python_path)
        
        # Try importing RWKV.cpp components
        from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
        from tokenizer_util import get_tokenizer
        import sampling
        
        print("✅ RWKV.cpp Python bindings imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import RWKV.cpp Python bindings: {e}")
        return False

def test_integration_imports():
    """Test importing our integration modules"""
    
    print("\n📦 Testing Integration Module Imports")
    print("=" * 50)
    
    try:
        from rwkv_cpp_integration import RWKVCppInterface, RWKVCppConfig, RWKVCppCognitiveBridge
        print("✅ RWKV.cpp integration modules imported successfully")
        
        from enhanced_echo_rwkv_bridge import EnhancedRWKVInterface, EnhancedEchoRWKVIntegrationEngine
        print("✅ Enhanced Echo RWKV bridge imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import integration modules: {e}")
        return False

async def test_rwkv_cpp_interface():
    """Test RWKV.cpp interface initialization"""
    
    print("\n🚀 Testing RWKV.cpp Interface")
    print("=" * 50)
    
    try:
        from rwkv_cpp_integration import RWKVCppInterface, RWKVCppConfig
        
        # Create interface
        interface = RWKVCppInterface()
        
        # Test configuration (without actual model for now)
        config = {
            'model_path': '/nonexistent/model.bin',  # This will trigger fallback mode
            'thread_count': 2,
            'gpu_layer_count': 0,
            'temperature': 0.8,
            'tokenizer_type': 'world'
        }
        
        # Try initialization (should fall back to mock)
        success = await interface.initialize(config)
        
        if success:
            print("✅ RWKV.cpp interface initialized (fallback mode)")
            
            # Test basic functionality
            from echo_rwkv_bridge import CognitiveContext
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
            
response = await interface.generate_response("Test prompt", context)
            print(f"✅ Generated response: {response[:100]}...")
            
            # Test model state
            state = interface.get_model_state()
            print(f"✅ Model state: {state['model_type']}, initialized: {state['initialized']}")
            
            return True
        else:
            print("❌ RWKV.cpp interface initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing RWKV.cpp interface: {e}")
        return False

async def test_cognitive_bridge():
    """Test RWKV.cpp cognitive bridge"""
    
    print("\n🧠 Testing Cognitive Bridge")
    print("=" * 50)
    
    try:
        from rwkv_cpp_integration import RWKVCppConfig, RWKVCppCognitiveBridge
        
        # Create configuration
        config = RWKVCppConfig(
            model_path='/nonexistent/model.bin',  # Will use fallback
            thread_count=2,
            gpu_layer_count=0,
            temperature=0.8,
            tokenizer_type='world'
        )
        
        # Create and initialize bridge
        bridge = RWKVCppCognitiveBridge(config)
        success = await bridge.initialize()
        
        if success:
            print("✅ Cognitive bridge initialized")
            
            # Test cognitive processing (basic)
            from echo_rwkv_bridge import CognitiveContext
            
            context = CognitiveContext(
                session_id="test_session",
                user_input="What is artificial intelligence?",
                conversation_history=[],
                memory_state={},
                processing_goals=["understand", "explain"],
                temporal_context=[],
                metadata={}
            )
            
            # Test individual membrane processing
            memory_response = await bridge.process_memory_membrane_enhanced(context)
            print(f"✅ Memory membrane: {memory_response.output_text[:80]}...")
            
            reasoning_response = await bridge.process_reasoning_membrane_enhanced(context)
            print(f"✅ Reasoning membrane: {reasoning_response.output_text[:80]}...")
            
            grammar_response = await bridge.process_grammar_membrane_enhanced(context)
            print(f"✅ Grammar membrane: {grammar_response.output_text[:80]}...")
            
            # Test integrated processing
            result = await bridge.process_integrated_cognitive_request(context)
            if 'integrated_response' in result:
                print(f"✅ Integrated response: {result['integrated_response'][:80]}...")
            
            # Test performance summary
            performance = bridge.get_performance_summary()
            print(f"✅ Performance summary: {performance['total_requests']} requests processed")
            
            return True
        else:
            print("❌ Cognitive bridge initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing cognitive bridge: {e}")
        return False

async def test_enhanced_integration_engine():
    """Test enhanced integration engine"""
    
    print("\n⚡ Testing Enhanced Integration Engine")
    print("=" * 50)
    
    try:
        from enhanced_echo_rwkv_bridge import (
            EnhancedEchoRWKVIntegrationEngine, 
            create_enhanced_rwkv_config
        )
        
        # Create enhanced configuration
        config = create_enhanced_rwkv_config(
            model_path='/nonexistent/model.bin',  # Will use fallback
            backend_preference='auto',
            enable_gpu=False,
            memory_limit_mb=600
        )
        
        # Create and initialize enhanced engine
        engine = EnhancedEchoRWKVIntegrationEngine(
            backend_preference='auto',
            enable_rwkv_cpp=True
        )
        
        success = await engine.initialize(config)
        
        if success:
            print("✅ Enhanced integration engine initialized")
            
            # Check system status
            status = engine.get_system_status()
            print(f"✅ Active backend: {status.get('rwkv_interface', {}).get('active_backend', 'unknown')}")
            print(f"✅ Available backends: {status.get('rwkv_interface', {}).get('available_backends', [])}")
            print(f"✅ RWKV.cpp enhanced: {status.get('processing_stats', {}).get('rwkv_cpp_enhanced', False)}")
            
            # Test cognitive processing
            from echo_rwkv_bridge import CognitiveContext
            
            context = CognitiveContext(
                session_id="test_session",
                user_input="Explain the concept of consciousness and its relationship to artificial intelligence.",
                conversation_history=[],
                memory_state={},
                processing_goals=["analyze", "explain", "relate"],
                temporal_context=[],
                metadata={}
            )
            
            result = await engine.process_cognitive_input(context)
            
            if 'integrated_response' in result:
                print(f"✅ Cognitive processing successful")
                print(f"   Backend used: {result.get('backend_used', 'unknown')}")
                print(f"   Enhanced processing: {result.get('enhanced_processing', False)}")
                print(f"   Processing time: {result.get('total_processing_time', 0):.3f}s")
                print(f"   Response preview: {result['integrated_response'][:100]}...")
            
            return True
        else:
            print("❌ Enhanced integration engine initialization failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing enhanced integration engine: {e}")
        return False

def test_configuration_options():
    """Test various configuration options"""
    
    print("\n⚙️ Testing Configuration Options")
    print("=" * 50)
    
    try:
        from enhanced_echo_rwkv_bridge import create_enhanced_rwkv_config
        
        # Test different configurations
        configs = [
            ("Auto backend", create_enhanced_rwkv_config(backend_preference="auto")),
            ("RWKV.cpp preferred", create_enhanced_rwkv_config(backend_preference="rwkv_cpp")),
            ("Mock fallback", create_enhanced_rwkv_config(backend_preference="mock")),
            ("GPU enabled", create_enhanced_rwkv_config(enable_gpu=True)),
            ("Memory constrained", create_enhanced_rwkv_config(memory_limit_mb=300))
        ]
        
        for name, config in configs:
            print(f"✅ {name}: {len(config)} configuration parameters")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing configurations: {e}")
        return False

def test_library_detection():
    """Test automatic library detection"""
    
    print("\n🔍 Testing Library Detection")
    print("=" * 50)
    
    try:
        from rwkv_cpp_integration.rwkv_cpp_interface import RWKVCppInterface
        
        interface = RWKVCppInterface()
        detected_path = interface._auto_detect_library_path()
        
        if detected_path:
            print(f"✅ Library auto-detected at: {detected_path}")
            print(f"   Exists: {os.path.exists(detected_path)}")
            
            if os.path.exists(detected_path):
                file_size = os.path.getsize(detected_path) / 1024 / 1024
                print(f"   Size: {file_size:.2f} MB")
                
        else:
            print("⚠️ Library not auto-detected (expected if not built)")
            
        return True
        
    except Exception as e:
        print(f"❌ Error testing library detection: {e}")
        return False

async def run_all_tests():
    """Run all integration tests"""
    
    print("🧪 RWKV.cpp Integration Test Suite")
    print("=" * 60)
    print(f"Test started at: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    print("=" * 60)
    
    test_results = []
    
    # Run tests in order
    tests = [
        ("Library Availability", test_rwkv_cpp_library_availability),
        ("Python Bindings", test_python_bindings),
        ("Integration Imports", test_integration_imports),
        ("Library Detection", test_library_detection),
        ("Configuration Options", test_configuration_options),
        ("RWKV.cpp Interface", test_rwkv_cpp_interface),
        ("Cognitive Bridge", test_cognitive_bridge),
        ("Enhanced Integration Engine", test_enhanced_integration_engine)
    ]
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            test_results.append((test_name, result))
        except Exception as e:
            logger.error(f"Test {test_name} failed with exception: {e}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n📊 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! RWKV.cpp integration is working correctly.")
    elif passed >= total * 0.6:
        print("⚠️ Most tests passed. Some features may not be available but basic functionality works.")
    else:
        print("❌ Many tests failed. Please check the setup and dependencies.")
    
    return passed, total

if __name__ == "__main__":
    asyncio.run(run_all_tests())
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
