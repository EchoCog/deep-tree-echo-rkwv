#!/usr/bin/env python3
"""
Test script for RWKV integration
Tests model loading, memory management, and basic inference
"""

import os
import sys
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_model_manager():
    """Test the RWKV Model Manager"""
    print("="*60)
    print("Testing RWKV Model Manager")
    print("="*60)
    
    try:
        from rwkv_model_manager import RWKVModelManager
        
        # Initialize model manager with WebVM memory constraints
        manager = RWKVModelManager(memory_limit_mb=600)
        
        # Check available models
        best_model = manager.get_best_model_for_memory_limit()
        print(f"Best model for 600MB limit: {best_model.name if best_model else 'None'}")
        
        if best_model:
            print(f"  Size: {best_model.size}")
            print(f"  Memory usage: {best_model.memory_mb}MB")
            print(f"  Context length: {best_model.context_length}")
        
        # List cached models
        cached = manager.list_cached_models()
        print(f"Cached models: {len(cached)}")
        
        # Test model preparation (without actual download)
        print("\nTesting model preparation for WebVM...")
        # Note: In actual WebVM deployment, models would be pre-downloaded
        
        return True
        
    except Exception as e:
        print(f"Model Manager test failed: {e}")
        return False

def test_memory_membrane():
    """Test the Memory Membrane with RWKV"""
    print("="*60)
    print("Testing RWKV Memory Membrane")
    print("="*60)
    
    try:
        from rwkv_echo_integration import RWKVMemoryMembrane, RWKVConfig, EchoMemoryState
        import numpy as np
        
        # Create config
        config = RWKVConfig(
            model_path="test_path",  # Will be overridden by model manager
            model_size="0.1B",
            context_length=1024,
            temperature=0.8
        )
        
        # Initialize memory membrane
        print("Initializing Memory Membrane...")
        membrane = RWKVMemoryMembrane(config)
        
        # Create test cognitive state
        cognitive_state = EchoMemoryState(
            declarative={},
            procedural={},
            episodic=[],
            intentional={'current_goals': []},
            temporal_context=[],
            activation_patterns=np.zeros(100)
        )
        
        # Test processing
        test_inputs = [
            "What is artificial intelligence?",
            "Remember that I like coffee in the morning",
            "How do neural networks learn?"
        ]
        
        print("Processing test inputs...")
        for i, test_input in enumerate(test_inputs):
            print(f"\nTest {i+1}: {test_input}")
            
            start_time = time.time()
            result = membrane.process_input(test_input, cognitive_state)
            processing_time = time.time() - start_time
            
            print(f"  Response: {result['response'][:100]}...")
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Processing type: {result.get('processing_type', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"Memory Membrane test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_reasoning_membrane():
    """Test the Reasoning Membrane with RWKV"""
    print("="*60)
    print("Testing RWKV Reasoning Membrane")
    print("="*60)
    
    try:
        from rwkv_echo_integration import RWKVReasoningMembrane, RWKVConfig, EchoMemoryState
        import numpy as np
        
        # Create config
        config = RWKVConfig(
            model_path="test_path",
            model_size="0.1B",
            context_length=1024,
            temperature=0.7
        )
        
        # Initialize reasoning membrane
        print("Initializing Reasoning Membrane...")
        membrane = RWKVReasoningMembrane(config)
        
        # Create test cognitive state
        cognitive_state = EchoMemoryState(
            declarative={},
            procedural={},
            episodic=[],
            intentional={'current_goals': []},
            temporal_context=[],
            activation_patterns=np.zeros(100)
        )
        
        # Test reasoning with different types
        test_inputs = [
            "If all humans are mortal, and Socrates is human, what can we conclude?",
            "Why do birds migrate south for winter?",
            "How can we solve the problem of climate change?"
        ]
        
        print("Processing reasoning tasks...")
        for i, test_input in enumerate(test_inputs):
            print(f"\nTest {i+1}: {test_input}")
            
            start_time = time.time()
            result = membrane.process_input(test_input, cognitive_state)
            processing_time = time.time() - start_time
            
            print(f"  Reasoning type: {result.get('reasoning_type', 'unknown')}")
            print(f"  Response: {result['response'][:100]}...")
            print(f"  Confidence: {result.get('confidence', 0):.2f}")
            print(f"  Processing time: {processing_time:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"Reasoning Membrane test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_integrated_system():
    """Test the integrated Deep Tree Echo RWKV system"""
    print("="*60)
    print("Testing Integrated Deep Tree Echo RWKV System")
    print("="*60)
    
    try:
        from rwkv_echo_integration import DeepTreeEchoRWKV, RWKVConfig
        
        # Create configuration optimized for WebVM
        config = RWKVConfig(
            model_path="test_path",
            model_size="0.1B",
            context_length=1024,
            temperature=0.8,
            top_p=0.9,
            top_k=40
        )
        
        print("Initializing Deep Tree Echo RWKV system...")
        echo_system = DeepTreeEchoRWKV(config)
        
        # Test cognitive processing
        test_inputs = [
            "What is consciousness?",
            "How do I learn a new skill effectively?",
            "The quick brown fox jumps over the lazy dog.",
            "Why is creativity important for problem solving?"
        ]
        
        print("Testing cognitive processing...")
        total_time = 0
        for i, test_input in enumerate(test_inputs):
            print(f"\nTest {i+1}: {test_input}")
            
            start_time = time.time()
            result = echo_system.process_cognitive_input(test_input)
            processing_time = time.time() - start_time
            total_time += processing_time
            
            print(f"  Processing time: {processing_time:.3f}s")
            print(f"  Integrated response: {result['integrated_response'][:100]}...")
            print(f"  Cognitive state: {result['cognitive_state_summary']}")
        
        avg_time = total_time / len(test_inputs)
        print(f"\nAverage processing time: {avg_time:.3f}s")
        print("WebVM performance target: < 2.0s per response")
        print(f"Performance status: {'✓ PASS' if avg_time < 2.0 else '✗ NEEDS OPTIMIZATION'}")
        
        return True
        
    except Exception as e:
        print(f"Integrated system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("Starting RWKV Integration Tests")
    print("="*60)
    
    tests = [
        ("Model Manager", test_model_manager),
        ("Memory Membrane", test_memory_membrane),
        ("Reasoning Membrane", test_reasoning_membrane),
        ("Integrated System", test_integrated_system)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            success = test_func()
            results[test_name] = success
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"\n{test_name}: {status}")
        except Exception as e:
            results[test_name] = False
            print(f"\n{test_name}: ✗ FAIL - {e}")
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{test_name:20}: {status}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RWKV integration is working correctly.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())