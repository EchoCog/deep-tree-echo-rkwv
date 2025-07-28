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