#!/usr/bin/env python3
"""
Demo: Simple RWKV Integration using pip install rwkv
This demonstrates the straightforward approach for RWKV integration
"""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from simple_rwkv_integration import SimpleEchoCognitiveBridge

async def main():
    """Main demo function"""
    print("🧠 Deep Tree Echo - Simple RWKV Integration Demo")
    print("Using: pip install rwkv")
    print("=" * 60)
    
    # Initialize the cognitive bridge
    bridge = SimpleEchoCognitiveBridge()
    
    # Configuration for WebVM compatibility
    config = {
        'rwkv': {
            'strategy': 'cpu fp32',  # Memory efficient for WebVM
            'model_path': 'RWKV-4-Raven-1B5-v12-Eng98%-Other2%-20230520-ctx4096.pth'
        }
    }
    
    print("Initializing RWKV cognitive bridge...")
    success = await bridge.initialize(config)
    
    if not success:
        print("❌ Failed to initialize bridge")
        print("💡 Make sure RWKV is installed: pip install rwkv")
        return
    
    print("✅ Bridge initialized successfully!")
    print()
    
    # Interactive demo
    while True:
        print("🤖 Ask me anything (or 'quit' to exit):")
        user_input = input("> ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        print("\n🔄 Processing through cognitive architecture...")
        
        context = {
            'session_id': 'interactive_demo',
            'user_id': 'demo_user'
        }
        
        result = await bridge.process_cognitive_query(user_input, context)
        
        if result['success']:
            print(f"\n🧠 Response: {result['response']}")
            print(f"⏱️  Processing Time: {result['processing_time']:.3f}s")
            print(f"🔧 Model: {result['model_info'].get('backend_type', 'unknown')}")
        else:
            print(f"\n❌ Error: {result.get('error', 'Unknown error')}")
        
        print("\n" + "-" * 60)
    
    print("\n👋 Thanks for using Deep Tree Echo RWKV Integration!")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")