#!/usr/bin/env python3
"""
Demo: Toroidal Cognitive System
Demonstrates the dual-hemisphere Echo + Marduk architecture
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any

from toroidal_cognitive_system import create_toroidal_cognitive_system
from toroidal_integration import create_toroidal_bridge, create_toroidal_api

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def print_separator(title: str = ""):
    """Print a decorative separator"""
    print("\n" + "="*80)
    if title:
        print(f"  {title}")
        print("="*80)
    print()

def print_response_section(title: str, content: str):
    """Print a formatted response section"""
    print(f"\n### {title}")
    print("-" * (len(title) + 4))
    print(content)

async def demo_basic_toroidal_system():
    """Demonstrate basic Toroidal Cognitive System functionality"""
    print_separator("🌳 BASIC TOROIDAL COGNITIVE SYSTEM DEMO")
    
    # Create the system
    system = create_toroidal_cognitive_system()
    
    # Test queries that should trigger different response patterns
    test_queries = [
        "What is the architecture of this cognitive system?",
        "How do you process patterns and meaning?",
        "Tell me about recursive thinking and intuitive synthesis.",
        "What makes you different from other AI systems?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print_separator(f"Query {i}: {query}")
        
        start_time = time.time()
        response = await system.process_input(query)
        processing_time = time.time() - start_time
        
        print(f"**Processing Time**: {processing_time:.3f}s")
        print(f"**System Processing Time**: {response.total_processing_time:.3f}s")
        
        print_response_section("SYNCHRONIZED OUTPUT", response.synchronized_output)
        print_response_section("REFLECTION", response.reflection)
        
        print(f"\n**Convergence Metrics**:")
        for metric, value in response.convergence_metrics.items():
            print(f"  - {metric}: {value:.3f}")
        
        print(f"\n**Hemisphere Details**:")
        print(f"  - Echo: {response.echo_response.confidence:.3f} confidence, {response.echo_response.processing_time:.3f}s")
        print(f"  - Marduk: {response.marduk_response.confidence:.3f} confidence, {response.marduk_response.processing_time:.3f}s")
        
        # Brief pause between queries
        await asyncio.sleep(1)

async def demo_toroidal_integration():
    """Demonstrate Toroidal Integration with Echo-RWKV Bridge"""
    print_separator("🔗 TOROIDAL INTEGRATION DEMO")
    
    # Create bridge (without real RWKV for this demo)
    bridge = create_toroidal_bridge(buffer_size=500, use_real_rwkv=False)
    
    # Initialize bridge
    initialized = await bridge.initialize()
    print(f"Bridge Initialized: {initialized}")
    
    # Get system status
    status = bridge.get_system_status()
    print(f"\n**System Status**:")
    print(json.dumps(status, indent=2))
    
    # Test integration
    test_input = "Explain the toroidal architecture and how Echo and Marduk work together."
    
    print_separator(f"Integration Test: {test_input}")
    
    response = await bridge.process_cognitive_input(
        user_input=test_input,
        session_id="demo_session",
        conversation_history=[],
        memory_state={},
        processing_goals=["demonstrate_integration", "show_dual_processing"]
    )
    
    print_response_section("INTEGRATED RESPONSE", response.synchronized_output)
    print_response_section("SYSTEM REFLECTION", response.reflection)

async def demo_api_interface():
    """Demonstrate the REST API interface"""
    print_separator("🚀 TOROIDAL REST API DEMO")
    
    # Create bridge and API
    bridge = create_toroidal_bridge(buffer_size=200, use_real_rwkv=False)
    api = create_toroidal_api(bridge)
    
    await bridge.initialize()
    
    # Test API query
    request_data = {
        "input": "How does the shared memory lattice work in the toroidal system?",
        "session_id": "api_demo",
        "conversation_history": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Greetings from the Toroidal Cognitive System!"}
        ],
        "memory_state": {"context": "api_demonstration"},
        "processing_goals": ["explain_memory_system", "demonstrate_api"]
    }
    
    print("**API Request**:")
    print(json.dumps(request_data, indent=2))
    
    print_separator("API Processing...")
    
    api_response = await api.process_query(request_data)
    
    print("**API Response**:")
    if api_response.get("success"):
        response_data = api_response["response"]
        print(f"Processing Time: {response_data['total_processing_time']:.3f}s")
        print_response_section("SYNCHRONIZED OUTPUT", response_data["synchronized_output"])
        print_response_section("REFLECTION", response_data["reflection"])
        
        print(f"\n**Hemisphere Performance**:")
        echo = response_data["echo_response"]
        marduk = response_data["marduk_response"]
        print(f"  - Echo ({echo['hemisphere']}): {echo['confidence']:.3f} confidence")
        print(f"  - Marduk ({marduk['hemisphere']}): {marduk['confidence']:.3f} confidence")
        
        print(f"\n**Convergence Metrics**:")
        for metric, value in response_data["convergence_metrics"].items():
            print(f"  - {metric}: {value:.3f}")
    else:
        print(f"API Error: {api_response.get('error', 'Unknown error')}")
    
    # Get system status through API
    print_separator("API System Status")
    status_response = await api.get_system_status()
    if status_response.get("success"):
        print("System Status Retrieved Successfully")
        toroidal_metrics = status_response["status"]["toroidal_system"]
        print(f"Memory Buffer Size: {toroidal_metrics['memory_buffer_size']}")
        print(f"Processing Depth: {toroidal_metrics['system_state']['processing_depth']}")
        print(f"Attention Allocation: {toroidal_metrics['system_state']['attention_allocation']}")

async def interactive_demo():
    """Interactive demo allowing user input"""
    print_separator("💬 INTERACTIVE TOROIDAL DEMO")
    print("Enter queries to interact with the Toroidal Cognitive System")
    print("Commands: 'status' for system status, 'quit' to exit")
    print()
    
    bridge = create_toroidal_bridge(buffer_size=1000, use_real_rwkv=False)
    await bridge.initialize()
    
    session_id = f"interactive_{int(time.time())}"
    conversation_history = []
    
    while True:
        try:
            user_input = input("🧠 Your Query: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 🌳")
                break
            elif user_input.lower() == 'status':
                status = bridge.get_system_status()
                print(json.dumps(status, indent=2))
                continue
            elif not user_input:
                continue
            
            print_separator(f"Processing: {user_input}")
            
            # Add to conversation history
            conversation_history.append({"role": "user", "content": user_input})
            
            # Process through system
            response = await bridge.process_cognitive_input(
                user_input=user_input,
                session_id=session_id,
                conversation_history=conversation_history,
                memory_state={},
                processing_goals=["interactive_response"]
            )
            
            # Add response to history
            conversation_history.append({"role": "assistant", "content": response.synchronized_output})
            
            # Keep history manageable
            if len(conversation_history) > 10:
                conversation_history = conversation_history[-10:]
            
            print(response.synchronized_output)
            print_response_section("REFLECTION", response.reflection)
            
            print(f"\n⚡ Processing: {response.total_processing_time:.3f}s | "
                  f"Echo: {response.echo_response.confidence:.3f} | "
                  f"Marduk: {response.marduk_response.confidence:.3f}")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🌳")
            break
        except Exception as e:
            print(f"Error: {e}")

async def main():
    """Main demo function"""
    print_separator("🌳 TOROIDAL COGNITIVE SYSTEM DEMONSTRATION")
    print("This demo showcases the dual-hemisphere architecture with Echo and Marduk")
    print("working together in a braided helix of insight and recursion.")
    
    try:
        # Run basic demo
        await demo_basic_toroidal_system()
        
        # Run integration demo
        await demo_toroidal_integration()
        
        # Run API demo
        await demo_api_interface()
        
        # Ask if user wants interactive demo
        print_separator("Demo Complete!")
        print("Would you like to try the interactive demo? (y/n)")
        choice = input().strip().lower()
        
        if choice in ['y', 'yes']:
            await interactive_demo()
        else:
            print("Demo complete. Thank you for exploring the Toroidal Cognitive System! 🌳")
            
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())