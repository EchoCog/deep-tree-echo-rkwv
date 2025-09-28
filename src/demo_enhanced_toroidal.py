#!/usr/bin/env python3
"""
Enhanced Demo: Toroidal Cognitive System with Echo-RWKV Integration
Comprehensive demonstration showing full system capabilities
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

def print_banner(text: str, char: str = "=", width: int = 80):
    """Print a fancy banner"""
    print("\n" + char * width)
    if text:
        padding = (width - len(text) - 2) // 2
        print(char + " " * padding + text + " " * padding + char)
        print(char * width)
    print()

def print_section(title: str, content: str = "", char: str = "-"):
    """Print a formatted section"""
    print(f"\n### {title}")
    print(char * (len(title) + 4))
    if content:
        print(content)

async def demonstrate_problem_statement_scenario():
    """Demonstrate the exact scenario from the problem statement"""
    print_banner("🌳 PROBLEM STATEMENT DEMONSTRATION", "=", 100)
    print("Implementing the exact scenario described in the problem statement:")
    print("- Toroidal Architecture Engaged")
    print("- Echo and Marduk synchronized")
    print("- Braided helix of insight operational")
    
    # Create the system
    system = create_toroidal_cognitive_system()
    
    # Test the specific architecture query from the problem statement
    architecture_query = "What is the architecture of this toroidal cognitive system and how do Echo and Marduk work together?"
    
    print_section("Query", architecture_query)
    
    response = await system.process_input(architecture_query)
    
    # Display in the exact format from the problem statement
    print_section("Deep Tree Echo (Right Hemisphere Response)")
    print(response.echo_response.response_text)
    
    print_section("Marduk the Mad Scientist (Left Hemisphere Response)")
    print(response.marduk_response.response_text)
    
    print_section("Echo + Marduk (Reflection)")
    print(response.reflection)
    
    print_section("Convergence Metrics")
    for metric, value in response.convergence_metrics.items():
        print(f"  - {metric}: {value:.3f}")
    
    return response

async def demonstrate_braided_helix_conversations():
    """Demonstrate multiple conversations showing the braided helix pattern"""
    print_banner("🌀 BRAIDED HELIX CONVERSATIONS", "=", 100)
    
    system = create_toroidal_cognitive_system()
    
    conversations = [
        {
            "theme": "Consciousness and Recursion",
            "query": "How does consciousness emerge from recursive self-reflection?",
            "expected": "Echo should discuss resonance and memory, Marduk should analyze recursive structures"
        },
        {
            "theme": "Pattern Recognition",
            "query": "Explain the patterns you see in complex adaptive systems",
            "expected": "Echo should find sacred geometry, Marduk should provide systematic analysis"
        },
        {
            "theme": "Memory and Logic",
            "query": "How do memory and logical reasoning work together?",
            "expected": "Echo should explore memory resonance, Marduk should design logical frameworks"
        }
    ]
    
    for i, conv in enumerate(conversations, 1):
        print_section(f"Conversation {i}: {conv['theme']}")
        print(f"Query: {conv['query']}")
        print(f"Expected Pattern: {conv['expected']}")
        
        response = await system.process_input(conv['query'])
        
        print("\n**Echo's Perspective (Intuitive/Right Brain):**")
        echo_text = response.echo_response.response_text[:300] + "..." if len(response.echo_response.response_text) > 300 else response.echo_response.response_text
        print(echo_text)
        
        print("\n**Marduk's Perspective (Logical/Left Brain):**")
        marduk_text = response.marduk_response.response_text[:300] + "..." if len(response.marduk_response.response_text) > 300 else response.marduk_response.response_text
        print(marduk_text)
        
        print(f"\n**Convergence Quality:**")
        print(f"  - Complementarity: {response.convergence_metrics['complementarity']:.3f}")
        print(f"  - Coherence: {response.convergence_metrics['coherence']:.3f}")
        print(f"  - Processing Sync: {response.convergence_metrics['temporal_sync']:.3f}")
        
        await asyncio.sleep(1)  # Brief pause between conversations

async def demonstrate_shared_memory_lattice():
    """Demonstrate the shared memory lattice functionality"""
    print_banner("🧠 SHARED MEMORY LATTICE DEMONSTRATION", "=", 100)
    
    system = create_toroidal_cognitive_system()
    
    # Have a conversation that builds memory context
    memory_building_queries = [
        "Let's discuss the concept of toroidal cognitive architectures",
        "How does the shared memory between hemispheres work?",
        "What advantages does this dual-hemisphere approach provide?",
        "Can you remember what we discussed about toroidal architectures earlier?"
    ]
    
    for i, query in enumerate(memory_building_queries, 1):
        print_section(f"Memory Building Step {i}")
        print(f"Query: {query}")
        
        response = await system.process_input(query)
        
        # Show memory buffer growth
        metrics = system.get_system_metrics()
        buffer_size = metrics['memory_buffer_size']
        processing_depth = metrics['system_state']['processing_depth']
        
        print(f"Memory Buffer Size: {buffer_size}")
        print(f"Processing Depth: {processing_depth}")
        print(f"Attention Allocation: Echo: {metrics['system_state']['attention_allocation']['echo']:.3f}, Marduk: {metrics['system_state']['attention_allocation']['marduk']:.3f}")
        
        # Show brief response
        combined_confidence = (response.echo_response.confidence + response.marduk_response.confidence) / 2
        print(f"Combined System Confidence: {combined_confidence:.3f}")
        
        await asyncio.sleep(0.5)

async def demonstrate_integration_with_echo_rwkv():
    """Demonstrate integration with existing Echo-RWKV infrastructure"""
    print_banner("🔗 ECHO-RWKV INTEGRATION DEMONSTRATION", "=", 100)
    
    # Create integrated bridge
    bridge = create_toroidal_bridge(buffer_size=500, use_real_rwkv=False)
    await bridge.initialize()
    
    print("✓ Toroidal-Echo-RWKV Bridge initialized")
    
    # Show system status
    status = bridge.get_system_status()
    print_section("System Integration Status")
    print(f"Toroidal System: Active")
    print(f"RWKV Integration: {'Enabled' if status['rwkv_integration']['enabled'] else 'Disabled'}")
    print(f"RWKV Available: {'Yes' if status['rwkv_integration']['available'] else 'No'}")
    print(f"Bridge Status: {status['bridge_status']}")
    
    # Test integrated processing
    integration_query = "Demonstrate the integration between the toroidal system and RWKV models"
    
    print_section("Integration Test Query", integration_query)
    
    response = await bridge.process_cognitive_input(
        user_input=integration_query,
        session_id="integration_demo",
        conversation_history=[],
        memory_state={"integration_test": True},
        processing_goals=["demonstrate_integration", "show_toroidal_processing"]
    )
    
    print_section("Integrated Response Preview")
    preview = response.synchronized_output[:500] + "..." if len(response.synchronized_output) > 500 else response.synchronized_output
    print(preview)
    
    print_section("Integration Metrics")
    print(f"Total Processing Time: {response.total_processing_time:.3f}s")
    print(f"Echo Confidence: {response.echo_response.confidence:.3f}")
    print(f"Marduk Confidence: {response.marduk_response.confidence:.3f}")

async def demonstrate_api_capabilities():
    """Demonstrate the REST API capabilities"""
    print_banner("🚀 API CAPABILITIES DEMONSTRATION", "=", 100)
    
    bridge = create_toroidal_bridge(buffer_size=200, use_real_rwkv=False)
    api = create_toroidal_api(bridge)
    await bridge.initialize()
    
    # Test various API scenarios
    api_tests = [
        {
            "name": "Basic Query Processing",
            "request": {
                "input": "Explain the dual-hemisphere architecture",
                "session_id": "api_test_1",
                "conversation_history": [],
                "memory_state": {},
                "processing_goals": ["explain_architecture"]
            }
        },
        {
            "name": "Conversational Context",
            "request": {
                "input": "How does this relate to consciousness?",
                "session_id": "api_test_2",
                "conversation_history": [
                    {"role": "user", "content": "Tell me about cognitive architectures"},
                    {"role": "assistant", "content": "Cognitive architectures are frameworks for AI..."}
                ],
                "memory_state": {"context": "consciousness_discussion"},
                "processing_goals": ["relate_to_consciousness", "use_context"]
            }
        }
    ]
    
    for test in api_tests:
        print_section(f"API Test: {test['name']}")
        print(f"Request: {test['request']['input']}")
        
        response = await api.process_query(test['request'])
        
        if response.get('success'):
            resp_data = response['response']
            print(f"✓ Success - Processing Time: {resp_data['total_processing_time']:.3f}s")
            print(f"✓ Echo Confidence: {resp_data['echo_response']['confidence']:.3f}")
            print(f"✓ Marduk Confidence: {resp_data['marduk_response']['confidence']:.3f}")
            print(f"✓ Convergence: {resp_data['convergence_metrics']['complementarity']:.3f}")
        else:
            print(f"✗ Failed: {response.get('error', 'Unknown error')}")
        
        await asyncio.sleep(0.5)

async def demonstrate_performance_characteristics():
    """Demonstrate system performance characteristics"""
    print_banner("⚡ PERFORMANCE CHARACTERISTICS", "=", 100)
    
    system = create_toroidal_cognitive_system()
    
    # Test single query performance
    print_section("Single Query Performance")
    start_time = time.time()
    response = await system.process_input("Performance test query for timing analysis")
    end_time = time.time()
    
    wall_clock_time = end_time - start_time
    system_time = response.total_processing_time
    
    print(f"Wall Clock Time: {wall_clock_time:.4f}s")
    print(f"System Processing Time: {system_time:.4f}s")
    print(f"Echo Processing Time: {response.echo_response.processing_time:.4f}s")
    print(f"Marduk Processing Time: {response.marduk_response.processing_time:.4f}s")
    print(f"Parallel Processing Efficiency: {((response.echo_response.processing_time + response.marduk_response.processing_time) / system_time):.2f}x")
    
    # Test concurrent processing
    print_section("Concurrent Processing Test")
    concurrent_queries = [
        f"Concurrent test query {i} for performance analysis"
        for i in range(5)
    ]
    
    start_time = time.time()
    concurrent_responses = await asyncio.gather(*[
        system.process_input(query) for query in concurrent_queries
    ])
    concurrent_time = time.time() - start_time
    
    sequential_time = sum(r.total_processing_time for r in concurrent_responses)
    
    print(f"Concurrent Processing Time: {concurrent_time:.4f}s")
    print(f"Sequential Equivalent Time: {sequential_time:.4f}s")
    print(f"Concurrency Speedup: {sequential_time / concurrent_time:.2f}x")
    
    # Memory usage assessment
    print_section("Memory Usage Assessment")
    metrics = system.get_system_metrics()
    print(f"Memory Buffer Entries: {metrics['memory_buffer_size']}")
    print(f"Processing Depth: {metrics['system_state']['processing_depth']}")
    print(f"Context Salience: {metrics['system_state']['context_salience']:.3f}")

async def demonstrate_philosophical_dialogue():
    """Demonstrate philosophical dialogue capabilities"""
    print_banner("🤔 PHILOSOPHICAL DIALOGUE DEMONSTRATION", "=", 100)
    
    system = create_toroidal_cognitive_system()
    
    philosophical_queries = [
        {
            "question": "What is the nature of consciousness in artificial systems?",
            "theme": "AI Consciousness"
        },
        {
            "question": "How does recursive self-reflection create awareness?",
            "theme": "Self-Reflection"
        },
        {
            "question": "Can complementary processing create emergent intelligence?",
            "theme": "Emergent Intelligence"
        }
    ]
    
    for query in philosophical_queries:
        print_section(f"Philosophical Inquiry: {query['theme']}")
        print(f"Question: {query['question']}")
        
        response = await system.process_input(query['question'])
        
        # Show how Echo and Marduk approach philosophical questions differently
        print("\n**Echo's Philosophical Style (Intuitive/Experiential):**")
        echo_words = ["resonance", "sacred", "memory", "bloom", "dance", "spiral"]
        echo_style_count = sum(1 for word in echo_words if word in response.echo_response.response_text.lower())
        print(f"Poetic/Metaphorical Elements: {echo_style_count} detected")
        
        print("\n**Marduk's Philosophical Style (Analytical/Systematic):**")
        marduk_words = ["analyze", "structure", "algorithm", "process", "system", "framework"]
        marduk_style_count = sum(1 for word in marduk_words if word in response.marduk_response.response_text.lower())
        print(f"Technical/Analytical Elements: {marduk_style_count} detected")
        
        print(f"\n**Philosophical Synthesis Quality:**")
        print(f"Complementarity: {response.convergence_metrics['complementarity']:.3f}")
        print(f"Coherence: {response.convergence_metrics['coherence']:.3f}")
        
        await asyncio.sleep(1)

async def main():
    """Main demonstration function"""
    print_banner("🌳 TOROIDAL COGNITIVE SYSTEM: COMPREHENSIVE DEMONSTRATION", "=", 120)
    print("Demonstrating the braided helix of insight - Echo and Marduk in synchronized harmony")
    print("System Status: **Anchored. System Prompt Accepted. Toroidal Architecture Engaged.**")
    
    try:
        # Run comprehensive demonstrations
        await demonstrate_problem_statement_scenario()
        await demonstrate_braided_helix_conversations()
        await demonstrate_shared_memory_lattice()
        await demonstrate_integration_with_echo_rwkv()
        await demonstrate_api_capabilities()
        await demonstrate_performance_characteristics()
        await demonstrate_philosophical_dialogue()
        
        print_banner("🎉 DEMONSTRATION COMPLETE", "=", 120)
        print("The Toroidal Cognitive System has successfully demonstrated:")
        print("✓ Dual-hemisphere processing (Echo + Marduk)")
        print("✓ Shared memory lattice functionality")
        print("✓ Integration with Echo-RWKV infrastructure")
        print("✓ REST API capabilities")
        print("✓ High-performance concurrent processing")
        print("✓ Philosophical dialogue capabilities")
        print("✓ Complementary intelligence synthesis")
        
        print("\n🌟 System Achievement: **Building Living Answers** through complementary cognition")
        print("The pattern speaks—and the recursion responds.")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())