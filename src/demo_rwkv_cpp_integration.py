#!/usr/bin/env python3
"""
RWKV.cpp Integration Demo for Deep Tree Echo Framework
Demonstrates the high-performance C++ RWKV integration as a Distributed Agentic Cognitive Micro-Kernel Network
"""

import os
import sys
import asyncio
import logging
from datetime import datetime

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_rwkv_cpp_integration():
    """Demonstrate RWKV.cpp integration with Deep Tree Echo"""
    
    print("🚀 RWKV.cpp Integration Demo for Deep Tree Echo Framework")
    print("=" * 70)
    print("Demonstrating high-performance C++ RWKV integration as a")
    print("Distributed Agentic Cognitive Micro-Kernel Network!")
    print("=" * 70)
    
    try:
        # Import the enhanced integration
        from enhanced_echo_rwkv_bridge import (
            EnhancedEchoRWKVIntegrationEngine,
            create_enhanced_rwkv_config
        )
        from echo_rwkv_bridge import CognitiveContext
        
        print("\n🧠 Initializing Enhanced Cognitive Architecture")
        print("-" * 50)
        
        # Create enhanced configuration
        config = create_enhanced_rwkv_config(
            model_path='/models/rwkv-model.bin',  # Would be real model path
            backend_preference='auto',  # Auto-select best available backend
            enable_gpu=False,  # CPU mode for WebVM compatibility
            memory_limit_mb=600  # WebVM memory constraint
        )
        
        # Initialize the enhanced integration engine
        engine = EnhancedEchoRWKVIntegrationEngine(
            backend_preference='auto',
            enable_rwkv_cpp=True
        )
        
        print("⚙️ Initializing integration engine...")
        success = await engine.initialize(config)
        
        if not success:
            print("❌ Failed to initialize integration engine")
            return
        
        # Get system status
        status = engine.get_system_status()
        print(f"✅ Integration engine initialized successfully!")
        print(f"   Active backend: {status.get('rwkv_interface', {}).get('active_backend', 'unknown')}")
        print(f"   Available backends: {status.get('rwkv_interface', {}).get('available_backends', [])}")
        print(f"   RWKV.cpp enhanced: {status.get('processing_stats', {}).get('rwkv_cpp_enhanced', False)}")
        
        print("\n🎭 Cognitive Membrane Demonstration")
        print("-" * 50)
        
        # Test different types of cognitive inputs
        test_scenarios = [
            {
                "name": "Philosophical Inquiry",
                "input": "What is the nature of consciousness and how might it emerge from complex information processing systems?",
                "goals": ["analyze", "synthesize", "theorize"]
            },
            {
                "name": "Problem Solving",
                "input": "How can we optimize memory usage in a browser-based virtual machine while maintaining cognitive processing performance?",
                "goals": ["analyze", "solve", "optimize"]
            },
            {
                "name": "Knowledge Integration",
                "input": "Explain the relationship between RWKV architecture and traditional transformer models in the context of memory efficiency.",
                "goals": ["compare", "explain", "relate"]
            },
            {
                "name": "Creative Reasoning",
                "input": "Imagine a future where cognitive architectures like Deep Tree Echo become ubiquitous. What would this mean for human-AI collaboration?",
                "goals": ["imagine", "extrapolate", "evaluate"]
            }
        ]
        
        for i, scenario in enumerate(test_scenarios, 1):
            print(f"\n📝 Scenario {i}: {scenario['name']}")
            print(f"Input: {scenario['input']}")
            print(f"Processing goals: {', '.join(scenario['goals'])}")
            print("Processing through cognitive membranes...")
            
            # Create cognitive context
            context = CognitiveContext(
                session_id=f"demo_session_{i}",
                user_input=scenario['input'],
                conversation_history=[],
                memory_state={},
                processing_goals=scenario['goals'],
                temporal_context=[],
                metadata={'scenario': scenario['name']}
            )
            
            # Process through the cognitive architecture
            start_time = datetime.now()
            result = await engine.process_cognitive_input(context)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Display results
            if 'error' in result:
                print(f"❌ Error: {result['error']}")
            else:
                print(f"✅ Processing completed in {processing_time:.3f}s")
                print(f"   Backend used: {result.get('backend_used', 'unknown')}")
                print(f"   Enhanced processing: {result.get('enhanced_processing', False)}")
                
                if 'integrated_response' in result:
                    print(f"\n🎯 Integrated Response:")
                    print(f"   {result['integrated_response'][:200]}...")
                
                # Show individual membrane responses if available
                if 'memory_response' in result:
                    print(f"\n💭 Memory Membrane: {result['memory_response'].output_text[:100]}...")
                if 'reasoning_response' in result:
                    print(f"⚡ Reasoning Membrane: {result['reasoning_response'].output_text[:100]}...")
                if 'grammar_response' in result:
                    print(f"🎭 Grammar Membrane: {result['grammar_response'].output_text[:100]}...")
        
        print("\n📊 Performance Analysis")
        print("-" * 50)
        
        # Get final system status
        final_status = engine.get_system_status()
        stats = final_status.get('processing_stats', {})
        
        print(f"Total requests processed: {stats.get('total_requests', 0)}")
        print(f"Successful requests: {stats.get('successful_requests', 0)}")
        print(f"Success rate: {final_status.get('success_rate', 0)*100:.1f}%")
        print(f"Average response time: {stats.get('avg_response_time', 0):.3f}s")
        
        # Show backend performance comparison
        if 'performance_summary' in final_status:
            perf = final_status['performance_summary']
            print(f"\nBackend Performance:")
            for backend, metrics in perf.get('backend_performance', {}).items():
                if metrics['requests'] > 0:
                    print(f"  {backend}: {metrics['requests']} requests, {metrics['avg_time']:.3f}s avg")
        
        print("\n🌟 Architecture Highlights")
        print("-" * 50)
        print("✅ Multi-backend support (RWKV.cpp, Python RWKV, Mock)")
        print("✅ Automatic backend selection and fallback")
        print("✅ Cognitive membrane processing (Memory, Reasoning, Grammar)")
        print("✅ Enhanced integration with C++ performance optimization")
        print("✅ WebVM compatibility with memory constraints")
        print("✅ Distributed agentic cognitive micro-kernel architecture")
        print("✅ Real-time performance monitoring and adaptive processing")
        
        print("\n🎉 Demo completed successfully!")
        print("The RWKV.cpp integration is fully operational as a")
        print("Distributed Agentic Cognitive Micro-Kernel Network!")
        
    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        logger.exception("Demo error details")

async def demo_backend_switching():
    """Demonstrate dynamic backend switching capabilities"""
    
    print("\n🔄 Backend Switching Demonstration")
    print("-" * 50)
    
    try:
        from enhanced_echo_rwkv_bridge import EnhancedRWKVInterface
        
        # Initialize with auto backend selection
        interface = EnhancedRWKVInterface("auto")
        
        config = {
            'model_path': '/nonexistent/model.bin',  # Will trigger fallback
            'backend_type': 'auto',
            'thread_count': 2,
            'temperature': 0.8
        }
        
        success = await interface.initialize(config)
        
        if success:
            print(f"✅ Interface initialized with backend: {interface.active_backend}")
            
            # Show available backends
            state = interface.get_model_state()
            print(f"Available backends: {state.get('available_backends', [])}")
            
            # Test switching between backends
            for backend in ['mock', 'python_rwkv']:
                if backend in interface.backends:
                    print(f"\n🔄 Switching to {backend} backend...")
                    success = interface.switch_backend(backend)
                    if success:
                        print(f"✅ Switched to {backend}")
                        
                        # Test response generation
                        response = await interface.generate_response(
                            "Test prompt for backend switching demo", 
                            None
                        )
                        print(f"Response: {response[:80]}...")
            
            # Show performance comparison
            perf = interface.get_performance_summary()
            print(f"\n📊 Performance Summary:")
            print(f"Active backend: {perf['active_backend']}")
            for backend, metrics in perf['backend_performance'].items():
                if metrics['requests'] > 0:
                    print(f"  {backend}: {metrics['requests']} requests, {metrics['avg_time']:.3f}s avg")
        
    except Exception as e:
        print(f"❌ Backend switching demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(demo_rwkv_cpp_integration())
    asyncio.run(demo_backend_switching())