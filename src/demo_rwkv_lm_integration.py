#!/usr/bin/env python3
"""
Demo script showing RWKV-LM integration with Deep Tree Echo
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the src directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import Deep Tree Echo integration
from echo_rwkv_bridge import EchoRWKVIntegrationEngine, CognitiveContext
from integrations.rwkv_repos import RWKVRepoManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def demo_repository_integration():
    """Demo the RWKV repository integration"""
    print("🧠 Deep Tree Echo RWKV-LM Integration Demo")
    print("=" * 50)
    
    # Initialize repository manager
    repo_manager = RWKVRepoManager()
    
    # List available repositories
    print("\n📦 Available RWKV Repositories:")
    for name, repo_info in repo_manager.repositories.items():
        status = "✅ Available" if repo_info.available else "❌ Not Available"
        print(f"  {name}: {status}")
        print(f"    Description: {repo_info.description}")
        if repo_info.available and repo_info.entry_points:
            print(f"    Entry Points: {list(repo_info.entry_points.keys())}")
        print()
    
    # Get RWKV-LM information
    rwkv_lm = repo_manager.get_repository("RWKV-LM")
    if rwkv_lm and rwkv_lm.available:
        print("🎯 RWKV-LM Repository Details:")
        print(f"  Path: {rwkv_lm.path}")
        print(f"  Available Versions: {rwkv_lm.metadata.get('versions', [])}")
        print(f"  Latest Version: {rwkv_lm.metadata.get('latest_version', 'unknown')}")
        print(f"  Architecture: {rwkv_lm.metadata.get('architecture', 'unknown')}")
        print(f"  Features: {rwkv_lm.metadata.get('features', [])}")
        
        # List demo files
        demo_path = rwkv_lm.path / "RWKV-v7"
        if demo_path.exists():
            demo_files = [f for f in demo_path.glob("*.py") if "demo" in f.name]
            print(f"  Demo Files: {[f.name for f in demo_files]}")
    
    return repo_manager

async def demo_cognitive_processing():
    """Demo cognitive processing with RWKV integration"""
    print("\n🧠 Deep Tree Echo Cognitive Processing Demo")
    print("=" * 50)
    
    # Create integration engine
    engine = EchoRWKVIntegrationEngine(
        use_real_rwkv=True,  # Use real RWKV if available
        use_cpp_backend=False  # Start with Python backend for demo
    )
    
    # Configure with minimal settings (no actual model for demo)
    config = {
        'rwkv': {
            'model_path': '',  # No model path for demo
            'strategy': 'cpu fp32',
            'max_tokens': 100,
            'temperature': 0.8,
        },
        'enable_advanced_cognitive': True
    }
    
    try:
        # Initialize the cognitive system
        await engine.initialize(config)
        print("✅ Cognitive engine initialized successfully")
        
        # Create cognitive context
        context = CognitiveContext(
            session_id="demo_session",
            user_input="What is RWKV and how does it work?",
            conversation_history=[],
            memory_state={},
            processing_goals=["explain_technology", "provide_context"],
            temporal_context=["current_ai_landscape"],
            metadata={"demo": True}
        )
        
        print(f"\n🎯 Processing Input: '{context.user_input}'")
        print("Processing through cognitive membranes...")
        
        # Process cognitive input
        response = await engine.process_cognitive_input(context)
        
        print("\n📊 Cognitive Processing Results:")
        print(f"Memory Response: {response.memory_response.output_text}")
        print(f"Reasoning Response: {response.reasoning_response.output_text}")
        print(f"Grammar Response: {response.grammar_response.output_text}")
        print(f"Integrated Output: {response.integrated_output}")
        print(f"Processing Time: {response.total_processing_time}ms")
        
    except Exception as e:
        print(f"❌ Error during cognitive processing: {e}")
        logger.exception("Cognitive processing error")

def demo_rwkv_v7_info():
    """Demo RWKV v7 information from the repository"""
    print("\n🚀 RWKV-7 'Goose' Information")
    print("=" * 50)
    
    rwkv_lm_path = Path(__file__).parent.parent / "external" / "RWKV-LM"
    
    if not rwkv_lm_path.exists():
        print("❌ RWKV-LM repository not found")
        return
    
    # Read RWKV-7 README if available
    v7_readme = rwkv_lm_path / "RWKV-v7" / "README.md"
    if v7_readme.exists():
        try:
            with open(v7_readme, 'r') as f:
                content = f.read()
                print("📖 RWKV-7 README snippet:")
                # Show first few lines
                lines = content.split('\n')[:10]
                for line in lines:
                    print(f"  {line}")
                print("  ...")
        except Exception as e:
            print(f"❌ Error reading RWKV-7 README: {e}")
    
    # List available demo files
    v7_path = rwkv_lm_path / "RWKV-v7"
    if v7_path.exists():
        print("\n🎮 Available RWKV-7 Demo Files:")
        demo_files = list(v7_path.glob("*demo*.py"))
        for demo_file in demo_files:
            print(f"  📄 {demo_file.name}")
            
        print("\n📚 Other RWKV-7 Files:")
        other_files = [f for f in v7_path.glob("*.py") if "demo" not in f.name]
        for other_file in other_files[:5]:  # Show first 5
            print(f"  📄 {other_file.name}")

async def main():
    """Main demo function"""
    print("🌟 Welcome to Deep Tree Echo RWKV-LM Integration Demo!")
    print("This demo showcases the integration of RWKV-LM with Deep Tree Echo")
    print()
    
    # Demo 1: Repository integration
    repo_manager = demo_repository_integration()
    
    # Demo 2: RWKV-7 information
    demo_rwkv_v7_info()
    
    # Demo 3: Cognitive processing
    await demo_cognitive_processing()
    
    print("\n🎉 Demo completed successfully!")
    print("Next steps:")
    print("1. Download an RWKV model to enable full functionality")
    print("2. Configure model path in the integration")
    print("3. Explore the RWKV-LM repository for training and inference examples")

if __name__ == "__main__":
    asyncio.run(main())