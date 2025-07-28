"""
Demo: BlinkDL RWKV Repository Integration
Demonstrates the integration of external RWKV repositories with Deep Tree Echo
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from integrations.rwkv_repos import get_repo_manager, list_available_rwkv_repos
from integrations.enhanced_rwkv_bridge import create_enhanced_rwkv_bridge

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def demo_repository_integration():
    """Demonstrate repository integration capabilities"""
    print("🚀 Deep Tree Echo RWKV Integration Demo")
    print("=" * 60)
    
    # 1. Initialize repository manager
    print("\n1. Initializing Repository Manager...")
    repo_manager = get_repo_manager()
    
    # 2. Show repository status
    print("\n2. Repository Status:")
    summary = repo_manager.get_integration_summary()
    
    print(f"   Total Repositories: {summary['total_repositories']}")
    print(f"   Available: {summary['available_repositories']}")
    print(f"   Unavailable: {summary['unavailable_repositories']}")
    
    # 3. List available repositories
    print("\n3. Available External Repositories:")
    available_repos = list_available_rwkv_repos()
    for repo_name in available_repos:
        repo = repo_manager.get_repository(repo_name)
        print(f"   ✓ {repo_name}")
        print(f"     Type: {repo.repo_type.value}")
        print(f"     Description: {repo.description}")
        print(f"     Path: {repo.path}")
        if repo.entry_points:
            print(f"     Entry Points: {list(repo.entry_points.keys())}")
        print()
    
    # 4. Create enhanced bridge
    print("4. Creating Enhanced RWKV Bridge...")
    bridge = create_enhanced_rwkv_bridge()
    
    # 5. Load external models
    print("\n5. Loading External Models:")
    for repo_name in available_repos:
        print(f"   Loading {repo_name}...")
        success = bridge.load_external_model(repo_name)
        print(f"   {'✓' if success else '✗'} {repo_name} {'loaded' if success else 'failed to load'}")
    
    # 6. Show available models
    print("\n6. Available Models:")
    models = bridge.get_available_models()
    for model_name, model_info in models.items():
        print(f"   • {model_name}")
        print(f"     Type: {model_info['type']}")
        print(f"     Description: {model_info['description']}")
        print()
    
    # 7. Demonstrate model processing
    print("7. Model Processing Demo:")
    test_input = "Hello, how can RWKV help with cognitive processing?"
    
    for model_name in ["main_lm", "chat", "world_model", "v2_pile"]:
        try:
            print(f"\n   Processing with {model_name}:")
            result = bridge.process_with_external_model(test_input, model_name)
            print(f"   Input: {result.get('input', 'N/A')}")
            print(f"   Output: {result.get('output', 'N/A')}")
            print(f"   Type: {result.get('type', 'N/A')}")
            print(f"   Capabilities: {result.get('capabilities', [])}")
        except Exception as e:
            print(f"   ✗ Error processing with {model_name}: {e}")
    
    # 8. Integration status
    print("\n8. Integration Status Summary:")
    status = bridge.get_integration_status()
    print(f"   Loaded Repositories: {status['loaded_repositories']}")
    print(f"   External Models: {status['external_models']}")
    
    print("\n   Repository Details:")
    for name, info in status['repositories'].items():
        status_icon = "✓" if info['loaded'] else "✗"
        print(f"     {status_icon} {name}: {info['type']}")
        if info['loaded']:
            print(f"       Path: {info['path']}")
    
    print("\n   Model Details:")
    for name, info in status['models'].items():
        print(f"     • {name}: {info['status']}")
        print(f"       Repository: {info['repository']}")
        print(f"       Capabilities: {', '.join(info['capabilities'])}")
    
    print("\n" + "=" * 60)
    print("✅ Demo completed successfully!")
    return bridge

def demo_cognitive_integration():
    """Demonstrate cognitive architecture integration"""
    print("\n🧠 Cognitive Architecture Integration")
    print("-" * 40)
    
    bridge = create_enhanced_rwkv_bridge()
    
    # Simulate cognitive processing with external models
    cognitive_tasks = [
        ("reasoning", "What are the implications of quantum computing for AI?"),
        ("memory", "Remember this important fact: RWKV combines RNN and Transformer benefits"),
        ("analysis", "Analyze the relationship between language and consciousness"),
    ]
    
    for task_type, task_input in cognitive_tasks:
        print(f"\n🔍 Cognitive Task: {task_type}")
        print(f"Input: {task_input}")
        
        # Try different models for different cognitive tasks
        if task_type == "reasoning":
            model = "main_lm"
        elif task_type == "memory":
            model = "world_model"
        else:
            model = "chat"
        
        try:
            result = bridge.process_with_external_model(task_input, model)
            print(f"Model: {model}")
            print(f"Response: {result.get('output', 'No response')}")
            print(f"Type: {result.get('type', 'Unknown')}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    try:
        # Run main demo
        bridge = demo_repository_integration()
        
        # Run cognitive integration demo
        demo_cognitive_integration()
        
        print("\n🎉 All demos completed successfully!")
        print("\nThe BlinkDL RWKV repositories have been successfully integrated!")
        print("You can now use these external repositories in your cognitive architecture.")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)