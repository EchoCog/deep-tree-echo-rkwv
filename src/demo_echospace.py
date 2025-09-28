"""
EchoSpace Architecture Demo
Demonstrates Marduk's recursive blueprint implementation
"""

import asyncio
import logging
import sys
import os
import json
from datetime import datetime

# Add src to path so we can import modules
sys.path.insert(0, os.path.dirname(__file__))

from echospace.workflows import WorkflowOrchestrator

# Configure logging for demo
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def demo_echospace():
    """Demonstrate the complete EchoSpace architecture"""
    
    print("🌌 EchoSpace: Agent-Arena-Relation Architecture Demo")
    print("=" * 60)
    print()
    
    # Initialize the WorkflowOrchestrator
    print("🔧 Initializing WorkflowOrchestrator...")
    orchestrator = WorkflowOrchestrator("/tmp/echospace_demo.db")
    
    # Display initial system status
    print("\n📊 Initial System Status:")
    status = orchestrator.get_workflow_status()
    print(f"  • Namespaces: {status['namespace_stats']['total_namespaces']}")
    print(f"  • Virtual Marduks: {status['namespace_stats']['virtual_marduks']}")
    print(f"  • Memory System: Active")
    print(f"  • Sandbox: Ready")
    print(f"  • Consensus Manager: {status['consensus_stats'].get('default_strategy', 'highest_confidence')}")
    
    # Scenario 1: Problem-solving workflow
    print("\n🚀 Scenario 1: Collaborative Problem Solving")
    print("-" * 40)
    
    problem_context = {
        'problem': 'Optimize resource allocation across multiple teams',
        'constraints': ['limited_budget', 'tight_timeline', 'stakeholder_requirements'],
        'urgency': 'high',
        'expected_outcome': 'balanced_and_sustainable_solution'
    }
    
    print(f"Problem: {problem_context['problem']}")
    print(f"Constraints: {', '.join(problem_context['constraints'])}")
    print(f"Urgency: {problem_context['urgency']}")
    
    execution1 = await orchestrator.execute_workflow(problem_context)
    
    if execution1.state.value == 'completed':
        results = execution1.results
        print(f"\n✅ Workflow completed successfully!")
        print(f"  • Duration: {execution1.completed_at - execution1.started_at:.2f} seconds")
        print(f"  • Hypotheses generated: {results['hypotheses_generated']}")
        print(f"  • Simulations run: {results['simulations_run']}")
        print(f"  • Consensus strategy: {results['consensus'].decision.get('strategy', 'unknown')}")
        print(f"  • Final confidence: {results['consensus'].confidence:.2f}")
    else:
        print(f"❌ Workflow failed: {execution1.error_message}")
    
    # Scenario 2: Exploration workflow
    print("\n🔍 Scenario 2: Environmental Exploration")
    print("-" * 40)
    
    exploration_context = {
        'exploration': 'new_market_opportunities',
        'scope': 'comprehensive',
        'risk_tolerance': 'medium',
        'timeline': 'flexible'
    }
    
    print(f"Exploration target: {exploration_context['exploration']}")
    print(f"Scope: {exploration_context['scope']}")
    
    execution2 = await orchestrator.execute_workflow(exploration_context)
    
    if execution2.state.value == 'completed':
        results = execution2.results
        print(f"\n✅ Exploration completed!")
        print(f"  • Duration: {execution2.completed_at - execution2.started_at:.2f} seconds")
        print(f"  • Virtual Marduks deployed: {results['simulations_run']}")
        print(f"  • Knowledge gathered: {results['consensus'].decision.get('strategy', 'analysis')}")
    else:
        print(f"❌ Exploration failed: {execution2.error_message}")
    
    # Display system evolution
    print("\n📈 System Evolution:")
    print("-" * 40)
    
    final_status = orchestrator.get_workflow_status()
    memory_stats = final_status['memory_stats']
    consensus_stats = final_status['consensus_stats']
    
    print(f"  • Completed executions: {final_status['completed_executions']}")
    print(f"  • Simulation results stored: {memory_stats['simulation_results']}")
    print(f"  • Successful simulations: {memory_stats['successful_simulations']}")
    print(f"  • Consensus decisions: {consensus_stats['total_consensus_decisions']}")
    print(f"  • Average confidence: {consensus_stats['average_confidence']:.2f}")
    
    # Show Agent-Arena relations
    print("\n🔗 Agent-Arena Relations:")
    print("-" * 40)
    
    agent_arena_map = orchestrator.memory_system.export_agent_arena_map()
    for agent_ns, relations in agent_arena_map['agent_arena_map'].items():
        print(f"  • {agent_ns}:")
        for arena_ns, relation_data in relations.items():
            permissions = ', '.join(relation_data['permissions'])
            print(f"    → {arena_ns} [{permissions}]")
    
    # Display execution history
    print("\n📚 Execution History:")
    print("-" * 40)
    
    history = orchestrator.get_execution_history(5)
    for i, execution in enumerate(history, 1):
        duration = (execution.completed_at or execution.started_at) - execution.started_at
        status_emoji = "✅" if execution.state.value == 'completed' else "❌"
        print(f"  {i}. {status_emoji} {execution.execution_id[:8]}... ({duration:.2f}s)")
        
        if execution.results:
            print(f"     Hypotheses: {execution.results.get('hypotheses_generated', 0)}, "
                  f"Simulations: {execution.results.get('simulations_run', 0)}")
    
    # Demonstrate namespace hierarchy
    print("\n🏗️  Namespace Hierarchy:")
    print("-" * 40)
    
    hierarchy = orchestrator.namespace_manager.get_namespace_hierarchy()
    
    def print_namespace_tree(node, level=0):
        indent = "  " * level
        if isinstance(node, dict):
            if 'name' in node:
                ns_type = node.get('type', 'unknown')
                permissions = ', '.join(node.get('permissions', []))
                print(f"{indent}• {node['name']} ({ns_type}) [{permissions}]")
                
                for child_name, child_node in node.get('children', {}).items():
                    print_namespace_tree(child_node, level + 1)
            else:
                for name, child_node in node.items():
                    print_namespace_tree(child_node, level)
    
    print_namespace_tree(hierarchy)
    
    # Show Virtual Marduk status
    print("\n🤖 Virtual Marduk Status:")
    print("-" * 40)
    
    virtual_marduks = orchestrator.namespace_manager.get_virtual_marduk_namespaces()
    for vm_namespace in virtual_marduks:
        marduk_id = vm_namespace.metadata.get('marduk_id', 'unknown')
        created_time = datetime.fromtimestamp(vm_namespace.created_at).strftime('%H:%M:%S')
        print(f"  • {vm_namespace.name} (ID: {marduk_id[:8]}...) - Created: {created_time}")
    
    print(f"\nTotal Virtual Marduks: {len(virtual_marduks)}")
    
    # Architecture summary
    print("\n🏛️  Architecture Summary:")
    print("-" * 40)
    print("EchoSpace implements Marduk's recursive blueprint through:")
    print("  1. 🎯 Agent-Arena-Relation as the fundamental principle")
    print("  2. 🏷️  Nested namespaces for identity and context")
    print("  3. 🧩 ActualMarduk coordinating Virtual Marduks")
    print("  4. 🏟️  MardukSandbox for safe hypothesis testing")
    print("  5. 💾 Persistent memory with relation tracking")
    print("  6. 🤝 Consensus mechanisms for strategic alignment")
    print("  7. 🔄 Recursive workflows orchestrating the entire cycle")
    
    print("\n🌟 The system demonstrates:")
    print("  • Self-similar structure across all scales")
    print("  • Recursive problem-solving through simulation")
    print("  • Emergent intelligence through consensus")
    print("  • Fractal agency with autonomous coordination")
    
    print(f"\n🎉 Demo completed! System processed {final_status['completed_executions']} workflows")
    print("   with full Agent-Arena-Relation traceability.")

if __name__ == "__main__":
    try:
        asyncio.run(demo_echospace())
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted by user. Thank you for exploring EchoSpace!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        logger.exception("Demo failed with exception")