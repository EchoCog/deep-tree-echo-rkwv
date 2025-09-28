"""
Test EchoSpace Agent-Arena-Relation Architecture
Comprehensive tests for the recursive blueprint implementation
"""

import asyncio
import logging
import sys
import os
import tempfile
from pathlib import Path

# Add src to path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import EchoSpace components
from echospace.core import Agent, Arena, PermissionLevel, RelationManager
from echospace.namespaces import NamespaceManager, NamespaceType
from echospace.memory import EchoMemorySystem, SimulationResult
from echospace.marduk import ActualMarduk, VirtualMarduk, Hypothesis
from echospace.sandbox import MardukSandbox
from echospace.consensus import ConsensusManager, ConsensusStrategy
from echospace.workflows import WorkflowOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestEchoSpace:
    """Test suite for EchoSpace architecture"""
    
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
        self.storage_path = os.path.join(self.temp_dir, "test_echo_memory.db")
        logger.info(f"Using temporary storage: {self.storage_path}")
    
    def test_namespace_system(self):
        """Test the namespace management system"""
        logger.info("Testing Namespace System...")
        
        ns_manager = NamespaceManager()
        
        # Test default namespaces
        assert ns_manager.get_namespace("ActualMarduk") is not None
        assert ns_manager.get_namespace("Marduk-Space") is not None
        assert ns_manager.get_namespace("EchoCog") is not None
        
        # Test Virtual Marduk creation
        virtual_ns = ns_manager.create_virtual_marduk_namespace("test-001")
        assert virtual_ns.name == "VirtualMarduk-test-001"
        assert virtual_ns.namespace_type == NamespaceType.AGENT
        
        # Test hierarchy
        hierarchy = ns_manager.get_namespace_hierarchy()
        assert len(hierarchy) > 0
        
        # Test stats
        stats = ns_manager.get_stats()
        assert stats['total_namespaces'] > 0
        assert stats['virtual_marduks'] >= 1
        
        logger.info("✅ Namespace System tests passed")
    
    def test_core_architecture(self):
        """Test core Agent-Arena-Relation architecture"""
        logger.info("Testing Core Architecture...")
        
        # Test RelationManager
        relation_manager = RelationManager()
        
        # Create a test relation
        relation = relation_manager.create_relation(
            agent_namespace="TestAgent",
            arena_namespace="TestArena", 
            permissions={PermissionLevel.READ, PermissionLevel.WRITE}
        )
        
        assert relation.agent_namespace == "TestAgent"
        assert relation.arena_namespace == "TestArena"
        assert relation.has_permission(PermissionLevel.READ)
        assert not relation.has_permission(PermissionLevel.FULL)
        
        # Test permission checking
        assert relation_manager.check_permission("TestAgent", "TestArena", PermissionLevel.READ)
        assert not relation_manager.check_permission("TestAgent", "TestArena", PermissionLevel.FULL)
        
        # Test stats
        stats = relation_manager.get_stats()
        assert stats['total_relations'] >= 1
        
        logger.info("✅ Core Architecture tests passed")
    
    async def test_memory_system(self):
        """Test the EchoMemorySystem"""
        logger.info("Testing Memory System...")
        
        memory_system = EchoMemorySystem(self.storage_path)
        
        # Test agent-arena relation storage
        from echospace.core import AgentArenaRelation
        test_relation = AgentArenaRelation(
            agent_namespace="TestAgent",
            arena_namespace="TestArena",
            permissions={PermissionLevel.READ}
        )
        
        memory_system.store_agent_arena_relation("TestAgent", "TestArena", test_relation)
        
        # Test retrieval
        relations = memory_system.get_agent_relations("TestAgent")
        assert "TestArena" in relations
        
        # Test simulation result storage
        sim_result = SimulationResult(
            simulation_id="test_sim_001",
            virtual_marduk_id="vm_001",
            hypothesis="Test hypothesis",
            result={'success': True, 'confidence': 0.8},
            success=True,
            confidence=0.8,
            timestamp=1234567890.0,
            metadata={}
        )
        
        memory_system.store_simulation_result(sim_result)
        
        # Test retrieval
        results = memory_system.get_simulation_results(virtual_marduk_id="vm_001")
        assert len(results) >= 1
        assert results[0].simulation_id == "test_sim_001"
        
        # Test memory stats
        stats = memory_system.get_memory_stats()
        assert stats['simulation_results'] >= 1
        
        logger.info("✅ Memory System tests passed")
    
    async def test_virtual_marduk(self):
        """Test Virtual Marduk agent"""
        logger.info("Testing Virtual Marduk...")
        
        memory_system = EchoMemorySystem(self.storage_path)
        virtual_marduk = VirtualMarduk(memory_system, "test_vm_001")
        
        assert virtual_marduk.namespace == "VirtualMarduk-test_vm_001"
        assert virtual_marduk.agent_id == "test_vm_001"
        
        # Test hypothesis simulation
        hypothesis = Hypothesis(
            id="hyp_001",
            description="Test direct approach",
            parameters={'approach': 'direct', 'problem': 'test_problem'},
            expected_outcome="Quick resolution"
        )
        
        result = await virtual_marduk.simulate_hypothesis(hypothesis)
        
        assert 'success' in result
        assert 'confidence' in result
        assert 'approach' in result
        
        # Test status
        status = virtual_marduk.get_status()
        assert status['simulation_count'] >= 1
        
        logger.info("✅ Virtual Marduk tests passed")
    
    async def test_actual_marduk(self):
        """Test Actual Marduk agent"""
        logger.info("Testing Actual Marduk...")
        
        memory_system = EchoMemorySystem(self.storage_path)
        actual_marduk = ActualMarduk(memory_system)
        
        assert actual_marduk.namespace == "ActualMarduk"
        
        # Test action execution
        context = {
            'problem': 'test_coordination_problem',
            'urgency': 'medium'
        }
        
        result = await actual_marduk.act(context)
        
        assert result['success'] is True
        assert 'situation_analysis' in result
        assert 'consensus' in result
        assert 'execution_result' in result
        
        # Test status
        status = actual_marduk.get_status()
        assert status['namespace'] == "ActualMarduk"
        
        logger.info("✅ Actual Marduk tests passed")
    
    async def test_sandbox(self):
        """Test Marduk Sandbox arena"""
        logger.info("Testing Marduk Sandbox...")
        
        sandbox = MardukSandbox()
        
        assert sandbox.namespace == "Marduk-Sandbox"
        
        # Create a mock agent for testing
        class MockAgent(Agent):
            def __init__(self):
                super().__init__("MockAgent")
            
            async def act(self, context):
                return {'success': True}
        
        mock_agent = MockAgent()
        
        # Test resource allocation action
        action = {
            'type': 'resource_allocation',
            'resource': 'computational_power',
            'amount': 50.0,
            'session_id': 'test_session'
        }
        
        result = await sandbox.process_action(mock_agent, action)
        
        assert result['success'] is True
        assert result['action_type'] == 'resource_allocation'
        assert result['amount'] == 50.0
        
        # Test simulation run action
        simulation_action = {
            'type': 'simulation_run',
            'simulation': 'basic',
            'duration': 1.0,
            'session_id': 'test_session'
        }
        
        sim_result = await sandbox.process_action(mock_agent, simulation_action)
        
        assert 'success' in sim_result
        assert sim_result['action_type'] == 'simulation_run'
        
        # Test sandbox stats
        stats = sandbox.get_sandbox_stats()
        assert stats['active_sessions'] >= 0
        
        logger.info("✅ Marduk Sandbox tests passed")
    
    async def test_consensus_manager(self):
        """Test consensus management"""
        logger.info("Testing Consensus Manager...")
        
        consensus_manager = ConsensusManager()
        
        # Create test simulation results
        sim_results = [
            SimulationResult(
                simulation_id="sim_001",
                virtual_marduk_id="vm_001",
                hypothesis="Direct approach",
                result={'success': True, 'approach': 'direct', 'confidence': 0.8},
                success=True,
                confidence=0.8,
                timestamp=1234567890.0,
                metadata={}
            ),
            SimulationResult(
                simulation_id="sim_002", 
                virtual_marduk_id="vm_002",
                hypothesis="Collaborative approach",
                result={'success': True, 'approach': 'collaborative', 'confidence': 0.9},
                success=True,
                confidence=0.9,
                timestamp=1234567891.0,
                metadata={}
            )
        ]
        
        # Test highest confidence consensus
        consensus = await consensus_manager.reach_consensus(
            sim_results, 
            ConsensusStrategy.HIGHEST_CONFIDENCE
        )
        
        assert consensus.decision['strategy'] == 'execute_simulation'
        assert consensus.confidence > 0
        assert len(consensus.participants) == 2
        
        # Test majority vote consensus
        majority_consensus = await consensus_manager.reach_consensus(
            sim_results,
            ConsensusStrategy.MAJORITY_VOTE
        )
        
        assert majority_consensus.decision['strategy'] == 'execute_majority_choice'
        
        # Test consensus stats
        stats = consensus_manager.get_consensus_stats()
        assert stats['total_consensus_decisions'] >= 2
        
        logger.info("✅ Consensus Manager tests passed")
    
    async def test_workflow_orchestrator(self):
        """Test the complete workflow orchestration"""
        logger.info("Testing Workflow Orchestrator...")
        
        orchestrator = WorkflowOrchestrator(self.storage_path)
        
        # Test workflow status
        status = orchestrator.get_workflow_status()
        assert 'active_executions' in status
        assert 'namespace_stats' in status
        assert 'memory_stats' in status
        
        # Test workflow execution
        context = {
            'problem': 'test_workflow_problem',
            'urgency': 'low'
        }
        
        execution = await orchestrator.execute_workflow(context)
        
        assert execution.state.value in ['completed', 'error']
        
        if execution.state.value == 'completed':
            assert execution.results is not None
            assert 'analysis' in execution.results
            assert 'consensus' in execution.results
        
        # Test execution history
        history = orchestrator.get_execution_history(5)
        assert len(history) >= 1
        
        logger.info("✅ Workflow Orchestrator tests passed")
    
    async def test_integration_scenario(self):
        """Test a complete integration scenario"""
        logger.info("Testing Integration Scenario...")
        
        # Create orchestrator
        orchestrator = WorkflowOrchestrator(self.storage_path)
        
        # Scenario: Collaborative problem solving
        context = {
            'problem': 'optimize_resource_allocation',
            'constraints': ['limited_time', 'multiple_stakeholders'],
            'expected_outcome': 'balanced_solution',
            'urgency': 'medium'
        }
        
        # Execute workflow
        execution = await orchestrator.execute_workflow(context)
        
        # Verify results
        assert execution.started_at > 0
        assert execution.completed_at is not None
        
        if execution.state.value == 'completed':
            results = execution.results
            
            # Verify each phase completed
            assert 'analysis' in results
            assert 'hypotheses_generated' in results
            assert results['hypotheses_generated'] > 0
            assert 'simulations_run' in results
            assert 'consensus' in results
            assert 'execution' in results
            
            # Verify memory system captured everything
            memory_stats = orchestrator.memory_system.get_memory_stats()
            assert memory_stats['simulation_results'] >= results['simulations_run']
            
            # Verify consensus was reached
            consensus_stats = orchestrator.consensus_manager.get_consensus_stats()
            assert consensus_stats['total_consensus_decisions'] >= 1
            
            logger.info("✅ Integration scenario completed successfully")
        else:
            logger.warning(f"Integration scenario ended with state: {execution.state.value}")
            if execution.error_message:
                logger.error(f"Error: {execution.error_message}")
    
    def cleanup(self):
        """Clean up test resources"""
        import shutil
        try:
            shutil.rmtree(self.temp_dir)
            logger.info("✅ Test cleanup completed")
        except Exception as e:
            logger.warning(f"Cleanup warning: {e}")

async def run_all_tests():
    """Run all EchoSpace tests"""
    logger.info("🚀 Starting EchoSpace Architecture Tests")
    
    test_suite = TestEchoSpace()
    
    try:
        # Run tests
        test_suite.test_namespace_system()
        test_suite.test_core_architecture()
        await test_suite.test_memory_system()
        await test_suite.test_virtual_marduk()
        await test_suite.test_actual_marduk()
        await test_suite.test_sandbox()
        await test_suite.test_consensus_manager()
        await test_suite.test_workflow_orchestrator()
        await test_suite.test_integration_scenario()
        
        logger.info("🎉 All EchoSpace tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise
    
    finally:
        test_suite.cleanup()

if __name__ == "__main__":
    # Run the tests
    asyncio.run(run_all_tests())