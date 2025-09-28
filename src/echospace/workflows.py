"""
Workflow Orchestrator
Coordinates the entire Agent-Arena-Relation workflow cycle
"""

import asyncio
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .core import Agent, Arena, PermissionLevel, RelationManager
from .namespaces import NamespaceManager, NamespaceType
from .memory import EchoMemorySystem, SimulationResult, ConsensusRecord
from .marduk import ActualMarduk, VirtualMarduk, Hypothesis
from .sandbox import MardukSandbox
from .consensus import ConsensusManager, ConsensusStrategy

logger = logging.getLogger(__name__)

class WorkflowState(Enum):
    """Workflow execution states"""
    IDLE = "idle"
    ANALYZING = "analyzing"
    GENERATING_HYPOTHESES = "generating_hypotheses"
    RUNNING_SIMULATIONS = "running_simulations"
    REACHING_CONSENSUS = "reaching_consensus"
    EXECUTING = "executing"
    STORING_FEEDBACK = "storing_feedback"
    ERROR = "error"
    COMPLETED = "completed"

@dataclass
class WorkflowExecution:
    """Represents a workflow execution instance"""
    execution_id: str
    state: WorkflowState
    context: Dict[str, Any]
    started_at: float
    completed_at: Optional[float] = None
    error_message: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class WorkflowOrchestrator:
    """
    Orchestrates the complete Agent-Arena-Relation workflow.
    Manages the cycle: Analysis -> Hypotheses -> Simulations -> Consensus -> Execution -> Feedback
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        # Initialize core components
        self.namespace_manager = NamespaceManager()
        self.relation_manager = RelationManager()
        self.memory_system = EchoMemorySystem(storage_path)
        self.consensus_manager = ConsensusManager()
        
        # Initialize agents and arenas
        self.actual_marduk = ActualMarduk(self.memory_system)
        self.sandbox = MardukSandbox()
        
        # Execution tracking
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.execution_history: List[WorkflowExecution] = []
        
        # Configuration
        self.max_virtual_marduks = 5
        self.simulation_timeout = 30.0  # seconds
        
        self._initialize_relations()
        logger.info("WorkflowOrchestrator initialized")
    
    def _initialize_relations(self):
        """Initialize default agent-arena relations"""
        
        # ActualMarduk -> Marduk-Space (full permissions)
        self.relation_manager.create_relation(
            agent_namespace="ActualMarduk",
            arena_namespace="Marduk-Space",
            permissions={PermissionLevel.FULL}
        )
        
        # ActualMarduk -> Marduk-Memory (read/write permissions)
        self.relation_manager.create_relation(
            agent_namespace="ActualMarduk",
            arena_namespace="Marduk-Memory", 
            permissions={PermissionLevel.READ, PermissionLevel.WRITE}
        )
        
        # Store relations in memory
        for agent_ns, arena_relations in self.relation_manager.relations.items():
            for arena_ns, relation in arena_relations.items():
                self.memory_system.store_agent_arena_relation(agent_ns, arena_ns, relation)
        
        logger.info("Default agent-arena relations initialized")
    
    async def execute_workflow(
        self, 
        context: Dict[str, Any],
        execution_id: Optional[str] = None
    ) -> WorkflowExecution:
        """
        Execute the complete Agent-Arena-Relation workflow
        """
        
        execution_id = execution_id or str(uuid.uuid4())
        
        execution = WorkflowExecution(
            execution_id=execution_id,
            state=WorkflowState.IDLE,
            context=context.copy(),
            started_at=time.time()
        )
        
        self.active_executions[execution_id] = execution
        
        try:
            logger.info(f"Starting workflow execution {execution_id}")
            
            # Step 1: Analysis
            execution.state = WorkflowState.ANALYZING
            analysis_result = await self._execute_analysis_phase(execution)
            
            # Step 2: Generate hypotheses
            execution.state = WorkflowState.GENERATING_HYPOTHESES
            hypotheses = await self._execute_hypothesis_generation(execution, analysis_result)
            
            # Step 3: Run simulations
            execution.state = WorkflowState.RUNNING_SIMULATIONS
            simulation_results = await self._execute_simulation_phase(execution, hypotheses)
            
            # Step 4: Reach consensus
            execution.state = WorkflowState.REACHING_CONSENSUS
            consensus = await self._execute_consensus_phase(execution, simulation_results)
            
            # Step 5: Execute chosen strategy
            execution.state = WorkflowState.EXECUTING
            execution_result = await self._execute_action_phase(execution, consensus)
            
            # Step 6: Store feedback
            execution.state = WorkflowState.STORING_FEEDBACK
            await self._execute_feedback_phase(execution, execution_result)
            
            # Complete workflow
            execution.state = WorkflowState.COMPLETED
            execution.completed_at = time.time()
            execution.results = {
                'analysis': analysis_result,
                'hypotheses_generated': len(hypotheses),
                'simulations_run': len(simulation_results),
                'consensus': consensus,
                'execution': execution_result
            }
            
            logger.info(f"Workflow execution {execution_id} completed successfully")
            
        except Exception as e:
            execution.state = WorkflowState.ERROR
            execution.completed_at = time.time()
            execution.error_message = str(e)
            logger.error(f"Workflow execution {execution_id} failed: {e}")
            
        finally:
            # Move to history
            if execution_id in self.active_executions:
                del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
            # Limit history size
            if len(self.execution_history) > 100:
                self.execution_history = self.execution_history[-100:]
        
        return execution
    
    async def _execute_analysis_phase(
        self, 
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute the situation analysis phase"""
        
        logger.debug(f"Executing analysis phase for {execution.execution_id}")
        
        # Get memory context for ActualMarduk
        memory_context = self.memory_system.get_agent_memory_context("ActualMarduk")
        
        # Get recent consensus history
        consensus_history = self.consensus_manager.get_consensus_history(5)
        
        # Analyze current namespace state
        namespace_stats = self.namespace_manager.get_stats()
        
        # Analyze sandbox state
        sandbox_stats = self.sandbox.get_sandbox_stats()
        
        analysis = {
            'execution_id': execution.execution_id,
            'context': execution.context,
            'memory_context': memory_context,
            'consensus_history': len(consensus_history),
            'namespace_stats': namespace_stats,
            'sandbox_stats': sandbox_stats,
            'analysis_timestamp': time.time(),
            'virtual_marduks_available': len([
                ns for ns in self.namespace_manager.list_namespaces(NamespaceType.AGENT)
                if ns.name.startswith('VirtualMarduk-')
            ])
        }
        
        return analysis
    
    async def _execute_hypothesis_generation(
        self, 
        execution: WorkflowExecution,
        analysis: Dict[str, Any]
    ) -> List[Hypothesis]:
        """Generate hypotheses based on analysis"""
        
        logger.debug(f"Generating hypotheses for {execution.execution_id}")
        
        hypotheses = []
        context = execution.context
        
        # Generate different types of hypotheses based on context
        if 'problem' in context:
            problem = context['problem']
            
            # Direct solution hypothesis
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                description=f"Direct solution to: {problem}",
                parameters={
                    'approach': 'direct',
                    'problem': problem,
                    'urgency': context.get('urgency', 'medium')
                },
                expected_outcome="Immediate problem resolution",
                confidence=0.6
            ))
            
            # Collaborative solution hypothesis
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                description=f"Collaborative solution to: {problem}",
                parameters={
                    'approach': 'collaborative',
                    'problem': problem,
                    'collaboration_level': 'high'
                },
                expected_outcome="Shared solution with multiple perspectives",
                confidence=0.7
            ))
            
            # Gradual solution hypothesis
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                description=f"Gradual solution to: {problem}",
                parameters={
                    'approach': 'gradual',
                    'problem': problem,
                    'steps': context.get('max_steps', 3)
                },
                expected_outcome="Stable long-term solution",
                confidence=0.75
            ))
            
        elif 'exploration' in context:
            # Exploration hypotheses
            exploration_target = context['exploration']
            
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                description=f"Systematic exploration of: {exploration_target}",
                parameters={
                    'action': 'explore',
                    'target': exploration_target,
                    'depth': 'comprehensive'
                },
                expected_outcome="Detailed understanding of target area",
                confidence=0.65
            ))
        
        else:
            # Default exploration hypothesis
            hypotheses.append(Hypothesis(
                id=str(uuid.uuid4()),
                description="General environment analysis",
                parameters={
                    'action': 'analyze',
                    'scope': 'general'
                },
                expected_outcome="Better situational awareness",
                confidence=0.5
            ))
        
        logger.info(f"Generated {len(hypotheses)} hypotheses for execution {execution.execution_id}")
        return hypotheses
    
    async def _execute_simulation_phase(
        self, 
        execution: WorkflowExecution,
        hypotheses: List[Hypothesis]
    ) -> List[SimulationResult]:
        """Execute simulations using Virtual Marduks"""
        
        logger.debug(f"Running simulations for {execution.execution_id}")
        
        # Ensure we have enough Virtual Marduks
        virtual_marduks = await self._ensure_virtual_marduks(len(hypotheses))
        
        # Create simulation tasks
        simulation_tasks = []
        for i, hypothesis in enumerate(hypotheses):
            virtual_marduk = virtual_marduks[i % len(virtual_marduks)]
            
            # Create sandbox session for this simulation
            session_id = f"sim_{execution.execution_id}_{i}"
            
            task = self._run_simulation_with_timeout(
                virtual_marduk, 
                hypothesis, 
                session_id
            )
            simulation_tasks.append(task)
        
        # Run simulations in parallel with timeout
        try:
            simulation_results = await asyncio.wait_for(
                asyncio.gather(*simulation_tasks, return_exceptions=True),
                timeout=self.simulation_timeout
            )
        except asyncio.TimeoutError:
            logger.warning(f"Simulations timed out for execution {execution.execution_id}")
            simulation_results = []
        
        # Process results
        valid_results = []
        for result in simulation_results:
            if isinstance(result, SimulationResult):
                valid_results.append(result)
                # Store in memory
                self.memory_system.store_simulation_result(result)
            else:
                logger.error(f"Simulation failed: {result}")
        
        logger.info(f"Completed {len(valid_results)} simulations for execution {execution.execution_id}")
        return valid_results
    
    async def _run_simulation_with_timeout(
        self,
        virtual_marduk: VirtualMarduk,
        hypothesis: Hypothesis,
        session_id: str
    ) -> SimulationResult:
        """Run a single simulation with timeout"""
        
        # Create relation for Virtual Marduk to sandbox
        if not self.relation_manager.check_permission(
            virtual_marduk.namespace, 
            "Marduk-Sandbox", 
            PermissionLevel.EXECUTE
        ):
            self.relation_manager.create_relation(
                agent_namespace=virtual_marduk.namespace,
                arena_namespace="Marduk-Sandbox",
                permissions={PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE}
            )
        
        # Run simulation
        simulation_start = time.time()
        
        try:
            # Prepare simulation context
            context = {
                'hypothesis': hypothesis,
                'session_id': session_id,
                'execution_id': session_id.split('_')[1]  # Extract execution ID
            }
            
            # Run the simulation
            result_data = await virtual_marduk.simulate_hypothesis(hypothesis)
            
            return SimulationResult(
                simulation_id=session_id,
                virtual_marduk_id=virtual_marduk.agent_id,
                hypothesis=hypothesis.description,
                result=result_data,
                success=result_data.get('success', False),
                confidence=result_data.get('confidence', hypothesis.confidence),
                timestamp=time.time(),
                metadata={
                    'hypothesis_id': hypothesis.id,
                    'simulation_time': time.time() - simulation_start
                }
            )
            
        except Exception as e:
            logger.error(f"Simulation failed for {virtual_marduk.agent_id}: {e}")
            return SimulationResult(
                simulation_id=session_id,
                virtual_marduk_id=virtual_marduk.agent_id,
                hypothesis=hypothesis.description,
                result={'success': False, 'error': str(e)},
                success=False,
                confidence=0.0,
                timestamp=time.time(),
                metadata={'error': str(e)}
            )
    
    async def _ensure_virtual_marduks(self, count: int) -> List[VirtualMarduk]:
        """Ensure we have the required number of Virtual Marduks"""
        
        # Get existing Virtual Marduks from namespace
        virtual_marduk_namespaces = self.namespace_manager.get_virtual_marduk_namespaces()
        
        virtual_marduks = []
        
        # Use existing ones first
        for namespace in virtual_marduk_namespaces:
            marduk_id = namespace.metadata.get('marduk_id')
            if marduk_id:
                virtual_marduk = VirtualMarduk(self.memory_system, marduk_id)
                virtual_marduks.append(virtual_marduk)
        
        # Create additional ones if needed
        needed_count = min(count, self.max_virtual_marduks) - len(virtual_marduks)
        for _ in range(needed_count):
            # Create namespace
            marduk_id = str(uuid.uuid4())
            self.namespace_manager.create_virtual_marduk_namespace(marduk_id)
            
            # Create agent
            virtual_marduk = VirtualMarduk(self.memory_system, marduk_id)
            virtual_marduks.append(virtual_marduk)
            
            logger.info(f"Created new Virtual Marduk: {virtual_marduk.namespace}")
        
        return virtual_marduks[:min(count, self.max_virtual_marduks)]
    
    async def _execute_consensus_phase(
        self, 
        execution: WorkflowExecution,
        simulation_results: List[SimulationResult]
    ) -> ConsensusRecord:
        """Reach consensus on simulation results"""
        
        logger.debug(f"Reaching consensus for execution {execution.execution_id}")
        
        if not simulation_results:
            # Create a "no action" consensus
            return ConsensusRecord(
                consensus_id=f"consensus_{execution.execution_id}",
                simulation_results=[],
                decision={'strategy': 'no_action', 'reason': 'no_simulations'},
                confidence=0.0,
                timestamp=time.time(),
                participants=[]
            )
        
        # Use consensus manager to reach consensus
        consensus = await self.consensus_manager.reach_consensus(
            simulation_results,
            strategy=ConsensusStrategy.RISK_ADJUSTED,  # Use risk-adjusted by default
            metadata={'execution_id': execution.execution_id}
        )
        
        # Store consensus in memory
        self.memory_system.store_consensus_record(consensus)
        
        return consensus
    
    async def _execute_action_phase(
        self, 
        execution: WorkflowExecution,
        consensus: ConsensusRecord
    ) -> Dict[str, Any]:
        """Execute the chosen action based on consensus"""
        
        logger.debug(f"Executing action for execution {execution.execution_id}")
        
        # Use ActualMarduk to execute the strategy
        action_context = {
            'consensus': consensus.decision,
            'execution_id': execution.execution_id,
            'original_context': execution.context
        }
        
        # Execute through ActualMarduk
        result = await self.actual_marduk.act(action_context)
        
        return result
    
    async def _execute_feedback_phase(
        self, 
        execution: WorkflowExecution,
        execution_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Store feedback and learning from execution"""
        
        logger.debug(f"Storing feedback for execution {execution.execution_id}")
        
        # Store execution feedback in ActualMarduk's memory
        feedback = {
            'execution_id': execution.execution_id,
            'original_context': execution.context,
            'execution_result': execution_result,
            'success': execution_result.get('success', False),
            'timestamp': time.time()
        }
        
        self.actual_marduk.store_memory(f'execution_feedback_{execution.execution_id}', feedback)
        
        return feedback
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow orchestrator status"""
        
        return {
            'active_executions': len(self.active_executions),
            'completed_executions': len(self.execution_history),
            'namespace_stats': self.namespace_manager.get_stats(),
            'memory_stats': self.memory_system.get_memory_stats(),
            'consensus_stats': self.consensus_manager.get_consensus_stats(),
            'sandbox_stats': self.sandbox.get_sandbox_stats(),
            'actual_marduk_status': self.actual_marduk.get_status(),
            'max_virtual_marduks': self.max_virtual_marduks,
            'simulation_timeout': self.simulation_timeout
        }
    
    def get_execution_history(self, limit: int = 10) -> List[WorkflowExecution]:
        """Get recent workflow execution history"""
        
        return sorted(
            self.execution_history,
            key=lambda e: e.started_at,
            reverse=True
        )[:limit]
    
    def get_active_executions(self) -> List[WorkflowExecution]:
        """Get currently active workflow executions"""
        
        return list(self.active_executions.values())
    
    async def cancel_execution(self, execution_id: str) -> bool:
        """Cancel an active workflow execution"""
        
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.state = WorkflowState.ERROR
            execution.error_message = "Execution cancelled by user"
            execution.completed_at = time.time()
            
            # Move to history
            del self.active_executions[execution_id]
            self.execution_history.append(execution)
            
            logger.info(f"Workflow execution {execution_id} cancelled")
            return True
        
        return False