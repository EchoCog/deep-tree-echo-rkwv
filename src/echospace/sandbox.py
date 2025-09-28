"""
Marduk Sandbox Environment
Isolated environment for Virtual Marduk simulations
"""

import asyncio
import logging
import time
import copy
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass

from .core import Arena, Agent, PermissionLevel

logger = logging.getLogger(__name__)

@dataclass
class SandboxResource:
    """A resource available in the sandbox"""
    name: str
    resource_type: str
    quantity: float
    renewable: bool = False
    constraints: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}

@dataclass
class SandboxState:
    """Snapshot of sandbox state"""
    timestamp: float
    resources: Dict[str, SandboxResource]
    active_agents: Set[str]
    actions_taken: int
    success_rate: float
    environment_health: float

class MardukSandbox(Arena):
    """
    Sandbox arena where Virtual Marduks can safely test hypotheses.
    Provides controlled environment with rollback capabilities.
    """
    
    def __init__(self):
        super().__init__("Marduk-Sandbox")
        self.base_state: Dict[str, Any] = {}
        self.state_history: List[SandboxState] = []
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.max_history_size = 100
        
        self._initialize_sandbox_resources()
        logger.info("MardukSandbox initialized")
    
    def _initialize_sandbox_resources(self):
        """Initialize default sandbox resources"""
        
        # Basic resources that mirror real-world equivalents
        base_resources = {
            'computational_power': SandboxResource(
                name='computational_power',
                resource_type='processing',
                quantity=1000.0,
                renewable=True,
                constraints={'max_per_action': 100}
            ),
            'memory_space': SandboxResource(
                name='memory_space', 
                resource_type='storage',
                quantity=10000.0,
                renewable=False,
                constraints={'allocation_unit': 1}
            ),
            'time_units': SandboxResource(
                name='time_units',
                resource_type='temporal',
                quantity=10000.0,
                renewable=True,
                constraints={'flow_rate': 1.0}
            ),
            'information': SandboxResource(
                name='information',
                resource_type='knowledge',
                quantity=500.0,
                renewable=True,
                constraints={'discovery_rate': 0.1}
            ),
            'collaboration_tokens': SandboxResource(
                name='collaboration_tokens',
                resource_type='social',
                quantity=50.0,
                renewable=True,
                constraints={'interaction_cost': 1}
            )
        }
        
        for name, resource in base_resources.items():
            self.add_resource(name, resource)
        
        # Initialize base state
        self.base_state = {
            'environment_health': 100.0,
            'stability_index': 1.0,
            'complexity_level': 1,
            'risk_factors': [],
            'available_tools': ['analyzer', 'predictor', 'optimizer'],
            'success_metrics': {
                'efficiency': 0.0,
                'effectiveness': 0.0,
                'sustainability': 0.0
            }
        }
        
        self.update_state(self.base_state)
        self._save_state_snapshot()
    
    async def process_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an action from a Virtual Marduk in the sandbox.
        All actions are isolated and reversible.
        """
        
        session_id = action.get('session_id', agent.agent_id)
        logger.debug(f"Processing sandbox action for session {session_id}: {action.get('type', 'unknown')}")
        
        # Create or get session state
        if session_id not in self.active_sessions:
            self.active_sessions[session_id] = self._create_session_state()
        
        session_state = self.active_sessions[session_id]
        
        try:
            # Process the specific action
            result = await self._execute_sandbox_action(session_state, action)
            
            # Update session metrics
            session_state['actions_taken'] += 1
            session_state['last_action_time'] = time.time()
            
            # Calculate success rate
            if result.get('success', False):
                session_state['successful_actions'] += 1
            
            session_state['success_rate'] = (
                session_state['successful_actions'] / session_state['actions_taken']
                if session_state['actions_taken'] > 0 else 0.0
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Sandbox action failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'session_id': session_id,
                'timestamp': time.time()
            }
    
    def _create_session_state(self) -> Dict[str, Any]:
        """Create a new session state"""
        return {
            'created_at': time.time(),
            'resources': copy.deepcopy({name: res for name, res in self.resources.items()}),
            'environment': copy.deepcopy(self.base_state),
            'actions_taken': 0,
            'successful_actions': 0,
            'success_rate': 0.0,
            'modifications': [],
            'last_action_time': time.time()
        }
    
    async def _execute_sandbox_action(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a specific sandbox action"""
        
        action_type = action.get('type', 'unknown')
        
        if action_type == 'resource_allocation':
            return await self._handle_resource_allocation(session_state, action)
        elif action_type == 'environment_modification':
            return await self._handle_environment_modification(session_state, action)
        elif action_type == 'simulation_run':
            return await self._handle_simulation_run(session_state, action)
        elif action_type == 'collaboration_test':
            return await self._handle_collaboration_test(session_state, action)
        elif action_type == 'prediction':
            return await self._handle_prediction(session_state, action)
        else:
            return await self._handle_generic_action(session_state, action)
    
    async def _handle_resource_allocation(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle resource allocation actions"""
        
        resource_name = action.get('resource', 'computational_power')
        requested_amount = action.get('amount', 10.0)
        
        # Check if resource exists
        if resource_name not in session_state['resources']:
            return {
                'success': False,
                'error': f'Resource not found: {resource_name}',
                'timestamp': time.time()
            }
        
        resource = session_state['resources'][resource_name]
        
        # Check if enough resource is available
        if resource.quantity < requested_amount:
            return {
                'success': False,
                'error': f'Insufficient {resource_name}: requested {requested_amount}, available {resource.quantity}',
                'timestamp': time.time()
            }
        
        # Allocate resource
        resource.quantity -= requested_amount
        
        # Simulate processing time based on resource amount
        processing_time = requested_amount / 100.0
        await asyncio.sleep(min(processing_time, 0.5))  # Cap at 0.5 seconds
        
        # Calculate effectiveness
        effectiveness = min(requested_amount / 100.0, 1.0)  # Normalized to [0,1]
        
        session_state['modifications'].append({
            'type': 'resource_allocation',
            'resource': resource_name,
            'amount': requested_amount,
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'action_type': 'resource_allocation',
            'resource_allocated': resource_name,
            'amount': requested_amount,
            'remaining': resource.quantity,
            'effectiveness': effectiveness,
            'processing_time': processing_time,
            'timestamp': time.time()
        }
    
    async def _handle_environment_modification(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle environment modification actions"""
        
        modification_type = action.get('modification', 'stability_adjustment')
        intensity = action.get('intensity', 0.1)
        
        environment = session_state['environment']
        
        # Apply modification based on type
        if modification_type == 'stability_adjustment':
            old_stability = environment['stability_index']
            environment['stability_index'] = max(0.1, min(2.0, old_stability + intensity))
            
        elif modification_type == 'complexity_increase':
            environment['complexity_level'] += max(1, int(intensity * 10))
            
        elif modification_type == 'risk_introduction':
            risk_factor = action.get('risk_factor', 'unknown_risk')
            environment['risk_factors'].append(risk_factor)
            
        elif modification_type == 'tool_addition':
            new_tool = action.get('tool', 'generic_tool')
            if new_tool not in environment['available_tools']:
                environment['available_tools'].append(new_tool)
        
        # Simulate processing time
        await asyncio.sleep(0.1)
        
        session_state['modifications'].append({
            'type': 'environment_modification',
            'modification': modification_type,
            'intensity': intensity,
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'action_type': 'environment_modification',
            'modification_applied': modification_type,
            'intensity': intensity,
            'new_state': copy.deepcopy(environment),
            'timestamp': time.time()
        }
    
    async def _handle_simulation_run(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle simulation run actions"""
        
        simulation_type = action.get('simulation', 'basic')
        duration = action.get('duration', 1.0)
        parameters = action.get('parameters', {})
        
        # Simulate processing time
        await asyncio.sleep(min(duration * 0.1, 1.0))
        
        # Generate simulation results
        if simulation_type == 'basic':
            success_probability = 0.7
            confidence = 0.6
            
        elif simulation_type == 'complex':
            success_probability = 0.5
            confidence = 0.8
            duration *= 1.5  # Complex simulations take longer
            
        elif simulation_type == 'collaborative':
            success_probability = 0.8
            confidence = 0.7
            
        else:
            success_probability = 0.6
            confidence = 0.5
        
        # Apply environment factors
        environment = session_state['environment']
        success_probability *= environment['stability_index']
        confidence *= (1.0 / max(1, environment['complexity_level'] * 0.1))
        
        success = success_probability > 0.5
        
        session_state['modifications'].append({
            'type': 'simulation_run',
            'simulation': simulation_type,
            'duration': duration,
            'timestamp': time.time()
        })
        
        return {
            'success': success,
            'action_type': 'simulation_run',
            'simulation_type': simulation_type,
            'duration': duration,
            'confidence': confidence,
            'success_probability': success_probability,
            'parameters': parameters,
            'timestamp': time.time()
        }
    
    async def _handle_collaboration_test(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle collaboration testing actions"""
        
        collaborators = action.get('collaborators', ['agent_1'])
        task = action.get('task', 'generic_collaboration')
        
        # Check collaboration tokens
        collab_resource = session_state['resources'].get('collaboration_tokens')
        tokens_needed = len(collaborators)
        
        if not collab_resource or collab_resource.quantity < tokens_needed:
            return {
                'success': False,
                'error': 'Insufficient collaboration tokens',
                'tokens_needed': tokens_needed,
                'tokens_available': collab_resource.quantity if collab_resource else 0,
                'timestamp': time.time()
            }
        
        # Use collaboration tokens
        collab_resource.quantity -= tokens_needed
        
        # Simulate collaboration
        await asyncio.sleep(0.2)
        
        # Calculate collaboration effectiveness
        base_effectiveness = 0.7
        collaborator_bonus = min(len(collaborators) * 0.1, 0.3)  # Max 30% bonus
        effectiveness = base_effectiveness + collaborator_bonus
        
        session_state['modifications'].append({
            'type': 'collaboration_test',
            'collaborators': collaborators,
            'task': task,
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'action_type': 'collaboration_test',
            'collaborators': collaborators,
            'task': task,
            'effectiveness': effectiveness,
            'tokens_used': tokens_needed,
            'collaboration_outcome': 'successful' if effectiveness > 0.6 else 'partial',
            'timestamp': time.time()
        }
    
    async def _handle_prediction(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle prediction actions"""
        
        target = action.get('target', 'future_state')
        horizon = action.get('horizon', 1.0)  # Time horizon
        
        await asyncio.sleep(0.1)
        
        # Generate prediction based on current state
        environment = session_state['environment']
        stability = environment['stability_index']
        complexity = environment['complexity_level']
        
        # Prediction accuracy decreases with horizon and complexity
        accuracy = max(0.3, min(0.95, stability / (1 + horizon * complexity * 0.1)))
        
        prediction_result = {
            'target': target,
            'predicted_outcome': 'positive' if stability > 0.7 else 'neutral',
            'confidence': accuracy,
            'factors_considered': ['stability', 'complexity', 'risk_factors'],
            'recommendation': 'proceed' if accuracy > 0.6 else 'caution'
        }
        
        session_state['modifications'].append({
            'type': 'prediction',
            'target': target,
            'horizon': horizon,
            'timestamp': time.time()
        })
        
        return {
            'success': True,
            'action_type': 'prediction',
            'prediction': prediction_result,
            'accuracy': accuracy,
            'timestamp': time.time()
        }
    
    async def _handle_generic_action(
        self, 
        session_state: Dict[str, Any], 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Handle generic/unknown actions"""
        
        await asyncio.sleep(0.05)
        
        return {
            'success': True,
            'action_type': 'generic',
            'action': action,
            'effect': 'minimal',
            'timestamp': time.time()
        }
    
    def rollback_session(self, session_id: str) -> bool:
        """Rollback a session to its initial state"""
        
        if session_id in self.active_sessions:
            self.active_sessions[session_id] = self._create_session_state()
            logger.info(f"Session {session_id} rolled back to initial state")
            return True
        
        return False
    
    def close_session(self, session_id: str) -> Dict[str, Any]:
        """Close a session and return final statistics"""
        
        if session_id not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_state = self.active_sessions[session_id]
        
        # Calculate final statistics
        stats = {
            'session_id': session_id,
            'duration': time.time() - session_state['created_at'],
            'actions_taken': session_state['actions_taken'],
            'success_rate': session_state['success_rate'],
            'modifications_count': len(session_state['modifications']),
            'final_environment_health': session_state['environment']['environment_health'],
            'resource_utilization': self._calculate_resource_utilization(session_state)
        }
        
        # Clean up session
        del self.active_sessions[session_id]
        logger.info(f"Session {session_id} closed with {stats['actions_taken']} actions")
        
        return stats
    
    def _calculate_resource_utilization(self, session_state: Dict[str, Any]) -> Dict[str, float]:
        """Calculate resource utilization percentages"""
        
        utilization = {}
        original_resources = {name: res for name, res in self.resources.items()}
        current_resources = session_state['resources']
        
        for resource_name in original_resources:
            if resource_name in current_resources:
                original_qty = original_resources[resource_name].quantity
                current_qty = current_resources[resource_name].quantity
                
                if original_qty > 0:
                    utilization[resource_name] = (original_qty - current_qty) / original_qty
                else:
                    utilization[resource_name] = 0.0
        
        return utilization
    
    def _save_state_snapshot(self):
        """Save current state as a snapshot"""
        
        snapshot = SandboxState(
            timestamp=time.time(),
            resources=copy.deepcopy(self.resources),
            active_agents=set(self.active_sessions.keys()),
            actions_taken=sum(s['actions_taken'] for s in self.active_sessions.values()),
            success_rate=sum(s['success_rate'] for s in self.active_sessions.values()) / max(1, len(self.active_sessions)),
            environment_health=self.state.get('environment_health', 100.0)
        )
        
        self.state_history.append(snapshot)
        
        # Limit history size
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
    
    def get_sandbox_stats(self) -> Dict[str, Any]:
        """Get comprehensive sandbox statistics"""
        
        active_session_count = len(self.active_sessions)
        total_actions = sum(s['actions_taken'] for s in self.active_sessions.values())
        avg_success_rate = (
            sum(s['success_rate'] for s in self.active_sessions.values()) / max(1, active_session_count)
        )
        
        return {
            'active_sessions': active_session_count,
            'total_actions_processed': total_actions,
            'average_success_rate': avg_success_rate,
            'state_snapshots': len(self.state_history),
            'resource_types': len(self.resources),
            'uptime': time.time() - self.created_at,
            'last_snapshot': self.state_history[-1] if self.state_history else None
        }