"""
Core Agent-Arena-Relation Architecture
Foundational classes implementing the recursive blueprint
"""

import uuid
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Set, Callable
from enum import Enum

logger = logging.getLogger(__name__)

class PermissionLevel(Enum):
    """Permission levels for agent-arena interactions"""
    NONE = "none"
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    FULL = "full"

@dataclass
class AgentArenaRelation:
    """
    The fundamental relation between an Agent and an Arena.
    This is the core abstraction that enables fractal scaling.
    """
    agent_namespace: str
    arena_namespace: str
    permissions: Set[PermissionLevel]
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def has_permission(self, permission: PermissionLevel) -> bool:
        """Check if relation allows specific permission"""
        return permission in self.permissions or PermissionLevel.FULL in self.permissions
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = time.time()
        self.access_count += 1

class Agent(ABC):
    """
    Abstract base class for all agents in the EchoSpace.
    Agents exist within namespaces and interact with arenas through relations.
    """
    
    def __init__(self, namespace: str, agent_id: Optional[str] = None):
        self.namespace = namespace
        self.agent_id = agent_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.active = True
        self.relations: Dict[str, AgentArenaRelation] = {}
        self.memory_store: Dict[str, Any] = {}
        self._event_handlers: Dict[str, List[Callable]] = {}
        
        logger.info(f"Agent created: {self.namespace}/{self.agent_id}")
    
    @abstractmethod
    async def act(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Primary action method for the agent.
        Must be implemented by all concrete agents.
        """
        pass
    
    def add_relation(self, relation: AgentArenaRelation):
        """Add an agent-arena relation"""
        self.relations[relation.arena_namespace] = relation
        logger.debug(f"Relation added: {self.namespace} -> {relation.arena_namespace}")
    
    def get_relation(self, arena_namespace: str) -> Optional[AgentArenaRelation]:
        """Get relation to specific arena"""
        return self.relations.get(arena_namespace)
    
    def has_arena_permission(self, arena_namespace: str, permission: PermissionLevel) -> bool:
        """Check if agent has permission for arena"""
        relation = self.get_relation(arena_namespace)
        return relation and relation.has_permission(permission)
    
    def store_memory(self, key: str, value: Any):
        """Store memory in agent's local store"""
        self.memory_store[key] = {
            'value': value,
            'timestamp': time.time(),
            'namespace': self.namespace
        }
    
    def retrieve_memory(self, key: str) -> Any:
        """Retrieve memory from agent's store"""
        memory = self.memory_store.get(key)
        return memory['value'] if memory else None
    
    def subscribe_to_event(self, event_type: str, handler: Callable):
        """Subscribe to events"""
        if event_type not in self._event_handlers:
            self._event_handlers[event_type] = []
        self._event_handlers[event_type].append(handler)
    
    def emit_event(self, event_type: str, data: Any):
        """Emit an event to registered handlers"""
        handlers = self._event_handlers.get(event_type, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Event handler error: {e}")

class Arena(ABC):
    """
    Abstract base class for all arenas in the EchoSpace.
    Arenas provide environments where agents can act.
    """
    
    def __init__(self, namespace: str, arena_id: Optional[str] = None):
        self.namespace = namespace
        self.arena_id = arena_id or str(uuid.uuid4())
        self.created_at = time.time()
        self.active = True
        self.resources: Dict[str, Any] = {}
        self.state: Dict[str, Any] = {}
        self._authorized_agents: Set[str] = set()
        
        logger.info(f"Arena created: {self.namespace}/{self.arena_id}")
    
    @abstractmethod
    async def process_action(self, agent: Agent, action: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an action from an agent.
        Must be implemented by all concrete arenas.
        """
        pass
    
    def add_resource(self, name: str, resource: Any):
        """Add a resource to the arena"""
        self.resources[name] = resource
        logger.debug(f"Resource added to {self.namespace}: {name}")
    
    def get_resource(self, name: str) -> Any:
        """Get a resource from the arena"""
        return self.resources.get(name)
    
    def update_state(self, updates: Dict[str, Any]):
        """Update arena state"""
        self.state.update(updates)
        self.state['last_updated'] = time.time()
    
    def authorize_agent(self, agent_namespace: str):
        """Authorize an agent to access this arena"""
        self._authorized_agents.add(agent_namespace)
        logger.debug(f"Agent authorized: {agent_namespace} -> {self.namespace}")
    
    def is_agent_authorized(self, agent_namespace: str) -> bool:
        """Check if agent is authorized"""
        return agent_namespace in self._authorized_agents
    
    def get_state_snapshot(self) -> Dict[str, Any]:
        """Get a snapshot of current arena state"""
        return {
            'namespace': self.namespace,
            'arena_id': self.arena_id,
            'state': self.state.copy(),
            'resource_count': len(self.resources),
            'authorized_agents': len(self._authorized_agents),
            'timestamp': time.time()
        }

class RelationManager:
    """
    Manages Agent-Arena relations and enforces permissions
    """
    
    def __init__(self):
        self.relations: Dict[str, Dict[str, AgentArenaRelation]] = {}
        self._lock = None  # Will be initialized as needed
    
    def create_relation(
        self, 
        agent_namespace: str, 
        arena_namespace: str,
        permissions: Set[PermissionLevel],
        metadata: Optional[Dict[str, Any]] = None
    ) -> AgentArenaRelation:
        """Create a new agent-arena relation"""
        
        relation = AgentArenaRelation(
            agent_namespace=agent_namespace,
            arena_namespace=arena_namespace,
            permissions=permissions,
            metadata=metadata or {}
        )
        
        if agent_namespace not in self.relations:
            self.relations[agent_namespace] = {}
        
        self.relations[agent_namespace][arena_namespace] = relation
        
        logger.info(f"Relation created: {agent_namespace} -> {arena_namespace} with {permissions}")
        return relation
    
    def get_relation(self, agent_namespace: str, arena_namespace: str) -> Optional[AgentArenaRelation]:
        """Get specific relation"""
        return self.relations.get(agent_namespace, {}).get(arena_namespace)
    
    def get_agent_relations(self, agent_namespace: str) -> Dict[str, AgentArenaRelation]:
        """Get all relations for an agent"""
        return self.relations.get(agent_namespace, {}).copy()
    
    def check_permission(
        self, 
        agent_namespace: str, 
        arena_namespace: str, 
        permission: PermissionLevel
    ) -> bool:
        """Check if agent has permission for arena"""
        relation = self.get_relation(agent_namespace, arena_namespace)
        if relation:
            relation.update_access()
            return relation.has_permission(permission)
        return False
    
    def revoke_relation(self, agent_namespace: str, arena_namespace: str) -> bool:
        """Revoke an agent-arena relation"""
        if (agent_namespace in self.relations and 
            arena_namespace in self.relations[agent_namespace]):
            del self.relations[agent_namespace][arena_namespace]
            logger.info(f"Relation revoked: {agent_namespace} -> {arena_namespace}")
            return True
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get relation statistics"""
        total_relations = sum(len(arena_relations) for arena_relations in self.relations.values())
        agent_count = len(self.relations)
        
        permission_stats = {}
        for agent_relations in self.relations.values():
            for relation in agent_relations.values():
                for perm in relation.permissions:
                    permission_stats[perm.value] = permission_stats.get(perm.value, 0) + 1
        
        return {
            'total_relations': total_relations,
            'agent_count': agent_count,
            'permission_distribution': permission_stats,
            'timestamp': time.time()
        }