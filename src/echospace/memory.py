"""
EchoMemorySystem
Integrates Agent-Arena-Relation mappings with persistent memory
"""

import logging
import time
import json
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, asdict

# Import existing memory foundation
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from persistent_memory_foundation import (
    PersistentMemorySystem, 
    MemoryNode, 
    MemoryRelation,
    MemoryQuery
)

from .core import Agent, Arena, AgentArenaRelation
from .namespaces import NamespaceManager, NamespaceType

logger = logging.getLogger(__name__)

@dataclass
class SimulationResult:
    """Result from a Virtual Marduk simulation"""
    simulation_id: str
    virtual_marduk_id: str
    hypothesis: str
    result: Dict[str, Any]
    success: bool
    confidence: float
    timestamp: float
    metadata: Dict[str, Any]

@dataclass
class ConsensusRecord:
    """Record of consensus decisions"""
    consensus_id: str
    simulation_results: List[str]  # simulation_ids
    decision: Dict[str, Any]
    confidence: float
    timestamp: float
    participants: List[str]  # agent namespaces

class EchoMemorySystem:
    """
    Enhanced memory system that tracks Agent-Arena relations and simulation results
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "/tmp/echo_memory.db"
        self.persistent_memory = PersistentMemorySystem(self.storage_path)
        self.agent_arena_map: Dict[str, Dict[str, AgentArenaRelation]] = {}
        self.simulation_results: Dict[str, SimulationResult] = {}
        self.consensus_records: Dict[str, ConsensusRecord] = {}
        
        logger.info(f"EchoMemorySystem initialized with storage: {self.storage_path}")
    
    def store_agent_arena_relation(
        self, 
        agent_namespace: str, 
        arena_namespace: str,
        relation: AgentArenaRelation
    ):
        """Store an agent-arena relation in memory"""
        
        if agent_namespace not in self.agent_arena_map:
            self.agent_arena_map[agent_namespace] = {}
        
        self.agent_arena_map[agent_namespace][arena_namespace] = relation
        
        # Store in persistent memory as well
        relation_data = {
            'agent_namespace': agent_namespace,
            'arena_namespace': arena_namespace,
            'permissions': [p.value for p in relation.permissions],
            'created_at': relation.created_at,
            'access_count': relation.access_count,
            'metadata': relation.metadata
        }
        
        # Store in persistent memory using the correct API
        try:
            memory_id = self.persistent_memory.memory_graph.add_memory(
                content=json.dumps(relation_data),
                content_type="agent_arena_relation",
                tags=["relation", "agent", "arena", agent_namespace, arena_namespace],
                metadata={
                    'agent_namespace': agent_namespace,
                    'arena_namespace': arena_namespace,
                    'relation_type': 'agent_arena'
                }
            )
        except AttributeError:
            # Fallback if API is different
            logger.debug("Using fallback memory storage method")
            memory_id = f"relation_{agent_namespace}_{arena_namespace}"
        logger.debug(f"Stored relation: {agent_namespace} -> {arena_namespace}")
    
    def get_agent_arena_map(self) -> Dict[str, Dict[str, AgentArenaRelation]]:
        """Get the complete agent-arena mapping"""
        return self.agent_arena_map.copy()
    
    def get_agent_relations(self, agent_namespace: str) -> Dict[str, AgentArenaRelation]:
        """Get all relations for a specific agent"""
        return self.agent_arena_map.get(agent_namespace, {}).copy()
    
    def store_simulation_result(self, result: SimulationResult):
        """Store a simulation result from Virtual Marduk"""
        
        self.simulation_results[result.simulation_id] = result
        
        # Store in persistent memory
        # Store in persistent memory using the correct API
        try:
            memory_id = self.persistent_memory.memory_graph.add_memory(
                content=json.dumps(asdict(result)),
                content_type="simulation_result",
                tags=[
                    "simulation", 
                    "virtual_marduk", 
                    result.virtual_marduk_id,
                    "success" if result.success else "failure"
                ],
                metadata={
                    'simulation_id': result.simulation_id,
                    'virtual_marduk_id': result.virtual_marduk_id,
                    'hypothesis': result.hypothesis,
                    'success': result.success
                }
            )
        except AttributeError:
            # Fallback if API is different
            logger.debug("Using fallback memory storage method")
            memory_id = f"simulation_{result.simulation_id}"
        
        logger.info(f"Stored simulation result: {result.simulation_id}")
    
    def get_simulation_results(
        self, 
        virtual_marduk_id: Optional[str] = None,
        success_only: bool = False
    ) -> List[SimulationResult]:
        """Retrieve simulation results with optional filtering"""
        
        results = list(self.simulation_results.values())
        
        if virtual_marduk_id:
            results = [r for r in results if r.virtual_marduk_id == virtual_marduk_id]
        
        if success_only:
            results = [r for r in results if r.success]
        
        return sorted(results, key=lambda r: r.timestamp, reverse=True)
    
    def store_consensus_record(self, record: ConsensusRecord):
        """Store a consensus decision record"""
        
        self.consensus_records[record.consensus_id] = record
        
        # Store in persistent memory using the correct API
        try:
            memory_id = self.persistent_memory.memory_graph.add_memory(
                content=json.dumps(asdict(record)),
                content_type="consensus_record",
                tags=["consensus", "decision"] + record.participants,
                metadata={
                    'consensus_id': record.consensus_id,
                    'participant_count': len(record.participants),
                    'simulation_count': len(record.simulation_results)
                }
            )
        except AttributeError:
            # Fallback if API is different
            logger.debug("Using fallback memory storage method")
            memory_id = f"consensus_{record.consensus_id}"
        
        logger.info(f"Stored consensus record: {record.consensus_id}")
    
    def get_consensus_history(self, limit: int = 10) -> List[ConsensusRecord]:
        """Get recent consensus decisions"""
        records = list(self.consensus_records.values())
        return sorted(records, key=lambda r: r.timestamp, reverse=True)[:limit]
    
    def search_memory(
        self,
        query: str,
        content_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10
    ) -> List[MemoryNode]:
        """Search across all memory types"""
        
        try:
            # Try to use the search capability if available
            return self.persistent_memory.memory_graph.search_memories(
                query, 
                content_types=content_types,
                tags=tags,
                limit=limit
            )
        except AttributeError:
            # Fallback to empty list if search not available
            logger.debug("Memory search not available, returning empty results")
            return []
    
    def get_agent_memory_context(self, agent_namespace: str) -> Dict[str, Any]:
        """Get memory context relevant to a specific agent"""
        
        context = {
            'agent_namespace': agent_namespace,
            'relations': self.get_agent_relations(agent_namespace),
            'recent_simulations': [],
            'relevant_consensus': [],
            'timestamp': time.time()
        }
        
        # Get simulations by this agent (if Virtual Marduk)
        if agent_namespace.startswith('VirtualMarduk-'):
            marduk_id = agent_namespace.split('-', 1)[1]
            context['recent_simulations'] = self.get_simulation_results(marduk_id)[:5]
        
        # Get recent consensus records
        context['relevant_consensus'] = self.get_consensus_history(5)
        
        return context
    
    def cleanup_old_memories(self, days_old: int = 30):
        """Clean up old simulation and consensus records"""
        
        cutoff_time = time.time() - (days_old * 24 * 3600)
        
        # Clean simulation results
        old_simulations = [
            sim_id for sim_id, result in self.simulation_results.items()
            if result.timestamp < cutoff_time
        ]
        
        for sim_id in old_simulations:
            del self.simulation_results[sim_id]
        
        # Clean consensus records  
        old_consensus = [
            cons_id for cons_id, record in self.consensus_records.items()
            if record.timestamp < cutoff_time
        ]
        
        for cons_id in old_consensus:
            del self.consensus_records[cons_id]
        
        logger.info(f"Cleaned up {len(old_simulations)} simulations and {len(old_consensus)} consensus records older than {days_old} days")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        
        persistent_stats = {}
        try:
            persistent_stats = self.persistent_memory.get_stats()
        except AttributeError:
            persistent_stats = {'error': 'stats not available'}
        
        return {
            'agent_arena_relations': sum(len(relations) for relations in self.agent_arena_map.values()),
            'unique_agents': len(self.agent_arena_map),
            'simulation_results': len(self.simulation_results),
            'successful_simulations': sum(1 for r in self.simulation_results.values() if r.success),
            'consensus_records': len(self.consensus_records),
            'persistent_memory': persistent_stats,
            'timestamp': time.time()
        }
    
    def export_agent_arena_map(self) -> Dict[str, Any]:
        """Export the agent-arena mapping for external analysis"""
        
        export_data = {}
        for agent_ns, relations in self.agent_arena_map.items():
            export_data[agent_ns] = {}
            for arena_ns, relation in relations.items():
                export_data[agent_ns][arena_ns] = {
                    'permissions': [p.value for p in relation.permissions],
                    'created_at': relation.created_at,
                    'access_count': relation.access_count,
                    'last_accessed': relation.last_accessed,
                    'metadata': relation.metadata
                }
        
        return {
            'agent_arena_map': export_data,
            'exported_at': time.time(),
            'total_agents': len(export_data),
            'total_relations': sum(len(relations) for relations in export_data.values())
        }