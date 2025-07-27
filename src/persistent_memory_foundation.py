"""
P0-002: Persistent Memory Architecture Foundation
This module provides the foundation for advanced persistent memory capabilities.
"""

import os
import json
import logging
import sqlite3
import time
import hashlib
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from pathlib import Path
from datetime import datetime, timedelta
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class MemoryNode:
    """A node in the knowledge graph representing a piece of information"""
    id: str
    content: str
    content_type: str  # "fact", "experience", "reasoning", "conversation"
    timestamp: float
    tags: List[str]
    confidence: float = 1.0
    source: str = "user"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MemoryRelation:
    """A relationship between memory nodes"""
    from_id: str
    to_id: str
    relation_type: str  # "relates_to", "caused_by", "follows", "contradicts"
    strength: float = 1.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

@dataclass
class MemoryQuery:
    """Query for retrieving memories"""
    query_text: str
    content_types: List[str] = None
    tags: List[str] = None
    time_range: Tuple[float, float] = None
    min_confidence: float = 0.0
    max_results: int = 10
    include_relations: bool = True

class MemoryStorage(ABC):
    """Abstract storage interface for memory persistence"""
    
    @abstractmethod
    def store_node(self, node: MemoryNode) -> bool:
        """Store a memory node"""
        pass
    
    @abstractmethod
    def store_relation(self, relation: MemoryRelation) -> bool:
        """Store a memory relation"""
        pass
    
    @abstractmethod
    def retrieve_nodes(self, query: MemoryQuery) -> List[MemoryNode]:
        """Retrieve memory nodes matching query"""
        pass
    
    @abstractmethod
    def retrieve_relations(self, node_ids: List[str]) -> List[MemoryRelation]:
        """Retrieve relations for given nodes"""
        pass
    
    @abstractmethod
    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node"""
        pass

class SQLiteMemoryStorage(MemoryStorage):
    """SQLite-based memory storage implementation"""
    
    def __init__(self, db_path: str = "/tmp/echo_memory.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_database()
        logger.info(f"SQLite memory storage initialized: {db_path}")
    
    def _init_database(self):
        """Initialize the database schema"""
        with sqlite3.connect(self.db_path) as conn:
            # Memory nodes table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_nodes (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    content_type TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    tags TEXT,  -- JSON array
                    confidence REAL DEFAULT 1.0,
                    source TEXT DEFAULT 'user',
                    metadata TEXT  -- JSON object
                )
            """)
            
            # Memory relations table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS memory_relations (
                    from_id TEXT NOT NULL,
                    to_id TEXT NOT NULL,
                    relation_type TEXT NOT NULL,
                    strength REAL DEFAULT 1.0,
                    metadata TEXT,  -- JSON object
                    PRIMARY KEY (from_id, to_id, relation_type),
                    FOREIGN KEY (from_id) REFERENCES memory_nodes(id),
                    FOREIGN KEY (to_id) REFERENCES memory_nodes(id)
                )
            """)
            
            # Create indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_timestamp ON memory_nodes(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_content_type ON memory_nodes(content_type)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_from ON memory_relations(from_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_relations_to ON memory_relations(to_id)")
            
            conn.commit()
    
    def store_node(self, node: MemoryNode) -> bool:
        """Store a memory node"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO memory_nodes 
                        (id, content, content_type, timestamp, tags, confidence, source, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        node.id,
                        node.content,
                        node.content_type,
                        node.timestamp,
                        json.dumps(node.tags),
                        node.confidence,
                        node.source,
                        json.dumps(node.metadata)
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store node {node.id}: {e}")
            return False
    
    def store_relation(self, relation: MemoryRelation) -> bool:
        """Store a memory relation"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute("""
                        INSERT OR REPLACE INTO memory_relations
                        (from_id, to_id, relation_type, strength, metadata)
                        VALUES (?, ?, ?, ?, ?)
                    """, (
                        relation.from_id,
                        relation.to_id,
                        relation.relation_type,
                        relation.strength,
                        json.dumps(relation.metadata)
                    ))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to store relation {relation.from_id}->{relation.to_id}: {e}")
            return False
    
    def retrieve_nodes(self, query: MemoryQuery) -> List[MemoryNode]:
        """Retrieve memory nodes matching query"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    sql_conditions = []
                    params = []
                    
                    # Content search (simple text matching)
                    if query.query_text:
                        sql_conditions.append("content LIKE ?")
                        params.append(f"%{query.query_text}%")
                    
                    # Content type filter
                    if query.content_types:
                        placeholders = ",".join(["?"] * len(query.content_types))
                        sql_conditions.append(f"content_type IN ({placeholders})")
                        params.extend(query.content_types)
                    
                    # Confidence filter
                    if query.min_confidence > 0:
                        sql_conditions.append("confidence >= ?")
                        params.append(query.min_confidence)
                    
                    # Time range filter
                    if query.time_range:
                        sql_conditions.append("timestamp BETWEEN ? AND ?")
                        params.extend(query.time_range)
                    
                    # Build query
                    sql = "SELECT * FROM memory_nodes"
                    if sql_conditions:
                        sql += " WHERE " + " AND ".join(sql_conditions)
                    
                    sql += " ORDER BY timestamp DESC LIMIT ?"
                    params.append(query.max_results)
                    
                    cursor = conn.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    nodes = []
                    for row in rows:
                        node = MemoryNode(
                            id=row[0],
                            content=row[1],
                            content_type=row[2],
                            timestamp=row[3],
                            tags=json.loads(row[4]) if row[4] else [],
                            confidence=row[5],
                            source=row[6],
                            metadata=json.loads(row[7]) if row[7] else {}
                        )
                        
                        # Tag filter (post-processing since SQLite doesn't have good JSON support)
                        if query.tags:
                            if not any(tag in node.tags for tag in query.tags):
                                continue
                        
                        nodes.append(node)
                    
                    return nodes[:query.max_results]
                    
        except Exception as e:
            logger.error(f"Failed to retrieve nodes: {e}")
            return []
    
    def retrieve_relations(self, node_ids: List[str]) -> List[MemoryRelation]:
        """Retrieve relations for given nodes"""
        if not node_ids:
            return []
        
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    placeholders = ",".join(["?"] * len(node_ids))
                    sql = f"""
                        SELECT * FROM memory_relations 
                        WHERE from_id IN ({placeholders}) OR to_id IN ({placeholders})
                    """
                    params = node_ids + node_ids
                    
                    cursor = conn.execute(sql, params)
                    rows = cursor.fetchall()
                    
                    relations = []
                    for row in rows:
                        relation = MemoryRelation(
                            from_id=row[0],
                            to_id=row[1],
                            relation_type=row[2],
                            strength=row[3],
                            metadata=json.loads(row[4]) if row[4] else {}
                        )
                        relations.append(relation)
                    
                    return relations
                    
        except Exception as e:
            logger.error(f"Failed to retrieve relations: {e}")
            return []
    
    def delete_node(self, node_id: str) -> bool:
        """Delete a memory node and its relations"""
        try:
            with self._lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Delete relations first
                    conn.execute("DELETE FROM memory_relations WHERE from_id = ? OR to_id = ?", 
                               (node_id, node_id))
                    # Delete node
                    conn.execute("DELETE FROM memory_nodes WHERE id = ?", (node_id,))
                    conn.commit()
            return True
        except Exception as e:
            logger.error(f"Failed to delete node {node_id}: {e}")
            return False

class MemoryGraph:
    """Knowledge graph for managing memory relationships"""
    
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self._cache = {}  # Simple in-memory cache
        self._cache_timeout = 300  # 5 minutes
        logger.info("Memory graph initialized")
    
    def add_memory(self, content: str, content_type: str = "fact", 
                   tags: List[str] = None, source: str = "user",
                   metadata: Dict[str, Any] = None) -> str:
        """Add a new memory to the graph"""
        if tags is None:
            tags = []
        if metadata is None:
            metadata = {}
        
        # Generate unique ID
        memory_id = self._generate_id(content)
        
        node = MemoryNode(
            id=memory_id,
            content=content,
            content_type=content_type,
            timestamp=time.time(),
            tags=tags,
            source=source,
            metadata=metadata
        )
        
        if self.storage.store_node(node):
            # Invalidate cache
            self._cache.clear()
            logger.info(f"Added memory: {memory_id}")
            return memory_id
        else:
            raise Exception("Failed to store memory")
    
    def add_relation(self, from_id: str, to_id: str, relation_type: str,
                    strength: float = 1.0, metadata: Dict[str, Any] = None) -> bool:
        """Add a relationship between memories"""
        if metadata is None:
            metadata = {}
        
        relation = MemoryRelation(
            from_id=from_id,
            to_id=to_id,
            relation_type=relation_type,
            strength=strength,
            metadata=metadata
        )
        
        result = self.storage.store_relation(relation)
        if result:
            self._cache.clear()
            logger.info(f"Added relation: {from_id} -> {to_id} ({relation_type})")
        
        return result
    
    def search_memories(self, query_text: str, content_types: List[str] = None,
                       max_results: int = 10) -> List[MemoryNode]:
        """Search for memories matching query text"""
        query = MemoryQuery(
            query_text=query_text,
            content_types=content_types,
            max_results=max_results
        )
        
        return self.storage.retrieve_nodes(query)
    
    def get_related_memories(self, memory_id: str, max_depth: int = 2) -> List[MemoryNode]:
        """Get memories related to a given memory through relationships"""
        visited = set()
        to_visit = [(memory_id, 0)]
        related_ids = set()
        
        while to_visit:
            current_id, depth = to_visit.pop(0)
            
            if current_id in visited or depth > max_depth:
                continue
            
            visited.add(current_id)
            
            # Get relations for current node
            relations = self.storage.retrieve_relations([current_id])
            
            for relation in relations:
                if relation.from_id == current_id:
                    related_id = relation.to_id
                else:
                    related_id = relation.from_id
                
                if related_id not in visited:
                    related_ids.add(related_id)
                    if depth < max_depth:
                        to_visit.append((related_id, depth + 1))
        
        # Retrieve the actual memory nodes
        if related_ids:
            all_nodes = []
            for node_id in related_ids:
                nodes = self.storage.retrieve_nodes(MemoryQuery(query_text="", max_results=1000))
                all_nodes.extend([n for n in nodes if n.id == node_id])
            return all_nodes
        
        return []
    
    def get_memory_stats(self) -> Dict[str, int]:
        """Get statistics about the memory graph"""
        try:
            # This is a simplified implementation
            all_nodes = self.storage.retrieve_nodes(MemoryQuery(query_text="", max_results=10000))
            
            stats = {
                "total_memories": len(all_nodes),
                "content_types": len(set(node.content_type for node in all_nodes)),
                "unique_tags": len(set(tag for node in all_nodes for tag in node.tags)),
                "average_confidence": sum(node.confidence for node in all_nodes) / len(all_nodes) if all_nodes else 0
            }
            
            return stats
        except Exception as e:
            logger.error(f"Failed to get memory stats: {e}")
            return {}
    
    def _generate_id(self, content: str) -> str:
        """Generate a unique ID for a piece of content"""
        timestamp = str(time.time())
        content_hash = hashlib.md5((content + timestamp).encode()).hexdigest()
        return f"mem_{content_hash[:12]}"

class PersistentMemorySystem:
    """Enhanced persistent memory system with knowledge graph capabilities"""
    
    def __init__(self, storage_path: str = "/tmp/echo_memory.db"):
        self.storage = SQLiteMemoryStorage(storage_path)
        self.memory_graph = MemoryGraph(self.storage)
        self.session_memories = defaultdict(list)  # Session-specific memories
        logger.info(f"Persistent memory system initialized: {storage_path}")
    
    def store_conversation_turn(self, session_id: str, user_input: str, 
                               ai_response: str, metadata: Dict[str, Any] = None) -> Tuple[str, str]:
        """Store a conversation turn with user input and AI response"""
        if metadata is None:
            metadata = {}
        
        # Add session context to metadata
        metadata.update({
            "session_id": session_id,
            "interaction_type": "conversation"
        })
        
        # Store user input
        user_memory_id = self.memory_graph.add_memory(
            content=user_input,
            content_type="user_input",
            tags=["conversation", session_id],
            source="user",
            metadata=metadata
        )
        
        # Store AI response
        ai_memory_id = self.memory_graph.add_memory(
            content=ai_response,
            content_type="ai_response", 
            tags=["conversation", session_id],
            source="ai",
            metadata=metadata
        )
        
        # Create relation between input and response
        self.memory_graph.add_relation(
            user_memory_id, ai_memory_id, "response_to", strength=1.0,
            metadata={"session_id": session_id}
        )
        
        # Add to session memory
        self.session_memories[session_id].extend([user_memory_id, ai_memory_id])
        
        return user_memory_id, ai_memory_id
    
    def get_conversation_history(self, session_id: str, max_turns: int = 10) -> List[Dict[str, str]]:
        """Get conversation history for a session"""
        nodes = self.memory_graph.search_memories(
            query_text="",  # Get all
            content_types=["user_input", "ai_response"],
            max_results=max_turns * 2
        )
        
        # Filter by session and organize into turns
        session_nodes = [n for n in nodes if session_id in n.tags]
        session_nodes.sort(key=lambda x: x.timestamp)
        
        conversation = []
        current_turn = {}
        
        for node in session_nodes:
            if node.content_type == "user_input":
                if current_turn:
                    conversation.append(current_turn)
                current_turn = {"user": node.content, "timestamp": node.timestamp}
            elif node.content_type == "ai_response" and current_turn:
                current_turn["ai"] = node.content
        
        if current_turn:
            conversation.append(current_turn)
        
        return conversation[-max_turns:]
    
    def search_knowledge(self, query: str, context_session: str = None) -> List[MemoryNode]:
        """Search the knowledge base for relevant information"""
        # Basic text search
        results = self.memory_graph.search_memories(query, max_results=20)
        
        # If session context provided, boost session-related memories
        if context_session:
            session_results = [r for r in results if context_session in r.tags]
            other_results = [r for r in results if context_session not in r.tags]
            results = session_results + other_results
        
        return results[:10]
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        stats = self.memory_graph.get_memory_stats()
        stats.update({
            "active_sessions": len(self.session_memories),
            "storage_type": self.storage.__class__.__name__,
            "storage_path": getattr(self.storage, 'db_path', 'unknown')
        })
        return stats

# Backwards compatibility - alias for legacy code
MemoryQueryLegacy = MemoryQuery

# Global instance for backwards compatibility
_global_memory_system = None

def get_persistent_memory_system(storage_path: str = "/tmp/echo_memory.db") -> PersistentMemorySystem:
    """Get or create global persistent memory system"""
    global _global_memory_system
    if _global_memory_system is None:
        _global_memory_system = PersistentMemorySystem(storage_path)
    return _global_memory_system