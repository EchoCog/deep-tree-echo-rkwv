"""
Persistent Memory Architecture for Deep Tree Echo
Implements multi-layered memory system with semantic search and persistence
"""

import sqlite3
import json
import pickle
import os
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Create a simple fallback for numpy functionality
    class np:
        @staticmethod
        def array(data, dtype=None):
            return data
        
        @staticmethod
        def dot(a, b):
            if isinstance(a, list) and isinstance(b, list):
                return sum(x * y for x, y in zip(a, b))
            return 0
        
        @staticmethod
        def linalg():
            pass
        
        class linalg:
            @staticmethod
            def norm(x):
                if isinstance(x, list):
                    return sum(val ** 2 for val in x) ** 0.5
                return 1.0
        
        float32 = float

from pathlib import Path
import hashlib
import uuid

logger = logging.getLogger(__name__)

@dataclass
class MemoryItem:
    """Represents a single memory item with metadata"""
    id: str
    content: str
    memory_type: str  # 'declarative', 'procedural', 'episodic', 'semantic'
    timestamp: str
    session_id: str
    importance: float  # 0.0 to 1.0
    access_count: int
    last_accessed: str
    encoding: Optional[bytes] = None  # Serialized embedding
    tags: Optional[List[str]] = None
    associations: Optional[List[str]] = None  # IDs of associated memories
    metadata: Optional[Dict[str, Any]] = None

@dataclass 
class MemoryQuery:
    """Represents a memory query with filters and options"""
    query_text: str
    memory_types: List[str] = None
    session_id: Optional[str] = None
    importance_threshold: float = 0.0
    max_results: int = 10
    similarity_threshold: float = 0.5
    include_associations: bool = True
    time_range: Optional[Tuple[datetime, datetime]] = None

@dataclass
class MemorySearchResult:
    """Result of memory search with relevance scoring"""
    item: MemoryItem
    relevance_score: float
    similarity_score: float
    context_score: float
    associations: List['MemorySearchResult'] = None

class MemoryEncoder(ABC):
    """Abstract base class for memory encoding"""
    
    @abstractmethod
    def encode(self, text: str) -> Union[List[float], Any]:
        """Encode text into vector representation"""
        pass
    
    @abstractmethod
    def similarity(self, encoding1: Union[List[float], Any], encoding2: Union[List[float], Any]) -> float:
        """Calculate similarity between two encodings"""
        pass

class SimpleMemoryEncoder(MemoryEncoder):
    """Simple memory encoder using basic text features"""
    
    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        
    def encode(self, text: str) -> Union[List[float], Any]:
        """Simple encoding based on text features"""
        # Create a simple hash-based encoding
        text_hash = hashlib.md5(text.lower().encode()).hexdigest()
        
        # Convert hash to numerical features
        features = []
        for i in range(0, min(len(text_hash), 32), 2):
            features.append(int(text_hash[i:i+2], 16) / 255.0)
        
        # Add text statistics
        features.extend([
            len(text.split()) / 100.0,  # Word count
            len(text) / 1000.0,         # Character count
            text.count('?') / 10.0,     # Questions
            text.count('!') / 10.0,     # Exclamations
            len(set(text.lower().split())) / len(text.split()) if text.split() else 0  # Uniqueness
        ])
        
        # Add semantic features
        features.extend(self._extract_semantic_features(text))
        
        # Pad or truncate to desired dimension
        while len(features) < self.embedding_dim:
            features.append(0.0)
        features = features[:self.embedding_dim]
        
        if NUMPY_AVAILABLE:
            return np.array(features, dtype=np.float32)
        else:
            return features
            features.append(0.0)
        features = features[:self.embedding_dim]
        
        return np.array(features, dtype=np.float32)
    
    def _extract_semantic_features(self, text: str) -> List[float]:
        """Extract enhanced semantic features"""
        words = text.lower().split()
        
        # Semantic categories
        question_words = ['what', 'why', 'how', 'when', 'where', 'who', 'which']
        emotion_words = ['happy', 'sad', 'angry', 'excited', 'worried', 'confused']
        action_words = ['do', 'make', 'create', 'build', 'solve', 'find', 'learn']
        cognitive_words = ['think', 'understand', 'know', 'remember', 'analyze']
        temporal_words = ['before', 'after', 'during', 'while', 'then', 'now']
        
        features = []
        
        # Semantic category presence
        features.append(sum(1 for w in words if w in question_words) / max(len(words), 1))
        features.append(sum(1 for w in words if w in emotion_words) / max(len(words), 1))
        features.append(sum(1 for w in words if w in action_words) / max(len(words), 1))
        features.append(sum(1 for w in words if w in cognitive_words) / max(len(words), 1))
        features.append(sum(1 for w in words if w in temporal_words) / max(len(words), 1))
        
        # Syntactic features
        features.append(text.count(',') / max(len(text), 1))  # Comma density
        features.append(text.count(';') / max(len(text), 1))  # Semicolon density
        features.append(text.count(':') / max(len(text), 1))  # Colon density
        
        # Memory type indicators
        procedural_indicators = ['how', 'step', 'process', 'method', 'way']
        episodic_indicators = ['i', 'me', 'my', 'happened', 'experience']
        declarative_indicators = ['is', 'are', 'fact', 'definition', 'means']
        
        features.append(sum(1 for w in words if w in procedural_indicators) / max(len(words), 1))
        features.append(sum(1 for w in words if w in episodic_indicators) / max(len(words), 1))
        features.append(sum(1 for w in words if w in declarative_indicators) / max(len(words), 1))
        
        return features
    
    def similarity(self, encoding1: Union[List[float], Any], encoding2: Union[List[float], Any]) -> float:
        """Calculate cosine similarity"""
        if NUMPY_AVAILABLE and hasattr(encoding1, 'shape'):
            dot_product = np.dot(encoding1, encoding2)
            norm1 = np.linalg.norm(encoding1)
            norm2 = np.linalg.norm(encoding2)
        else:
            # Fallback for non-numpy
            if isinstance(encoding1, list) and isinstance(encoding2, list):
                dot_product = sum(a * b for a, b in zip(encoding1, encoding2))
                norm1 = sum(a * a for a in encoding1) ** 0.5
                norm2 = sum(b * b for b in encoding2) ** 0.5
            else:
                return 0.0
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)

class PersistentMemoryDatabase:
    """SQLite-based persistent storage for memory items"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.connection = None
        self.lock = threading.RLock()
        self._ensure_database()
    
    def _ensure_database(self):
        """Ensure database exists with proper schema"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    content TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    importance REAL NOT NULL,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT NOT NULL,
                    encoding BLOB,
                    tags TEXT,  -- JSON array
                    associations TEXT,  -- JSON array
                    metadata TEXT  -- JSON object
                )
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_memory_type ON memories(memory_type)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_session_id ON memories(session_id)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_timestamp ON memories(timestamp)
            ''')
            
            conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_importance ON memories(importance)
            ''')
            
            conn.commit()
    
    def store_memory(self, memory: MemoryItem) -> bool:
        """Store a memory item in the database"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        INSERT OR REPLACE INTO memories (
                            id, content, memory_type, timestamp, session_id,
                            importance, access_count, last_accessed, encoding,
                            tags, associations, metadata
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        memory.id,
                        memory.content,
                        memory.memory_type,
                        memory.timestamp,
                        memory.session_id,
                        memory.importance,
                        memory.access_count,
                        memory.last_accessed,
                        memory.encoding,
                        json.dumps(memory.tags) if memory.tags else None,
                        json.dumps(memory.associations) if memory.associations else None,
                        json.dumps(memory.metadata) if memory.metadata else None
                    ))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error storing memory {memory.id}: {e}")
            return False
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory by ID"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT id, content, memory_type, timestamp, session_id,
                               importance, access_count, last_accessed, encoding,
                               tags, associations, metadata
                        FROM memories WHERE id = ?
                    ''', (memory_id,))
                    
                    row = cursor.fetchone()
                    if row:
                        return self._row_to_memory(row)
                    return None
        except Exception as e:
            logger.error(f"Error retrieving memory {memory_id}: {e}")
            return None
    
    def search_memories(self, query: MemoryQuery) -> List[MemoryItem]:
        """Search for memories based on query criteria"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    # Build query conditions
                    conditions = []
                    params = []
                    
                    if query.memory_types:
                        placeholders = ','.join('?' * len(query.memory_types))
                        conditions.append(f'memory_type IN ({placeholders})')
                        params.extend(query.memory_types)
                    
                    if query.session_id:
                        conditions.append('session_id = ?')
                        params.append(query.session_id)
                    
                    if query.importance_threshold > 0:
                        conditions.append('importance >= ?')
                        params.append(query.importance_threshold)
                    
                    if query.time_range:
                        conditions.append('timestamp BETWEEN ? AND ?')
                        params.extend([
                            query.time_range[0].isoformat(),
                            query.time_range[1].isoformat()
                        ])
                    
                    # Add text search
                    if query.query_text:
                        conditions.append('content LIKE ?')
                        params.append(f'%{query.query_text}%')
                    
                    where_clause = ' AND '.join(conditions) if conditions else '1=1'
                    
                    cursor = conn.execute(f'''
                        SELECT id, content, memory_type, timestamp, session_id,
                               importance, access_count, last_accessed, encoding,
                               tags, associations, metadata
                        FROM memories 
                        WHERE {where_clause}
                        ORDER BY importance DESC, last_accessed DESC
                        LIMIT ?
                    ''', params + [query.max_results])
                    
                    return [self._row_to_memory(row) for row in cursor.fetchall()]
                    
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def update_access(self, memory_id: str) -> bool:
        """Update access count and last accessed time for a memory"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute('''
                        UPDATE memories 
                        SET access_count = access_count + 1,
                            last_accessed = ?
                        WHERE id = ?
                    ''', (datetime.now().isoformat(), memory_id))
                    conn.commit()
                    return True
        except Exception as e:
            logger.error(f"Error updating access for memory {memory_id}: {e}")
            return False
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            with self.lock:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.execute('''
                        SELECT 
                            COUNT(*) as total_memories,
                            COUNT(DISTINCT session_id) as unique_sessions,
                            COUNT(DISTINCT memory_type) as memory_types,
                            AVG(importance) as avg_importance,
                            MAX(timestamp) as latest_memory
                        FROM memories
                    ''')
                    
                    row = cursor.fetchone()
                    
                    # Get type breakdown
                    cursor = conn.execute('''
                        SELECT memory_type, COUNT(*) 
                        FROM memories 
                        GROUP BY memory_type
                    ''')
                    
                    type_counts = dict(cursor.fetchall())
                    
                    return {
                        'total_memories': row[0],
                        'unique_sessions': row[1],
                        'memory_types': row[2],
                        'avg_importance': row[3] or 0.0,
                        'latest_memory': row[4],
                        'type_breakdown': type_counts
                    }
        except Exception as e:
            logger.error(f"Error getting memory stats: {e}")
            return {}
    
    def _row_to_memory(self, row) -> MemoryItem:
        """Convert database row to MemoryItem"""
        return MemoryItem(
            id=row[0],
            content=row[1],
            memory_type=row[2],
            timestamp=row[3],
            session_id=row[4],
            importance=row[5],
            access_count=row[6],
            last_accessed=row[7],
            encoding=row[8],
            tags=json.loads(row[9]) if row[9] else None,
            associations=json.loads(row[10]) if row[10] else None,
            metadata=json.loads(row[11]) if row[11] else None
        )

class PersistentMemorySystem:
    """Main persistent memory system coordinating storage, encoding, and retrieval"""
    
    def __init__(self, data_dir: str, encoder: MemoryEncoder = None):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.encoder = encoder or SimpleMemoryEncoder()
        self.database = PersistentMemoryDatabase(str(self.data_dir / "memories.db"))
        
        self.stats = {
            'memories_stored': 0,
            'memories_retrieved': 0,
            'searches_performed': 0,
            'start_time': datetime.now()
        }
        
        logger.info(f"Persistent memory system initialized: {data_dir}")
    
    def store_memory(self, content: str, memory_type: str, session_id: str, 
                    importance: Optional[float] = None, tags: Optional[List[str]] = None,
                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """Store a new memory item"""
        
        # Generate unique ID
        memory_id = str(uuid.uuid4())
        
        # Calculate importance if not provided
        if importance is None:
            importance = self._calculate_importance(content, memory_type)
        
        # Encode the content
        encoding = self.encoder.encode(content)
        encoded_bytes = pickle.dumps(encoding)
        
        # Create memory item
        memory = MemoryItem(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            timestamp=datetime.now().isoformat(),
            session_id=session_id,
            importance=importance,
            access_count=0,
            last_accessed=datetime.now().isoformat(),
            encoding=encoded_bytes,
            tags=tags or [],
            associations=[],
            metadata=metadata or {}
        )
        
        # Store in database
        if self.database.store_memory(memory):
            self.stats['memories_stored'] += 1
            logger.debug(f"Stored memory {memory_id}: {content[:50]}...")
            return memory_id
        else:
            logger.error(f"Failed to store memory: {content[:50]}...")
            return None
    
    def retrieve_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """Retrieve a specific memory by ID and update access count"""
        memory = self.database.retrieve_memory(memory_id)
        if memory:
            self.database.update_access(memory_id)
            self.stats['memories_retrieved'] += 1
            # Retrieve updated memory with incremented access count
            memory = self.database.retrieve_memory(memory_id)
        return memory
    
    def search_memories(self, query_text: str, memory_types: List[str] = None,
                       session_id: Optional[str] = None, max_results: int = 10,
                       similarity_threshold: float = 0.3) -> List[MemorySearchResult]:
        """Search for relevant memories using semantic similarity"""
        
        self.stats['searches_performed'] += 1
        
        # Create query
        query = MemoryQuery(
            query_text=query_text,
            memory_types=memory_types,
            session_id=session_id,
            max_results=max_results * 2,  # Get more to filter by similarity
            similarity_threshold=similarity_threshold
        )
        
        # Search database
        candidate_memories = self.database.search_memories(query)
        
        if not candidate_memories:
            return []
        
        # Encode query for similarity comparison
        query_encoding = self.encoder.encode(query_text)
        
        # Calculate similarities and rank results
        results = []
        for memory in candidate_memories:
            if memory.encoding:
                try:
                    memory_encoding = pickle.loads(memory.encoding)
                    similarity = self.encoder.similarity(query_encoding, memory_encoding)
                    
                    if similarity >= similarity_threshold:
                        # Calculate composite relevance score
                        relevance = self._calculate_relevance_score(
                            memory, query_text, similarity
                        )
                        
                        result = MemorySearchResult(
                            item=memory,
                            relevance_score=relevance,
                            similarity_score=similarity,
                            context_score=self._calculate_context_score(memory, query_text)
                        )
                        
                        results.append(result)
                        
                except Exception as e:
                    logger.warning(f"Error processing memory encoding {memory.id}: {e}")
        
        # Sort by relevance and limit results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        final_results = results[:max_results]
        
        # Update access counts for retrieved memories
        for result in final_results:
            self.database.update_access(result.item.id)
        
        logger.debug(f"Memory search for '{query_text}' returned {len(final_results)} results")
        return final_results
    
    def _calculate_relevance_score(self, memory: MemoryItem, query_text: str, similarity: float) -> float:
        """Calculate composite relevance score for memory"""
        # Base relevance from similarity
        relevance = similarity * 0.6
        
        # Boost for importance
        relevance += memory.importance * 0.2
        
        # Boost for recent access
        try:
            last_accessed = datetime.fromisoformat(memory.last_accessed)
            days_since_access = (datetime.now() - last_accessed).days
            recency_boost = max(0, (30 - days_since_access) / 30) * 0.1
            relevance += recency_boost
        except:
            pass
        
        # Boost for access frequency
        frequency_boost = min(memory.access_count / 10.0, 0.1)
        relevance += frequency_boost
        
        return min(relevance, 1.0)
    
    def _calculate_context_score(self, memory: MemoryItem, query_text: str) -> float:
        """Calculate contextual relevance score"""
        score = 0.0
        
        # Check for exact word matches
        memory_words = set(memory.content.lower().split())
        query_words = set(query_text.lower().split())
        word_overlap = len(memory_words.intersection(query_words))
        if len(query_words) > 0:
            score += (word_overlap / len(query_words)) * 0.5
        
        # Check for semantic category matches
        if self._get_semantic_category(memory.content) == self._get_semantic_category(query_text):
            score += 0.3
        
        # Check memory type appropriateness
        if self._is_memory_type_appropriate(memory.memory_type, query_text):
            score += 0.2
        
        return min(score, 1.0)
    
    def _get_semantic_category(self, text: str) -> str:
        """Determine semantic category of text"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['how', 'step', 'process', 'method']):
            return 'procedural'
        elif any(word in text_lower for word in ['i', 'me', 'my', 'happened', 'experience']):
            return 'episodic'
        elif any(word in text_lower for word in ['what', 'is', 'define', 'fact']):
            return 'declarative'
        else:
            return 'general'
    
    def _is_memory_type_appropriate(self, memory_type: str, query_text: str) -> bool:
        """Check if memory type is appropriate for query"""
        query_category = self._get_semantic_category(query_text)
        return memory_type == query_category or query_category == 'general'
    
    def associate_memories(self, memory_id1: str, memory_id2: str) -> bool:
        """Create association between two memories"""
        memory1 = self.retrieve_memory(memory_id1)
        memory2 = self.retrieve_memory(memory_id2)
        
        if not memory1 or not memory2:
            return False
        
        # Add mutual associations
        if not memory1.associations:
            memory1.associations = []
        if not memory2.associations:
            memory2.associations = []
        
        if memory_id2 not in memory1.associations:
            memory1.associations.append(memory_id2)
        if memory_id1 not in memory2.associations:
            memory2.associations.append(memory_id1)
        
        # Update in database
        return (self.database.store_memory(memory1) and 
                self.database.store_memory(memory2))
    
    def consolidate_memories(self, session_id: Optional[str] = None) -> int:
        """Consolidate similar memories to reduce redundancy"""
        query = MemoryQuery(
            query_text="",
            session_id=session_id,
            max_results=1000
        )
        
        memories = self.database.search_memories(query)
        consolidated_count = 0
        
        # Group memories by type for more efficient comparison
        memory_groups = {}
        for memory in memories:
            if memory.memory_type not in memory_groups:
                memory_groups[memory.memory_type] = []
            memory_groups[memory.memory_type].append(memory)
        
        # Find and merge similar memories within each group
        for memory_type, group_memories in memory_groups.items():
            for i, memory1 in enumerate(group_memories):
                for j, memory2 in enumerate(group_memories[i+1:], i+1):
                    if self._should_consolidate(memory1, memory2):
                        if self._merge_memories(memory1, memory2):
                            consolidated_count += 1
        
        logger.info(f"Consolidated {consolidated_count} memories")
        return consolidated_count
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        db_stats = self.database.get_memory_stats()
        
        return {
            'processing_stats': self.stats,
            'database_stats': db_stats,
            'uptime_seconds': (datetime.now() - self.stats['start_time']).total_seconds(),
            'memory_types': db_stats.get('type_breakdown', {})
        }
    
    def _calculate_importance(self, content: str, memory_type: str) -> float:
        """Calculate importance score for memory content"""
        base_importance = 0.5
        
        # Adjust based on content features
        words = content.split()
        
        # Length factor
        if len(words) > 20:
            base_importance += 0.1
        elif len(words) < 5:
            base_importance -= 0.1
        
        # Question or exclamation increases importance
        if '?' in content or '!' in content:
            base_importance += 0.1
        
        # Memory type adjustments
        type_weights = {
            'episodic': 0.8,      # Personal experiences
            'declarative': 0.7,   # Facts and knowledge
            'procedural': 0.9,    # Skills and procedures
            'semantic': 0.6       # General knowledge
        }
        
        importance_multiplier = type_weights.get(memory_type, 0.5)
        final_importance = min(1.0, base_importance * importance_multiplier)
        
        return final_importance
    
    def _calculate_relevance_score(self, memory: MemoryItem, query: str, 
                                 similarity: float) -> float:
        """Calculate composite relevance score"""
        # Weight factors
        similarity_weight = 0.4
        importance_weight = 0.3
        recency_weight = 0.2
        access_weight = 0.1
        
        # Recency score (more recent = higher score)
        try:
            memory_time = datetime.fromisoformat(memory.timestamp)
            now = datetime.now()
            hours_old = (now - memory_time).total_seconds() / 3600
            recency_score = max(0, 1 - (hours_old / (24 * 7)))  # Decay over week
        except:
            recency_score = 0.5
        
        # Access frequency score
        access_score = min(1.0, memory.access_count / 10.0)
        
        # Composite score
        relevance = (
            similarity * similarity_weight +
            memory.importance * importance_weight +
            recency_score * recency_weight +
            access_score * access_weight
        )
        
        return min(1.0, relevance)
    
    def _calculate_context_score(self, memory: MemoryItem, query: str) -> float:
        """Calculate contextual relevance score"""
        # Simple context scoring based on word overlap
        memory_words = set(memory.content.lower().split())
        query_words = set(query.lower().split())
        
        if not memory_words or not query_words:
            return 0.0
        
        intersection = memory_words.intersection(query_words)
        union = memory_words.union(query_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _should_consolidate(self, memory1: MemoryItem, memory2: MemoryItem) -> bool:
        """Determine if two memories should be consolidated"""
        # Don't consolidate different types
        if memory1.memory_type != memory2.memory_type:
            return False
        
        # Don't consolidate if from different sessions (unless very similar)
        if memory1.session_id != memory2.session_id:
            return False
        
        # Check content similarity
        try:
            encoding1 = pickle.loads(memory1.encoding) if memory1.encoding else None
            encoding2 = pickle.loads(memory2.encoding) if memory2.encoding else None
            
            if encoding1 is not None and encoding2 is not None:
                similarity = self.encoder.similarity(encoding1, encoding2)
                return similarity > 0.9  # Very high threshold for consolidation
        except:
            pass
        
        # Basic content similarity check
        words1 = set(memory1.content.lower().split())
        words2 = set(memory2.content.lower().split())
        
        if words1 and words2:
            overlap = len(words1.intersection(words2)) / len(words1.union(words2))
            return overlap > 0.8
        
        return False
    
    def _merge_memories(self, memory1: MemoryItem, memory2: MemoryItem) -> bool:
        """Merge two similar memories"""
        try:
            # Keep the more important memory as base
            if memory2.importance > memory1.importance:
                memory1, memory2 = memory2, memory1
            
            # Merge content
            merged_content = f"{memory1.content}. {memory2.content}"
            
            # Combine access counts and update timestamp
            memory1.content = merged_content
            memory1.access_count += memory2.access_count
            memory1.importance = max(memory1.importance, memory2.importance)
            
            # Merge tags and associations
            if memory1.tags and memory2.tags:
                memory1.tags = list(set(memory1.tags + memory2.tags))
            elif memory2.tags:
                memory1.tags = memory2.tags
            
            if memory1.associations and memory2.associations:
                memory1.associations = list(set(memory1.associations + memory2.associations))
            elif memory2.associations:
                memory1.associations = memory2.associations
            
            # Re-encode merged content
            encoding = self.encoder.encode(memory1.content)
            memory1.encoding = pickle.dumps(encoding)
            
            # Store updated memory and remove old one
            return self.database.store_memory(memory1)
            
        except Exception as e:
            logger.error(f"Error merging memories: {e}")
            return False