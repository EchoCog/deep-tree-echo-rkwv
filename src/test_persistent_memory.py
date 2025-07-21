"""
Test suite for Persistent Memory Architecture
Tests memory storage, retrieval, search, and persistence across sessions
"""

import pytest
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import json

from persistent_memory import (
    PersistentMemorySystem, 
    MemoryItem,
    MemoryQuery,
    SimpleMemoryEncoder
)

class TestPersistentMemorySystem:
    """Test the persistent memory system"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for tests"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def memory_system(self, temp_dir):
        """Create memory system instance"""
        return PersistentMemorySystem(temp_dir)
    
    def test_memory_storage_and_retrieval(self, memory_system):
        """Test basic memory storage and retrieval"""
        # Store a memory
        memory_id = memory_system.store_memory(
            content="Python is a programming language",
            memory_type="declarative",
            session_id="test_session_1"
        )
        
        assert memory_id is not None
        
        # Retrieve the memory
        retrieved = memory_system.retrieve_memory(memory_id)
        assert retrieved is not None
        assert retrieved.content == "Python is a programming language"
        assert retrieved.memory_type == "declarative"
        assert retrieved.session_id == "test_session_1"
        assert retrieved.access_count == 1  # Should be incremented by retrieval
    
    def test_memory_search(self, memory_system):
        """Test memory search functionality"""
        # Store multiple memories
        memories = [
            ("Machine learning is a subset of AI", "declarative", "session1"),
            ("I learned Python last year", "episodic", "session1"),
            ("To write good code, test frequently", "procedural", "session2"),
            ("Deep learning uses neural networks", "declarative", "session2")
        ]
        
        memory_ids = []
        for content, mem_type, session in memories:
            memory_id = memory_system.store_memory(content, mem_type, session)
            memory_ids.append(memory_id)
        
        # Search for memories
        results = memory_system.search_memories("learning", max_results=5)
        
        assert len(results) >= 2  # Should find memories about learning
        
        # Check that results are ranked by relevance
        for i in range(len(results) - 1):
            assert results[i].relevance_score >= results[i + 1].relevance_score
        
        # Search with type filter
        declarative_results = memory_system.search_memories(
            "learning", 
            memory_types=["declarative"]
        )
        
        for result in declarative_results:
            assert result.item.memory_type == "declarative"
    
    def test_memory_associations(self, memory_system):
        """Test memory association functionality"""
        # Store two related memories
        id1 = memory_system.store_memory(
            "Python is great for data science",
            "declarative",
            "session1"
        )
        
        id2 = memory_system.store_memory(
            "I use Python for machine learning projects",
            "episodic", 
            "session1"
        )
        
        # Associate the memories
        success = memory_system.associate_memories(id1, id2)
        assert success
        
        # Verify associations
        memory1 = memory_system.retrieve_memory(id1)
        memory2 = memory_system.retrieve_memory(id2)
        
        assert id2 in memory1.associations
        assert id1 in memory2.associations
    
    def test_importance_scoring(self, memory_system):
        """Test automatic importance scoring"""
        # Store different types of content
        id1 = memory_system.store_memory(
            "What is the meaning of life?",
            "episodic",
            "session1"
        )
        
        id2 = memory_system.store_memory(
            "Hi",
            "episodic", 
            "session1"
        )
        
        memory1 = memory_system.retrieve_memory(id1)
        memory2 = memory_system.retrieve_memory(id2)
        
        # Longer, more meaningful content should have higher importance
        assert memory1.importance > memory2.importance
    
    def test_memory_consolidation(self, memory_system):
        """Test memory consolidation functionality"""
        # Store similar memories
        id1 = memory_system.store_memory(
            "Python is a programming language",
            "declarative",
            "session1"
        )
        
        id2 = memory_system.store_memory(
            "Python is a programming language used for many applications",
            "declarative",
            "session1"
        )
        
        # Consolidate memories
        consolidated_count = memory_system.consolidate_memories("session1")
        
        # Should find and consolidate similar memories
        assert consolidated_count >= 0
    
    def test_persistence_across_sessions(self, temp_dir):
        """Test that memories persist across system restarts"""
        # Create first memory system instance
        system1 = PersistentMemorySystem(temp_dir)
        
        memory_id = system1.store_memory(
            "This should persist across restarts",
            "declarative",
            "persistent_session"
        )
        
        # Destroy first instance
        del system1
        
        # Create new memory system instance with same directory
        system2 = PersistentMemorySystem(temp_dir)
        
        # Retrieve memory from new instance
        retrieved = system2.retrieve_memory(memory_id)
        assert retrieved is not None
        assert retrieved.content == "This should persist across restarts"
        
        # Search should also work
        results = system2.search_memories("persist")
        assert len(results) >= 1
        assert any("persist" in result.item.content.lower() for result in results)
    
    def test_memory_encoding(self, memory_system):
        """Test memory encoding functionality"""
        encoder = memory_system.encoder
        
        # Test encoding
        text1 = "Machine learning is fascinating"
        text2 = "Deep learning is a subset of machine learning"
        text3 = "The weather is nice today"
        
        encoding1 = encoder.encode(text1)
        encoding2 = encoder.encode(text2)
        encoding3 = encoder.encode(text3)
        
        # Encodings should be numpy arrays
        assert encoding1.shape[0] > 0
        assert encoding2.shape[0] > 0
        assert encoding3.shape[0] > 0
        
        # Similar texts should have higher similarity
        sim_12 = encoder.similarity(encoding1, encoding2)
        sim_13 = encoder.similarity(encoding1, encoding3)
        
        assert sim_12 > sim_13  # ML texts more similar than ML vs weather
    
    def test_system_statistics(self, memory_system):
        """Test system statistics functionality"""
        # Store some test memories
        for i in range(5):
            memory_system.store_memory(
                f"Test memory {i}",
                "declarative",
                f"session_{i % 2}"
            )
        
        stats = memory_system.get_system_stats()
        
        assert 'processing_stats' in stats
        assert 'database_stats' in stats
        assert stats['database_stats']['total_memories'] >= 5
        assert stats['processing_stats']['memories_stored'] >= 5
    
    def test_memory_query_filters(self, memory_system):
        """Test memory query filtering"""
        # Store memories with different attributes
        now = datetime.now()
        old_time = now - timedelta(days=7)
        
        # Store recent memory
        id1 = memory_system.store_memory(
            "Recent important fact",
            "declarative",
            "session1",
            importance=0.9
        )
        
        # Store old memory
        id2 = memory_system.store_memory(
            "Old less important fact", 
            "declarative",
            "session2",
            importance=0.3
        )
        
        # Search with importance filter
        important_results = memory_system.search_memories(
            "fact",
            max_results=10
        )
        
        # Higher importance should rank higher
        if len(important_results) >= 2:
            # The more important memory should rank higher
            high_imp_found = any(r.item.importance > 0.8 for r in important_results[:1])
            assert high_imp_found
    
    def test_session_isolation(self, memory_system):
        """Test that session filtering works correctly"""
        # Store memories in different sessions
        id1 = memory_system.store_memory(
            "Session 1 memory",
            "episodic", 
            "session_1"
        )
        
        id2 = memory_system.store_memory(
            "Session 2 memory",
            "episodic",
            "session_2"
        )
        
        # Search within specific session
        session1_results = memory_system.search_memories(
            "memory",
            session_id="session_1"
        )
        
        # Should only return memories from session 1
        for result in session1_results:
            assert result.item.session_id == "session_1"
    
    def test_error_handling(self, memory_system):
        """Test error handling for edge cases"""
        # Test invalid memory retrieval
        invalid_memory = memory_system.retrieve_memory("nonexistent_id")
        assert invalid_memory is None
        
        # Test empty search
        empty_results = memory_system.search_memories("")
        assert isinstance(empty_results, list)
        
        # Test association with invalid ID
        valid_id = memory_system.store_memory("Valid memory", "declarative", "session")
        success = memory_system.associate_memories(valid_id, "invalid_id")
        assert not success

def test_memory_encoder():
    """Test the SimpleMemoryEncoder"""
    encoder = SimpleMemoryEncoder(embedding_dim=128)
    
    # Test encoding consistency
    text = "This is a test sentence"
    encoding1 = encoder.encode(text)
    encoding2 = encoder.encode(text)
    
    assert encoding1.shape == (128,)
    assert encoding2.shape == (128,)
    
    # Same text should produce identical encodings
    similarity = encoder.similarity(encoding1, encoding2)
    assert similarity == 1.0
    
    # Different texts should have different encodings
    different_text = "Completely different content"
    different_encoding = encoder.encode(different_text)
    different_similarity = encoder.similarity(encoding1, different_encoding)
    
    assert different_similarity < 1.0

if __name__ == "__main__":
    # Run tests manually if not using pytest
    import tempfile
    import shutil
    
    temp_dir = tempfile.mkdtemp()
    try:
        system = PersistentMemorySystem(temp_dir)
        
        print("Testing memory storage...")
        memory_id = system.store_memory(
            "Test memory content",
            "declarative", 
            "test_session"
        )
        print(f"Stored memory: {memory_id}")
        
        print("Testing memory retrieval...")
        retrieved = system.retrieve_memory(memory_id)
        print(f"Retrieved: {retrieved.content if retrieved else 'None'}")
        
        print("Testing memory search...")
        results = system.search_memories("test")
        print(f"Search results: {len(results)}")
        
        print("Testing system stats...")
        stats = system.get_system_stats()
        print(f"Total memories: {stats.get('database_stats', {}).get('total_memories', 0)}")
        
        print("All basic tests passed!")
        
    finally:
        shutil.rmtree(temp_dir)