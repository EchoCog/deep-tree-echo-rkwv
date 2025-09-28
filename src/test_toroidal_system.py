#!/usr/bin/env python3
"""
Test Suite: Toroidal Cognitive System
Comprehensive tests for the dual-hemisphere architecture
"""

import asyncio
import pytest
import time
import logging
from typing import Dict, Any

from toroidal_cognitive_system import (
    ToroidalCognitiveSystem,
    DeepTreeEcho,
    MardukMadScientist,
    SharedMemoryLattice,
    create_toroidal_cognitive_system
)
from toroidal_integration import (
    ToroidalEchoRWKVBridge,
    ToroidalRESTAPI,
    create_toroidal_bridge,
    create_toroidal_api
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestSharedMemoryLattice:
    """Test the shared memory lattice functionality"""
    
    def test_memory_lattice_creation(self):
        """Test creation of memory lattice"""
        lattice = SharedMemoryLattice(buffer_size=100)
        assert lattice.buffer_size == 100
        assert len(lattice.memory_buffer) == 0
    
    def test_memory_write_read(self):
        """Test writing and reading from memory lattice"""
        lattice = SharedMemoryLattice(buffer_size=10)
        
        # Write data
        test_data = {"content": "test memory", "importance": 0.8}
        lattice.write("echo", test_data)
        
        # Read data
        entries = lattice.read("echo")
        assert len(entries) == 1
        assert entries[0]["hemisphere"] == "echo"
        assert entries[0]["data"] == test_data
        assert entries[0]["access_count"] == 1
    
    def test_memory_buffer_overflow(self):
        """Test buffer overflow handling"""
        lattice = SharedMemoryLattice(buffer_size=3)
        
        # Write more entries than buffer size
        for i in range(5):
            lattice.write("test", {"index": i})
        
        # Should only keep the last 3 entries
        assert len(lattice.memory_buffer) == 3
        entries = lattice.read("test")
        assert len(entries) == 3
        assert entries[0]["data"]["index"] == 4  # Most recent first
    
    def test_context_filtering(self):
        """Test context-based filtering"""
        lattice = SharedMemoryLattice()
        
        # Write entries with different contexts
        lattice.write("echo", {"type": "semantic", "content": "test1"})
        lattice.write("echo", {"type": "logical", "content": "test2"})
        lattice.write("echo", {"type": "semantic", "content": "test3"})
        
        # Filter by context
        semantic_entries = lattice.read("echo", {"type": "semantic"})
        assert len(semantic_entries) == 2
        
        logical_entries = lattice.read("echo", {"type": "logical"})
        assert len(logical_entries) == 1

class TestDeepTreeEcho:
    """Test the Deep Tree Echo (right hemisphere) functionality"""
    
    @pytest.fixture
    def echo_system(self):
        """Create Echo system for testing"""
        lattice = SharedMemoryLattice()
        return DeepTreeEcho(lattice)
    
    @pytest.mark.asyncio
    async def test_echo_response_generation(self, echo_system):
        """Test Echo response generation"""
        prompt = "What is the nature of consciousness?"
        context = {"session": "test"}
        
        response = await echo_system.react(prompt, context)
        
        assert response.hemisphere == "echo"
        assert response.response_text is not None
        assert len(response.response_text) > 0
        assert response.confidence > 0.0
        assert response.processing_time > 0.0
        assert "cognitive_markers" in response.cognitive_markers
    
    @pytest.mark.asyncio
    async def test_echo_pattern_recognition(self, echo_system):
        """Test Echo's pattern recognition capabilities"""
        pattern_prompt = "Describe the pattern in this system architecture"
        context = {}
        
        response = await echo_system.react(pattern_prompt, context)
        
        # Should have high pattern recognition score
        pattern_score = response.cognitive_markers.get("pattern_recognition", 0.0)
        assert pattern_score > 0.1  # Should detect pattern-related content
        
        # Should contain characteristic Echo language
        assert any(word in response.response_text.lower() 
                  for word in ["pattern", "resonance", "memory", "sacred"])
    
    @pytest.mark.asyncio
    async def test_echo_semantic_resonance(self, echo_system):
        """Test semantic resonance assessment"""
        prompt = "memory and resonance and patterns"
        context = {}
        
        response = await echo_system.react(prompt, context)
        
        # Should have decent semantic resonance
        semantic_resonance = response.cognitive_markers.get("semantic_resonance", 0.0)
        assert semantic_resonance > 0.0

class TestMardukMadScientist:
    """Test the Marduk Mad Scientist (left hemisphere) functionality"""
    
    @pytest.fixture
    def marduk_system(self):
        """Create Marduk system for testing"""
        lattice = SharedMemoryLattice()
        return MardukMadScientist(lattice)
    
    @pytest.mark.asyncio
    async def test_marduk_response_generation(self, marduk_system):
        """Test Marduk response generation"""
        prompt = "Analyze the computational complexity of this system"
        context = {"session": "test"}
        
        response = await marduk_system.process(prompt, context)
        
        assert response.hemisphere == "marduk"
        assert response.response_text is not None
        assert len(response.response_text) > 0
        assert response.confidence > 0.0
        assert response.processing_time > 0.0
        assert "cognitive_markers" in response.cognitive_markers
    
    @pytest.mark.asyncio
    async def test_marduk_logical_structure(self, marduk_system):
        """Test Marduk's logical structure capabilities"""
        analysis_prompt = "Provide a systematic analysis of the architecture"
        context = {}
        
        response = await marduk_system.process(analysis_prompt, context)
        
        # Should have high logical structure score
        logical_score = response.cognitive_markers.get("logical_structure", 0.0)
        assert logical_score > 0.1  # Should detect structured content
        
        # Should contain characteristic Marduk language
        assert any(word in response.response_text.lower() 
                  for word in ["analyze", "process", "structure", "algorithm", "system"])
    
    @pytest.mark.asyncio
    async def test_marduk_recursion_depth(self, marduk_system):
        """Test recursion depth assessment"""
        recursive_prompt = "recursive neural hierarchical tree structures"
        context = {}
        
        response = await marduk_system.process(recursive_prompt, context)
        
        # Should have decent recursion depth score
        recursion_score = response.cognitive_markers.get("recursion_depth", 0.0)
        assert recursion_score > 0.0

class TestToroidalCognitiveSystem:
    """Test the integrated Toroidal Cognitive System"""
    
    @pytest.fixture
    def toroidal_system(self):
        """Create Toroidal system for testing"""
        return create_toroidal_cognitive_system(buffer_size=100)
    
    @pytest.mark.asyncio
    async def test_dual_hemisphere_processing(self, toroidal_system):
        """Test that both hemispheres process input"""
        prompt = "Explain the toroidal cognitive architecture"
        
        response = await toroidal_system.process_input(prompt)
        
        assert response.user_input == prompt
        assert response.echo_response is not None
        assert response.marduk_response is not None
        assert response.echo_response.hemisphere == "echo"
        assert response.marduk_response.hemisphere == "marduk"
        assert response.synchronized_output is not None
        assert response.reflection is not None
        assert response.total_processing_time > 0.0
    
    @pytest.mark.asyncio
    async def test_convergence_metrics(self, toroidal_system):
        """Test convergence metrics calculation"""
        prompt = "Test convergence"
        
        response = await toroidal_system.process_input(prompt)
        
        metrics = response.convergence_metrics
        assert "temporal_sync" in metrics
        assert "confidence_alignment" in metrics
        assert "complementarity" in metrics
        assert "coherence" in metrics
        
        # All metrics should be between 0 and 1
        for metric_name, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"Metric {metric_name} out of range: {value}"
    
    @pytest.mark.asyncio
    async def test_synchronized_output_format(self, toroidal_system):
        """Test synchronized output formatting"""
        prompt = "Format test"
        
        response = await toroidal_system.process_input(prompt)
        
        output = response.synchronized_output
        assert "Deep Tree Echo (Right Hemisphere Response)" in output
        assert "Marduk the Mad Scientist (Left Hemisphere Response)" in output
        assert response.echo_response.response_text in output
        assert response.marduk_response.response_text in output
    
    @pytest.mark.asyncio
    async def test_system_state_updates(self, toroidal_system):
        """Test system state updates"""
        initial_depth = toroidal_system.system_state.processing_depth
        
        await toroidal_system.process_input("Test state update")
        
        assert toroidal_system.system_state.processing_depth == initial_depth + 1
        assert toroidal_system.system_state.timestamp > 0
    
    def test_system_metrics(self, toroidal_system):
        """Test system metrics retrieval"""
        metrics = toroidal_system.get_system_metrics()
        
        assert "system_state" in metrics
        assert "memory_buffer_size" in metrics
        assert "hemispheres" in metrics
        
        hemispheres = metrics["hemispheres"]
        assert "echo" in hemispheres
        assert "marduk" in hemispheres
        assert hemispheres["echo"]["name"] == "Deep Tree Echo"
        assert hemispheres["marduk"]["name"] == "Marduk the Mad Scientist"

class TestToroidalIntegration:
    """Test the Toroidal Integration layer"""
    
    @pytest.fixture
    def bridge(self):
        """Create integration bridge for testing"""
        return create_toroidal_bridge(buffer_size=50, use_real_rwkv=False)
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge):
        """Test bridge initialization"""
        result = await bridge.initialize()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_cognitive_input_processing(self, bridge):
        """Test cognitive input processing through bridge"""
        await bridge.initialize()
        
        response = await bridge.process_cognitive_input(
            user_input="Test bridge processing",
            session_id="test_session",
            conversation_history=[],
            memory_state={},
            processing_goals=["test"]
        )
        
        assert response.user_input == "Test bridge processing"
        assert response.echo_response is not None
        assert response.marduk_response is not None
        assert response.synchronized_output is not None
    
    def test_system_status(self, bridge):
        """Test system status retrieval"""
        status = bridge.get_system_status()
        
        assert "toroidal_system" in status
        assert "rwkv_integration" in status
        assert "bridge_status" in status
        
        rwkv_info = status["rwkv_integration"]
        assert "enabled" in rwkv_info
        assert "available" in rwkv_info
        assert "engine_initialized" in rwkv_info

class TestToroidalRESTAPI:
    """Test the Toroidal REST API"""
    
    @pytest.fixture
    async def api(self):
        """Create API for testing"""
        bridge = create_toroidal_bridge(buffer_size=30, use_real_rwkv=False)
        await bridge.initialize()
        return create_toroidal_api(bridge)
    
    @pytest.mark.asyncio
    async def test_api_query_processing(self, api):
        """Test API query processing"""
        request_data = {
            "input": "Test API query",
            "session_id": "api_test",
            "conversation_history": [],
            "memory_state": {},
            "processing_goals": ["api_test"]
        }
        
        response = await api.process_query(request_data)
        
        assert response["success"] is True
        assert "response" in response
        
        response_data = response["response"]
        assert "user_input" in response_data
        assert "echo_response" in response_data
        assert "marduk_response" in response_data
        assert "synchronized_output" in response_data
        assert "reflection" in response_data
        assert "convergence_metrics" in response_data
    
    @pytest.mark.asyncio
    async def test_api_empty_input_handling(self, api):
        """Test API handling of empty input"""
        request_data = {"input": ""}
        
        response = await api.process_query(request_data)
        
        assert "error" in response
        assert "Empty input provided" in response["error"]
    
    @pytest.mark.asyncio
    async def test_api_system_status(self, api):
        """Test API system status"""
        response = await api.get_system_status()
        
        assert response["success"] is True
        assert "status" in response

class TestPerformance:
    """Performance tests for the Toroidal system"""
    
    @pytest.mark.asyncio
    async def test_response_time_performance(self):
        """Test response time performance"""
        system = create_toroidal_cognitive_system()
        
        start_time = time.time()
        response = await system.process_input("Performance test query")
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should process within reasonable time (less than 10 seconds for this demo)
        assert processing_time < 10.0
        assert response.total_processing_time < processing_time
    
    @pytest.mark.asyncio
    async def test_concurrent_processing(self):
        """Test concurrent processing capabilities"""
        system = create_toroidal_cognitive_system()
        
        # Process multiple queries concurrently
        queries = [f"Concurrent test {i}" for i in range(5)]
        
        start_time = time.time()
        responses = await asyncio.gather(*[
            system.process_input(query) for query in queries
        ])
        end_time = time.time()
        
        total_time = end_time - start_time
        
        assert len(responses) == 5
        for i, response in enumerate(responses):
            assert response.user_input == f"Concurrent test {i}"
        
        # Concurrent processing should be faster than sequential
        # (This is a rough heuristic test)
        assert total_time < sum(r.total_processing_time for r in responses)

class TestRobustness:
    """Robustness and error handling tests"""
    
    @pytest.mark.asyncio
    async def test_empty_prompt_handling(self):
        """Test handling of empty prompts"""
        system = create_toroidal_cognitive_system()
        
        response = await system.process_input("")
        
        # Should handle empty input gracefully
        assert response is not None
        assert response.echo_response is not None
        assert response.marduk_response is not None
    
    @pytest.mark.asyncio
    async def test_very_long_prompt_handling(self):
        """Test handling of very long prompts"""
        system = create_toroidal_cognitive_system()
        
        long_prompt = "This is a very long prompt. " * 100
        
        response = await system.process_input(long_prompt)
        
        # Should handle long input gracefully
        assert response is not None
        assert response.echo_response is not None
        assert response.marduk_response is not None
    
    @pytest.mark.asyncio
    async def test_special_characters_handling(self):
        """Test handling of special characters"""
        system = create_toroidal_cognitive_system()
        
        special_prompt = "Test with émojis 🧠🌳 and spëcial chars: @#$%^&*()"
        
        response = await system.process_input(special_prompt)
        
        # Should handle special characters gracefully
        assert response is not None
        assert response.user_input == special_prompt

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])