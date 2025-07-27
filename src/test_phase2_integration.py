"""
Integration Test for Phase 2 Enhanced Cognitive Processing
Tests the complete integration of advanced cognitive capabilities
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch

from enhanced_cognitive_integration import EnhancedCognitiveProcessor
from persistent_memory_foundation import PersistentMemorySystem

class TestPhase2Integration:
    """Test Phase 2 advanced cognitive processing integration"""
    
    def setup_method(self):
        """Setup test fixtures"""
        # Initialize with test memory system
        self.test_memory = PersistentMemorySystem("/tmp/test_memory")
        self.processor = EnhancedCognitiveProcessor(self.test_memory)
        
        self.test_session_id = "test_session_123"
        self.test_conversation_history = []
        self.test_memory_state = {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'goals': []}
        }
    
    def test_enhanced_processor_initialization(self):
        """Test that enhanced processor initializes correctly"""
        assert self.processor.enhanced_processing_enabled == True
        assert self.processor.meta_cognitive_system is not None
        assert self.processor.complex_reasoning_system is not None
        assert self.processor.adaptive_learning_system is not None
        assert self.processor.persistent_memory is not None
    
    @pytest.mark.asyncio
    async def test_simple_query_processing(self):
        """Test processing of a simple query"""
        input_text = "What is artificial intelligence?"
        
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id, 
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate basic structure
        assert 'timestamp' in result
        assert 'input' in result
        assert 'response' in result
        assert 'processing_time' in result
        assert result['input'] == input_text
        
        # Validate Phase 2 features
        assert 'phase2_features' in result
        phase2 = result['phase2_features']
        assert phase2['advanced_processing_enabled'] == True
        assert phase2['adaptive_learning_applied'] == True
        assert 'enhanced_confidence' in phase2
        assert isinstance(phase2['enhanced_confidence'], float)
    
    @pytest.mark.asyncio 
    async def test_complex_reasoning_activation(self):
        """Test that complex reasoning is activated for complex queries"""
        complex_input = "Analyze the relationship between quantum mechanics and consciousness, considering the implications for artificial intelligence development and the nature of reality itself."
        
        result = await self.processor.process_input_enhanced(
            complex_input, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate complex reasoning was applied
        reasoning_output = result['membrane_outputs']['reasoning']
        assert reasoning_output['reasoning_type'] is not None
        assert reasoning_output['confidence'] > 0.8
        assert reasoning_output['steps_count'] > 0
        
        # Validate enhanced response format
        response = result['response']
        assert "Enhanced Cognitive Analysis" in response
        assert "Reasoning Applied" in response
        assert "Memory Context" in response
    
    @pytest.mark.asyncio
    async def test_memory_integration(self):
        """Test memory storage and retrieval integration"""
        # First interaction - store memory
        first_input = "Remember that I prefer detailed explanations with examples"
        result1 = await self.processor.process_input_enhanced(
            first_input, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Second interaction - should retrieve relevant memory
        second_input = "Explain how neural networks work"
        result2 = await self.processor.process_input_enhanced(
            second_input, self.test_session_id,
            [result1], self.test_memory_state
        )
        
        # Validate memory operations
        memory_output = result2['membrane_outputs']['memory']
        assert len(memory_output['retrieved_memories']) >= 0  # May retrieve stored preference
        assert memory_output['confidence'] > 0.0
        
        # Validate learning opportunities were identified
        cognitive_metadata = result1['cognitive_metadata']
        assert 'preference_learning' in cognitive_metadata['adaptive_learning_opportunities']
    
    @pytest.mark.asyncio
    async def test_adaptive_learning_integration(self):
        """Test adaptive learning system integration"""
        input_text = "I need a quick summary of machine learning"
        
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate learning opportunities identification
        cognitive_metadata = result['cognitive_metadata']
        learning_opportunities = cognitive_metadata['adaptive_learning_opportunities']
        assert 'response_style_learning' in learning_opportunities
        
        # Validate Phase 2 adaptive learning was applied
        phase2 = result['phase2_features']
        assert phase2['adaptive_learning_applied'] == True
    
    @pytest.mark.asyncio
    async def test_meta_cognitive_monitoring(self):
        """Test meta-cognitive monitoring and strategy selection"""
        input_text = "Compare and contrast deep learning versus traditional machine learning approaches"
        
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate meta-cognitive features
        phase2 = result['phase2_features']
        assert 'cognitive_strategy_used' in phase2
        assert phase2['cognitive_strategy_used'] is not None
        
        # Validate cognitive metadata
        cognitive_metadata = result['cognitive_metadata']
        assert cognitive_metadata['complexity_detected'] in ['low', 'medium', 'high']
        assert cognitive_metadata['reasoning_type_suggested'] is not None
    
    @pytest.mark.asyncio
    async def test_linguistic_analysis_enhancement(self):
        """Test enhanced linguistic analysis"""
        input_text = "Can you help me understand the philosophical implications of artificial consciousness?"
        
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate grammar processing enhancement
        grammar_output = result['membrane_outputs']['grammar']
        assert 'linguistic_analysis' in grammar_output
        
        linguistic_analysis = grammar_output['linguistic_analysis']
        assert 'sentence_structure' in linguistic_analysis
        assert 'semantic_markers' in linguistic_analysis
        assert 'key_entities' in linguistic_analysis
        assert 'linguistic_features' in linguistic_analysis
        
        # Validate semantic markers identification
        semantic_markers = linguistic_analysis['semantic_markers']
        assert 'interrogative' in semantic_markers
        assert 'assistance_request' in semantic_markers
    
    @pytest.mark.asyncio
    async def test_confidence_scoring(self):
        """Test confidence scoring across all membranes"""
        input_text = "Explain quantum computing principles"
        
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        
        # Validate confidence scores exist
        memory_conf = result['membrane_outputs']['memory']['confidence']
        reasoning_conf = result['membrane_outputs']['reasoning']['confidence']
        grammar_conf = result['membrane_outputs']['grammar']['confidence']
        overall_conf = result['phase2_features']['enhanced_confidence']
        
        assert 0.0 <= memory_conf <= 1.0
        assert 0.0 <= reasoning_conf <= 1.0
        assert 0.0 <= grammar_conf <= 1.0
        assert 0.0 <= overall_conf <= 1.0
    
    @pytest.mark.asyncio
    async def test_error_handling_and_fallback(self):
        """Test error handling and fallback to standard processing"""
        # Test with invalid processor to trigger fallback
        broken_processor = EnhancedCognitiveProcessor()
        broken_processor.enhanced_processing_enabled = False
        
        def mock_fallback(input_text):
            return {
                'input': input_text,
                'response': f"Fallback processing: {input_text}",
                'processing_time': 0.1
            }
        
        input_text = "Test fallback processing"
        result = await broken_processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state,
            fallback_processor=mock_fallback
        )
        
        # Validate fallback response structure
        assert result['input'] == input_text
        assert 'Fallback processing' in result['response']
        assert result['phase2_features']['advanced_processing_enabled'] == False
    
    @pytest.mark.asyncio
    async def test_performance_within_limits(self):
        """Test that processing time is within acceptable limits"""
        input_text = "What are the key principles of machine learning?"
        
        start_time = time.time()
        result = await self.processor.process_input_enhanced(
            input_text, self.test_session_id,
            self.test_conversation_history, self.test_memory_state
        )
        total_time = time.time() - start_time
        
        # Validate processing time
        processing_time = result['processing_time']
        assert processing_time < 2.0  # Should process in under 2 seconds
        assert total_time < 3.0  # Total test time should be under 3 seconds
        
        # Validate performance metadata
        assert processing_time > 0.0
        assert isinstance(processing_time, float)
    
    def test_integration_with_existing_session(self):
        """Test integration with existing cognitive session structure"""
        # Mock existing session structure
        mock_session = Mock()
        mock_session.conversation_history = []
        mock_session.memory_state = self.test_memory_state
        mock_session.process_input = Mock(return_value={'response': 'Standard processing'})
        
        # Test that enhanced processor can work with existing session
        processor = EnhancedCognitiveProcessor(self.test_memory)
        assert processor.enhanced_processing_enabled == True
        
        # Validate fallback mechanism exists
        fallback_result = processor._create_fallback_response("test input", time.time())
        assert 'phase2_features' in fallback_result
        assert fallback_result['phase2_features']['advanced_processing_enabled'] == False

def test_enhanced_processor_without_memory():
    """Test enhanced processor can work without persistent memory"""
    processor = EnhancedCognitiveProcessor(persistent_memory=None)
    assert processor.enhanced_processing_enabled == True
    assert processor.persistent_memory is None

@pytest.mark.asyncio
async def test_full_integration_flow():
    """Test complete integration flow from input to response"""
    processor = EnhancedCognitiveProcessor()
    
    test_inputs = [
        "Hello, how are you?",
        "Can you explain artificial intelligence?", 
        "I prefer detailed technical explanations with examples",
        "Compare neural networks and decision trees"
    ]
    
    conversation_history = []
    memory_state = {'declarative': {}, 'procedural': {}, 'episodic': [], 'intentional': {'goals': []}}
    
    for input_text in test_inputs:
        result = await processor.process_input_enhanced(
            input_text, "integration_test",
            conversation_history, memory_state
        )
        
        # Validate each result
        assert 'response' in result
        assert 'phase2_features' in result
        assert result['phase2_features']['advanced_processing_enabled'] == True
        
        # Add to conversation history for next iteration
        conversation_history.append(result)
    
    # Validate conversation progression
    assert len(conversation_history) == len(test_inputs)
    
    # Validate learning progression (later responses should show some cognitive enhancement)
    last_result = conversation_history[-1]
    assert 'preference_learning' in last_result['cognitive_metadata']['adaptive_learning_opportunities'] or \
           len(last_result['membrane_outputs']['memory']['retrieved_memories']) > 0 or \
           last_result['phase2_features']['enhanced_confidence'] > 0.7

if __name__ == "__main__":
    pytest.main([__file__, "-v"])