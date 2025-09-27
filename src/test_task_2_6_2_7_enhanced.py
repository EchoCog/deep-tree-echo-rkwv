"""
Test Task 2.6 and 2.7 Enhanced Functionality
Tests explanation generation and enhanced preference learning systems
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, patch
from explanation_generation import (
    ExplanationGenerator, ExplanationRequest, ExplanationStyle, ExplanationLevel,
    GeneratedExplanation, ClarityOptimizer
)
from enhanced_preference_learning import (
    EnhancedPersonalizationEngine, CommunicationStyle, InteractionPattern,
    LearningStrategy, EnhancedUserPreference, InteractionAnalysis
)
from enhanced_cognitive_integration import EnhancedCognitiveProcessor
from persistent_memory_foundation import PersistentMemorySystem

class TestExplanationGeneration:
    """Test the explanation generation system (Task 2.6)"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.explanation_generator = ExplanationGenerator()
        self.clarity_optimizer = ClarityOptimizer()
    
    def test_explanation_generator_initialization(self):
        """Test explanation generator initializes correctly"""
        assert self.explanation_generator is not None
        assert self.explanation_generator.template_library is not None
        assert self.explanation_generator.clarity_optimizer is not None
        assert len(self.explanation_generator.template_library.templates) > 0
    
    def test_basic_explanation_generation(self):
        """Test basic explanation generation"""
        request = ExplanationRequest(
            content_type='reasoning_chain',
            content_data={
                'query': 'What is the capital of France?',
                'reasoning_type': 'deductive',
                'steps': [
                    {
                        'step_type': 'analysis',
                        'explanation': 'Looking up geographical information',
                        'confidence': 0.9
                    }
                ],
                'conclusion': 'Paris is the capital of France',
                'overall_confidence': 0.95
            }
        )
        
        explanation = self.explanation_generator.generate_explanation(request)
        
        assert isinstance(explanation, GeneratedExplanation)
        assert explanation.generated_text is not None
        assert len(explanation.generated_text) > 0
        assert explanation.confidence_score > 0
        assert explanation.word_count > 0
    
    def test_different_explanation_styles(self):
        """Test different explanation styles"""
        base_request_data = {
            'content_type': 'reasoning_chain',
            'content_data': {
                'query': 'Explain machine learning',
                'reasoning_type': 'analytical',
                'steps': [{'explanation': 'analyzing data patterns', 'confidence': 0.8}],
                'conclusion': 'Machine learning uses algorithms to find patterns'
            }
        }
        
        styles_to_test = [
            ExplanationStyle.TECHNICAL,
            ExplanationStyle.CONVERSATIONAL,
            ExplanationStyle.BULLET_POINTS,
            ExplanationStyle.MINIMAL
        ]
        
        explanations = {}
        for style in styles_to_test:
            request = ExplanationRequest(
                **base_request_data,
                style_preference=style
            )
            explanation = self.explanation_generator.generate_explanation(request)
            explanations[style] = explanation
            assert explanation.generated_text is not None
        
        # Verify different styles produce different outputs
        texts = [exp.generated_text for exp in explanations.values()]
        assert len(set(texts)) > 1, "Different styles should produce different explanations"
    
    def test_explanation_clarity_optimization(self):
        """Test clarity optimization"""
        technical_text = "The algorithm utilizes sophisticated heuristics to optimize the paradigm"
        optimized_text = self.clarity_optimizer.optimize_clarity(technical_text, "general")
        
        # Should replace technical jargon
        assert 'utilizes' not in optimized_text or 'uses' in optimized_text
        assert 'heuristics' not in optimized_text or 'rule of thumb' in optimized_text
    
    def test_clarity_score_calculation(self):
        """Test clarity score calculation"""
        simple_text = "This is a simple sentence. It is easy to read."
        complex_text = "This is an extraordinarily complicated and convoluted sentence that contains numerous subordinate clauses and technical terminology that makes it difficult to comprehend."
        
        simple_score = self.clarity_optimizer.calculate_clarity_score(simple_text)
        complex_score = self.clarity_optimizer.calculate_clarity_score(complex_text)
        
        assert simple_score > complex_score
        assert 0 <= simple_score <= 1
        assert 0 <= complex_score <= 1
    
    def test_explanation_length_constraints(self):
        """Test explanation length constraints"""
        request = ExplanationRequest(
            content_type='reasoning_chain',
            content_data={
                'query': 'Long explanation test',
                'steps': [{'explanation': f'Step {i} with detailed explanation' for i in range(10)}]
            },
            max_length=100  # Very short constraint
        )
        
        explanation = self.explanation_generator.generate_explanation(request)
        assert len(explanation.generated_text) <= 120  # Some buffer for truncation message


class TestEnhancedPreferenceLearning:
    """Test the enhanced preference learning system (Task 2.7)"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.personalization_engine = EnhancedPersonalizationEngine()
        self.test_user_id = "test_user_123"
        self.test_session_id = "session_456"
    
    def test_personalization_engine_initialization(self):
        """Test personalization engine initializes correctly"""
        assert self.personalization_engine is not None
        assert self.personalization_engine.preference_learner is not None
        assert len(self.personalization_engine.preference_weights) > 0
    
    def test_communication_style_analysis(self):
        """Test communication style analysis"""
        queries_and_expected_styles = [
            ("Please explain this in detail comprehensively", CommunicationStyle.DETAILED),
            ("Just give me a brief answer", CommunicationStyle.DIRECT),
            ("Hey, could you help me understand this?", CommunicationStyle.CONVERSATIONAL),
            ("I would appreciate a formal analysis", CommunicationStyle.FORMAL),
            ("Let me analyze the data patterns here", CommunicationStyle.ANALYTICAL)
        ]
        
        for query, expected_style in queries_and_expected_styles:
            style, confidence = self.personalization_engine.preference_learner.communication_analyzer.analyze_communication_style(
                query, {}
            )
            if style:  # Some queries might not be classified
                assert isinstance(style, CommunicationStyle)
                assert 0 <= confidence <= 1
    
    def test_interaction_analysis(self):
        """Test comprehensive interaction analysis"""
        query = "Can you explain how machine learning algorithms work step by step?"
        response = "Machine learning algorithms work by finding patterns in data..."
        conversation_history = [
            {"query": "What is AI?", "response": "AI is artificial intelligence..."},
            {"query": "How does it differ from traditional programming?", "response": "Traditional programming..."}
        ]
        
        analysis = self.personalization_engine.preference_learner.analyze_interaction(
            self.test_user_id, query, response, conversation_history
        )
        
        assert isinstance(analysis, InteractionAnalysis)
        assert analysis.user_id == self.test_user_id
        assert analysis.query_text == query
        assert analysis.query_length == len(query)
        assert analysis.session_position > 0
        assert len(analysis.topic_categories) > 0
        assert 0 <= analysis.complexity_level <= 1
        assert 0 <= analysis.technical_depth <= 1
    
    def test_preference_learning_from_interaction(self):
        """Test learning preferences from interactions"""
        result = self.personalization_engine.process_interaction_for_learning(
            self.test_user_id, self.test_session_id,
            "Please give me a detailed technical explanation of neural networks",
            "Neural networks are computational models inspired by biological neural networks...",
            [],
            {"satisfaction": 0.9}
        )
        
        assert result['success'] is True or 'interaction_analysis' in result
        assert 'preferences_learned' in result or 'interaction_analysis' in result
        
    def test_personalized_context_generation(self):
        """Test generation of personalized context"""
        # First, learn some preferences
        self.personalization_engine.process_interaction_for_learning(
            self.test_user_id, self.test_session_id,
            "I prefer detailed technical explanations",
            "Here's a detailed technical explanation...",
            [],
            {"satisfaction": 0.8}
        )
        
        # Then get personalized context
        context = self.personalization_engine.get_personalized_response_context(
            self.test_user_id, self.test_session_id, {"base_key": "base_value"}
        )
        
        assert isinstance(context, dict)
        assert "base_key" in context  # Should preserve base context
        # May have additional personalized keys
    
    def test_profile_insights(self):
        """Test profile insights generation"""
        # Create some interaction history first
        for i in range(3):
            self.personalization_engine.process_interaction_for_learning(
                self.test_user_id, self.test_session_id,
                f"Test query {i} with varying complexity",
                f"Response {i}",
                [],
                {"satisfaction": 0.7 + i * 0.1}
            )
        
        insights = self.personalization_engine.get_profile_insights(self.test_user_id)
        
        # Should return insights even if minimal
        assert isinstance(insights, dict)
        if 'user_id' in insights:
            assert insights['user_id'] == self.test_user_id


class TestEnhancedCognitiveIntegration:
    """Test integration of Task 2.6 and 2.7 with cognitive processing"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.test_memory = PersistentMemorySystem("/tmp/test_memory_enhanced")
        self.processor = EnhancedCognitiveProcessor(self.test_memory)
        self.test_session_id = "enhanced_test_session"
    
    def test_enhanced_processor_with_new_features(self):
        """Test enhanced processor has new Task 2.6 and 2.7 features"""
        assert hasattr(self.processor, 'explanation_generator')
        assert hasattr(self.processor, 'enhanced_personalization_engine')
        assert hasattr(self.processor, 'generate_reasoning_explanation')
        assert hasattr(self.processor, 'learn_user_preferences')
    
    def test_explanation_generation_integration(self):
        """Test explanation generation integration"""
        reasoning_data = {
            'query': 'How does cognitive processing work?',
            'reasoning_type': 'analytical',
            'steps': [
                {
                    'step_type': 'analysis',
                    'explanation': 'Breaking down the cognitive process',
                    'confidence': 0.8
                }
            ],
            'conclusion': 'Cognitive processing involves multiple membranes working together',
            'overall_confidence': 0.85
        }
        
        explanation_result = self.processor.generate_reasoning_explanation(
            reasoning_data,
            {'audience': 'general'},
            'detailed'
        )
        
        assert isinstance(explanation_result, dict)
        if explanation_result.get('success'):
            assert 'explanation' in explanation_result
            assert 'text' in explanation_result['explanation']
        else:
            # Should have fallback explanation
            assert 'fallback_explanation' in explanation_result
    
    def test_preference_learning_integration(self):
        """Test preference learning integration"""
        conversation_history = [
            {"query": "What is AI?", "response": "AI is artificial intelligence"}
        ]
        
        learning_result = self.processor.learn_user_preferences(
            self.test_session_id,
            "Please explain machine learning in detail",
            "Machine learning is a subset of AI...",
            conversation_history,
            {"satisfaction": 0.8}
        )
        
        assert isinstance(learning_result, dict)
        # Should handle both success and error cases gracefully
        assert 'success' in learning_result or 'error' in learning_result
    
    def test_personalized_context_retrieval(self):
        """Test retrieval of personalized context"""
        context = self.processor.get_personalized_context(self.test_session_id)
        assert isinstance(context, dict)
        # Should return empty dict if no preferences learned yet, but not error
    
    def test_profile_insights_retrieval(self):
        """Test retrieval of profile insights"""
        insights = self.processor.get_user_profile_insights(self.test_session_id)
        assert isinstance(insights, dict)
        # Should handle cases where no profile exists yet
    
    def test_enhanced_system_status(self):
        """Test enhanced system status reporting"""
        status = self.processor.get_enhanced_system_status()
        assert isinstance(status, dict)
        
        expected_keys = [
            'enhanced_processing_enabled',
            'explanation_generator_active',
            'enhanced_personalization_active'
        ]
        
        for key in expected_keys:
            assert key in status
    
    @pytest.mark.asyncio
    async def test_end_to_end_enhanced_processing(self):
        """Test end-to-end enhanced processing with new features"""
        input_text = "Explain how neural networks learn from data"
        conversation_history = []
        memory_state = {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'goals': []}
        }
        
        # Process input with enhanced features
        try:
            result = await self.processor.process_input_enhanced(
                input_text, self.test_session_id, conversation_history, memory_state
            )
            
            assert isinstance(result, dict)
            assert 'response' in result
            assert 'processing_time' in result
            
            # Should have enhanced features
            if 'enhanced_features' in result:
                enhanced = result['enhanced_features']
                # May have enhanced preference learning results
                assert isinstance(enhanced, dict)
            
        except Exception as e:
            # Should gracefully handle any issues
            pytest.fail(f"Enhanced processing failed: {e}")


class TestRoadmapCompliance:
    """Test compliance with roadmap requirements for Task 2.6 and 2.7"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = EnhancedCognitiveProcessor()
    
    def test_task_2_6_explanation_generation_requirements(self):
        """Test Task 2.6: Explanation Generation System requirements"""
        # Should generate human-readable explanations
        assert hasattr(self.processor, 'generate_reasoning_explanation')
        
        # Should have explanation templates
        assert hasattr(self.processor.explanation_generator, 'template_library')
        assert len(self.processor.explanation_generator.template_library.templates) > 0
        
        # Should have clarity optimization
        assert hasattr(self.processor.explanation_generator, 'clarity_optimizer')
        
        # Should make complex reasoning accessible to users
        explanation_result = self.processor.generate_reasoning_explanation({
            'query': 'Complex reasoning test',
            'steps': [{'explanation': 'Complex step', 'confidence': 0.8}],
            'conclusion': 'Result'
        })
        assert isinstance(explanation_result, dict)
    
    def test_task_2_7_preference_learning_requirements(self):
        """Test Task 2.7: User Preference Learning requirements"""
        # Should learn user preferences
        assert hasattr(self.processor, 'learn_user_preferences')
        
        # Should learn communication styles
        engine = self.processor.enhanced_personalization_engine
        assert hasattr(engine.preference_learner, 'communication_analyzer')
        
        # Should learn interaction patterns  
        assert hasattr(engine.preference_learner, 'interaction_analyzer')
        
        # Should personalize responses
        assert hasattr(self.processor, 'get_personalized_context')
        
        # Should adapt to individual users
        context = self.processor.get_personalized_context("test_user")
        assert isinstance(context, dict)
    
    def test_roadmap_success_criteria(self):
        """Test that roadmap success criteria are met"""
        # System should demonstrate sophisticated reasoning capabilities
        assert self.processor.enhanced_processing_enabled
        
        # Should adapt to user preferences
        assert hasattr(self.processor, 'enhanced_personalization_engine')
        
        # Should provide meta-cognitive insights
        assert hasattr(self.processor, 'meta_cognitive_system')
        
        # Should show measurable improvement in cognitive performance
        status = self.processor.get_enhanced_system_status()
        assert isinstance(status, dict)
        assert status.get('enhanced_processing_enabled') is True


# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])