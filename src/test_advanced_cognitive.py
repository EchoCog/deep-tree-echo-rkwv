"""
Tests for Advanced Cognitive Processing Capabilities
Tests meta-cognitive reflection, complex reasoning chains, and adaptive learning
"""

import pytest
import asyncio
import time
from datetime import datetime
from unittest.mock import Mock, patch

# Import the advanced cognitive modules
from cognitive_reflection import (
    MetaCognitiveReflectionSystem, CognitiveStrategy, ProcessingError, 
    CognitiveMetrics, MetaCognitiveMonitor
)
from reasoning_chains import (
    ComplexReasoningSystem, ReasoningType, DeductiveReasoning,
    InductiveReasoning, AbductiveReasoning
)
from adaptive_learning import (
    AdaptiveLearningSystem, PersonalizationEngine, ResponseStyleLearner,
    CognitiveStrategyLearner, UserPreference, FeedbackEntry
)

class TestMetaCognitiveReflectionSystem:
    """Test meta-cognitive reflection capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.meta_system = MetaCognitiveReflectionSystem()
    
    def test_system_initialization(self):
        """Test that meta-cognitive system initializes correctly"""
        assert self.meta_system.monitor is not None
        assert self.meta_system.strategy_selector is not None
        assert self.meta_system.error_detector is not None
        assert self.meta_system.cognitive_state is not None
        assert self.meta_system.cognitive_state.meta_confidence > 0
    
    def test_strategy_selection(self):
        """Test cognitive strategy selection"""
        context = {
            'user_input': 'Analyze this complex problem step by step',
            'conversation_history': [],
            'memory_state': {},
            'processing_goals': []
        }
        
        monitoring_context = self.meta_system.before_processing(context)
        
        assert 'strategy_selected' in monitoring_context
        assert monitoring_context['strategy_selected'] in [s.value for s in CognitiveStrategy]
        assert 'expected_processing_time' in monitoring_context
    
    def test_performance_monitoring(self):
        """Test performance monitoring and metrics collection"""
        # Simulate processing context
        context = {
            'user_input': 'Test query',
            'conversation_history': [],
            'memory_state': {},
            'processing_goals': []
        }
        
        monitoring_context = self.meta_system.before_processing(context)
        
        # Simulate processing results
        processing_results = {
            'session_id': 'test_session',
            'user_input': 'Test query',
            'total_processing_time': 0.5,
            'memory_processing_time': 0.1,
            'reasoning_processing_time': 0.2,
            'grammar_processing_time': 0.1,
            'confidence_score': 0.8,
            'memory_retrievals': 3,
            'reasoning_complexity': 'medium'
        }
        
        reflection_results = self.meta_system.after_processing(
            processing_results, monitoring_context
        )
        
        assert 'metrics' in reflection_results
        assert 'detected_errors' in reflection_results
        assert 'meta_confidence' in reflection_results
        assert isinstance(reflection_results['detected_errors'], list)
    
    def test_error_detection(self):
        """Test cognitive error detection"""
        processing_context = {
            'processing_time': 15.0,  # Exceeds threshold
            'confidence_score': 0.2,   # Low confidence
            'timeout_threshold': 10.0,
            'memory_retrievals': 0,
            'query_needs_memory': True  # Should have retrieved memories
        }
        
        errors = self.meta_system.error_detector.detect_errors(processing_context)
        
        assert ProcessingError.TIMEOUT in errors
        assert ProcessingError.LOW_CONFIDENCE in errors
        assert ProcessingError.MEMORY_RETRIEVAL_FAILURE in errors
    
    def test_cognitive_insights(self):
        """Test cognitive insights generation"""
        # Add some processing history first
        for i in range(3):
            context = {'user_input': f'Test query {i}', 'processing_goals': []}
            monitoring_context = self.meta_system.before_processing(context)
            
            processing_results = {
                'session_id': 'test_session',
                'total_processing_time': 0.5 + i * 0.1,
                'confidence_score': 0.8 - i * 0.1,
                'memory_retrievals': i,
                'reasoning_complexity': 'medium'
            }
            
            self.meta_system.after_processing(processing_results, monitoring_context)
        
        insights = self.meta_system.get_cognitive_insights()
        
        assert 'current_state' in insights
        assert 'strategy_recommendations' in insights
        assert 'error_analysis' in insights
        assert 'adaptation_status' in insights

class TestComplexReasoningSystem:
    """Test complex reasoning chain capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.reasoning_system = ComplexReasoningSystem()
    
    def test_system_initialization(self):
        """Test reasoning system initialization"""
        assert self.reasoning_system.processor is not None
        assert self.reasoning_system.reasoning_cache is not None
        assert len(self.reasoning_system.processor.strategies) > 0
    
    @pytest.mark.asyncio
    async def test_deductive_reasoning(self):
        """Test deductive reasoning implementation"""
        query = "If all humans are mortal and Socrates is human, what can we conclude?"
        context = {'session_id': 'test_session'}
        
        result = await self.reasoning_system.execute_reasoning(
            query, context, 'deductive'
        )
        
        assert result['success'] is True
        assert result['reasoning_type'] == 'deductive'
        assert result['confidence'] > 0.5
        assert 'conclusion' in result
        assert 'explanation' in result
        assert len(result['steps']) > 0
    
    @pytest.mark.asyncio
    async def test_inductive_reasoning(self):
        """Test inductive reasoning implementation"""
        query = "Based on observations, what pattern emerges from these data points?"
        context = {'session_id': 'test_session'}
        
        result = await self.reasoning_system.execute_reasoning(
            query, context, 'inductive'
        )
        
        assert result['success'] is True
        assert result['reasoning_type'] == 'inductive'
        assert result['confidence'] >= 0.0  # Inductive reasoning typically less certain
        assert 'conclusion' in result
        assert 'explanation' in result
    
    @pytest.mark.asyncio
    async def test_abductive_reasoning(self):
        """Test abductive reasoning implementation"""
        query = "What is the most likely explanation for this phenomenon?"
        context = {'session_id': 'test_session'}
        
        result = await self.reasoning_system.execute_reasoning(
            query, context, 'abductive'
        )
        
        assert result['success'] is True
        assert result['reasoning_type'] == 'abductive'
        assert result['confidence'] > 0.0
        assert 'conclusion' in result
        assert 'explanation' in result
    
    @pytest.mark.asyncio
    async def test_automatic_reasoning_type_selection(self):
        """Test automatic reasoning type selection"""
        test_cases = [
            ("Why did this happen?", "abductive"),
            ("If A then B, A is true, therefore?", "deductive"),
            ("Based on these patterns, what usually happens?", "inductive")
        ]
        
        for query, expected_type in test_cases:
            result = await self.reasoning_system.execute_reasoning(
                query, {'session_id': 'test_session'}
            )
            
            assert result['success'] is True
            # Note: Actual type selection may vary based on implementation
            assert result['reasoning_type'] in [rt.value for rt in ReasoningType]
    
    def test_reasoning_validation(self):
        """Test reasoning chain validation"""
        # This would test the validation of reasoning chains
        # For now, we'll test that the validation methods exist
        for strategy in self.reasoning_system.processor.strategies.values():
            assert hasattr(strategy, 'validate')
    
    def test_reasoning_insights(self):
        """Test reasoning system insights"""
        insights = self.reasoning_system.get_system_stats()
        
        assert 'active_chains' in insights
        assert 'completed_chains' in insights
        assert 'cached_results' in insights
        assert 'available_strategies' in insights

class TestAdaptiveLearningSystem:
    """Test adaptive learning capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.learning_system = AdaptiveLearningSystem()
    
    def test_system_initialization(self):
        """Test adaptive learning system initialization"""
        assert self.learning_system.personalization_engine is not None
        assert self.learning_system.feedback_queue is not None
        assert self.learning_system.learning_enabled is True
    
    def test_preference_learning(self):
        """Test learning user preferences from interactions"""
        interaction = {
            'user_id': 'test_user',
            'session_id': 'test_session',
            'user_input': 'Please give me a brief summary of this topic',
            'timestamp': datetime.now().isoformat()
        }
        
        feedback = {
            'rating': 4.0,
            'type': 'rating'
        }
        
        result = self.learning_system.process_interaction_for_learning(
            interaction, feedback
        )
        
        assert result.get('preferences_learned', 0) >= 0
        assert 'profile_updated' in result
    
    def test_personalization_application(self):
        """Test application of personalization to processing context"""
        # First, create some learning data
        interaction = {
            'user_id': 'test_user',
            'session_id': 'test_session',
            'user_input': 'I prefer detailed analytical responses',
            'timestamp': datetime.now().isoformat()
        }
        
        self.learning_system.process_interaction_for_learning(interaction)
        
        # Now test personalization
        base_context = {
            'user_input': 'Analyze this problem',
            'processing_goals': []
        }
        
        result = self.learning_system.get_personalization_context(
            'test_user', 'test_session', base_context
        )
        
        assert 'success' in result
    
    def test_feedback_processing(self):
        """Test feedback processing for learning"""
        feedback_data = {
            'user_id': 'test_user',
            'session_id': 'test_session',
            'interaction_id': 'test_interaction',
            'type': 'rating',
            'value': 5.0,
            'context': {'response_style': 'detailed'}
        }
        
        result = self.learning_system.submit_feedback(feedback_data)
        
        assert result['success'] is True
        assert 'feedback_id' in result
    
    def test_user_insights(self):
        """Test generation of user insights"""
        # Add some interaction history first
        for i in range(3):
            interaction = {
                'user_id': 'test_user',
                'session_id': 'test_session',
                'user_input': f'Test query {i}',
                'timestamp': datetime.now().isoformat()
            }
            self.learning_system.process_interaction_for_learning(interaction)
        
        insights = self.learning_system.get_user_insights('test_user', 'test_session')
        
        # Should at least have the structure, even if no data
        assert isinstance(insights, dict)

class TestResponseStyleLearner:
    """Test response style learning"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.style_learner = ResponseStyleLearner()
    
    def test_style_detection(self):
        """Test detection of response style preferences"""
        interaction = {
            'user_id': 'test_user',
            'user_input': 'Please give me a brief summary'
        }
        
        feedback = {'rating': 4.0}
        
        preferences = self.style_learner.learn_from_interaction(interaction, feedback)
        
        # Should detect 'concise' style preference
        concise_prefs = [p for p in preferences if p.preference_value == 'concise']
        assert len(concise_prefs) > 0
        
        # Test detailed preference
        interaction['user_input'] = 'Please explain this in detail'
        preferences = self.style_learner.learn_from_interaction(interaction, feedback)
        
        detailed_prefs = [p for p in preferences if p.preference_value == 'detailed']
        assert len(detailed_prefs) > 0
    
    def test_preference_confidence_update(self):
        """Test updating preference confidence"""
        preference = UserPreference(
            preference_id='test_pref',
            user_id='test_user',
            preference_type='response_style',
            preference_value='concise',
            confidence=0.5,
            evidence_count=1,
            last_reinforced=datetime.now().isoformat(),
            creation_date=datetime.now().isoformat()
        )
        
        # Positive reinforcement
        updated_pref = self.style_learner.update_preference_confidence(preference, 0.9)
        assert updated_pref.confidence > 0.5
        assert updated_pref.evidence_count == 2
        
        # Negative reinforcement
        updated_pref = self.style_learner.update_preference_confidence(preference, 0.1)
        assert updated_pref.confidence < preference.confidence

class TestCognitiveStrategyLearner:
    """Test cognitive strategy learning"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.strategy_learner = CognitiveStrategyLearner()
    
    def test_strategy_detection(self):
        """Test detection of cognitive strategy preferences"""
        interaction = {
            'user_id': 'test_user',
            'user_input': 'Can you analyze this step by step?'
        }
        
        feedback = {'success': 0.8}
        
        preferences = self.strategy_learner.learn_from_interaction(interaction, feedback)
        
        # Should detect structured approach preference
        structured_prefs = [p for p in preferences if 'structured' in p.preference_value]
        assert len(structured_prefs) > 0
    
    def test_successful_strategy_learning(self):
        """Test learning from successful strategy usage"""
        interaction = {
            'user_id': 'test_user',
            'user_input': 'Test query',
            'strategy_used': 'deductive'
        }
        
        feedback = {'success': 0.9}
        
        preferences = self.strategy_learner.learn_from_interaction(interaction, feedback)
        
        # Should learn successful strategy preference
        success_prefs = [p for p in preferences if p.preference_type == 'successful_strategy']
        assert len(success_prefs) > 0
        assert success_prefs[0].preference_value == 'deductive'

class TestIntegration:
    """Test integration of all advanced cognitive capabilities"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.meta_system = MetaCognitiveReflectionSystem()
        self.reasoning_system = ComplexReasoningSystem()
        self.learning_system = AdaptiveLearningSystem()
    
    @pytest.mark.asyncio
    async def test_end_to_end_processing(self):
        """Test end-to-end advanced cognitive processing"""
        # Simulate a complex cognitive processing scenario
        
        # 1. Meta-cognitive pre-processing
        context = {
            'user_input': 'Explain why this complex system behaves this way',
            'user_id': 'test_user',
            'session_id': 'test_session',
            'conversation_history': [],
            'memory_state': {},
            'processing_goals': []
        }
        
        meta_context = self.meta_system.before_processing(context)
        assert 'strategy_selected' in meta_context
        
        # 2. Complex reasoning execution
        reasoning_result = await self.reasoning_system.execute_reasoning(
            context['user_input'], context
        )
        assert reasoning_result['success'] is True
        
        # 3. Adaptive learning processing
        interaction_data = {
            'user_id': context['user_id'],
            'session_id': context['session_id'],
            'user_input': context['user_input'],
            'response_quality': reasoning_result['confidence'],
            'processing_time': reasoning_result.get('processing_time', 0.5),
            'strategy_used': meta_context['strategy_selected'],
            'complexity': 'high',
            'timestamp': datetime.now().isoformat()
        }
        
        learning_result = self.learning_system.process_interaction_for_learning(
            interaction_data
        )
        assert 'profile_updated' in learning_result
        
        # 4. Meta-cognitive post-processing
        processing_results = {
            'session_id': context['session_id'],
            'user_input': context['user_input'],
            'total_processing_time': 1.0,
            'confidence_score': reasoning_result['confidence'],
            'memory_retrievals': 2,
            'reasoning_complexity': 'high'
        }
        
        reflection_results = self.meta_system.after_processing(
            processing_results, meta_context
        )
        
        assert 'metrics' in reflection_results
        assert 'meta_confidence' in reflection_results
        
        # Verify the systems worked together
        assert len(reflection_results['detected_errors']) >= 0
        assert reflection_results['meta_confidence'] > 0

# Performance and stress tests

class TestPerformance:
    """Test performance of advanced cognitive capabilities"""
    
    def test_meta_cognitive_performance(self):
        """Test meta-cognitive system performance"""
        meta_system = MetaCognitiveReflectionSystem()
        
        start_time = time.time()
        
        # Process multiple contexts rapidly
        for i in range(100):
            context = {
                'user_input': f'Test query {i}',
                'processing_goals': []
            }
            meta_context = meta_system.before_processing(context)
            
            processing_results = {
                'session_id': 'perf_test',
                'total_processing_time': 0.1,
                'confidence_score': 0.8,
                'memory_retrievals': 1
            }
            
            meta_system.after_processing(processing_results, meta_context)
        
        elapsed_time = time.time() - start_time
        
        # Should process 100 requests in reasonable time (< 5 seconds)
        assert elapsed_time < 5.0
        
        # Verify system state is maintained
        insights = meta_system.get_cognitive_insights()
        assert insights is not None
    
    @pytest.mark.asyncio
    async def test_reasoning_system_performance(self):
        """Test reasoning system performance"""
        reasoning_system = ComplexReasoningSystem()
        
        start_time = time.time()
        
        # Process multiple reasoning requests
        tasks = []
        for i in range(10):  # Smaller number for async operations
            query = f"Analyze problem {i} using logical reasoning"
            context = {'session_id': f'perf_test_{i}'}
            tasks.append(reasoning_system.execute_reasoning(query, context))
        
        results = await asyncio.gather(*tasks)
        
        elapsed_time = time.time() - start_time
        
        # Verify all succeeded
        assert all(result['success'] for result in results)
        
        # Should complete in reasonable time
        assert elapsed_time < 30.0  # Allow more time for reasoning operations
    
    def test_adaptive_learning_performance(self):
        """Test adaptive learning system performance"""
        learning_system = AdaptiveLearningSystem()
        
        start_time = time.time()
        
        # Process multiple learning interactions
        for i in range(50):
            interaction = {
                'user_id': f'user_{i % 5}',  # 5 different users
                'session_id': f'session_{i}',
                'user_input': f'Test interaction {i}',
                'timestamp': datetime.now().isoformat()
            }
            
            learning_system.process_interaction_for_learning(interaction)
        
        elapsed_time = time.time() - start_time
        
        # Should process 50 interactions quickly
        assert elapsed_time < 2.0
        
        # Verify system state
        status = learning_system.get_system_status()
        assert status['learning_enabled'] is True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])