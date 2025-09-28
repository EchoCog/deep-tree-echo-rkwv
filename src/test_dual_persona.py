#!/usr/bin/env python3
"""
Test Dual Persona System - Deep Tree Echo & Marduk
Tests the complementary dual-persona cognitive architecture.
"""

import os
import sys
import time
import logging
# import pytest
from datetime import datetime
from typing import Dict, Any, List

# Add src directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dual_persona_kernel import (
    DualPersonaProcessor, PersonaKernel, PersonaType, PersonaTraitType,
    PersonaTrait, PersonaResponse, DualPersonaResponse, 
    format_dual_response_for_ui, create_dual_persona_processor
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TestPersonaKernel:
    """Test individual persona functionality"""
    
    def test_deep_tree_echo_initialization(self):
        """Test Deep Tree Echo persona initialization"""
        persona = PersonaKernel(PersonaType.DEEP_TREE_ECHO)
        
        assert persona.persona_type == PersonaType.DEEP_TREE_ECHO
        assert len(persona.traits) == 7  # All trait types
        
        # Check specific Deep Tree Echo trait strengths
        assert persona.traits[PersonaTraitType.CANOPY].strength >= 0.9  # High creativity
        assert persona.traits[PersonaTraitType.LEAVES].strength >= 0.85  # High expression
        assert persona.traits[PersonaTraitType.NETWORK].strength >= 0.8  # High collaboration
        
        logger.info("✓ Deep Tree Echo persona initialized correctly")
    
    def test_marduk_initialization(self):
        """Test Marduk Mad Scientist persona initialization"""
        persona = PersonaKernel(PersonaType.MARDUK_MAD_SCIENTIST)
        
        assert persona.persona_type == PersonaType.MARDUK_MAD_SCIENTIST
        assert len(persona.traits) == 7  # All trait types
        
        # Check specific Marduk trait strengths
        assert persona.traits[PersonaTraitType.BRANCHES].strength >= 0.9  # High reasoning
        assert persona.traits[PersonaTraitType.ROOTS].strength >= 0.85  # High systematic knowledge
        assert persona.traits[PersonaTraitType.CANOPY].strength >= 0.8  # High systematic innovation
        
        logger.info("✓ Marduk Mad Scientist persona initialized correctly")
    
    def test_trait_evolution(self):
        """Test trait evolution mechanism"""
        persona = PersonaKernel(PersonaType.DEEP_TREE_ECHO)
        original_strength = persona.traits[PersonaTraitType.CANOPY].strength
        
        # Evolve trait
        persona.traits[PersonaTraitType.CANOPY].evolve("Creative challenge", 0.1)
        
        new_strength = persona.traits[PersonaTraitType.CANOPY].strength
        assert new_strength > original_strength
        assert len(persona.traits[PersonaTraitType.CANOPY].evolution_history) == 1
        
        logger.info("✓ Trait evolution mechanism working correctly")
    
    def test_persona_response_generation(self):
        """Test persona response generation"""
        echo_persona = PersonaKernel(PersonaType.DEEP_TREE_ECHO)
        marduk_persona = PersonaKernel(PersonaType.MARDUK_MAD_SCIENTIST)
        
        test_query = "How can we solve this complex problem?"
        context = {'session_id': 'test_001'}
        
        # Test Deep Tree Echo response
        echo_response = echo_persona.process_query(test_query, context)
        assert isinstance(echo_response, PersonaResponse)
        assert echo_response.persona_type == PersonaType.DEEP_TREE_ECHO
        assert echo_response.confidence > 0
        assert len(echo_response.reasoning_process) > 0
        assert len(echo_response.content) > 0
        
        # Test Marduk response
        marduk_response = marduk_persona.process_query(test_query, context)
        assert isinstance(marduk_response, PersonaResponse)
        assert marduk_response.persona_type == PersonaType.MARDUK_MAD_SCIENTIST
        assert marduk_response.confidence > 0
        assert len(marduk_response.reasoning_process) > 0
        assert len(marduk_response.content) > 0
        
        logger.info("✓ Both personas generate valid responses")

class TestDualPersonaProcessor:
    """Test dual persona interaction and processing"""
    
    def test_dual_processor_initialization(self):
        """Test dual persona processor initialization"""
        processor = DualPersonaProcessor()
        
        assert processor.deep_tree_echo is not None
        assert processor.marduk is not None
        assert processor.deep_tree_echo.persona_type == PersonaType.DEEP_TREE_ECHO
        assert processor.marduk.persona_type == PersonaType.MARDUK_MAD_SCIENTIST
        assert len(processor.interaction_history) == 0
        
        logger.info("✓ Dual persona processor initialized correctly")
    
    def test_dual_query_processing(self):
        """Test processing query through both personas"""
        processor = DualPersonaProcessor()
        
        test_query = "What's the best approach to building a recursive memory system?"
        context = {
            'user_input': test_query,
            'conversation_history': [],
            'memory_state': {},
            'session_id': 'test_session'
        }
        session_id = 'test_session_001'
        
        # Process through dual persona system
        dual_response = processor.process_dual_query(test_query, context, session_id)
        
        # Validate response structure
        assert isinstance(dual_response, DualPersonaResponse)
        assert dual_response.session_id == session_id
        assert dual_response.deep_tree_echo_response is not None
        assert dual_response.marduk_response is not None
        assert dual_response.reflection_content is not None
        assert 0 <= dual_response.convergence_score <= 1
        assert dual_response.total_processing_time > 0
        
        # Validate persona responses
        echo_response = dual_response.deep_tree_echo_response
        marduk_response = dual_response.marduk_response
        
        assert echo_response.persona_type == PersonaType.DEEP_TREE_ECHO
        assert marduk_response.persona_type == PersonaType.MARDUK_MAD_SCIENTIST
        assert len(echo_response.content) > 0
        assert len(marduk_response.content) > 0
        
        # Check interaction history
        assert len(processor.interaction_history) == 1
        
        logger.info("✓ Dual query processing working correctly")
    
    def test_persona_convergence_calculation(self):
        """Test persona convergence scoring"""
        processor = DualPersonaProcessor()
        
        # Test with similar responses (should have high convergence)
        test_query = "Explain the concept of memory"
        context = {'session_id': 'test_convergence'}
        
        dual_response = processor.process_dual_query(test_query, context, 'test_conv_001')
        
        # Convergence should be a valid score
        assert 0 <= dual_response.convergence_score <= 1
        
        logger.info(f"✓ Convergence score calculated: {dual_response.convergence_score:.3f}")
    
    def test_reflection_generation(self):
        """Test reflection between personas"""
        processor = DualPersonaProcessor()
        
        test_query = "How do emotions and logic work together?"
        context = {'session_id': 'test_reflection'}
        
        dual_response = processor.process_dual_query(test_query, context, 'test_refl_001')
        
        reflection = dual_response.reflection_content
        assert reflection is not None
        assert len(reflection) > 0
        assert "Deep Tree Echo" in reflection
        assert "Marduk" in reflection
        
        logger.info("✓ Reflection generation working correctly")
    
    def test_ui_format_conversion(self):
        """Test conversion to UI format"""
        processor = DualPersonaProcessor()
        
        test_query = "Design a cognitive architecture"
        context = {'session_id': 'test_ui'}
        
        dual_response = processor.process_dual_query(test_query, context, 'test_ui_001')
        ui_format = format_dual_response_for_ui(dual_response)
        
        # Check UI format structure
        assert 'session_id' in ui_format
        assert 'responses' in ui_format
        assert 'deep_tree_echo' in ui_format['responses']
        assert 'marduk' in ui_format['responses']
        assert 'reflection' in ui_format
        assert 'convergence_score' in ui_format
        
        # Check response details
        echo_ui = ui_format['responses']['deep_tree_echo']
        marduk_ui = ui_format['responses']['marduk']
        
        assert 'content' in echo_ui
        assert 'confidence' in echo_ui
        assert 'reasoning' in echo_ui
        assert 'processing_time' in echo_ui
        
        assert 'content' in marduk_ui
        assert 'confidence' in marduk_ui
        assert 'reasoning' in marduk_ui
        assert 'processing_time' in marduk_ui
        
        logger.info("✓ UI format conversion working correctly")

class TestPersonaTraits:
    """Test persona trait system"""
    
    def test_trait_activation_calculation(self):
        """Test trait activation based on query content"""
        processor = DualPersonaProcessor()
        
        # Test creative query (should activate CANOPY)
        creative_query = "Let's create something innovative and new"
        context = {'session_id': 'test_creative'}
        
        dual_response = processor.process_dual_query(creative_query, context, 'test_creative_001')
        
        echo_activations = dual_response.deep_tree_echo_response.trait_activations
        marduk_activations = dual_response.marduk_response.trait_activations
        
        # Creative queries should activate CANOPY trait
        assert echo_activations[PersonaTraitType.CANOPY] > 0.5
        assert marduk_activations[PersonaTraitType.CANOPY] > 0.5
        
        logger.info("✓ Trait activation calculation working correctly")
    
    def test_memory_trait_activation(self):
        """Test memory-related trait activation"""
        processor = DualPersonaProcessor()
        
        memory_query = "Remember what we discussed about knowledge foundations"
        context = {'session_id': 'test_memory'}
        
        dual_response = processor.process_dual_query(memory_query, context, 'test_memory_001')
        
        echo_activations = dual_response.deep_tree_echo_response.trait_activations
        marduk_activations = dual_response.marduk_response.trait_activations
        
        # Memory queries should activate ROOTS trait
        assert echo_activations[PersonaTraitType.ROOTS] > 0.5
        assert marduk_activations[PersonaTraitType.ROOTS] > 0.5
        
        logger.info("✓ Memory trait activation working correctly")

class TestPersonaCharacteristics:
    """Test persona-specific characteristics and behavior"""
    
    def test_echo_empathetic_response(self):
        """Test Deep Tree Echo empathetic characteristics"""
        echo_persona = PersonaKernel(PersonaType.DEEP_TREE_ECHO)
        
        emotional_query = "I'm feeling overwhelmed by this complex situation"
        context = {'session_id': 'test_empathy'}
        
        response = echo_persona.process_query(emotional_query, context)
        
        # Check for empathetic language patterns
        content_lower = response.content.lower()
        empathetic_indicators = ['sense', 'feel', 'understand', 'connect', 'journey', 'patterns']
        
        empathy_score = sum(1 for indicator in empathetic_indicators if indicator in content_lower)
        assert empathy_score > 0, "Deep Tree Echo should show empathetic characteristics"
        
        logger.info("✓ Deep Tree Echo shows empathetic characteristics")
    
    def test_marduk_analytical_response(self):
        """Test Marduk analytical characteristics"""
        marduk_persona = PersonaKernel(PersonaType.MARDUK_MAD_SCIENTIST)
        
        technical_query = "How should we architect this system for scalability?"
        context = {'session_id': 'test_analytical'}
        
        response = marduk_persona.process_query(technical_query, context)
        
        # Check for analytical language patterns
        content_lower = response.content.lower()
        analytical_indicators = ['systematic', 'framework', 'architecture', 'logic', 'analysis', 'recursive']
        
        analytical_score = sum(1 for indicator in analytical_indicators if indicator in content_lower)
        assert analytical_score > 0, "Marduk should show analytical characteristics"
        
        logger.info("✓ Marduk shows analytical characteristics")

class TestIntegrationTests:
    """Integration tests for the complete dual persona system"""
    
    def test_factory_function(self):
        """Test factory function for creating processor"""
        processor = create_dual_persona_processor()
        
        assert isinstance(processor, DualPersonaProcessor)
        assert processor.deep_tree_echo is not None
        assert processor.marduk is not None
        
        logger.info("✓ Factory function working correctly")
    
    def test_persona_status_reporting(self):
        """Test persona status reporting"""
        processor = DualPersonaProcessor()
        
        # Generate some interactions first
        test_queries = [
            "What is creativity?",
            "How do logical systems work?",
            "Can you help me understand memory?"
        ]
        
        for i, query in enumerate(test_queries):
            context = {'session_id': f'test_status_{i}'}
            processor.process_dual_query(query, context, f'session_{i}')
        
        status = processor.get_persona_status()
        
        # Check status structure
        assert 'deep_tree_echo' in status
        assert 'marduk' in status
        assert 'interaction_history_count' in status
        assert 'avg_convergence' in status
        
        # Check persona details
        echo_status = status['deep_tree_echo']
        marduk_status = status['marduk']
        
        assert 'traits' in echo_status
        assert 'response_count' in echo_status
        assert echo_status['response_count'] == len(test_queries)
        
        assert 'traits' in marduk_status
        assert 'response_count' in marduk_status
        assert marduk_status['response_count'] == len(test_queries)
        
        assert status['interaction_history_count'] == len(test_queries)
        
        logger.info("✓ Persona status reporting working correctly")

def run_comprehensive_test():
    """Run comprehensive test suite"""
    print("="*80)
    print("DUAL PERSONA SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*80)
    
    test_classes = [
        TestPersonaKernel(),
        TestDualPersonaProcessor(),
        TestPersonaTraits(),
        TestPersonaCharacteristics(),
        TestIntegrationTests()
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n--- Running {class_name} ---")
        
        test_methods = [method for method in dir(test_class) if method.startswith('test_')]
        
        for test_method_name in test_methods:
            total_tests += 1
            try:
                test_method = getattr(test_class, test_method_name)
                test_method()
                passed_tests += 1
                print(f"✓ {test_method_name}")
            except Exception as e:
                print(f"✗ {test_method_name}: {e}")
                logger.error(f"Test {test_method_name} failed: {e}")
    
    print(f"\n{'='*80}")
    print(f"TEST RESULTS: {passed_tests}/{total_tests} tests passed")
    print(f"{'='*80}")
    
    if passed_tests == total_tests:
        print("🎉 All tests passed! Dual persona system is working correctly.")
        return True
    else:
        print(f"⚠️  {total_tests - passed_tests} tests failed. Please review the issues above.")
        return False

def run_interactive_demo():
    """Run interactive demonstration of dual persona system"""
    print("\n" + "="*80)
    print("DUAL PERSONA INTERACTIVE DEMONSTRATION")
    print("="*80)
    
    processor = create_dual_persona_processor()
    
    demo_queries = [
        "How can we build more empathetic AI systems?",
        "What's the most efficient way to process large datasets?",
        "How do creativity and logic work together in problem solving?",
        "Design a system that learns from human emotions",
        "What are the key principles of recursive cognitive architectures?"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        print(f"\n--- Demo Query {i} ---")
        print(f"Query: {query}")
        print("-" * 60)
        
        context = {'session_id': f'demo_{i}'}
        dual_response = processor.process_dual_query(query, context, f'demo_session_{i}')
        
        print(f"🌳 Deep Tree Echo (Right Hemisphere):")
        print(f"   Confidence: {dual_response.deep_tree_echo_response.confidence:.2f}")
        print(f"   Response: {dual_response.deep_tree_echo_response.content[:200]}...")
        
        print(f"\n🧬 Marduk the Mad Scientist (Left Hemisphere):")
        print(f"   Confidence: {dual_response.marduk_response.confidence:.2f}")
        print(f"   Response: {dual_response.marduk_response.content[:200]}...")
        
        print(f"\n🤝 Reflection & Integration:")
        print(f"   Convergence Score: {dual_response.convergence_score:.2f}")
        print(f"   Reflection: {dual_response.reflection_content[:150]}...")
        
        if dual_response.synthesis:
            print(f"   Synthesis: {dual_response.synthesis[:150]}...")
        
        print(f"\n⏱️  Processing Time: {dual_response.total_processing_time:.3f}s")
    
    # Show final system status
    print(f"\n--- System Status After Demo ---")
    status = processor.get_persona_status()
    print(f"Total Interactions: {status['interaction_history_count']}")
    print(f"Average Convergence: {status['avg_convergence']:.3f}")
    print(f"Deep Tree Echo Responses: {status['deep_tree_echo']['response_count']}")
    print(f"Marduk Responses: {status['marduk']['response_count']}")

if __name__ == "__main__":
    # Run comprehensive tests
    test_success = run_comprehensive_test()
    
    if test_success:
        # If tests pass, run interactive demo
        run_interactive_demo()
    else:
        print("\n⚠️  Tests failed. Skipping interactive demo.")
        sys.exit(1)