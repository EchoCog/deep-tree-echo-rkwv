"""
Test suite for Research and Innovation Features
Tests experimental models, data collection, AI integration, and innovation testing
"""

import asyncio
import time
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from research.experimental_models import (
    AlternativeReasoningEngine, 
    ExperimentalMemoryArchitecture,
    CognitiveModelComparator,
    ReasoningAlgorithm,
    MemoryArchitectureType
)

from research.data_collection import (
    ResearchDataCollector,
    CognitiveBenchmark,
    InteractionAnalyzer,
    PatternRecognizer,
    DataPrivacyLevel,
    BenchmarkType,
    InteractionType
)

from research.ai_integration import (
    MultiModelAIIntegrator,
    ModelComparator,
    HybridCognitiveArchitecture,
    AIOptimizer,
    MockLanguageModel,
    MockVisionModel,
    AIModelConfig,
    AIModelType,
    OptimizationMethod
)

from research.innovation_testing import (
    ABTestingFramework,
    FeatureFlags,
    InnovationMetrics,
    ExperimentTracker,
    ExperimentVariant,
    UserSegment,
    MetricType
)

class MockCognitiveSystem:
    """Mock cognitive system for testing"""
    def process(self, input_data):
        return {
            'response': f"Cognitive analysis of: {input_data.get('query', 'unknown')}",
            'confidence': 0.8,
            'reasoning_steps': 3
        }

def test_experimental_models():
    """Test experimental cognitive models"""
    print("Testing Experimental Cognitive Models...")
    
    # Test Alternative Reasoning Engine
    reasoning_config = {'generations': 3, 'population_size': 5}
    reasoning_engine = AlternativeReasoningEngine(
        model_id="test_reasoning",
        algorithm=ReasoningAlgorithm.EVOLUTIONARY,
        config=reasoning_config
    )
    
    test_input = {'query': 'What is the meaning of intelligence?'}
    result = reasoning_engine.process(test_input)
    
    assert 'algorithm_used' in result
    assert result.algorithm_used == 'evolutionary'
    assert result.processing_time > 0
    print(f"✓ Reasoning engine test passed: {result.confidence_score:.2f} confidence")
    
    # Test Experimental Memory Architecture
    memory_config = {'num_nodes': 3, 'consensus_threshold': 0.6}
    memory_arch = ExperimentalMemoryArchitecture(
        model_id="test_memory",
        architecture_type=MemoryArchitectureType.DISTRIBUTED_CONSENSUS,
        config=memory_config
    )
    
    store_input = {
        'operation': 'store',
        'content': 'Test knowledge about AI systems',
        'memory_type': 'semantic'
    }
    
    store_result = memory_arch.process(store_input)
    assert 'algorithm_used' in store_result
    assert store_result.memory_operations > 0
    print(f"✓ Memory architecture test passed: {store_result.processing_time:.3f}s")
    
    # Test Model Comparator
    comparator = CognitiveModelComparator()
    comparator.register_model(reasoning_engine)
    comparator.register_model(memory_arch)
    
    test_inputs = [
        {'query': 'Test query 1'},
        {'operation': 'retrieve', 'query': 'Test query 2'}
    ]
    
    comparison = comparator.compare_models(test_inputs)
    assert 'models_compared' in comparison['summary']
    assert len(comparison['detailed_results']) == 2
    print(f"✓ Model comparison test passed: {comparison['summary']['models_compared']} models compared")

def test_data_collection():
    """Test research data collection system"""
    print("\nTesting Research Data Collection...")
    
    # Test Research Data Collector
    collector_config = {
        'privacy_level': 'anonymous',
        'enabled': True,
        'data_store': 'memory'
    }
    
    collector = ResearchDataCollector(collector_config)
    
    # Collect some test data
    test_data = {
        'query': 'What is machine learning?',
        'response': 'Machine learning is a type of AI...',
        'user_action': 'query_submitted'
    }
    
    data_id = collector.collect_interaction_data(
        interaction_type=InteractionType.QUERY_RESPONSE,
        raw_data=test_data,
        session_id='test_session_123',
        performance_metrics={'latency': 0.5, 'confidence': 0.8}
    )
    
    assert data_id is not None
    collected_data = collector.get_collected_data()
    assert len(collected_data) == 1
    print(f"✓ Data collection test passed: {len(collected_data)} data points collected")
    
    # Test Cognitive Benchmark
    benchmark_config = {}
    benchmark = CognitiveBenchmark(benchmark_config)
    
    # Create mock cognitive system for testing
    mock_system = MockCognitiveSystem()
    
    benchmark_result = benchmark.run_benchmark(
        benchmark_type=BenchmarkType.REASONING_SPEED,
        cognitive_system=mock_system
    )
    
    assert benchmark_result.benchmark_type == 'reasoning_speed'
    assert benchmark_result.performance_score >= 0
    print(f"✓ Benchmark test passed: {benchmark_result.performance_score:.2f} score")
    
    # Test Interaction Analyzer
    analyzer_config = {'buffer_size': 1000}
    analyzer = InteractionAnalyzer(analyzer_config)
    
    # Record some interactions
    for i in range(5):
        analyzer.record_interaction(
            interaction_type=InteractionType.QUERY_RESPONSE,
            interaction_data={'query': f'Test query {i}', 'response': f'Response {i}'},
            timestamp=datetime.now()
        )
    
    patterns = analyzer.analyze_patterns(min_frequency=2)
    report = analyzer.generate_interaction_report()
    
    assert 'total_interactions' in report['analysis_period']
    print(f"✓ Interaction analysis test passed: {len(patterns)} patterns detected")

async def test_ai_integration():
    """Test AI integration system"""
    print("\nTesting AI Integration...")
    
    # Create mock AI models
    language_config = AIModelConfig(
        model_id="mock_language",
        model_type=AIModelType.LANGUAGE_MODEL,
        model_name="Mock Language Model",
        api_endpoint=None,
        parameters={'temperature': 0.7},
        capabilities=['text_generation', 'question_answering'],
        performance_metrics={'latency': 0.2, 'accuracy': 0.8},
        cost_per_request=0.01,
        latency_ms=200
    )
    
    vision_config = AIModelConfig(
        model_id="mock_vision",
        model_type=AIModelType.VISION_MODEL,
        model_name="Mock Vision Model",
        api_endpoint=None,
        parameters={'confidence_threshold': 0.5},
        capabilities=['object_detection', 'scene_classification'],
        performance_metrics={'latency': 0.3, 'accuracy': 0.75},
        cost_per_request=0.02,
        latency_ms=300
    )
    
    language_model = MockLanguageModel(language_config)
    vision_model = MockVisionModel(vision_config)
    
    # Test Multi-Model AI Integrator
    integrator_config = {}
    integrator = MultiModelAIIntegrator(integrator_config)
    
    integrator.register_model(language_model)
    integrator.register_model(vision_model)
    
    # Test integration
    test_input = {'query': 'Describe what you see in this image', 'task_type': 'multimodal'}
    result = await integrator.process_with_integration(test_input, 'multimodal')
    
    assert 'response' in result
    assert 'integration_metadata' in result
    print(f"✓ AI integration test passed: {result.get('confidence', 0):.2f} confidence")
    
    # Test Model Comparator
    comparator_config = {}
    comparator = ModelComparator(comparator_config)
    
    comparison_result = await comparator.compare_models(
        model_list=[language_model, vision_model],
        test_suite_name='language_understanding'
    )
    
    assert comparison_result.models_compared == ['mock_language', 'mock_vision']
    assert 'best_model_overall' in comparison_result.__dict__
    print(f"✓ Model comparison test passed: {comparison_result.best_model_overall} is best")
    
    # Test Hybrid Cognitive Architecture
    mock_cognitive = MockCognitiveSystem()
    hybrid_arch = HybridCognitiveArchitecture(
        cognitive_system=mock_cognitive,
        ai_integrator=integrator,
        config={}
    )
    
    hybrid_result = await hybrid_arch.process_hybrid(
        input_data={'query': 'What is the future of AI?'},
        fusion_strategy='adaptive_fusion'
    )
    
    assert hybrid_result.response_id is not None
    assert hybrid_result.confidence_score >= 0
    print(f"✓ Hybrid architecture test passed: {hybrid_result.fusion_strategy} fusion")

def test_innovation_testing():
    """Test innovation testing environment"""
    print("\nTesting Innovation Testing Environment...")
    
    # Test Feature Flags
    flags_config = {}
    feature_flags = FeatureFlags(flags_config)
    
    # Create a test flag
    flag = feature_flags.create_flag(
        flag_id='test_feature',
        name='Test Feature',
        description='A test feature flag',
        default_value=False,
        enabled_segments=[UserSegment.BETA_USERS],
        rollout_percentage=50.0
    )
    
    # Test flag evaluation
    user_context = {'user_id': 'test_user', 'user_type': 'beta'}
    flag_result = feature_flags.evaluate_flag('test_feature', user_context)
    
    # Get flag statistics
    flag_stats = feature_flags.get_flag_statistics()
    assert 'total_flags' in flag_stats
    print(f"✓ Feature flags test passed: {flag_stats['total_flags']} flags created")
    
    # Test A/B Testing Framework
    ab_config = {'minimum_sample_size': 10, 'statistical_significance': 0.05}
    ab_framework = ABTestingFramework(ab_config)
    
    # Create test variants
    variants = [
        ExperimentVariant('control', 'Control', 'Original version', {}, 50.0, is_control=True),
        ExperimentVariant('variant_a', 'Variant A', 'New version', {'feature': 'enabled'}, 50.0)
    ]
    
    # Create experiment
    experiment_id = ab_framework.create_experiment(
        name='Test A/B Experiment',
        description='Testing new feature',
        variants=variants
    )
    
    # Start experiment
    started = ab_framework.start_experiment(experiment_id)
    assert started
    
    # Simulate some results
    for i in range(20):
        user_id = f'user_{i}'
        user_context = {'user_id': user_id, 'session_id': f'session_{i}'}
        
        variant_id = ab_framework.assign_user_to_variant(experiment_id, user_context)
        if variant_id:
            ab_framework.record_result(
                experiment_id=experiment_id,
                variant_id=variant_id,
                user_id=user_id,
                session_id=f'session_{i}',
                metric_values={'conversion_rate': 1.0 if i % 3 == 0 else 0.0, 'engagement_time': i * 10},
                conversion_event=i % 3 == 0
            )
    
    # Analyze experiment
    analysis = ab_framework.analyze_experiment(experiment_id)
    assert analysis is not None
    assert analysis.experiment_id == experiment_id
    print(f"✓ A/B testing test passed: {len(analysis.sample_sizes)} variants analyzed")
    
    # Test Innovation Metrics
    metrics_config = {}
    innovation_metrics = InnovationMetrics(metrics_config)
    
    # Record some metrics
    for i in range(10):
        innovation_metrics.record_innovation_metric(
            innovation_id='test_innovation',
            metric_type='performance_score',
            value=70 + i * 2,  # Improving trend
            context={'iteration': i}
        )
    
    # Calculate innovation score
    innovation_score = innovation_metrics.calculate_innovation_score('test_innovation')
    assert 'overall_score' in innovation_score
    print(f"✓ Innovation metrics test passed: {innovation_score['overall_score']:.1f} score")
    
    # Test Experiment Tracker
    tracker_config = {}
    tracker = ExperimentTracker(tracker_config)
    
    # Create and track an experiment
    exp_id = tracker.create_ab_test_experiment(
        experiment_name='Tracked Test Experiment',
        variants=[
            {'id': 'control', 'name': 'Control', 'traffic': 50, 'is_control': True},
            {'id': 'test', 'name': 'Test', 'traffic': 50, 'is_control': False}
        ]
    )
    
    # Start the experiment
    started = tracker.start_experiment(exp_id)
    assert started
    
    # Record some results
    tracker.record_experiment_result(
        experiment_id=exp_id,
        user_context={'user_id': 'test_user', 'session_id': 'test_session'},
        metrics={'conversion': True, 'engagement_time': 120}
    )
    
    # Get dashboard
    dashboard = tracker.get_experiment_dashboard()
    assert 'total_experiments' in dashboard
    print(f"✓ Experiment tracker test passed: {dashboard['total_experiments']} experiments tracked")

async def run_all_tests():
    """Run all research and innovation tests"""
    print("=" * 60)
    print("RESEARCH AND INNOVATION FEATURES - TEST SUITE")
    print("=" * 60)
    
    try:
        # Test experimental models
        test_experimental_models()
        
        # Test data collection
        test_data_collection()
        
        # Test AI integration (async)
        await test_ai_integration()
        
        # Test innovation testing
        test_innovation_testing()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS PASSED SUCCESSFULLY!")
        print("Research and Innovation Features are working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run the test suite
    success = asyncio.run(run_all_tests())
    
    if not success:
        exit(1)