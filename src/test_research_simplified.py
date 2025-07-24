"""
Simplified test for Research and Innovation Features
Tests core functionality without heavy dependencies
"""

import time
import sys
import os
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

def test_basic_research_framework():
    """Test basic research framework functionality"""
    print("Testing Basic Research Framework...")
    
    # Test that modules can be imported
    try:
        from research import experimental_models, data_collection, ai_integration, innovation_testing
        print("✓ All research modules imported successfully")
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    
    # Test basic functionality without numpy dependencies
    try:
        # Test enum classes
        from research.experimental_models import ReasoningAlgorithm, MemoryArchitectureType
        from research.data_collection import DataPrivacyLevel, BenchmarkType, InteractionType
        from research.ai_integration import AIModelType, IntegrationStrategy, OptimizationMethod
        from research.innovation_testing import ExperimentType, ExperimentStatus, UserSegment, MetricType
        
        print("✓ All enums and constants defined correctly")
        
        # Test basic classes can be instantiated
        from research.innovation_testing import FeatureFlag, UserSegment
        
        flag = FeatureFlag(
            flag_id="test_flag",
            name="Test Flag", 
            description="A test flag",
            default_value=False,
            enabled_segments=[UserSegment.ALL_USERS],
            rollout_percentage=50.0
        )
        
        user_context = {"user_id": "test_user", "user_type": "regular"}
        result = flag.evaluate(user_context)
        
        print(f"✓ Feature flag test passed: result={result}")
        
        # Test basic data structures
        from research.experimental_models import ExperimentalResult
        from research.data_collection import AnonymizedDataPoint
        from research.ai_integration import AIModelConfig, AIModelType
        from research.innovation_testing import ExperimentVariant
        
        # Create test data structures
        exp_result = ExperimentalResult(
            algorithm_used="test_algorithm",
            processing_time=0.1,
            confidence_score=0.8,
            result_data={"test": "data"},
            memory_operations=1,
            reasoning_steps=3,
            error_log=[]
        )
        
        ai_config = AIModelConfig(
            model_id="test_model",
            model_type=AIModelType.LANGUAGE_MODEL,
            model_name="Test Model",
            api_endpoint=None,
            parameters={},
            capabilities=["test"],
            performance_metrics={},
            cost_per_request=0.01,
            latency_ms=100
        )
        
        variant = ExperimentVariant(
            variant_id="test_variant",
            name="Test Variant",
            description="A test variant",
            feature_config={},
            traffic_percentage=50.0,
            is_control=True
        )
        
        print("✓ All data structures created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_research_framework_integration():
    """Test integration with existing cognitive system"""
    print("\nTesting Research Framework Integration...")
    
    try:
        # Test that research framework can work with existing components
        from research.data_collection import ResearchDataCollector, InteractionType
        from research.innovation_testing import FeatureFlags, UserSegment
        
        # Create data collector
        collector_config = {
            'privacy_level': 'anonymous',
            'enabled': True,
            'data_store': 'memory'
        }
        
        collector = ResearchDataCollector(collector_config)
        
        # Simulate collecting cognitive interaction data
        test_data = {
            'query': 'What is deep tree echo?',
            'response': 'Deep Tree Echo is a cognitive architecture...',
            'session_info': 'test_session'
        }
        
        data_id = collector.collect_interaction_data(
            interaction_type=InteractionType.QUERY_RESPONSE,
            raw_data=test_data,
            session_id='test_session_456'
        )
        
        assert data_id is not None
        print("✓ Data collection integration test passed")
        
        # Test feature flag for cognitive features
        feature_flags = FeatureFlags({})
        
        cognitive_flag = feature_flags.create_flag(
            flag_id='enhanced_reasoning',
            name='Enhanced Reasoning',
            description='Enable enhanced reasoning algorithms',
            default_value=False,
            enabled_segments=[UserSegment.RESEARCHERS, UserSegment.DEVELOPERS],
            rollout_percentage=25.0
        )
        
        # Test flag evaluation for different user types
        researcher_context = {'user_id': 'researcher_1', 'user_type': 'researcher'}
        regular_context = {'user_id': 'user_1', 'user_type': 'regular'}
        
        researcher_result = feature_flags.evaluate_flag('enhanced_reasoning', researcher_context)
        regular_result = feature_flags.evaluate_flag('enhanced_reasoning', regular_context)
        
        print(f"✓ Feature flag integration test passed: researcher={researcher_result}, regular={regular_result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_innovation_metrics():
    """Test innovation metrics without dependencies"""
    print("\nTesting Innovation Metrics...")
    
    try:
        from research.innovation_testing import InnovationMetrics
        
        metrics_config = {}
        innovation_metrics = InnovationMetrics(metrics_config)
        
        # Record some test metrics
        test_metrics = [
            ('cognitive_performance', 75.0),
            ('response_accuracy', 0.85),
            ('processing_speed', 120.0),
            ('user_satisfaction', 4.2),
            ('error_rate', 0.05)
        ]
        
        for metric_type, value in test_metrics:
            innovation_metrics.record_innovation_metric(
                innovation_id='test_cognitive_feature',
                metric_type=metric_type,
                value=value,
                context={'test': True}
            )
        
        # Set baseline metrics
        baseline = {
            'cognitive_performance': 70.0,
            'response_accuracy': 0.80,
            'processing_speed': 150.0,
            'user_satisfaction': 4.0,
            'error_rate': 0.08
        }
        
        innovation_metrics.set_baseline_metrics(baseline)
        
        # Calculate innovation score
        score_data = innovation_metrics.calculate_innovation_score('test_cognitive_feature')
        
        assert 'overall_score' in score_data
        assert score_data['overall_score'] >= 0
        print(f"✓ Innovation metrics test passed: score={score_data['overall_score']:.1f}")
        
        # Generate report
        report = innovation_metrics.generate_innovation_report()
        assert 'total_metrics' in report['summary']
        print(f"✓ Innovation report generated: {report['summary']['total_metrics']} metrics")
        
        return True
        
    except Exception as e:
        print(f"❌ Innovation metrics test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_simplified_tests():
    """Run simplified test suite"""
    print("=" * 60)
    print("RESEARCH AND INNOVATION FEATURES - SIMPLIFIED TEST SUITE")
    print("=" * 60)
    
    tests_passed = 0
    total_tests = 3
    
    # Test 1: Basic Framework
    if test_basic_research_framework():
        tests_passed += 1
    
    # Test 2: Integration
    if test_research_framework_integration():
        tests_passed += 1
    
    # Test 3: Innovation Metrics
    if test_innovation_metrics():
        tests_passed += 1
    
    print("\n" + "=" * 60)
    if tests_passed == total_tests:
        print("✅ ALL SIMPLIFIED TESTS PASSED!")
        print("Research and Innovation Features core functionality is working.")
    else:
        print(f"⚠️  {tests_passed}/{total_tests} tests passed")
        print("Some features may need additional dependencies or fixes.")
    
    print("=" * 60)
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = run_simplified_tests()
    
    if not success:
        exit(1)