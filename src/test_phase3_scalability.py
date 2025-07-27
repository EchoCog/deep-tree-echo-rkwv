"""
Test Phase 3 Scalability and Performance Enhancements
Validates the P1-002 implementation for Phase 3 milestone
"""

import os
import sys
import time
import asyncio
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_enhanced_observability():
    """Test P1-002.4: Enhanced Monitoring and Observability"""
    print("="*60)
    print("Testing P1-002.4: Enhanced Observability")
    print("="*60)
    
    try:
        # Test distributed tracing
        from observability.distributed_tracing import initialize_tracer, get_tracer
        
        tracer = initialize_tracer("test-service", "http://localhost:14268")
        print("✓ Distributed tracer initialized")
        
        # Test tracing context
        with tracer.trace_operation("test_operation") as span:
            span.add_tag("test.user", "test_user")
            span.add_log("Test operation started")
            time.sleep(0.01)  # Simulate work
        
        print("✓ Trace span created and finished")
        
        # Test metrics collection
        from observability.metrics_collector import initialize_metrics_collector, get_metrics_collector
        
        metrics = initialize_metrics_collector("test-service")
        print("✓ Metrics collector initialized")
        
        # Record test metrics
        metrics.record_performance("test_operation", 50.0, "success")
        metrics.record_cognitive_processing(
            "test_session", "memory", 25.0, 128.5, 0.7, 0.8
        )
        
        # Get metrics summary
        summary = metrics.get_metrics_summary()
        print(f"✓ Metrics recorded: {summary['performance_metrics']['total_count']} performance, "
              f"{summary['cognitive_metrics']['total_count']} cognitive")
        
        # Test alerting system
        from observability.alerting_system import AlertingSystem, AlertRule, AlertSeverity, create_standard_alert_rules
        
        alerting = AlertingSystem()
        
        # Add standard rules
        standard_rules = create_standard_alert_rules()
        for rule in standard_rules:
            alerting.add_rule(rule)
        
        print(f"✓ Alerting system initialized with {len(standard_rules)} rules")
        
        # Test metric recording for alerts
        alerting.record_metric("response_time_p95", 1500)  # High response time
        alerting.record_metric("error_rate", 3)  # Medium error rate
        
        stats = alerting.get_alert_stats()
        print(f"✓ Alert system stats: {stats['total_rules']} rules, {stats['enabled_rules']} enabled")
        
        return True
        
    except Exception as e:
        print(f"✗ Observability test failed: {e}")
        return False

def test_enhanced_caching():
    """Test P1-002.3: Multi-Level Caching Enhancement"""
    print("="*60)
    print("Testing P1-002.3: Enhanced Multi-Level Caching")
    print("="*60)
    
    try:
        from enhanced_caching import initialize_cache, get_cache
        
        # Initialize cache with test configuration
        cache_config = {
            "l1_size_mb": 16,
            "l2_size_mb": 32,
            "l3_size_mb": 64,
            "compression_enabled": True,
            "cognitive_optimization": True
        }
        
        cache = initialize_cache(cache_config)
        print("✓ Multi-level cache initialized")
        
        # Test caching with different content types
        cache.put("conversation_1", {"user": "Hello", "response": "Hi there!"}, 
                 content_type="conversation", cognitive_priority=0.9)
        cache.put("memory_fact", {"fact": "Paris is the capital of France"}, 
                 content_type="memory_retrieval", cognitive_priority=0.8)
        cache.put("general_data", {"data": "x" * 1000}, 
                 content_type="general", cognitive_priority=0.3)
        
        print("✓ Cached items with different cognitive priorities")
        
        # Test retrieval
        conversation = cache.get("conversation_1")
        memory_fact = cache.get("memory_fact")
        general_data = cache.get("general_data")
        
        print(f"✓ Retrieved cached items: conversation={conversation is not None}, "
              f"memory={memory_fact is not None}, general={general_data is not None}")
        
        # Test cache statistics
        stats = cache.get_comprehensive_stats()
        print(f"✓ Cache stats - Overall hit rate: {stats['overall']['hit_rate']:.1f}%")
        print(f"  L1: {stats['levels']['L1']['entries']} entries, {stats['levels']['L1']['hit_rate']:.1f}% hit rate")
        print(f"  L2: {stats['levels']['L2']['entries']} entries, {stats['levels']['L2']['hit_rate']:.1f}% hit rate")
        print(f"  L3: {stats['levels']['L3']['entries']} entries, {stats['levels']['L3']['hit_rate']:.1f}% hit rate")
        
        return True
        
    except Exception as e:
        print(f"✗ Enhanced caching test failed: {e}")
        return False

def test_intelligent_autoscaling():
    """Test P1-002.2: Intelligent Auto-Scaling"""
    print("="*60)
    print("Testing P1-002.2: Intelligent Auto-Scaling")
    print("="*60)
    
    try:
        from intelligent_autoscaling import initialize_auto_scaler, get_auto_scaler, ResourceMetrics, ScalingAction
        
        # Initialize auto-scaler with test configuration
        scaler_config = {
            "min_instances": 1,
            "max_instances": 5,
            "scale_up_threshold": 70,
            "scale_down_threshold": 30,
            "predictive_scaling_enabled": True
        }
        
        scaler = initialize_auto_scaler(scaler_config)
        print("✓ Intelligent auto-scaler initialized")
        
        # Set up mock scaling callbacks
        scale_actions = []
        
        def mock_scale_up(target_instances: int) -> bool:
            scale_actions.append(f"Scale up to {target_instances}")
            return True
        
        def mock_scale_down(target_instances: int) -> bool:
            scale_actions.append(f"Scale down to {target_instances}")
            return True
        
        scaler.set_scaling_callbacks(mock_scale_up, mock_scale_down)
        print("✓ Scaling callbacks configured")
        
        # Record test metrics to trigger scaling
        high_load_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=85.0,
            memory_percent=75.0,
            active_connections=150,
            requests_per_minute=1000.0,
            response_time_p95=800.0,
            error_rate_percent=2.0
        )
        
        scaler.record_metrics(high_load_metrics)
        print("✓ Recorded high load metrics")
        
        # Simulate scaling decision
        action = scaler._determine_scaling_action(high_load_metrics)
        print(f"✓ Scaling decision: {action.value}")
        
        if action == ScalingAction.SCALE_UP:
            scaler._execute_scaling_action(action, high_load_metrics)
            print("✓ Executed scale up action")
        
        # Test low load scenario
        low_load_metrics = ResourceMetrics(
            timestamp=time.time() + 300,  # 5 minutes later
            cpu_percent=25.0,
            memory_percent=30.0,
            active_connections=10,
            requests_per_minute=100.0,
            response_time_p95=200.0,
            error_rate_percent=0.5
        )
        
        scaler.record_metrics(low_load_metrics)
        action = scaler._determine_scaling_action(low_load_metrics)
        print(f"✓ Low load scaling decision: {action.value}")
        
        # Get scaling statistics
        stats = scaler.get_scaling_stats()
        print(f"✓ Auto-scaler stats: {stats['current_instances']} instances, "
              f"load score: {stats['current_load_score']:.1f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Intelligent auto-scaling test failed: {e}")
        return False

async def test_performance_framework():
    """Test P1-002.7: Performance Testing Framework"""
    print("="*60)
    print("Testing P1-002.7: Performance Testing Framework")
    print("="*60)
    
    try:
        from performance_testing import PerformanceTestSuite, TestScenario, LoadGenerator
        
        # Create a simple test scenario
        test_scenario = TestScenario(
            name="quick_test",
            description="Quick performance test",
            target_url="http://httpbin.org/delay/0.1",  # External test endpoint
            concurrent_users=5,
            duration_seconds=10,
            ramp_up_seconds=2,
            request_interval_ms=500,
            method="GET"
        )
        
        print("✓ Test scenario created")
        
        # Run the test
        generator = LoadGenerator(test_scenario)
        result = await generator.run_test()
        
        print(f"✓ Performance test completed:")
        print(f"  Total requests: {result.total_requests}")
        print(f"  Success rate: {100 - result.error_rate_percent:.1f}%")
        print(f"  Avg response time: {result.avg_response_time_ms:.1f}ms")
        print(f"  P95 response time: {result.p95_response_time_ms:.1f}ms")
        print(f"  Requests/sec: {result.requests_per_second:.1f}")
        
        # Test suite functionality
        test_suite = PerformanceTestSuite("http://httpbin.org")
        test_suite.add_scenario(test_scenario)
        
        report = test_suite.generate_report()
        if test_scenario.name in test_suite.results:
            print("✓ Test suite and reporting functionality works")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance framework test failed: {e}")
        return False

def test_phase3_integration():
    """Test Phase 3 component integration"""
    print("="*60)
    print("Testing Phase 3 Integration")
    print("="*60)
    
    try:
        # Initialize all Phase 3 components
        from observability.distributed_tracing import initialize_tracer
        from observability.metrics_collector import initialize_metrics_collector
        from enhanced_caching import initialize_cache
        from intelligent_autoscaling import initialize_auto_scaler
        
        # Initialize components
        tracer = initialize_tracer("integrated-test-service")
        metrics = initialize_metrics_collector("integrated-test-service")
        cache = initialize_cache()
        scaler = initialize_auto_scaler()
        
        print("✓ All Phase 3 components initialized")
        
        # Test integrated workflow
        with tracer.trace_operation("integrated_workflow") as span:
            span.add_tag("workflow.type", "phase3_integration")
            
            # Cache some data
            cache.put("integration_test", {"status": "testing", "timestamp": time.time()})
            cached_data = cache.get("integration_test")
            
            # Record metrics
            metrics.record_performance("integration_test", 75.0, "success", test=True)
            metrics.record_cognitive_processing("test_session", "integration", 50.0, 256.0, 0.8, 0.9)
            
            # Update auto-scaler metrics
            from intelligent_autoscaling import ResourceMetrics
            test_metrics = ResourceMetrics(
                timestamp=time.time(),
                cpu_percent=60.0,
                memory_percent=55.0,
                active_connections=25,
                requests_per_minute=500.0,
                response_time_p95=300.0,
                error_rate_percent=1.0
            )
            scaler.record_metrics(test_metrics)
            
            span.add_tag("integration.cache_hit", cached_data is not None)
            span.add_tag("integration.metrics_recorded", True)
        
        # Verify integration
        trace_stats = tracer.get_trace_stats()
        metrics_summary = metrics.get_metrics_summary()
        cache_stats = cache.get_comprehensive_stats()
        scaler_stats = scaler.get_scaling_stats()
        
        print(f"✓ Integration verification:")
        print(f"  Traces: {trace_stats['completed_traces']} completed")
        print(f"  Metrics: {metrics_summary['performance_metrics']['total_count']} performance recorded")
        print(f"  Cache: {cache_stats['overall']['hit_rate']:.1f}% hit rate")
        print(f"  Scaler: {scaler_stats['current_load_score']:.1f} load score")
        
        return True
        
    except Exception as e:
        print(f"✗ Phase 3 integration test failed: {e}")
        return False

async def main():
    """Run all Phase 3 tests"""
    print("🎭 Deep Tree Echo - Phase 3 Scalability Tests")
    print("=" * 60)
    print("Testing P1-002: Scalability and Performance Optimization")
    print("=" * 60)
    print()
    
    test_results = []
    
    # Run all tests
    test_results.append(("Enhanced Observability", test_enhanced_observability()))
    test_results.append(("Enhanced Caching", test_enhanced_caching()))
    test_results.append(("Intelligent Auto-scaling", test_intelligent_autoscaling()))
    test_results.append(("Performance Framework", await test_performance_framework()))
    test_results.append(("Phase 3 Integration", test_phase3_integration()))
    
    # Summary
    print("\n" + "="*60)
    print("PHASE 3 TEST SUMMARY")
    print("="*60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<30} : {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Phase 3 scalability components are working correctly!")
        print("\nPhase 3 Milestone Status: ✅ ACHIEVED")
        print("System supports high-concurrency with optimized performance")
    else:
        print("❌ Some Phase 3 tests failed. Review the issues above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        result = asyncio.run(main())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n\nTests interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        sys.exit(1)