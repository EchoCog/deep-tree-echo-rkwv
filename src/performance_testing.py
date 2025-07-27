"""
Performance Testing Framework for Phase 3 Scalability
Comprehensive load testing and scalability validation for P1-002.7
"""

import os
import time
import asyncio
import aiohttp
import threading
import logging
import statistics
import json
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from collections import deque
import random

logger = logging.getLogger(__name__)

@dataclass
class TestScenario:
    """Definition of a performance test scenario"""
    name: str
    description: str
    target_url: str
    concurrent_users: int
    duration_seconds: int
    ramp_up_seconds: int = 30
    ramp_down_seconds: int = 30
    request_interval_ms: int = 1000
    test_data: Dict[str, Any] = field(default_factory=dict)
    headers: Dict[str, str] = field(default_factory=dict)
    method: str = "GET"

@dataclass
class TestResult:
    """Result of a single request"""
    timestamp: float
    response_time_ms: float
    status_code: int
    success: bool
    error_message: Optional[str] = None
    response_size_bytes: int = 0
    user_id: int = 0

@dataclass
class ScenarioResults:
    """Aggregated results for a test scenario"""
    scenario_name: str
    start_time: float
    end_time: float
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    p50_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    requests_per_second: float
    error_rate_percent: float
    throughput_mb_per_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

class LoadGenerator:
    """Generates load for performance testing"""
    
    def __init__(self, scenario: TestScenario):
        self.scenario = scenario
        self.results: List[TestResult] = []
        self.active_users = 0
        self.start_time = 0
        self.stop_event = asyncio.Event()
        
    async def run_test(self) -> ScenarioResults:
        """Run the load test scenario"""
        logger.info(f"Starting load test: {self.scenario.name}")
        logger.info(f"Target: {self.scenario.target_url}")
        logger.info(f"Users: {self.scenario.concurrent_users}, Duration: {self.scenario.duration_seconds}s")
        
        self.start_time = time.time()
        self.stop_event.clear()
        
        # Start user tasks
        tasks = []
        
        # Ramp up phase
        if self.scenario.ramp_up_seconds > 0:
            ramp_interval = self.scenario.ramp_up_seconds / self.scenario.concurrent_users
            for user_id in range(self.scenario.concurrent_users):
                task = asyncio.create_task(self._user_simulation(user_id, user_id * ramp_interval))
                tasks.append(task)
        else:
            for user_id in range(self.scenario.concurrent_users):
                task = asyncio.create_task(self._user_simulation(user_id, 0))
                tasks.append(task)
        
        # Wait for test duration
        await asyncio.sleep(self.scenario.duration_seconds)
        
        # Stop all users
        self.stop_event.set()
        
        # Wait for ramp down
        if self.scenario.ramp_down_seconds > 0:
            await asyncio.sleep(self.scenario.ramp_down_seconds)
        
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        
        # Wait for all tasks to complete
        await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        
        # Analyze results
        return self._analyze_results(end_time)
    
    async def _user_simulation(self, user_id: int, start_delay: float):
        """Simulate a single user's behavior"""
        # Wait for ramp-up delay
        if start_delay > 0:
            await asyncio.sleep(start_delay)
        
        self.active_users += 1
        
        # Create session for this user
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            
            while not self.stop_event.is_set():
                try:
                    # Make request
                    start_time = time.time()
                    
                    if self.scenario.method.upper() == "GET":
                        async with session.get(
                            self.scenario.target_url,
                            headers=self.scenario.headers
                        ) as response:
                            content = await response.read()
                            result = TestResult(
                                timestamp=start_time,
                                response_time_ms=(time.time() - start_time) * 1000,
                                status_code=response.status,
                                success=response.status < 400,
                                response_size_bytes=len(content),
                                user_id=user_id
                            )
                    
                    elif self.scenario.method.upper() == "POST":
                        async with session.post(
                            self.scenario.target_url,
                            headers=self.scenario.headers,
                            json=self.scenario.test_data
                        ) as response:
                            content = await response.read()
                            result = TestResult(
                                timestamp=start_time,
                                response_time_ms=(time.time() - start_time) * 1000,
                                status_code=response.status,
                                success=response.status < 400,
                                response_size_bytes=len(content),
                                user_id=user_id
                            )
                    
                    else:
                        raise ValueError(f"Unsupported method: {self.scenario.method}")
                    
                    self.results.append(result)
                    
                    if not result.success:
                        logger.warning(f"Request failed: {result.status_code}")
                
                except Exception as e:
                    error_result = TestResult(
                        timestamp=start_time,
                        response_time_ms=(time.time() - start_time) * 1000,
                        status_code=0,
                        success=False,
                        error_message=str(e),
                        user_id=user_id
                    )
                    self.results.append(error_result)
                    logger.error(f"Request error: {e}")
                
                # Wait before next request
                if self.scenario.request_interval_ms > 0:
                    await asyncio.sleep(self.scenario.request_interval_ms / 1000)
        
        self.active_users -= 1
    
    def _analyze_results(self, end_time: float) -> ScenarioResults:
        """Analyze test results and generate summary"""
        if not self.results:
            return ScenarioResults(
                scenario_name=self.scenario.name,
                start_time=self.start_time,
                end_time=end_time,
                total_requests=0,
                successful_requests=0,
                failed_requests=0,
                avg_response_time_ms=0,
                p50_response_time_ms=0,
                p95_response_time_ms=0,
                p99_response_time_ms=0,
                min_response_time_ms=0,
                max_response_time_ms=0,
                requests_per_second=0,
                error_rate_percent=0,
                throughput_mb_per_sec=0
            )
        
        successful_results = [r for r in self.results if r.success]
        failed_results = [r for r in self.results if not r.success]
        
        response_times = [r.response_time_ms for r in self.results]
        successful_response_times = [r.response_time_ms for r in successful_results]
        
        total_duration = end_time - self.start_time
        total_bytes = sum(r.response_size_bytes for r in successful_results)
        
        # Calculate percentiles
        sorted_times = sorted(successful_response_times) if successful_response_times else [0]
        
        return ScenarioResults(
            scenario_name=self.scenario.name,
            start_time=self.start_time,
            end_time=end_time,
            total_requests=len(self.results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            avg_response_time_ms=statistics.mean(successful_response_times) if successful_response_times else 0,
            p50_response_time_ms=self._percentile(sorted_times, 50),
            p95_response_time_ms=self._percentile(sorted_times, 95),
            p99_response_time_ms=self._percentile(sorted_times, 99),
            min_response_time_ms=min(successful_response_times) if successful_response_times else 0,
            max_response_time_ms=max(successful_response_times) if successful_response_times else 0,
            requests_per_second=len(self.results) / total_duration if total_duration > 0 else 0,
            error_rate_percent=(len(failed_results) / len(self.results) * 100) if self.results else 0,
            throughput_mb_per_sec=(total_bytes / (1024 * 1024)) / total_duration if total_duration > 0 else 0
        )
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data"""
        if not sorted_data:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1] if int(index) + 1 < len(sorted_data) else lower
            return lower + (upper - lower) * (index - int(index))

class PerformanceTestSuite:
    """
    Comprehensive performance testing framework for Phase 3 scalability
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.scenarios: List[TestScenario] = []
        self.results: Dict[str, ScenarioResults] = {}
        
    def add_scenario(self, scenario: TestScenario):
        """Add a test scenario"""
        self.scenarios.append(scenario)
        logger.info(f"Added test scenario: {scenario.name}")
    
    def create_cognitive_test_scenarios(self):
        """Create standard test scenarios for cognitive architecture"""
        
        # Light load test
        self.add_scenario(TestScenario(
            name="light_load",
            description="Light load with 10 concurrent users",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=10,
            duration_seconds=60,
            ramp_up_seconds=10,
            request_interval_ms=2000,
            method="POST",
            test_data={
                "input": "What is the meaning of life?",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
        
        # Medium load test
        self.add_scenario(TestScenario(
            name="medium_load",
            description="Medium load with 50 concurrent users",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=50,
            duration_seconds=120,
            ramp_up_seconds=30,
            request_interval_ms=1000,
            method="POST",
            test_data={
                "input": "Explain quantum computing",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
        
        # High load test
        self.add_scenario(TestScenario(
            name="high_load",
            description="High load with 100 concurrent users",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=100,
            duration_seconds=180,
            ramp_up_seconds=60,
            request_interval_ms=500,
            method="POST",
            test_data={
                "input": "Solve this complex problem step by step",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
        
        # Stress test
        self.add_scenario(TestScenario(
            name="stress_test",
            description="Stress test with 200 concurrent users",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=200,
            duration_seconds=300,
            ramp_up_seconds=120,
            request_interval_ms=250,
            method="POST",
            test_data={
                "input": "Perform complex reasoning task",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
        
        # Spike test
        self.add_scenario(TestScenario(
            name="spike_test",
            description="Spike test with sudden load increase",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=500,
            duration_seconds=60,
            ramp_up_seconds=5,  # Rapid ramp-up
            ramp_down_seconds=5,
            request_interval_ms=100,
            method="POST",
            test_data={
                "input": "Quick response needed",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
        
        # Endurance test
        self.add_scenario(TestScenario(
            name="endurance_test",
            description="Endurance test with sustained load",
            target_url=f"{self.base_url}/api/cognitive_process",
            concurrent_users=30,
            duration_seconds=3600,  # 1 hour
            ramp_up_seconds=300,
            request_interval_ms=2000,
            method="POST",
            test_data={
                "input": "Long running cognitive task",
                "session_id": "test_session",
                "context": {"test": True}
            },
            headers={"Content-Type": "application/json"}
        ))
    
    async def run_all_scenarios(self) -> Dict[str, ScenarioResults]:
        """Run all test scenarios"""
        logger.info(f"Starting performance test suite with {len(self.scenarios)} scenarios")
        
        for scenario in self.scenarios:
            logger.info(f"Running scenario: {scenario.name}")
            
            generator = LoadGenerator(scenario)
            result = await generator.run_test()
            self.results[scenario.name] = result
            
            logger.info(f"Completed scenario: {scenario.name}")
            logger.info(f"Requests: {result.total_requests}, Success rate: {100 - result.error_rate_percent:.1f}%")
            logger.info(f"Avg response time: {result.avg_response_time_ms:.1f}ms, P95: {result.p95_response_time_ms:.1f}ms")
            
            # Wait between scenarios
            await asyncio.sleep(30)
        
        return self.results
    
    async def run_scenario(self, scenario_name: str) -> Optional[ScenarioResults]:
        """Run a specific scenario by name"""
        scenario = next((s for s in self.scenarios if s.name == scenario_name), None)
        if not scenario:
            logger.error(f"Scenario not found: {scenario_name}")
            return None
        
        generator = LoadGenerator(scenario)
        result = await generator.run_test()
        self.results[scenario_name] = result
        
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        # Aggregate statistics
        total_requests = sum(r.total_requests for r in self.results.values())
        total_successful = sum(r.successful_requests for r in self.results.values())
        avg_response_times = [r.avg_response_time_ms for r in self.results.values() if r.avg_response_time_ms > 0]
        p95_response_times = [r.p95_response_time_ms for r in self.results.values() if r.p95_response_time_ms > 0]
        
        summary = {
            "total_scenarios": len(self.results),
            "total_requests": total_requests,
            "total_successful_requests": total_successful,
            "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "avg_response_time_ms": statistics.mean(avg_response_times) if avg_response_times else 0,
            "avg_p95_response_time_ms": statistics.mean(p95_response_times) if p95_response_times else 0,
            "test_timestamp": datetime.now().isoformat()
        }
        
        # Detailed results
        detailed_results = {name: result.to_dict() for name, result in self.results.items()}
        
        # Performance assessment
        assessment = self._assess_performance()
        
        return {
            "summary": summary,
            "detailed_results": detailed_results,
            "assessment": assessment
        }
    
    def _assess_performance(self) -> Dict[str, Any]:
        """Assess overall performance and provide recommendations"""
        assessment = {
            "overall_grade": "A",
            "issues": [],
            "recommendations": [],
            "scalability_rating": "High"
        }
        
        # Check for common issues
        for name, result in self.results.items():
            if result.error_rate_percent > 5:
                assessment["issues"].append(f"High error rate in {name}: {result.error_rate_percent:.1f}%")
                assessment["overall_grade"] = "C" if assessment["overall_grade"] == "A" else "D"
            
            if result.p95_response_time_ms > 2000:
                assessment["issues"].append(f"High response times in {name}: P95 = {result.p95_response_time_ms:.1f}ms")
                assessment["overall_grade"] = "B" if assessment["overall_grade"] == "A" else assessment["overall_grade"]
            
            if result.requests_per_second < 10:
                assessment["issues"].append(f"Low throughput in {name}: {result.requests_per_second:.1f} req/s")
        
        # Generate recommendations
        if assessment["issues"]:
            assessment["recommendations"].extend([
                "Consider optimizing slow endpoints",
                "Review auto-scaling configuration",
                "Check resource utilization and bottlenecks",
                "Implement additional caching strategies"
            ])
        else:
            assessment["recommendations"].append("Performance looks good! Consider testing with higher loads.")
        
        # Determine scalability rating
        if len(assessment["issues"]) == 0:
            assessment["scalability_rating"] = "High"
        elif len(assessment["issues"]) <= 2:
            assessment["scalability_rating"] = "Medium"
        else:
            assessment["scalability_rating"] = "Low"
        
        return assessment
    
    def save_results(self, filename: str):
        """Save test results to file"""
        report = self.generate_report()
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Test results saved to {filename}")

async def run_cognitive_performance_tests(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """
    Run comprehensive performance tests for cognitive architecture
    """
    test_suite = PerformanceTestSuite(base_url)
    test_suite.create_cognitive_test_scenarios()
    
    # Run all scenarios
    await test_suite.run_all_scenarios()
    
    # Generate and return report
    report = test_suite.generate_report()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_suite.save_results(f"performance_test_results_{timestamp}.json")
    
    return report