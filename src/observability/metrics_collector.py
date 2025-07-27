"""
Advanced Metrics Collection for Phase 3 Scalability
Implements cognitive-specific metrics and performance monitoring for P1-002.4
"""

import os
import time
import threading
import logging
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from datetime import datetime, timedelta
import json

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    timestamp: float
    service_name: str
    operation_name: str
    duration_ms: float
    status: str = "success"
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class CognitiveMetrics:
    """Container for cognitive processing metrics"""
    timestamp: float
    session_id: str
    membrane_type: str  # memory, reasoning, grammar
    processing_time_ms: float
    memory_usage_mb: float
    complexity_score: float
    quality_score: float
    metadata: Dict[str, Any] = field(default_factory=dict)

class MetricsBuffer:
    """Thread-safe circular buffer for metrics"""
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.lock = threading.Lock()
    
    def add(self, metric: Any):
        """Add metric to buffer"""
        with self.lock:
            self.buffer.append(metric)
    
    def get_recent(self, count: int) -> List[Any]:
        """Get recent metrics"""
        with self.lock:
            return list(self.buffer)[-count:] if count < len(self.buffer) else list(self.buffer)
    
    def get_count(self) -> int:
        """Get total metrics count"""
        with self.lock:
            return len(self.buffer)

class CognitiveMetricsCollector:
    """
    Advanced metrics collector for Phase 3 observability
    Collects cognitive-specific and performance metrics
    """
    
    def __init__(self, service_name: str):
        self.service_name = service_name
        self.performance_metrics = MetricsBuffer()
        self.cognitive_metrics = MetricsBuffer()
        self.counters = defaultdict(int)
        self.gauges = defaultdict(float)
        self.histograms = defaultdict(list)
        
        # Metric aggregations
        self.response_times = deque(maxlen=1000)
        self.cognitive_processing_times = defaultdict(lambda: deque(maxlen=1000))
        self.error_rates = defaultdict(int)
        
        # Background thread for metric aggregation
        self._aggregation_thread = None
        self._stop_aggregation = threading.Event()
        self.start_aggregation()
        
        logger.info(f"Metrics collector initialized for service: {service_name}")
    
    def start_aggregation(self):
        """Start background metric aggregation"""
        if self._aggregation_thread is None or not self._aggregation_thread.is_alive():
            self._stop_aggregation.clear()
            self._aggregation_thread = threading.Thread(target=self._aggregate_metrics, daemon=True)
            self._aggregation_thread.start()
    
    def stop_aggregation(self):
        """Stop background metric aggregation"""
        self._stop_aggregation.set()
        if self._aggregation_thread:
            self._aggregation_thread.join(timeout=5)
    
    def record_performance(self, operation_name: str, duration_ms: float, 
                          status: str = "success", **metadata):
        """Record performance metric"""
        metric = PerformanceMetrics(
            timestamp=time.time(),
            service_name=self.service_name,
            operation_name=operation_name,
            duration_ms=duration_ms,
            status=status,
            metadata=metadata
        )
        
        self.performance_metrics.add(metric)
        self.response_times.append(duration_ms)
        
        # Update counters
        self.counters[f"requests_total_{operation_name}"] += 1
        if status != "success":
            self.counters[f"errors_total_{operation_name}"] += 1
            self.error_rates[operation_name] += 1
        
        logger.debug(f"Recorded performance metric: {operation_name} ({duration_ms:.2f}ms)")
    
    def record_cognitive_processing(self, session_id: str, membrane_type: str,
                                  processing_time_ms: float, memory_usage_mb: float,
                                  complexity_score: float, quality_score: float,
                                  **metadata):
        """Record cognitive processing metric"""
        metric = CognitiveMetrics(
            timestamp=time.time(),
            session_id=session_id,
            membrane_type=membrane_type,
            processing_time_ms=processing_time_ms,
            memory_usage_mb=memory_usage_mb,
            complexity_score=complexity_score,
            quality_score=quality_score,
            metadata=metadata
        )
        
        self.cognitive_metrics.add(metric)
        self.cognitive_processing_times[membrane_type].append(processing_time_ms)
        
        # Update gauges
        self.gauges[f"cognitive_memory_usage_{membrane_type}"] = memory_usage_mb
        self.gauges[f"cognitive_complexity_{membrane_type}"] = complexity_score
        self.gauges[f"cognitive_quality_{membrane_type}"] = quality_score
        
        logger.debug(f"Recorded cognitive metric: {membrane_type} ({processing_time_ms:.2f}ms)")
    
    def increment_counter(self, name: str, value: int = 1, **tags):
        """Increment a counter metric"""
        key = f"{name}_{'_'.join(f'{k}_{v}' for k, v in tags.items())}" if tags else name
        self.counters[key] += value
    
    def set_gauge(self, name: str, value: float, **tags):
        """Set a gauge metric"""
        key = f"{name}_{'_'.join(f'{k}_{v}' for k, v in tags.items())}" if tags else name
        self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, **tags):
        """Record histogram metric"""
        key = f"{name}_{'_'.join(f'{k}_{v}' for k, v in tags.items())}" if tags else name
        self.histograms[key].append(value)
        
        # Keep only recent values
        if len(self.histograms[key]) > 1000:
            self.histograms[key] = self.histograms[key][-1000:]
    
    def _aggregate_metrics(self):
        """Background metric aggregation"""
        while not self._stop_aggregation.wait(30):  # Aggregate every 30 seconds
            try:
                self._calculate_aggregated_metrics()
            except Exception as e:
                logger.error(f"Error in metric aggregation: {e}")
    
    def _calculate_aggregated_metrics(self):
        """Calculate aggregated metrics"""
        current_time = time.time()
        
        # Calculate response time percentiles
        if self.response_times:
            sorted_times = sorted(self.response_times)
            self.gauges["response_time_p50"] = self._percentile(sorted_times, 50)
            self.gauges["response_time_p95"] = self._percentile(sorted_times, 95)
            self.gauges["response_time_p99"] = self._percentile(sorted_times, 99)
            self.gauges["response_time_avg"] = sum(sorted_times) / len(sorted_times)
        
        # Calculate cognitive processing percentiles
        for membrane_type, times in self.cognitive_processing_times.items():
            if times:
                sorted_times = sorted(times)
                self.gauges[f"cognitive_processing_p95_{membrane_type}"] = self._percentile(sorted_times, 95)
                self.gauges[f"cognitive_processing_avg_{membrane_type}"] = sum(sorted_times) / len(sorted_times)
        
        # Calculate error rates
        total_requests = sum(v for k, v in self.counters.items() if "requests_total" in k)
        total_errors = sum(v for k, v in self.counters.items() if "errors_total" in k)
        self.gauges["error_rate"] = (total_errors / total_requests * 100) if total_requests > 0 else 0
        
        logger.debug("Calculated aggregated metrics")
    
    def _percentile(self, sorted_data: List[float], percentile: int) -> float:
        """Calculate percentile from sorted data"""
        if not sorted_data:
            return 0.0
        
        index = (percentile / 100.0) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary"""
        return {
            "service_name": self.service_name,
            "timestamp": time.time(),
            "performance_metrics": {
                "total_count": self.performance_metrics.get_count(),
                "response_time_p95": self.gauges.get("response_time_p95", 0),
                "response_time_avg": self.gauges.get("response_time_avg", 0),
                "error_rate": self.gauges.get("error_rate", 0)
            },
            "cognitive_metrics": {
                "total_count": self.cognitive_metrics.get_count(),
                "memory_avg": {
                    membrane: self.gauges.get(f"cognitive_processing_avg_{membrane}", 0)
                    for membrane in ["memory", "reasoning", "grammar"]
                },
                "complexity_avg": {
                    membrane: self.gauges.get(f"cognitive_complexity_{membrane}", 0)
                    for membrane in ["memory", "reasoning", "grammar"]
                },
                "quality_avg": {
                    membrane: self.gauges.get(f"cognitive_quality_{membrane}", 0)
                    for membrane in ["memory", "reasoning", "grammar"]
                }
            },
            "counters": dict(self.counters),
            "gauges": dict(self.gauges)
        }
    
    def get_recent_performance_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent performance metrics"""
        metrics = self.performance_metrics.get_recent(count)
        return [
            {
                "timestamp": m.timestamp,
                "operation": m.operation_name,
                "duration_ms": m.duration_ms,
                "status": m.status,
                "metadata": m.metadata
            }
            for m in metrics
        ]
    
    def get_recent_cognitive_metrics(self, count: int = 100) -> List[Dict[str, Any]]:
        """Get recent cognitive metrics"""
        metrics = self.cognitive_metrics.get_recent(count)
        return [
            {
                "timestamp": m.timestamp,
                "session_id": m.session_id,
                "membrane_type": m.membrane_type,
                "processing_time_ms": m.processing_time_ms,
                "memory_usage_mb": m.memory_usage_mb,
                "complexity_score": m.complexity_score,
                "quality_score": m.quality_score,
                "metadata": m.metadata
            }
            for m in metrics
        ]

# Global metrics collector instance
_metrics_collector: Optional[CognitiveMetricsCollector] = None

def initialize_metrics_collector(service_name: str) -> CognitiveMetricsCollector:
    """Initialize global metrics collector"""
    global _metrics_collector
    _metrics_collector = CognitiveMetricsCollector(service_name)
    return _metrics_collector

def get_metrics_collector() -> Optional[CognitiveMetricsCollector]:
    """Get global metrics collector instance"""
    return _metrics_collector