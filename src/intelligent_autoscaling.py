"""
Intelligent Auto-Scaling System for Phase 3 Scalability
Implements predictive scaling and resource optimization for P1-002.2
"""

import os
import time
import logging
import threading
import statistics
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import deque
from enum import Enum

logger = logging.getLogger(__name__)

class ScalingAction(Enum):
    """Types of scaling actions"""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"

@dataclass
class ResourceMetrics:
    """Container for resource utilization metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    active_connections: int
    requests_per_minute: float
    response_time_p95: float
    error_rate_percent: float
    queue_length: int = 0

@dataclass
class ScalingEvent:
    """Record of a scaling event"""
    timestamp: float
    action: ScalingAction
    reason: str
    target_instances: int
    previous_instances: int
    metrics: ResourceMetrics
    success: bool = False

@dataclass
class LoadPattern:
    """Detected load pattern"""
    name: str
    description: str
    time_periods: List[str]  # e.g., ["09:00-12:00", "13:00-17:00"]
    expected_load_multiplier: float
    confidence: float

class IntelligentAutoScaler:
    """
    Intelligent auto-scaling system for Phase 3 scalability
    Implements predictive scaling and advanced resource optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "min_instances": 1,
            "max_instances": 10,
            "target_cpu_percent": 70,
            "target_response_time_ms": 500,
            "scale_up_threshold": 80,
            "scale_down_threshold": 30,
            "scale_up_cooldown_seconds": 300,
            "scale_down_cooldown_seconds": 600,
            "predictive_scaling_enabled": True,
            "learning_window_hours": 24,
            "min_data_points_for_prediction": 50
        }
        
        self.config = {**default_config, **(config or {})}
        self.current_instances = self.config["min_instances"]
        
        # Metrics storage
        self.metrics_history: deque[ResourceMetrics] = deque(maxlen=10000)
        self.scaling_events: List[ScalingEvent] = []
        
        # Load pattern detection
        self.detected_patterns: List[LoadPattern] = []
        self.hourly_load_averages: Dict[int, List[float]] = {i: [] for i in range(24)}
        
        # Cooldown tracking
        self.last_scale_up = 0
        self.last_scale_down = 0
        
        # Callbacks for scaling actions
        self.scale_up_callback: Optional[Callable[[int], bool]] = None
        self.scale_down_callback: Optional[Callable[[int], bool]] = None
        
        # Background threads
        self._monitoring_thread = None
        self._pattern_detection_thread = None
        self._stop_monitoring = threading.Event()
        
        logger.info("Intelligent auto-scaler initialized")
    
    def set_scaling_callbacks(self, scale_up_fn: Callable[[int], bool], 
                            scale_down_fn: Callable[[int], bool]):
        """Set callbacks for scaling actions"""
        self.scale_up_callback = scale_up_fn
        self.scale_down_callback = scale_down_fn
    
    def start_monitoring(self):
        """Start background monitoring and scaling"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self._monitoring_thread.start()
            
        if self._pattern_detection_thread is None or not self._pattern_detection_thread.is_alive():
            self._pattern_detection_thread = threading.Thread(
                target=self._pattern_detection_loop, daemon=True
            )
            self._pattern_detection_thread.start()
        
        logger.info("Auto-scaler monitoring started")
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        if self._pattern_detection_thread:
            self._pattern_detection_thread.join(timeout=5)
        logger.info("Auto-scaler monitoring stopped")
    
    def record_metrics(self, metrics: ResourceMetrics):
        """Record new resource metrics"""
        self.metrics_history.append(metrics)
        
        # Update hourly load averages for pattern detection
        hour = datetime.fromtimestamp(metrics.timestamp).hour
        load_score = self._calculate_load_score(metrics)
        self.hourly_load_averages[hour].append(load_score)
        
        # Keep only recent data for each hour
        if len(self.hourly_load_averages[hour]) > 100:
            self.hourly_load_averages[hour] = self.hourly_load_averages[hour][-100:]
    
    def _monitoring_loop(self):
        """Main monitoring and scaling loop"""
        while not self._stop_monitoring.wait(30):  # Check every 30 seconds
            try:
                if len(self.metrics_history) > 0:
                    latest_metrics = self.metrics_history[-1]
                    action = self._determine_scaling_action(latest_metrics)
                    
                    if action != ScalingAction.NO_ACTION:
                        self._execute_scaling_action(action, latest_metrics)
                        
            except Exception as e:
                logger.error(f"Error in auto-scaler monitoring: {e}")
    
    def _pattern_detection_loop(self):
        """Background pattern detection and learning"""
        while not self._stop_monitoring.wait(3600):  # Check every hour
            try:
                self._detect_load_patterns()
            except Exception as e:
                logger.error(f"Error in pattern detection: {e}")
    
    def _determine_scaling_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Determine if scaling action is needed"""
        current_time = time.time()
        
        # Check cooldown periods
        if (current_time - self.last_scale_up < self.config["scale_up_cooldown_seconds"] or
            current_time - self.last_scale_down < self.config["scale_down_cooldown_seconds"]):
            return ScalingAction.NO_ACTION
        
        # Calculate load score
        load_score = self._calculate_load_score(metrics)
        
        # Get predictive recommendation if enabled
        predictive_action = ScalingAction.NO_ACTION
        if self.config["predictive_scaling_enabled"]:
            predictive_action = self._get_predictive_scaling_action(metrics)
        
        # Reactive scaling logic
        reactive_action = self._get_reactive_scaling_action(metrics, load_score)
        
        # Combine reactive and predictive decisions
        if predictive_action == ScalingAction.SCALE_UP or reactive_action == ScalingAction.SCALE_UP:
            if self.current_instances < self.config["max_instances"]:
                return ScalingAction.SCALE_UP
        elif predictive_action == ScalingAction.SCALE_DOWN and reactive_action != ScalingAction.SCALE_UP:
            if self.current_instances > self.config["min_instances"]:
                return ScalingAction.SCALE_DOWN
        elif reactive_action == ScalingAction.SCALE_DOWN:
            if self.current_instances > self.config["min_instances"]:
                return ScalingAction.SCALE_DOWN
        
        return ScalingAction.NO_ACTION
    
    def _calculate_load_score(self, metrics: ResourceMetrics) -> float:
        """Calculate composite load score (0-100)"""
        # Weighted combination of metrics
        cpu_weight = 0.3
        memory_weight = 0.2
        response_time_weight = 0.3
        error_rate_weight = 0.1
        queue_weight = 0.1
        
        # Normalize metrics to 0-100 scale
        cpu_score = min(100, metrics.cpu_percent)
        memory_score = min(100, metrics.memory_percent)
        response_time_score = min(100, (metrics.response_time_p95 / self.config["target_response_time_ms"]) * 50)
        error_rate_score = min(100, metrics.error_rate_percent * 10)  # Scale up error impact
        queue_score = min(100, metrics.queue_length * 5)  # Approximate queue impact
        
        load_score = (
            cpu_score * cpu_weight +
            memory_score * memory_weight +
            response_time_score * response_time_weight +
            error_rate_score * error_rate_weight +
            queue_score * queue_weight
        )
        
        return load_score
    
    def _get_reactive_scaling_action(self, metrics: ResourceMetrics, load_score: float) -> ScalingAction:
        """Get reactive scaling action based on current metrics"""
        
        # Scale up conditions
        if (load_score > self.config["scale_up_threshold"] or
            metrics.cpu_percent > self.config["scale_up_threshold"] or
            metrics.response_time_p95 > self.config["target_response_time_ms"] * 2 or
            metrics.error_rate_percent > 5):
            return ScalingAction.SCALE_UP
        
        # Scale down conditions
        if (load_score < self.config["scale_down_threshold"] and
            metrics.cpu_percent < self.config["scale_down_threshold"] and
            metrics.response_time_p95 < self.config["target_response_time_ms"] * 0.5 and
            metrics.error_rate_percent < 1):
            return ScalingAction.SCALE_DOWN
        
        return ScalingAction.NO_ACTION
    
    def _get_predictive_scaling_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Get predictive scaling action based on learned patterns"""
        if len(self.metrics_history) < self.config["min_data_points_for_prediction"]:
            return ScalingAction.NO_ACTION
        
        current_hour = datetime.fromtimestamp(metrics.timestamp).hour
        next_hour = (current_hour + 1) % 24
        
        # Look for matching patterns
        for pattern in self.detected_patterns:
            if self._matches_pattern_time(pattern, current_hour):
                predicted_load = self._predict_next_hour_load(next_hour, pattern)
                current_load = self._calculate_load_score(metrics)
                
                # If significant load increase expected, scale up proactively
                if predicted_load > current_load * 1.5 and predicted_load > 60:
                    logger.info(f"Predictive scaling: Load increase expected (pattern: {pattern.name})")
                    return ScalingAction.SCALE_UP
                
                # If significant load decrease expected, consider scaling down
                elif predicted_load < current_load * 0.7 and predicted_load < 40:
                    logger.info(f"Predictive scaling: Load decrease expected (pattern: {pattern.name})")
                    return ScalingAction.SCALE_DOWN
        
        return ScalingAction.NO_ACTION
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ResourceMetrics):
        """Execute the scaling action"""
        current_time = time.time()
        
        if action == ScalingAction.SCALE_UP:
            target_instances = min(self.current_instances + 1, self.config["max_instances"])
            reason = "High load detected"
            
            if self.scale_up_callback:
                success = self.scale_up_callback(target_instances)
                if success:
                    self.current_instances = target_instances
                    self.last_scale_up = current_time
                    logger.info(f"Scaled up to {target_instances} instances")
                else:
                    logger.error("Scale up failed")
            else:
                success = False
                logger.warning("No scale up callback configured")
        
        elif action == ScalingAction.SCALE_DOWN:
            target_instances = max(self.current_instances - 1, self.config["min_instances"])
            reason = "Low load detected"
            
            if self.scale_down_callback:
                success = self.scale_down_callback(target_instances)
                if success:
                    self.current_instances = target_instances
                    self.last_scale_down = current_time
                    logger.info(f"Scaled down to {target_instances} instances")
                else:
                    logger.error("Scale down failed")
            else:
                success = False
                logger.warning("No scale down callback configured")
        
        else:
            return
        
        # Record scaling event
        event = ScalingEvent(
            timestamp=current_time,
            action=action,
            reason=reason,
            target_instances=target_instances,
            previous_instances=self.current_instances if not success else target_instances,
            metrics=metrics,
            success=success
        )
        self.scaling_events.append(event)
    
    def _detect_load_patterns(self):
        """Detect recurring load patterns"""
        if len(self.metrics_history) < 100:
            return
        
        # Analyze hourly patterns
        self.detected_patterns = []
        
        # Daily business hours pattern
        business_hours_loads = []
        for hour in range(9, 18):  # 9 AM to 5 PM
            if self.hourly_load_averages[hour]:
                business_hours_loads.extend(self.hourly_load_averages[hour])
        
        if business_hours_loads:
            avg_business_load = statistics.mean(business_hours_loads)
            if avg_business_load > 50:  # Significant business hours load
                pattern = LoadPattern(
                    name="business_hours",
                    description="Higher load during business hours",
                    time_periods=["09:00-18:00"],
                    expected_load_multiplier=avg_business_load / 30,  # Relative to baseline
                    confidence=0.8
                )
                self.detected_patterns.append(pattern)
        
        # Weekend vs weekday patterns could be added here
        
        logger.debug(f"Detected {len(self.detected_patterns)} load patterns")
    
    def _matches_pattern_time(self, pattern: LoadPattern, current_hour: int) -> bool:
        """Check if current time matches pattern"""
        for period in pattern.time_periods:
            start_hour, end_hour = map(int, period.split('-')[0].split(':')[0]), map(int, period.split('-')[1].split(':')[0])
            if start_hour <= current_hour < end_hour:
                return True
        return False
    
    def _predict_next_hour_load(self, next_hour: int, pattern: LoadPattern) -> float:
        """Predict load for next hour based on pattern"""
        if self.hourly_load_averages[next_hour]:
            base_load = statistics.mean(self.hourly_load_averages[next_hour])
            return base_load * pattern.expected_load_multiplier
        return 50  # Default prediction
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get auto-scaling statistics"""
        recent_events = [e for e in self.scaling_events if e.timestamp > time.time() - 86400]  # Last 24h
        
        scale_ups = len([e for e in recent_events if e.action == ScalingAction.SCALE_UP])
        scale_downs = len([e for e in recent_events if e.action == ScalingAction.SCALE_DOWN])
        
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            current_load_score = self._calculate_load_score(latest_metrics)
        else:
            current_load_score = 0
        
        return {
            "current_instances": self.current_instances,
            "current_load_score": current_load_score,
            "scale_ups_24h": scale_ups,
            "scale_downs_24h": scale_downs,
            "total_scaling_events": len(self.scaling_events),
            "detected_patterns": len(self.detected_patterns),
            "last_scale_up": self.last_scale_up,
            "last_scale_down": self.last_scale_down,
            "config": self.config
        }
    
    def get_recent_events(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get recent scaling events"""
        cutoff_time = time.time() - (hours * 3600)
        recent_events = [e for e in self.scaling_events if e.timestamp > cutoff_time]
        
        return [
            {
                "timestamp": e.timestamp,
                "action": e.action.value,
                "reason": e.reason,
                "target_instances": e.target_instances,
                "previous_instances": e.previous_instances,
                "success": e.success,
                "load_score": self._calculate_load_score(e.metrics)
            }
            for e in recent_events
        ]

# Global auto-scaler instance
_auto_scaler: Optional[IntelligentAutoScaler] = None

def initialize_auto_scaler(config: Optional[Dict[str, Any]] = None) -> IntelligentAutoScaler:
    """Initialize global auto-scaler"""
    global _auto_scaler
    _auto_scaler = IntelligentAutoScaler(config)
    return _auto_scaler

def get_auto_scaler() -> Optional[IntelligentAutoScaler]:
    """Get global auto-scaler instance"""
    return _auto_scaler