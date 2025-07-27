"""
Alerting System for Phase 3 Observability
Implements intelligent alerting and notification for P1-002.4
"""

import os
import time
import logging
import threading
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class AlertRule:
    """Definition of an alert rule"""
    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    threshold: Union[int, float]
    duration_seconds: int = 60  # How long condition must be true
    cooldown_seconds: int = 300  # Cooldown before re-alerting
    enabled: bool = True
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Alert:
    """Active alert instance"""
    rule_name: str
    severity: AlertSeverity
    message: str
    triggered_at: datetime
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_resolved(self) -> bool:
        return self.resolved_at is not None
    
    @property
    def duration_seconds(self) -> float:
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.triggered_at).total_seconds()

class AlertingSystem:
    """
    Intelligent alerting system for Phase 3 observability
    Monitors metrics and triggers alerts based on configurable rules
    """
    
    def __init__(self):
        self.rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.metric_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.last_evaluation: Dict[str, float] = {}
        self.cooldown_tracking: Dict[str, float] = {}
        
        # Notification handlers
        self.notification_handlers: List[Callable[[Alert], None]] = []
        
        # Background monitoring thread
        self._monitoring_thread = None
        self._stop_monitoring = threading.Event()
        self.evaluation_interval = 30  # seconds
        
        logger.info("Alerting system initialized")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule"""
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name} ({rule.severity.value})")
    
    def remove_rule(self, rule_name: str):
        """Remove an alert rule"""
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
    
    def add_notification_handler(self, handler: Callable[[Alert], None]):
        """Add a notification handler for alerts"""
        self.notification_handlers.append(handler)
        logger.info("Added notification handler")
    
    def start_monitoring(self):
        """Start background alert monitoring"""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitoring_thread.start()
            logger.info("Alert monitoring started")
    
    def stop_monitoring(self):
        """Stop background alert monitoring"""
        self._stop_monitoring.set()
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
        logger.info("Alert monitoring stopped")
    
    def record_metric(self, metric_name: str, value: Union[int, float], timestamp: Optional[float] = None):
        """Record a metric value for alert evaluation"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metric_buffer[metric_name].append((timestamp, value))
    
    def update_metrics(self, metrics: Dict[str, Union[int, float]]):
        """Update multiple metrics at once"""
        timestamp = time.time()
        for name, value in metrics.items():
            self.record_metric(name, value, timestamp)
    
    def _monitor_loop(self):
        """Background monitoring loop"""
        while not self._stop_monitoring.wait(self.evaluation_interval):
            try:
                self._evaluate_rules()
            except Exception as e:
                logger.error(f"Error in alert monitoring: {e}")
    
    def _evaluate_rules(self):
        """Evaluate all alert rules"""
        current_time = time.time()
        
        for rule_name, rule in self.rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Check if we're in cooldown period
                if rule_name in self.cooldown_tracking:
                    if current_time - self.cooldown_tracking[rule_name] < rule.cooldown_seconds:
                        continue
                
                # Evaluate the rule condition
                is_triggered = self._evaluate_condition(rule, current_time)
                
                if is_triggered:
                    # Check if condition has been true for required duration
                    if rule_name not in self.last_evaluation or \
                       current_time - self.last_evaluation[rule_name] >= rule.duration_seconds:
                        
                        # Trigger alert if not already active
                        if rule_name not in self.active_alerts:
                            self._trigger_alert(rule, current_time)
                        
                        self.last_evaluation[rule_name] = current_time
                else:
                    # Condition no longer true, resolve alert if active
                    if rule_name in self.active_alerts:
                        self._resolve_alert(rule_name, current_time)
                    
                    # Reset evaluation tracking
                    if rule_name in self.last_evaluation:
                        del self.last_evaluation[rule_name]
                        
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    def _evaluate_condition(self, rule: AlertRule, current_time: float) -> bool:
        """Evaluate if a rule condition is met"""
        try:
            # Parse condition (simplified expression parser)
            # Format: "metric_name operator threshold"
            # Example: "response_time_p95 > 1000"
            
            parts = rule.condition.strip().split()
            if len(parts) != 3:
                logger.error(f"Invalid condition format: {rule.condition}")
                return False
            
            metric_name, operator, threshold_str = parts
            
            # Get recent metric values
            if metric_name not in self.metric_buffer:
                return False
            
            recent_values = list(self.metric_buffer[metric_name])
            if not recent_values:
                return False
            
            # Get most recent value within time window
            time_window = 300  # 5 minutes
            cutoff_time = current_time - time_window
            
            valid_values = [value for timestamp, value in recent_values if timestamp >= cutoff_time]
            if not valid_values:
                return False
            
            # Use average of recent values
            current_value = sum(valid_values) / len(valid_values)
            threshold = float(threshold_str)
            
            # Evaluate condition
            if operator == '>':
                return current_value > threshold
            elif operator == '>=':
                return current_value >= threshold
            elif operator == '<':
                return current_value < threshold
            elif operator == '<=':
                return current_value <= threshold
            elif operator == '==':
                return abs(current_value - threshold) < 0.001
            elif operator == '!=':
                return abs(current_value - threshold) >= 0.001
            else:
                logger.error(f"Unknown operator: {operator}")
                return False
                
        except Exception as e:
            logger.error(f"Error evaluating condition {rule.condition}: {e}")
            return False
    
    def _trigger_alert(self, rule: AlertRule, timestamp: float):
        """Trigger a new alert"""
        alert = Alert(
            rule_name=rule.name,
            severity=rule.severity,
            message=f"Alert: {rule.description or rule.name} - Condition: {rule.condition}",
            triggered_at=datetime.fromtimestamp(timestamp),
            metadata=rule.metadata.copy()
        )
        
        self.active_alerts[rule.name] = alert
        self.alert_history.append(alert)
        
        # Send notifications
        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Error in notification handler: {e}")
        
        logger.warning(f"ALERT TRIGGERED: {alert.message} (Severity: {alert.severity.value})")
    
    def _resolve_alert(self, rule_name: str, timestamp: float):
        """Resolve an active alert"""
        if rule_name in self.active_alerts:
            alert = self.active_alerts[rule_name]
            alert.resolved_at = datetime.fromtimestamp(timestamp)
            
            # Move to history and remove from active
            del self.active_alerts[rule_name]
            
            # Set cooldown
            self.cooldown_tracking[rule_name] = timestamp
            
            logger.info(f"ALERT RESOLVED: {alert.message} (Duration: {alert.duration_seconds:.1f}s)")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts"""
        return list(self.active_alerts.values())
    
    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for specified time period"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [alert for alert in self.alert_history if alert.triggered_at >= cutoff_time]
    
    def get_alert_stats(self) -> Dict[str, Any]:
        """Get alerting system statistics"""
        total_alerts = len(self.alert_history)
        active_count = len(self.active_alerts)
        
        # Count by severity
        severity_counts = defaultdict(int)
        for alert in self.alert_history:
            severity_counts[alert.severity.value] += 1
        
        # Average resolution time
        resolved_alerts = [alert for alert in self.alert_history if alert.is_resolved]
        avg_resolution_time = (
            sum(alert.duration_seconds for alert in resolved_alerts) / len(resolved_alerts)
            if resolved_alerts else 0
        )
        
        return {
            "total_rules": len(self.rules),
            "enabled_rules": sum(1 for rule in self.rules.values() if rule.enabled),
            "active_alerts": active_count,
            "total_alerts": total_alerts,
            "severity_counts": dict(severity_counts),
            "avg_resolution_time_seconds": avg_resolution_time,
            "monitoring_active": self._monitoring_thread is not None and self._monitoring_thread.is_alive()
        }

# Built-in notification handlers
def console_notification_handler(alert: Alert):
    """Simple console notification handler"""
    print(f"🚨 ALERT [{alert.severity.value.upper()}]: {alert.message}")

def log_notification_handler(alert: Alert):
    """Log-based notification handler"""
    level = logging.CRITICAL if alert.severity == AlertSeverity.CRITICAL else logging.WARNING
    logger.log(level, f"Alert triggered: {alert.message}")

# Common alert rules for cognitive architecture
def create_standard_alert_rules() -> List[AlertRule]:
    """Create standard alert rules for cognitive architecture monitoring"""
    return [
        AlertRule(
            name="high_response_time",
            condition="response_time_p95 > 1000",
            severity=AlertSeverity.HIGH,
            threshold=1000,
            duration_seconds=120,
            description="95th percentile response time is too high"
        ),
        AlertRule(
            name="high_error_rate",
            condition="error_rate > 5",
            severity=AlertSeverity.HIGH,
            threshold=5,
            duration_seconds=60,
            description="Error rate exceeds 5%"
        ),
        AlertRule(
            name="cognitive_memory_usage_high",
            condition="cognitive_memory_usage_memory > 400",
            severity=AlertSeverity.MEDIUM,
            threshold=400,
            duration_seconds=180,
            description="Memory membrane usage is high"
        ),
        AlertRule(
            name="cognitive_processing_slow",
            condition="cognitive_processing_p95_reasoning > 500",
            severity=AlertSeverity.MEDIUM,
            threshold=500,
            duration_seconds=120,
            description="Reasoning membrane processing is slow"
        ),
        AlertRule(
            name="system_overload",
            condition="response_time_p99 > 5000",
            severity=AlertSeverity.CRITICAL,
            threshold=5000,
            duration_seconds=60,
            description="System appears to be overloaded"
        )
    ]