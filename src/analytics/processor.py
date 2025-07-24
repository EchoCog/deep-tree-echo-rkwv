"""
Analytics Processing Engine
Implements real-time analytics, predictive analytics, and anomaly detection.
"""

import statistics
import math
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple
from collections import defaultdict, deque
from dataclasses import dataclass

from .data_models import (
    AnalyticsEvent, UserSession, SystemMetrics, CognitiveProcessingMetrics, 
    DataWarehouse, EventType
)


@dataclass
class AnalyticsResult:
    """Result from analytics processing"""
    metric_name: str
    value: float
    timestamp: datetime
    metadata: Dict[str, Any]
    confidence: float = 1.0


@dataclass
class Anomaly:
    """Detected anomaly"""
    timestamp: datetime
    metric_name: str
    actual_value: float
    expected_value: float
    deviation: float
    severity: str  # low, medium, high, critical
    description: str


class RealTimeAnalytics:
    """Real-time analytics processing engine"""
    
    def __init__(self, window_size_minutes: int = 60):
        self.window_size = timedelta(minutes=window_size_minutes)
        self.metrics_cache = {}
        self.sliding_windows = defaultdict(lambda: deque())
        
    def process_event(self, event: AnalyticsEvent) -> List[AnalyticsResult]:
        """Process an individual event and update real-time metrics"""
        results = []
        
        # Update sliding windows
        self._update_sliding_windows(event)
        
        # Calculate real-time metrics
        results.extend(self._calculate_user_metrics(event))
        results.extend(self._calculate_system_metrics(event))
        results.extend(self._calculate_cognitive_metrics(event))
        
        return results
    
    def _update_sliding_windows(self, event: AnalyticsEvent):
        """Update sliding time windows for metrics calculation"""
        current_time = event.timestamp
        
        # Add event to appropriate windows
        self.sliding_windows['all_events'].append((current_time, event))
        self.sliding_windows[f'events_{event.event_type.value}'].append((current_time, event))
        
        if event.user_id:
            self.sliding_windows[f'user_{event.user_id}'].append((current_time, event))
        
        # Clean old entries from all windows
        cutoff_time = current_time - self.window_size
        for window_key, window in self.sliding_windows.items():
            while window and window[0][0] < cutoff_time:
                window.popleft()
    
    def _calculate_user_metrics(self, event: AnalyticsEvent) -> List[AnalyticsResult]:
        """Calculate user-related metrics"""
        results = []
        
        # Active users in window
        active_users = set()
        for timestamp, evt in self.sliding_windows['all_events']:
            if evt.user_id:
                active_users.add(evt.user_id)
        
        results.append(AnalyticsResult(
            metric_name="active_users",
            value=len(active_users),
            timestamp=event.timestamp,
            metadata={"window_minutes": self.window_size.total_seconds() / 60}
        ))
        
        # User activity rate (events per user)
        if len(active_users) > 0:
            total_events = len(self.sliding_windows['all_events'])
            activity_rate = total_events / len(active_users)
            
            results.append(AnalyticsResult(
                metric_name="user_activity_rate",
                value=activity_rate,
                timestamp=event.timestamp,
                metadata={"total_events": total_events, "active_users": len(active_users)}
            ))
        
        return results
    
    def _calculate_system_metrics(self, event: AnalyticsEvent) -> List[AnalyticsResult]:
        """Calculate system-related metrics"""
        results = []
        
        # Event rate (events per minute)
        total_events = len(self.sliding_windows['all_events'])
        window_minutes = self.window_size.total_seconds() / 60
        event_rate = total_events / window_minutes if window_minutes > 0 else 0
        
        results.append(AnalyticsResult(
            metric_name="event_rate_per_minute",
            value=event_rate,
            timestamp=event.timestamp,
            metadata={"total_events": total_events, "window_minutes": window_minutes}
        ))
        
        # Error rate
        error_events = len(self.sliding_windows['events_error'])
        error_rate = (error_events / total_events * 100) if total_events > 0 else 0
        
        results.append(AnalyticsResult(
            metric_name="error_rate_percentage",
            value=error_rate,
            timestamp=event.timestamp,
            metadata={"error_events": error_events, "total_events": total_events}
        ))
        
        # API request performance
        api_events = self.sliding_windows['events_api_request']
        if api_events:
            response_times = []
            for timestamp, evt in api_events:
                if 'response_time' in evt.data:
                    response_times.append(evt.data['response_time'])
            
            if response_times:
                avg_response_time = statistics.mean(response_times)
                p95_response_time = self._percentile(response_times, 95)
                
                results.append(AnalyticsResult(
                    metric_name="avg_response_time",
                    value=avg_response_time,
                    timestamp=event.timestamp,
                    metadata={"sample_size": len(response_times)}
                ))
                
                results.append(AnalyticsResult(
                    metric_name="p95_response_time",
                    value=p95_response_time,
                    timestamp=event.timestamp,
                    metadata={"sample_size": len(response_times)}
                ))
        
        return results
    
    def _calculate_cognitive_metrics(self, event: AnalyticsEvent) -> List[AnalyticsResult]:
        """Calculate cognitive processing metrics"""
        results = []
        
        cognitive_events = self.sliding_windows['events_cognitive_process']
        if cognitive_events:
            processing_times = []
            success_count = 0
            membrane_counts = defaultdict(int)
            
            for timestamp, evt in cognitive_events:
                if 'processing_time' in evt.data:
                    processing_times.append(evt.data['processing_time'])
                if evt.data.get('success', True):
                    success_count += 1
                if 'membrane_type' in evt.data:
                    membrane_counts[evt.data['membrane_type']] += 1
            
            total_cognitive = len(cognitive_events)
            
            # Average cognitive processing time
            if processing_times:
                avg_processing_time = statistics.mean(processing_times)
                results.append(AnalyticsResult(
                    metric_name="avg_cognitive_processing_time",
                    value=avg_processing_time,
                    timestamp=event.timestamp,
                    metadata={"sample_size": len(processing_times)}
                ))
            
            # Cognitive success rate
            success_rate = (success_count / total_cognitive * 100) if total_cognitive > 0 else 0
            results.append(AnalyticsResult(
                metric_name="cognitive_success_rate",
                value=success_rate,
                timestamp=event.timestamp,
                metadata={"successful": success_count, "total": total_cognitive}
            ))
            
            # Membrane usage distribution
            for membrane_type, count in membrane_counts.items():
                usage_percentage = (count / total_cognitive * 100) if total_cognitive > 0 else 0
                results.append(AnalyticsResult(
                    metric_name=f"membrane_{membrane_type}_usage",
                    value=usage_percentage,
                    timestamp=event.timestamp,
                    metadata={"count": count, "total": total_cognitive}
                ))
        
        return results
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of numbers"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_current_metrics(self) -> Dict[str, AnalyticsResult]:
        """Get current real-time metrics snapshot"""
        return self.metrics_cache.copy()


class PredictiveAnalytics:
    """Predictive analytics using simple statistical methods"""
    
    def __init__(self, history_days: int = 30):
        self.history_days = history_days
        self.prediction_models = {}
    
    def build_prediction_model(self, metric_name: str, historical_data: List[Tuple[datetime, float]]):
        """Build a simple prediction model for a metric"""
        if len(historical_data) < 7:  # Need at least a week of data
            return
        
        # Simple linear trend analysis
        timestamps = [t.timestamp() for t, _ in historical_data]
        values = [v for _, v in historical_data]
        
        # Calculate trend
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(t * v for t, v in zip(timestamps, values))
        sum_x2 = sum(t * t for t in timestamps)
        
        # Linear regression coefficients
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
        intercept = (sum_y - slope * sum_x) / n
        
        # Calculate R-squared for model quality
        mean_y = statistics.mean(values)
        ss_tot = sum((y - mean_y) ** 2 for y in values)
        ss_res = sum((y - (slope * t + intercept)) ** 2 for t, y in zip(timestamps, values))
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        self.prediction_models[metric_name] = {
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared,
            'last_update': datetime.now(),
            'data_points': len(historical_data)
        }
    
    def predict_metric(self, metric_name: str, future_timestamp: datetime) -> Optional[Tuple[float, float]]:
        """Predict metric value for future timestamp"""
        if metric_name not in self.prediction_models:
            return None
        
        model = self.prediction_models[metric_name]
        
        # Check model quality
        if model['r_squared'] < 0.5:  # Poor model quality
            return None
        
        future_ts = future_timestamp.timestamp()
        predicted_value = model['slope'] * future_ts + model['intercept']
        
        # Calculate prediction confidence based on R-squared
        confidence = model['r_squared']
        
        return predicted_value, confidence
    
    def detect_trends(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Detect trends in metric data"""
        if metric_name not in self.prediction_models:
            return None
        
        model = self.prediction_models[metric_name]
        slope = model['slope']
        
        # Classify trend
        if abs(slope) < 1e-6:  # Very small slope
            trend = "stable"
        elif slope > 0:
            trend = "increasing"
        else:
            trend = "decreasing"
        
        # Calculate trend strength
        trend_strength = min(abs(slope) * 1000, 100)  # Scale for readability
        
        return {
            'trend': trend,
            'strength': trend_strength,
            'slope': slope,
            'confidence': model['r_squared'],
            'data_points': model['data_points']
        }
    
    def generate_forecasts(self, days_ahead: int = 7) -> Dict[str, List[Tuple[datetime, float, float]]]:
        """Generate forecasts for all metrics"""
        forecasts = {}
        base_time = datetime.now()
        
        for metric_name in self.prediction_models:
            forecast_points = []
            
            for day in range(1, days_ahead + 1):
                future_time = base_time + timedelta(days=day)
                prediction = self.predict_metric(metric_name, future_time)
                
                if prediction:
                    value, confidence = prediction
                    forecast_points.append((future_time, value, confidence))
            
            if forecast_points:
                forecasts[metric_name] = forecast_points
        
        return forecasts


class AnomalyDetector:
    """Anomaly detection using statistical methods"""
    
    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard deviations for anomaly threshold
        self.baseline_stats = {}
        self.detected_anomalies = []
    
    def update_baseline(self, metric_name: str, historical_values: List[float]):
        """Update baseline statistics for a metric"""
        if len(historical_values) < 10:  # Need sufficient data
            return
        
        mean_val = statistics.mean(historical_values)
        std_val = statistics.stdev(historical_values) if len(historical_values) > 1 else 0
        
        # Calculate additional statistics
        median_val = statistics.median(historical_values)
        q1 = self._percentile(historical_values, 25)
        q3 = self._percentile(historical_values, 75)
        iqr = q3 - q1
        
        self.baseline_stats[metric_name] = {
            'mean': mean_val,
            'std': std_val,
            'median': median_val,
            'q1': q1,
            'q3': q3,
            'iqr': iqr,
            'min': min(historical_values),
            'max': max(historical_values),
            'sample_size': len(historical_values),
            'last_update': datetime.now()
        }
    
    def detect_anomaly(self, metric_name: str, value: float, timestamp: datetime) -> Optional[Anomaly]:
        """Detect if a value is anomalous"""
        if metric_name not in self.baseline_stats:
            return None
        
        stats = self.baseline_stats[metric_name]
        
        # Z-score method
        if stats['std'] > 0:
            z_score = abs(value - stats['mean']) / stats['std']
            is_z_anomaly = z_score > self.sensitivity
        else:
            is_z_anomaly = False
        
        # IQR method (outlier detection)
        lower_bound = stats['q1'] - 1.5 * stats['iqr']
        upper_bound = stats['q3'] + 1.5 * stats['iqr']
        is_iqr_anomaly = value < lower_bound or value > upper_bound
        
        # Combine methods
        if is_z_anomaly or is_iqr_anomaly:
            # Calculate expected value
            expected_value = stats['median']  # Use median as it's more robust
            
            # Calculate deviation
            deviation = abs(value - expected_value)
            relative_deviation = (deviation / expected_value * 100) if expected_value != 0 else 0
            
            # Determine severity
            if relative_deviation > 100:
                severity = "critical"
            elif relative_deviation > 50:
                severity = "high"
            elif relative_deviation > 25:
                severity = "medium"
            else:
                severity = "low"
            
            # Create anomaly description
            direction = "above" if value > expected_value else "below"
            description = f"{metric_name} is {relative_deviation:.1f}% {direction} expected value"
            
            anomaly = Anomaly(
                timestamp=timestamp,
                metric_name=metric_name,
                actual_value=value,
                expected_value=expected_value,
                deviation=deviation,
                severity=severity,
                description=description
            )
            
            self.detected_anomalies.append(anomaly)
            return anomaly
        
        return None
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of numbers"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = (percentile / 100) * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        else:
            lower = sorted_data[int(index)]
            upper = sorted_data[int(index) + 1]
            return lower + (upper - lower) * (index - int(index))
    
    def get_recent_anomalies(self, hours: int = 24) -> List[Anomaly]:
        """Get anomalies detected in recent hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.detected_anomalies if a.timestamp > cutoff_time]
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies"""
        recent_anomalies = self.get_recent_anomalies()
        
        severity_counts = defaultdict(int)
        metric_counts = defaultdict(int)
        
        for anomaly in recent_anomalies:
            severity_counts[anomaly.severity] += 1
            metric_counts[anomaly.metric_name] += 1
        
        return {
            'total_anomalies_24h': len(recent_anomalies),
            'severity_distribution': dict(severity_counts),
            'metrics_affected': dict(metric_counts),
            'most_affected_metric': max(metric_counts.items(), key=lambda x: x[1])[0] if metric_counts else None
        }


class AnalyticsProcessor:
    """Main analytics processing coordinator"""
    
    def __init__(self, warehouse: DataWarehouse):
        self.warehouse = warehouse
        self.real_time_analytics = RealTimeAnalytics()
        self.predictive_analytics = PredictiveAnalytics()
        self.anomaly_detector = AnomalyDetector()
        
        self.kpi_definitions = {
            'user_engagement': {
                'description': 'User engagement score based on session activity',
                'calculation': 'weighted_average',
                'components': ['session_duration', 'events_per_session', 'cognitive_interactions']
            },
            'system_health': {
                'description': 'Overall system health score',
                'calculation': 'weighted_average',
                'components': ['uptime', 'response_time', 'error_rate', 'resource_utilization']
            },
            'cognitive_performance': {
                'description': 'Cognitive processing performance score',
                'calculation': 'weighted_average',
                'components': ['processing_speed', 'accuracy', 'success_rate']
            }
        }
        
        self.processing_stats = {
            'total_events_processed': 0,
            'anomalies_detected': 0,
            'predictions_generated': 0,
            'last_processing_time': None
        }
    
    def process_analytics_batch(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Process a batch of analytics events"""
        results = {
            'real_time_metrics': [],
            'anomalies': [],
            'trends': {},
            'kpis': {}
        }
        
        # Process each event through real-time analytics
        for event in events:
            rt_metrics = self.real_time_analytics.process_event(event)
            results['real_time_metrics'].extend(rt_metrics)
            
            # Check for anomalies in real-time metrics
            for metric in rt_metrics:
                anomaly = self.anomaly_detector.detect_anomaly(
                    metric.metric_name, metric.value, metric.timestamp
                )
                if anomaly:
                    results['anomalies'].append(anomaly)
        
        # Update baselines for anomaly detection
        self._update_anomaly_baselines()
        
        # Build prediction models
        self._update_prediction_models()
        
        # Calculate KPIs
        results['kpis'] = self._calculate_kpis()
        
        # Update processing stats
        self.processing_stats['total_events_processed'] += len(events)
        self.processing_stats['anomalies_detected'] += len(results['anomalies'])
        self.processing_stats['last_processing_time'] = datetime.now()
        
        return results
    
    def _update_anomaly_baselines(self):
        """Update anomaly detection baselines with historical data"""
        # Get historical data for each metric type
        for metric_name in ['active_users', 'event_rate_per_minute', 'error_rate_percentage']:
            historical_values = self._get_historical_metric_values(metric_name)
            if historical_values:
                self.anomaly_detector.update_baseline(metric_name, historical_values)
    
    def _update_prediction_models(self):
        """Update prediction models with historical data"""
        for metric_name in ['active_users', 'event_rate_per_minute', 'avg_response_time']:
            historical_data = self._get_historical_metric_data(metric_name)
            if historical_data:
                self.predictive_analytics.build_prediction_model(metric_name, historical_data)
    
    def _get_historical_metric_values(self, metric_name: str, days: int = 30) -> List[float]:
        """Get historical values for a metric"""
        # This would typically query stored analytics results
        # For now, simulate with some sample data
        import random
        return [random.uniform(10, 100) for _ in range(days * 24)]  # Hourly data
    
    def _get_historical_metric_data(self, metric_name: str, days: int = 30) -> List[Tuple[datetime, float]]:
        """Get historical timestamp-value pairs for a metric"""
        # Simulate historical data
        import random
        base_time = datetime.now() - timedelta(days=days)
        data = []
        
        for hour in range(days * 24):
            timestamp = base_time + timedelta(hours=hour)
            value = random.uniform(10, 100)
            data.append((timestamp, value))
        
        return data
    
    def _calculate_kpis(self) -> Dict[str, float]:
        """Calculate key performance indicators"""
        kpis = {}
        
        # User engagement KPI
        active_sessions = len(self.warehouse.get_active_sessions())
        total_events = len(self.warehouse.events)
        avg_events_per_session = total_events / max(active_sessions, 1)
        
        # Normalize to 0-100 scale
        user_engagement = min(avg_events_per_session * 10, 100)
        kpis['user_engagement'] = user_engagement
        
        # System health KPI (based on recent error rate)
        recent_events = self.warehouse.query_events(
            start_time=datetime.now() - timedelta(hours=1)
        )
        error_events = [e for e in recent_events if e.event_type == EventType.ERROR]
        error_rate = len(error_events) / max(len(recent_events), 1) * 100
        
        system_health = max(0, 100 - error_rate * 10)  # Inverse of error rate
        kpis['system_health'] = system_health
        
        # Cognitive performance KPI
        cognitive_events = [e for e in recent_events if e.event_type == EventType.COGNITIVE_PROCESS]
        if cognitive_events:
            successful = sum(1 for e in cognitive_events if e.data.get('success', True))
            success_rate = successful / len(cognitive_events) * 100
            kpis['cognitive_performance'] = success_rate
        else:
            kpis['cognitive_performance'] = 100.0  # Default when no data
        
        return kpis
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        return {
            'processing_stats': self.processing_stats,
            'current_metrics': self.real_time_analytics.get_current_metrics(),
            'recent_anomalies': self.anomaly_detector.get_recent_anomalies(),
            'anomaly_summary': self.anomaly_detector.get_anomaly_summary(),
            'forecasts': self.predictive_analytics.generate_forecasts(),
            'trends': {
                metric: self.predictive_analytics.detect_trends(metric)
                for metric in self.predictive_analytics.prediction_models
            },
            'kpis': self._calculate_kpis(),
            'data_warehouse_stats': {
                'total_events': len(self.warehouse.events),
                'active_sessions': len(self.warehouse.get_active_sessions()),
                'system_metrics_count': len(self.warehouse.system_metrics),
                'cognitive_metrics_count': len(self.warehouse.cognitive_metrics)
            }
        }