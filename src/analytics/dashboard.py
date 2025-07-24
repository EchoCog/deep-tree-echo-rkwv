"""
Dashboard and Visualization Engine
Implements interactive dashboards, visualization components, and report generation.
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict

from .data_models import DataWarehouse, AnalyticsEvent, EventType
from .processor import AnalyticsProcessor


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    widget_id: str
    widget_type: str  # chart, metric, table, gauge
    title: str
    data_source: str
    configuration: Dict[str, Any]
    position: Dict[str, int]  # x, y, width, height
    refresh_interval: int = 30  # seconds


@dataclass
class Dashboard:
    """Dashboard configuration"""
    dashboard_id: str
    name: str
    description: str
    widgets: List[DashboardWidget]
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    last_modified: Optional[datetime] = None


class VisualizationEngine:
    """Generates visualization data for charts and graphs"""
    
    def __init__(self, warehouse: DataWarehouse, analytics_processor: AnalyticsProcessor):
        self.warehouse = warehouse
        self.analytics_processor = analytics_processor
    
    def generate_time_series_data(self, 
                                 metric_name: str,
                                 start_time: datetime,
                                 end_time: datetime,
                                 interval: str = "1h") -> Dict[str, Any]:
        """Generate time series data for charts"""
        
        # Calculate interval delta
        interval_delta = self._parse_interval(interval)
        
        # Generate time buckets
        time_buckets = []
        current_time = start_time
        while current_time <= end_time:
            time_buckets.append(current_time)
            current_time += interval_delta
        
        # Aggregate data by time buckets
        data_points = []
        for bucket_start in time_buckets:
            bucket_end = bucket_start + interval_delta
            
            # Get events in this time bucket
            bucket_events = self.warehouse.query_events(
                start_time=bucket_start,
                end_time=bucket_end
            )
            
            # Calculate metric value for this bucket
            value = self._calculate_metric_for_bucket(metric_name, bucket_events)
            
            data_points.append({
                'timestamp': bucket_start.isoformat(),
                'value': value
            })
        
        return {
            'metric_name': metric_name,
            'data_points': data_points,
            'interval': interval,
            'start_time': start_time.isoformat(),
            'end_time': end_time.isoformat()
        }
    
    def generate_distribution_data(self, 
                                  field_name: str,
                                  start_time: datetime,
                                  end_time: datetime) -> Dict[str, Any]:
        """Generate distribution data for histograms and pie charts"""
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        
        # Count occurrences of each value
        distribution = {}
        for event in events:
            if field_name == 'event_type':
                value = event.event_type.value
            elif field_name in event.data:
                value = event.data[field_name]
            else:
                continue
            
            distribution[str(value)] = distribution.get(str(value), 0) + 1
        
        # Convert to chart format
        chart_data = [
            {'label': label, 'value': count}
            for label, count in sorted(distribution.items(), key=lambda x: x[1], reverse=True)
        ]
        
        return {
            'field_name': field_name,
            'distribution': chart_data,
            'total_events': len(events),
            'unique_values': len(distribution)
        }
    
    def generate_correlation_data(self, 
                                 metric1: str,
                                 metric2: str,
                                 start_time: datetime,
                                 end_time: datetime) -> Dict[str, Any]:
        """Generate correlation data between two metrics"""
        
        # This is a simplified correlation analysis
        # In practice, you'd want more sophisticated statistical analysis
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        
        metric1_values = []
        metric2_values = []
        
        for event in events:
            val1 = self._extract_metric_value(event, metric1)
            val2 = self._extract_metric_value(event, metric2)
            
            if val1 is not None and val2 is not None:
                metric1_values.append(val1)
                metric2_values.append(val2)
        
        # Calculate simple correlation coefficient
        correlation = self._calculate_correlation(metric1_values, metric2_values)
        
        # Generate scatter plot data
        scatter_data = [
            {'x': x, 'y': y}
            for x, y in zip(metric1_values, metric2_values)
        ]
        
        return {
            'metric1': metric1,
            'metric2': metric2,
            'correlation': correlation,
            'scatter_data': scatter_data[:1000],  # Limit for performance
            'sample_size': len(scatter_data)
        }
    
    def generate_heatmap_data(self, 
                             x_field: str,
                             y_field: str,
                             start_time: datetime,
                             end_time: datetime) -> Dict[str, Any]:
        """Generate heatmap data"""
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        
        # Build 2D distribution
        heatmap_data = {}
        for event in events:
            x_val = self._extract_field_value(event, x_field)
            y_val = self._extract_field_value(event, y_field)
            
            if x_val is not None and y_val is not None:
                key = f"{x_val},{y_val}"
                heatmap_data[key] = heatmap_data.get(key, 0) + 1
        
        # Convert to chart format
        chart_data = []
        for key, count in heatmap_data.items():
            x_val, y_val = key.split(',', 1)
            chart_data.append({
                'x': x_val,
                'y': y_val,
                'count': count
            })
        
        return {
            'x_field': x_field,
            'y_field': y_field,
            'heatmap_data': chart_data,
            'max_count': max(heatmap_data.values()) if heatmap_data else 0
        }
    
    def _parse_interval(self, interval: str) -> timedelta:
        """Parse interval string to timedelta"""
        if interval.endswith('m'):
            return timedelta(minutes=int(interval[:-1]))
        elif interval.endswith('h'):
            return timedelta(hours=int(interval[:-1]))
        elif interval.endswith('d'):
            return timedelta(days=int(interval[:-1]))
        else:
            return timedelta(hours=1)  # Default
    
    def _calculate_metric_for_bucket(self, metric_name: str, events: List[AnalyticsEvent]) -> float:
        """Calculate metric value for a time bucket"""
        if metric_name == 'event_count':
            return len(events)
        elif metric_name == 'unique_users':
            users = set(e.user_id for e in events if e.user_id)
            return len(users)
        elif metric_name == 'avg_response_time':
            response_times = [
                e.data.get('response_time', 0) 
                for e in events 
                if e.event_type == EventType.API_REQUEST and 'response_time' in e.data
            ]
            return sum(response_times) / len(response_times) if response_times else 0
        elif metric_name == 'error_rate':
            error_events = [e for e in events if e.event_type == EventType.ERROR]
            return (len(error_events) / len(events) * 100) if events else 0
        else:
            return 0
    
    def _extract_metric_value(self, event: AnalyticsEvent, metric_name: str) -> Optional[float]:
        """Extract metric value from an event"""
        if metric_name in event.data:
            try:
                return float(event.data[metric_name])
            except (ValueError, TypeError):
                return None
        return None
    
    def _extract_field_value(self, event: AnalyticsEvent, field_name: str) -> Optional[str]:
        """Extract field value from an event"""
        if field_name == 'event_type':
            return event.event_type.value
        elif field_name == 'hour':
            return str(event.timestamp.hour)
        elif field_name == 'day_of_week':
            return event.timestamp.strftime('%A')
        elif field_name in event.data:
            return str(event.data[field_name])
        return None
    
    def _calculate_correlation(self, x_values: List[float], y_values: List[float]) -> float:
        """Calculate Pearson correlation coefficient"""
        if len(x_values) < 2 or len(y_values) < 2:
            return 0.0
        
        n = len(x_values)
        sum_x = sum(x_values)
        sum_y = sum(y_values)
        sum_xy = sum(x * y for x, y in zip(x_values, y_values))
        sum_x2 = sum(x * x for x in x_values)
        sum_y2 = sum(y * y for y in y_values)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
        
        return numerator / denominator if denominator != 0 else 0.0


class ReportGenerator:
    """Generates various types of reports"""
    
    def __init__(self, warehouse: DataWarehouse, analytics_processor: AnalyticsProcessor):
        self.warehouse = warehouse
        self.analytics_processor = analytics_processor
        self.visualization_engine = VisualizationEngine(warehouse, analytics_processor)
    
    def generate_executive_summary(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate executive summary report"""
        
        # Get basic metrics
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        active_sessions = self.warehouse.get_active_sessions()
        
        # Calculate key metrics
        total_events = len(events)
        unique_users = len(set(e.user_id for e in events if e.user_id))
        active_user_count = len(active_sessions)
        
        # Error analysis
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        error_rate = (len(error_events) / total_events * 100) if total_events > 0 else 0
        
        # Cognitive processing analysis
        cognitive_events = [e for e in events if e.event_type == EventType.COGNITIVE_PROCESS]
        cognitive_success_rate = 0
        if cognitive_events:
            successful_cognitive = sum(1 for e in cognitive_events if e.data.get('success', True))
            cognitive_success_rate = (successful_cognitive / len(cognitive_events) * 100)
        
        # Performance metrics
        api_events = [e for e in events if e.event_type == EventType.API_REQUEST]
        avg_response_time = 0
        if api_events:
            response_times = [e.data.get('response_time', 0) for e in api_events if 'response_time' in e.data]
            avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        # Get KPIs
        kpis = self.analytics_processor._calculate_kpis()
        
        return {
            'report_type': 'executive_summary',
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat(),
                'duration_hours': (end_time - start_time).total_seconds() / 3600
            },
            'key_metrics': {
                'total_events': total_events,
                'unique_users': unique_users,
                'active_users': active_user_count,
                'error_rate': round(error_rate, 2),
                'cognitive_success_rate': round(cognitive_success_rate, 2),
                'avg_response_time': round(avg_response_time, 2)
            },
            'kpis': kpis,
            'trends': {
                'user_growth': self._calculate_user_growth(start_time, end_time),
                'performance_trend': self._calculate_performance_trend(start_time, end_time),
                'error_trend': self._calculate_error_trend(start_time, end_time)
            },
            'recommendations': self._generate_recommendations(kpis, error_rate, cognitive_success_rate),
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_technical_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate detailed technical report"""
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        system_metrics = [m for m in self.warehouse.system_metrics 
                         if start_time <= m.timestamp <= end_time]
        
        # Performance analysis
        performance_data = self._analyze_performance(events, system_metrics)
        
        # Error analysis
        error_analysis = self._analyze_errors(events)
        
        # Resource utilization
        resource_analysis = self._analyze_resources(system_metrics)
        
        # Cognitive processing analysis
        cognitive_analysis = self._analyze_cognitive_processing(events)
        
        return {
            'report_type': 'technical_report',
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'performance': performance_data,
            'errors': error_analysis,
            'resources': resource_analysis,
            'cognitive_processing': cognitive_analysis,
            'anomalies': [
                {
                    'timestamp': a.timestamp.isoformat(),
                    'metric': a.metric_name,
                    'severity': a.severity,
                    'description': a.description
                }
                for a in self.analytics_processor.anomaly_detector.get_recent_anomalies(24)
            ],
            'generated_at': datetime.now().isoformat()
        }
    
    def generate_user_behavior_report(self, start_time: datetime, end_time: datetime) -> Dict[str, Any]:
        """Generate user behavior analysis report"""
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        sessions = self.warehouse.sessions
        
        # User engagement metrics
        user_sessions = {}
        for event in events:
            if event.user_id:
                if event.user_id not in user_sessions:
                    user_sessions[event.user_id] = {
                        'sessions': set(),
                        'events': 0,
                        'cognitive_interactions': 0
                    }
                
                user_sessions[event.user_id]['events'] += 1
                if event.session_id:
                    user_sessions[event.user_id]['sessions'].add(event.session_id)
                if event.event_type == EventType.COGNITIVE_PROCESS:
                    user_sessions[event.user_id]['cognitive_interactions'] += 1
        
        # Calculate engagement metrics
        engagement_metrics = []
        for user_id, data in user_sessions.items():
            session_count = len(data['sessions'])
            avg_events_per_session = data['events'] / session_count if session_count > 0 else 0
            cognitive_ratio = data['cognitive_interactions'] / data['events'] if data['events'] > 0 else 0
            
            engagement_metrics.append({
                'user_id': user_id,
                'session_count': session_count,
                'total_events': data['events'],
                'avg_events_per_session': round(avg_events_per_session, 2),
                'cognitive_interaction_ratio': round(cognitive_ratio, 2)
            })
        
        # Sort by engagement
        engagement_metrics.sort(key=lambda x: x['total_events'], reverse=True)
        
        # Usage patterns
        hourly_usage = self._analyze_hourly_usage(events)
        daily_usage = self._analyze_daily_usage(events)
        
        return {
            'report_type': 'user_behavior',
            'period': {
                'start': start_time.isoformat(),
                'end': end_time.isoformat()
            },
            'summary': {
                'unique_users': len(user_sessions),
                'total_sessions': len(set(e.session_id for e in events if e.session_id)),
                'avg_session_duration': self._calculate_avg_session_duration(sessions),
                'most_active_users': engagement_metrics[:10]
            },
            'usage_patterns': {
                'hourly_distribution': hourly_usage,
                'daily_distribution': daily_usage
            },
            'engagement_analysis': {
                'high_engagement_users': [u for u in engagement_metrics if u['total_events'] > 100],
                'cognitive_power_users': [u for u in engagement_metrics if u['cognitive_interaction_ratio'] > 0.5]
            },
            'generated_at': datetime.now().isoformat()
        }
    
    def _calculate_user_growth(self, start_time: datetime, end_time: datetime) -> str:
        """Calculate user growth trend"""
        mid_time = start_time + (end_time - start_time) / 2
        
        first_half_users = len(set(e.user_id for e in self.warehouse.query_events(start_time, mid_time) if e.user_id))
        second_half_users = len(set(e.user_id for e in self.warehouse.query_events(mid_time, end_time) if e.user_id))
        
        if first_half_users == 0:
            return "stable"
        
        growth_rate = ((second_half_users - first_half_users) / first_half_users) * 100
        
        if growth_rate > 10:
            return "growing"
        elif growth_rate < -10:
            return "declining"
        else:
            return "stable"
    
    def _calculate_performance_trend(self, start_time: datetime, end_time: datetime) -> str:
        """Calculate performance trend"""
        # Simplified trend analysis based on error rate
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        error_rate = (len(error_events) / len(events) * 100) if events else 0
        
        if error_rate < 1:
            return "excellent"
        elif error_rate < 5:
            return "good"
        elif error_rate < 10:
            return "fair"
        else:
            return "needs_attention"
    
    def _calculate_error_trend(self, start_time: datetime, end_time: datetime) -> str:
        """Calculate error trend"""
        # Simple trend based on error rate
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        error_rate = (len(error_events) / len(events) * 100) if events else 0
        
        if error_rate < 1:
            return "improving"
        elif error_rate < 5:
            return "stable"
        else:
            return "degrading"
    
    def _generate_recommendations(self, kpis: Dict[str, float], error_rate: float, cognitive_success_rate: float) -> List[str]:
        """Generate actionable recommendations"""
        recommendations = []
        
        if error_rate > 5:
            recommendations.append("High error rate detected. Review error logs and implement additional error handling.")
        
        if cognitive_success_rate < 90:
            recommendations.append("Cognitive processing success rate is below optimal. Review cognitive processing pipeline.")
        
        if kpis.get('user_engagement', 0) < 50:
            recommendations.append("User engagement is low. Consider improving user interface and experience.")
        
        if kpis.get('system_health', 0) < 80:
            recommendations.append("System health score is suboptimal. Review system performance metrics.")
        
        if not recommendations:
            recommendations.append("All metrics are within acceptable ranges. Continue monitoring for any changes.")
        
        return recommendations
    
    def _analyze_performance(self, events: List[AnalyticsEvent], system_metrics: List) -> Dict[str, Any]:
        """Analyze performance metrics"""
        api_events = [e for e in events if e.event_type == EventType.API_REQUEST]
        
        if not api_events:
            return {'response_times': [], 'throughput': 0, 'availability': 100}
        
        response_times = [e.data.get('response_time', 0) for e in api_events if 'response_time' in e.data]
        
        return {
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'p95_response_time': self._percentile(response_times, 95) if response_times else 0,
            'throughput': len(api_events) / 3600,  # requests per hour
            'availability': 99.9  # Simplified calculation
        }
    
    def _analyze_errors(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Analyze error patterns"""
        error_events = [e for e in events if e.event_type == EventType.ERROR]
        
        # Group errors by type
        error_types = {}
        for event in error_events:
            error_type = event.data.get('error_type', 'unknown')
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        return {
            'total_errors': len(error_events),
            'error_rate': (len(error_events) / len(events) * 100) if events else 0,
            'error_distribution': error_types,
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }
    
    def _analyze_resources(self, system_metrics: List) -> Dict[str, Any]:
        """Analyze resource utilization"""
        if not system_metrics:
            return {}
        
        cpu_usage = [m.cpu_usage for m in system_metrics]
        memory_usage = [m.memory_usage for m in system_metrics]
        disk_usage = [m.disk_usage for m in system_metrics]
        
        return {
            'avg_cpu_usage': sum(cpu_usage) / len(cpu_usage),
            'max_cpu_usage': max(cpu_usage),
            'avg_memory_usage': sum(memory_usage) / len(memory_usage),
            'max_memory_usage': max(memory_usage),
            'avg_disk_usage': sum(disk_usage) / len(disk_usage),
            'max_disk_usage': max(disk_usage)
        }
    
    def _analyze_cognitive_processing(self, events: List[AnalyticsEvent]) -> Dict[str, Any]:
        """Analyze cognitive processing metrics"""
        cognitive_events = [e for e in events if e.event_type == EventType.COGNITIVE_PROCESS]
        
        if not cognitive_events:
            return {}
        
        processing_times = [e.data.get('processing_time', 0) for e in cognitive_events if 'processing_time' in e.data]
        success_count = sum(1 for e in cognitive_events if e.data.get('success', True))
        
        # Membrane usage
        membrane_usage = {}
        for event in cognitive_events:
            membrane = event.data.get('membrane_type', 'unknown')
            membrane_usage[membrane] = membrane_usage.get(membrane, 0) + 1
        
        return {
            'total_cognitive_processes': len(cognitive_events),
            'success_rate': (success_count / len(cognitive_events) * 100),
            'avg_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'membrane_distribution': membrane_usage
        }
    
    def _analyze_hourly_usage(self, events: List[AnalyticsEvent]) -> Dict[int, int]:
        """Analyze usage by hour of day"""
        hourly_counts = {}
        for event in events:
            hour = event.timestamp.hour
            hourly_counts[hour] = hourly_counts.get(hour, 0) + 1
        return hourly_counts
    
    def _analyze_daily_usage(self, events: List[AnalyticsEvent]) -> Dict[str, int]:
        """Analyze usage by day of week"""
        daily_counts = {}
        for event in events:
            day = event.timestamp.strftime('%A')
            daily_counts[day] = daily_counts.get(day, 0) + 1
        return daily_counts
    
    def _calculate_avg_session_duration(self, sessions: Dict[str, Any]) -> float:
        """Calculate average session duration in minutes"""
        durations = []
        for session in sessions.values():
            if hasattr(session, 'duration'):
                durations.append(session.duration.total_seconds() / 60)
        
        return sum(durations) / len(durations) if durations else 0
    
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


class DashboardManager:
    """Manages dashboards and their configurations"""
    
    def __init__(self, warehouse: DataWarehouse, analytics_processor: AnalyticsProcessor):
        self.warehouse = warehouse
        self.analytics_processor = analytics_processor
        self.visualization_engine = VisualizationEngine(warehouse, analytics_processor)
        self.report_generator = ReportGenerator(warehouse, analytics_processor)
        
        self.dashboards = {}
        self._load_default_dashboards()
    
    def _load_default_dashboards(self):
        """Load default dashboard configurations"""
        
        # Executive Dashboard
        executive_dashboard = Dashboard(
            dashboard_id="executive",
            name="Executive Dashboard",
            description="High-level business metrics and KPIs",
            widgets=[
                DashboardWidget(
                    widget_id="kpi_overview",
                    widget_type="metric",
                    title="Key Performance Indicators",
                    data_source="kpis",
                    configuration={"metrics": ["user_engagement", "system_health", "cognitive_performance"]},
                    position={"x": 0, "y": 0, "width": 12, "height": 3}
                ),
                DashboardWidget(
                    widget_id="user_growth",
                    widget_type="chart",
                    title="User Growth Trend",
                    data_source="time_series",
                    configuration={"metric": "unique_users", "interval": "1d"},
                    position={"x": 0, "y": 3, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_id="system_health",
                    widget_type="gauge",
                    title="System Health Score",
                    data_source="kpis",
                    configuration={"metric": "system_health", "min": 0, "max": 100},
                    position={"x": 6, "y": 3, "width": 6, "height": 4}
                )
            ]
        )
        
        # Technical Dashboard
        technical_dashboard = Dashboard(
            dashboard_id="technical",
            name="Technical Dashboard",
            description="System performance and technical metrics",
            widgets=[
                DashboardWidget(
                    widget_id="response_time",
                    widget_type="chart",
                    title="Average Response Time",
                    data_source="time_series",
                    configuration={"metric": "avg_response_time", "interval": "1h"},
                    position={"x": 0, "y": 0, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_id="error_rate",
                    widget_type="chart",
                    title="Error Rate",
                    data_source="time_series",
                    configuration={"metric": "error_rate", "interval": "1h"},
                    position={"x": 6, "y": 0, "width": 6, "height": 4}
                ),
                DashboardWidget(
                    widget_id="event_distribution",
                    widget_type="chart",
                    title="Event Type Distribution",
                    data_source="distribution",
                    configuration={"field": "event_type"},
                    position={"x": 0, "y": 4, "width": 12, "height": 4}
                )
            ]
        )
        
        self.dashboards["executive"] = executive_dashboard
        self.dashboards["technical"] = technical_dashboard
    
    def get_dashboard_data(self, dashboard_id: str, time_range: Dict[str, datetime]) -> Dict[str, Any]:
        """Get data for a specific dashboard"""
        if dashboard_id not in self.dashboards:
            return {"error": "Dashboard not found"}
        
        dashboard = self.dashboards[dashboard_id]
        widget_data = {}
        
        start_time = time_range['start']
        end_time = time_range['end']
        
        for widget in dashboard.widgets:
            try:
                if widget.data_source == "time_series":
                    metric = widget.configuration.get("metric")
                    interval = widget.configuration.get("interval", "1h")
                    widget_data[widget.widget_id] = self.visualization_engine.generate_time_series_data(
                        metric, start_time, end_time, interval
                    )
                
                elif widget.data_source == "distribution":
                    field = widget.configuration.get("field")
                    widget_data[widget.widget_id] = self.visualization_engine.generate_distribution_data(
                        field, start_time, end_time
                    )
                
                elif widget.data_source == "kpis":
                    widget_data[widget.widget_id] = self.analytics_processor._calculate_kpis()
                
                elif widget.data_source == "correlation":
                    metric1 = widget.configuration.get("metric1")
                    metric2 = widget.configuration.get("metric2")
                    widget_data[widget.widget_id] = self.visualization_engine.generate_correlation_data(
                        metric1, metric2, start_time, end_time
                    )
                
            except Exception as e:
                widget_data[widget.widget_id] = {"error": str(e)}
        
        return {
            "dashboard_id": dashboard_id,
            "dashboard_name": dashboard.name,
            "widgets": widget_data,
            "generated_at": datetime.now().isoformat(),
            "time_range": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            }
        }
    
    def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a custom dashboard"""
        dashboard = Dashboard(
            dashboard_id=dashboard_config["id"],
            name=dashboard_config["name"],
            description=dashboard_config.get("description", ""),
            widgets=[
                DashboardWidget(**widget_config)
                for widget_config in dashboard_config["widgets"]
            ],
            created_at=datetime.now()
        )
        
        self.dashboards[dashboard.dashboard_id] = dashboard
        return dashboard.dashboard_id
    
    def get_available_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return [
            {
                "dashboard_id": dashboard.dashboard_id,
                "name": dashboard.name,
                "description": dashboard.description,
                "widget_count": len(dashboard.widgets)
            }
            for dashboard in self.dashboards.values()
        ]