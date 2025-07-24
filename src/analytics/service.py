"""
Analytics Service
Main service that integrates analytics capabilities with the existing Flask application.
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .data_models import (
    DataWarehouse, AnalyticsEvent, UserSession, SystemMetrics, 
    CognitiveProcessingMetrics, EventType, SessionStatus
)
from .etl import ETLPipeline, ETLConfig
from .processor import AnalyticsProcessor
from .dashboard import DashboardManager
from .integrations import IntegrationManager, BIConnection, ExportConfig


class AnalyticsService:
    """Main analytics service that coordinates all analytics components"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Initialize configuration
        self.config = config or {}
        
        # Initialize core components
        self.warehouse = DataWarehouse()
        self.etl_config = ETLConfig(
            batch_size=self.config.get('batch_size', 1000),
            processing_interval=self.config.get('processing_interval', 60),
            data_directory=self.config.get('data_directory', 'data/analytics'),
            backup_directory=self.config.get('backup_directory', 'data/backups')
        )
        
        self.etl_pipeline = ETLPipeline(self.etl_config, self.warehouse)
        self.analytics_processor = AnalyticsProcessor(self.warehouse)
        self.dashboard_manager = DashboardManager(self.warehouse, self.analytics_processor)
        self.integration_manager = IntegrationManager(self.warehouse, self.analytics_processor)
        
        # Service state
        self.running = False
        self.service_stats = {
            'start_time': None,
            'events_processed': 0,
            'real_time_analytics_calculated': 0,
            'anomalies_detected': 0,
            'dashboards_served': 0,
            'reports_generated': 0,
            'bi_syncs_completed': 0
        }
        
        # Session tracking
        self.active_sessions = {}
        
        # Start ETL pipeline
        self.etl_pipeline.start()
    
    def start_service(self):
        """Start the analytics service"""
        if not self.running:
            self.running = True
            self.service_stats['start_time'] = datetime.now()
            print(f"Analytics service started at {self.service_stats['start_time']}")
    
    def stop_service(self):
        """Stop the analytics service"""
        if self.running:
            self.running = False
            self.etl_pipeline.stop()
            print("Analytics service stopped")
    
    def track_user_action(self, 
                         user_id: Optional[str],
                         session_id: Optional[str],
                         action: str,
                         details: Dict[str, Any] = None) -> str:
        """Track a user action"""
        
        event = AnalyticsEvent(
            event_id=None,  # Auto-generated
            event_type=EventType.USER_ACTION,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            data={
                'action': action,
                'details': details or {}
            }
        )
        
        # Add to warehouse
        self.warehouse.add_event(event)
        
        # Update session if exists
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id].update_activity()
            self.active_sessions[session_id].events_count += 1
        
        # Process through real-time analytics
        self.analytics_processor.real_time_analytics.process_event(event)
        
        self.service_stats['events_processed'] += 1
        self.service_stats['real_time_analytics_calculated'] += 1
        
        return event.event_id
    
    def track_api_request(self,
                         endpoint: str,
                         method: str,
                         status_code: int,
                         response_time: float,
                         user_id: Optional[str] = None,
                         session_id: Optional[str] = None,
                         ip_address: Optional[str] = None) -> str:
        """Track an API request"""
        
        event = AnalyticsEvent(
            event_id=None,
            event_type=EventType.API_REQUEST,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            data={
                'endpoint': endpoint,
                'method': method,
                'status_code': status_code,
                'response_time': response_time,
                'ip_address': ip_address
            }
        )
        
        self.warehouse.add_event(event)
        
        # Update session API calls count
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id].api_calls += 1
        
        # Process through real-time analytics
        self.analytics_processor.real_time_analytics.process_event(event)
        
        self.service_stats['events_processed'] += 1
        
        return event.event_id
    
    def track_cognitive_process(self,
                               membrane_type: str,
                               processing_time: float,
                               input_tokens: int,
                               output_tokens: int,
                               success: bool,
                               user_id: Optional[str] = None,
                               session_id: Optional[str] = None,
                               error_message: Optional[str] = None) -> str:
        """Track a cognitive processing operation"""
        
        # Create cognitive metrics
        cognitive_metrics = CognitiveProcessingMetrics(
            timestamp=datetime.now(),
            session_id=session_id or "unknown",
            membrane_type=membrane_type,
            processing_time=processing_time,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            success=success,
            error_message=error_message
        )
        
        self.warehouse.add_cognitive_metrics(cognitive_metrics)
        
        # Create analytics event
        event = AnalyticsEvent(
            event_id=None,
            event_type=EventType.COGNITIVE_PROCESS,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            data={
                'membrane_type': membrane_type,
                'processing_time': processing_time,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'success': success,
                'error_message': error_message
            }
        )
        
        self.warehouse.add_event(event)
        
        # Update session cognitive processes count
        if session_id and session_id in self.active_sessions:
            self.active_sessions[session_id].cognitive_processes += 1
        
        # Process through real-time analytics
        self.analytics_processor.real_time_analytics.process_event(event)
        
        self.service_stats['events_processed'] += 1
        
        return event.event_id
    
    def track_system_metrics(self,
                            cpu_usage: float,
                            memory_usage: float,
                            disk_usage: float,
                            network_io: Dict[str, float],
                            response_time: float,
                            throughput: float,
                            error_rate: float,
                            active_sessions: int,
                            cache_hit_rate: float) -> None:
        """Track system performance metrics"""
        
        metrics = SystemMetrics(
            timestamp=datetime.now(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            disk_usage=disk_usage,
            network_io=network_io,
            response_time=response_time,
            throughput=throughput,
            error_rate=error_rate,
            active_sessions=active_sessions,
            cache_hit_rate=cache_hit_rate
        )
        
        self.warehouse.add_system_metrics(metrics)
        
        # Create system metric event for real-time processing
        event = AnalyticsEvent(
            event_id=None,
            event_type=EventType.SYSTEM_METRIC,
            timestamp=datetime.now(),
            data={
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'response_time': response_time,
                'throughput': throughput,
                'error_rate': error_rate,
                'active_sessions': active_sessions,
                'cache_hit_rate': cache_hit_rate
            }
        )
        
        self.warehouse.add_event(event)
        self.analytics_processor.real_time_analytics.process_event(event)
        
        self.service_stats['events_processed'] += 1
    
    def track_error(self,
                   error_type: str,
                   error_message: str,
                   stack_trace: Optional[str] = None,
                   user_id: Optional[str] = None,
                   session_id: Optional[str] = None,
                   context: Dict[str, Any] = None) -> str:
        """Track an error occurrence"""
        
        event = AnalyticsEvent(
            event_id=None,
            event_type=EventType.ERROR,
            timestamp=datetime.now(),
            user_id=user_id,
            session_id=session_id,
            data={
                'error_type': error_type,
                'error_message': error_message,
                'stack_trace': stack_trace,
                'context': context or {}
            }
        )
        
        self.warehouse.add_event(event)
        self.analytics_processor.real_time_analytics.process_event(event)
        
        self.service_stats['events_processed'] += 1
        
        return event.event_id
    
    def start_user_session(self,
                          user_id: Optional[str] = None,
                          ip_address: Optional[str] = None,
                          user_agent: Optional[str] = None) -> str:
        """Start a new user session"""
        
        session = UserSession(
            session_id=None,  # Auto-generated
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            status=SessionStatus.ACTIVE,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        self.active_sessions[session.session_id] = session
        self.warehouse.add_session(session)
        
        # Track session start event
        self.track_user_action(user_id, session.session_id, "session_start", {
            'ip_address': ip_address,
            'user_agent': user_agent
        })
        
        return session.session_id
    
    def end_user_session(self, session_id: str) -> bool:
        """End a user session"""
        
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            session.status = SessionStatus.TERMINATED
            session.update_activity()
            
            # Track session end event
            self.track_user_action(session.user_id, session_id, "session_end", {
                'duration_minutes': session.duration.total_seconds() / 60,
                'events_count': session.events_count,
                'cognitive_processes': session.cognitive_processes,
                'api_calls': session.api_calls
            })
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            return True
        
        return False
    
    def get_real_time_analytics(self) -> Dict[str, Any]:
        """Get current real-time analytics"""
        
        # Get current metrics from processor
        current_metrics = self.analytics_processor.real_time_analytics.get_current_metrics()
        
        # Get basic warehouse stats
        warehouse_stats = {
            'total_events': len(self.warehouse.events),
            'active_sessions': len(self.active_sessions),
            'system_metrics_count': len(self.warehouse.system_metrics),
            'cognitive_metrics_count': len(self.warehouse.cognitive_metrics)
        }
        
        # Get recent anomalies
        recent_anomalies = self.analytics_processor.anomaly_detector.get_recent_anomalies(1)  # Last hour
        
        return {
            'timestamp': datetime.now().isoformat(),
            'current_metrics': {name: result.value for name, result in current_metrics.items()},
            'warehouse_stats': warehouse_stats,
            'recent_anomalies': [
                {
                    'metric': anomaly.metric_name,
                    'severity': anomaly.severity,
                    'description': anomaly.description,
                    'timestamp': anomaly.timestamp.isoformat()
                }
                for anomaly in recent_anomalies
            ],
            'service_stats': self.service_stats
        }
    
    def get_dashboard_data(self, dashboard_id: str, hours_back: int = 24) -> Dict[str, Any]:
        """Get dashboard data for a specific time range"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        time_range = {'start': start_time, 'end': end_time}
        
        dashboard_data = self.dashboard_manager.get_dashboard_data(dashboard_id, time_range)
        
        self.service_stats['dashboards_served'] += 1
        
        return dashboard_data
    
    def generate_report(self, report_type: str, hours_back: int = 24) -> Dict[str, Any]:
        """Generate an analytics report"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        if report_type == 'executive_summary':
            report = self.dashboard_manager.report_generator.generate_executive_summary(start_time, end_time)
        elif report_type == 'technical_report':
            report = self.dashboard_manager.report_generator.generate_technical_report(start_time, end_time)
        elif report_type == 'user_behavior':
            report = self.dashboard_manager.report_generator.generate_user_behavior_report(start_time, end_time)
        else:
            raise ValueError(f"Unknown report type: {report_type}")
        
        self.service_stats['reports_generated'] += 1
        
        return report
    
    def export_data(self, 
                   export_type: str,
                   format_type: str = 'json',
                   hours_back: int = 24,
                   include_metadata: bool = True) -> str:
        """Export analytics data"""
        
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours_back)
        
        config = ExportConfig(
            format=format_type,
            include_metadata=include_metadata
        )
        
        return self.integration_manager.export_data(export_type, start_time, end_time, config)
    
    def add_bi_connection(self, connection_config: Dict[str, Any]) -> bool:
        """Add a new BI tool connection"""
        
        connection = BIConnection(
            connection_id=connection_config['connection_id'],
            connection_type=connection_config['connection_type'],
            connection_params=connection_config['connection_params'],
            api_key=connection_config.get('api_key'),
            sync_frequency=connection_config.get('sync_frequency', 'daily')
        )
        
        success = self.integration_manager.add_bi_connection(connection)
        
        if success:
            self.service_stats['bi_syncs_completed'] += 1
        
        return success
    
    def sync_to_bi(self, connection_id: str, data_type: str = 'events') -> bool:
        """Sync data to a BI tool"""
        
        success = self.integration_manager.sync_data_to_bi(connection_id, data_type)
        
        if success:
            self.service_stats['bi_syncs_completed'] += 1
        
        return success
    
    def get_analytics_summary(self) -> Dict[str, Any]:
        """Get comprehensive analytics summary"""
        
        summary = self.analytics_processor.get_analytics_summary()
        
        # Add service-specific information
        summary['service_info'] = {
            'service_stats': self.service_stats,
            'active_sessions_count': len(self.active_sessions),
            'etl_pipeline_status': self.etl_pipeline.get_pipeline_status(),
            'integration_summary': self.integration_manager.get_integration_summary()
        }
        
        return summary
    
    def get_available_dashboards(self) -> List[Dict[str, Any]]:
        """Get list of available dashboards"""
        return self.dashboard_manager.get_available_dashboards()
    
    def create_custom_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a custom dashboard"""
        return self.dashboard_manager.create_custom_dashboard(dashboard_config)
    
    def cleanup_old_data(self, retention_days: int = 365) -> Dict[str, int]:
        """Clean up old analytics data"""
        return self.warehouse.cleanup_old_data(retention_days)
    
    def get_service_health(self) -> Dict[str, Any]:
        """Get service health status"""
        
        # Check ETL pipeline health
        etl_status = self.etl_pipeline.get_pipeline_status()
        etl_healthy = etl_status['running'] and len(etl_status['stats']['errors']) < 10
        
        # Check data freshness
        latest_event_time = None
        if self.warehouse.events:
            latest_event_time = max(event.timestamp for event in self.warehouse.events)
            data_fresh = (datetime.now() - latest_event_time) < timedelta(minutes=30)
        else:
            data_fresh = False
        
        # Check anomaly levels
        recent_anomalies = self.analytics_processor.anomaly_detector.get_recent_anomalies(24)
        critical_anomalies = [a for a in recent_anomalies if a.severity == 'critical']
        anomaly_level = 'normal'
        if len(critical_anomalies) > 5:
            anomaly_level = 'critical'
        elif len(critical_anomalies) > 2:
            anomaly_level = 'warning'
        
        overall_health = 'healthy'
        if not etl_healthy or not data_fresh or anomaly_level == 'critical':
            overall_health = 'unhealthy'
        elif anomaly_level == 'warning':
            overall_health = 'warning'
        
        return {
            'overall_health': overall_health,
            'components': {
                'etl_pipeline': 'healthy' if etl_healthy else 'unhealthy',
                'data_freshness': 'fresh' if data_fresh else 'stale',
                'anomaly_detection': anomaly_level
            },
            'metrics': {
                'events_processed_today': self.service_stats['events_processed'],
                'active_sessions': len(self.active_sessions),
                'recent_anomalies': len(recent_anomalies),
                'critical_anomalies': len(critical_anomalies),
                'uptime_hours': (
                    (datetime.now() - self.service_stats['start_time']).total_seconds() / 3600
                    if self.service_stats['start_time'] else 0
                )
            },
            'last_updated': datetime.now().isoformat()
        }


# Global analytics service instance
analytics_service = None


def initialize_analytics_service(config: Optional[Dict[str, Any]] = None) -> AnalyticsService:
    """Initialize the global analytics service instance"""
    global analytics_service
    
    if analytics_service is None:
        analytics_service = AnalyticsService(config)
        analytics_service.start_service()
    
    return analytics_service


def get_analytics_service() -> Optional[AnalyticsService]:
    """Get the global analytics service instance"""
    return analytics_service


def shutdown_analytics_service():
    """Shutdown the global analytics service"""
    global analytics_service
    
    if analytics_service:
        analytics_service.stop_service()
        analytics_service = None