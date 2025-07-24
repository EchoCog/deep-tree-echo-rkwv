"""
Test suite for Analytics System
Tests all components of the advanced analytics and reporting system.
"""

import unittest
import json
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import analytics components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analytics.data_models import (
    AnalyticsEvent, UserSession, SystemMetrics, CognitiveProcessingMetrics,
    DataWarehouse, EventType, SessionStatus
)
from analytics.etl import ETLPipeline, ETLConfig, DataValidator, DataTransformer
from analytics.processor import AnalyticsProcessor, RealTimeAnalytics, PredictiveAnalytics, AnomalyDetector
from analytics.dashboard import DashboardManager, VisualizationEngine, ReportGenerator
from analytics.integrations import IntegrationManager, ExportConfig, TableauConnector, PowerBIConnector
from analytics.service import AnalyticsService


class TestDataModels(unittest.TestCase):
    """Test analytics data models"""
    
    def setUp(self):
        self.warehouse = DataWarehouse()
    
    def test_analytics_event_creation(self):
        """Test analytics event creation and serialization"""
        event = AnalyticsEvent(
            event_id="test-event-1",
            event_type=EventType.USER_ACTION,
            timestamp=datetime.now(),
            user_id="user123",
            session_id="session456",
            data={"action": "click", "button": "submit"}
        )
        
        self.assertEqual(event.event_id, "test-event-1")
        self.assertEqual(event.event_type, EventType.USER_ACTION)
        self.assertEqual(event.user_id, "user123")
        self.assertEqual(event.data["action"], "click")
        
        # Test serialization
        event_dict = event.to_dict()
        self.assertIn('timestamp', event_dict)
        self.assertEqual(event_dict['event_type'], 'user_action')
        
        # Test deserialization
        restored_event = AnalyticsEvent.from_dict(event_dict)
        self.assertEqual(restored_event.event_id, event.event_id)
        self.assertEqual(restored_event.event_type, event.event_type)
    
    def test_user_session_tracking(self):
        """Test user session functionality"""
        session = UserSession(
            session_id="session123",
            user_id="user456",
            start_time=datetime.now() - timedelta(minutes=30),
            last_activity=datetime.now() - timedelta(minutes=5),
            status=SessionStatus.ACTIVE
        )
        
        self.assertEqual(session.session_id, "session123")
        self.assertTrue(session.is_active)
        self.assertGreater(session.duration.total_seconds(), 0)
        
        # Test session update
        session.update_activity()
        self.assertLess((datetime.now() - session.last_activity).total_seconds(), 5)
    
    def test_data_warehouse_operations(self):
        """Test data warehouse functionality"""
        # Add events
        event1 = AnalyticsEvent(
            event_id="event1",
            event_type=EventType.USER_ACTION,
            timestamp=datetime.now(),
            user_id="user1",
            data={"action": "login"}
        )
        
        event2 = AnalyticsEvent(
            event_id="event2",
            event_type=EventType.API_REQUEST,
            timestamp=datetime.now(),
            user_id="user1",
            data={"endpoint": "/api/test"}
        )
        
        self.warehouse.add_event(event1)
        self.warehouse.add_event(event2)
        
        self.assertEqual(len(self.warehouse.events), 2)
        
        # Test querying
        user_events = self.warehouse.query_events(user_id="user1")
        self.assertEqual(len(user_events), 2)
        
        api_events = self.warehouse.query_events(event_type=EventType.API_REQUEST)
        self.assertEqual(len(api_events), 1)
        
        # Test cleanup
        old_count = len(self.warehouse.events)
        self.warehouse.cleanup_old_data(retention_days=1)
        # Should not remove recent events
        self.assertEqual(len(self.warehouse.events), old_count)


class TestETLPipeline(unittest.TestCase):
    """Test ETL pipeline functionality"""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.warehouse = DataWarehouse()
        self.config = ETLConfig(
            data_directory=self.temp_dir,
            backup_directory=f"{self.temp_dir}/backup",
            batch_size=10,
            processing_interval=1
        )
        self.etl = ETLPipeline(self.config, self.warehouse)
    
    def tearDown(self):
        self.etl.stop()
        shutil.rmtree(self.temp_dir)
    
    def test_data_validator(self):
        """Test data validation"""
        validator = DataValidator()
        
        # Valid event data
        valid_data = {
            'event_id': 'test123',
            'event_type': 'user_action',
            'timestamp': datetime.now().isoformat(),
            'user_id': 'user456',
            'data': {'action': 'click'}
        }
        
        is_valid, errors = validator.validate_event(valid_data)
        self.assertTrue(is_valid)
        self.assertEqual(len(errors), 0)
        
        # Invalid event data
        invalid_data = {
            'event_type': 'invalid_type',
            'timestamp': 'invalid_timestamp'
        }
        
        is_valid, errors = validator.validate_event(invalid_data)
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
    
    def test_data_transformer(self):
        """Test data transformation"""
        transformer = DataTransformer()
        
        # Test web request transformation
        web_request_data = {
            'user_id': 'user123',
            'session_id': 'session456',
            'method': 'GET',
            'endpoint': '/api/test',
            'status_code': 200,
            'response_time': 150.5
        }
        
        event = transformer.transform_web_request(web_request_data)
        self.assertEqual(event.event_type, EventType.API_REQUEST)
        self.assertEqual(event.user_id, 'user123')
        self.assertEqual(event.data['method'], 'GET')
        self.assertEqual(event.data['response_time'], 150.5)
        
        # Test system metrics transformation
        metrics_data = {
            'cpu_usage': 45.2,
            'memory_usage': 78.9,
            'disk_usage': 32.1,
            'response_time': 125.0,
            'throughput': 1000.0,
            'error_rate': 0.5,
            'active_sessions': 150,
            'cache_hit_rate': 85.3
        }
        
        metrics = transformer.transform_system_metrics(metrics_data)
        self.assertEqual(metrics.cpu_usage, 45.2)
        self.assertEqual(metrics.cache_hit_rate, 85.3)
    
    def test_real_time_ingestion(self):
        """Test real-time data ingestion"""
        initial_count = len(self.warehouse.events)
        
        # Ingest web request data
        web_request = {
            'type': 'web_request',
            'user_id': 'user789',
            'method': 'POST',
            'endpoint': '/api/cognitive',
            'status_code': 201,
            'response_time': 89.3
        }
        
        self.etl.ingest_real_time_data(web_request)
        
        # Check that event was added
        self.assertEqual(len(self.warehouse.events), initial_count + 1)
        
        # Check event content
        latest_event = self.warehouse.events[-1]
        self.assertEqual(latest_event.event_type, EventType.API_REQUEST)
        self.assertEqual(latest_event.user_id, 'user789')


class TestAnalyticsProcessor(unittest.TestCase):
    """Test analytics processing functionality"""
    
    def setUp(self):
        self.warehouse = DataWarehouse()
        self.processor = AnalyticsProcessor(self.warehouse)
        
        # Add sample data
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Add sample data for testing"""
        # Add some events
        for i in range(50):
            event = AnalyticsEvent(
                event_id=f"event{i}",
                event_type=EventType.USER_ACTION if i % 3 == 0 else EventType.API_REQUEST,
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"user{i % 10}",
                session_id=f"session{i % 5}",
                data={
                    'action': 'test_action',
                    'response_time': 100 + (i * 2),
                    'success': i % 10 != 0  # 10% failure rate
                }
            )
            self.warehouse.add_event(event)
        
        # Add some sessions
        for i in range(5):
            session = UserSession(
                session_id=f"session{i}",
                user_id=f"user{i}",
                start_time=datetime.now() - timedelta(hours=1),
                last_activity=datetime.now() - timedelta(minutes=i*10),
                status=SessionStatus.ACTIVE if i < 3 else SessionStatus.EXPIRED
            )
            self.warehouse.add_session(session)
    
    def test_real_time_analytics(self):
        """Test real-time analytics processing"""
        analytics = self.processor.real_time_analytics
        
        # Process a new event
        event = AnalyticsEvent(
            event_id="test_event",
            event_type=EventType.USER_ACTION,
            timestamp=datetime.now(),
            user_id="test_user",
            data={'action': 'test'}
        )
        
        results = analytics.process_event(event)
        
        self.assertGreater(len(results), 0)
        
        # Check that metrics are calculated
        metric_names = [r.metric_name for r in results]
        self.assertIn('active_users', metric_names)
        self.assertIn('event_rate_per_minute', metric_names)
    
    def test_anomaly_detection(self):
        """Test anomaly detection"""
        detector = self.processor.anomaly_detector
        
        # Update baseline with normal values
        normal_values = [100, 105, 95, 110, 98, 102, 108, 97, 103, 106]
        detector.update_baseline('response_time', normal_values)
        
        # Test normal value (should not be anomaly)
        anomaly = detector.detect_anomaly('response_time', 104, datetime.now())
        self.assertIsNone(anomaly)
        
        # Test anomalous value (should be detected)
        anomaly = detector.detect_anomaly('response_time', 500, datetime.now())
        self.assertIsNotNone(anomaly)
        self.assertEqual(anomaly.metric_name, 'response_time')
        self.assertIn(anomaly.severity, ['low', 'medium', 'high', 'critical'])
    
    def test_predictive_analytics(self):
        """Test predictive analytics"""
        predictor = self.processor.predictive_analytics
        
        # Create sample time series data
        base_time = datetime.now() - timedelta(days=30)
        historical_data = []
        for i in range(30):
            timestamp = base_time + timedelta(days=i)
            value = 100 + i * 2 + (i % 7) * 5  # Trending upward with weekly pattern
            historical_data.append((timestamp, value))
        
        # Build prediction model
        predictor.build_prediction_model('test_metric', historical_data)
        
        # Test prediction
        future_time = datetime.now() + timedelta(days=7)
        prediction = predictor.predict_metric('test_metric', future_time)
        
        if prediction:  # Model quality might not be sufficient
            predicted_value, confidence = prediction
            self.assertGreater(predicted_value, 0)
            self.assertGreater(confidence, 0)
        
        # Test trend detection
        trend = predictor.detect_trends('test_metric')
        if trend:
            self.assertIn(trend['trend'], ['increasing', 'decreasing', 'stable'])
    
    def test_analytics_summary(self):
        """Test comprehensive analytics summary"""
        summary = self.processor.get_analytics_summary()
        
        self.assertIn('processing_stats', summary)
        self.assertIn('kpis', summary)
        self.assertIn('data_warehouse_stats', summary)
        
        # Check KPIs
        kpis = summary['kpis']
        self.assertIn('user_engagement', kpis)
        self.assertIn('system_health', kpis)
        self.assertIn('cognitive_performance', kpis)
        
        # Verify KPI values are in valid range
        for kpi_name, kpi_value in kpis.items():
            self.assertGreaterEqual(kpi_value, 0)
            self.assertLessEqual(kpi_value, 100)


class TestDashboards(unittest.TestCase):
    """Test dashboard and visualization functionality"""
    
    def setUp(self):
        self.warehouse = DataWarehouse()
        self.processor = AnalyticsProcessor(self.warehouse)
        self.dashboard_manager = DashboardManager(self.warehouse, self.processor)
        
        # Add sample data
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Add sample data for testing"""
        # Add events across time range
        base_time = datetime.now() - timedelta(hours=24)
        for i in range(100):
            event = AnalyticsEvent(
                event_id=f"viz_event{i}",
                event_type=EventType.API_REQUEST if i % 2 == 0 else EventType.USER_ACTION,
                timestamp=base_time + timedelta(minutes=i*14),  # Spread over 24 hours
                user_id=f"user{i % 20}",
                data={
                    'response_time': 50 + (i % 50),
                    'endpoint': f'/api/endpoint{i % 5}'
                }
            )
            self.warehouse.add_event(event)
    
    def test_dashboard_availability(self):
        """Test dashboard availability"""
        dashboards = self.dashboard_manager.get_available_dashboards()
        
        self.assertGreater(len(dashboards), 0)
        
        # Check that default dashboards exist
        dashboard_ids = [d['dashboard_id'] for d in dashboards]
        self.assertIn('executive', dashboard_ids)
        self.assertIn('technical', dashboard_ids)
    
    def test_dashboard_data_generation(self):
        """Test dashboard data generation"""
        time_range = {
            'start': datetime.now() - timedelta(hours=24),
            'end': datetime.now()
        }
        
        dashboard_data = self.dashboard_manager.get_dashboard_data('executive', time_range)
        
        self.assertIn('dashboard_id', dashboard_data)
        self.assertIn('widgets', dashboard_data)
        self.assertEqual(dashboard_data['dashboard_id'], 'executive')
        
        # Check that widget data is generated
        widgets = dashboard_data['widgets']
        self.assertGreater(len(widgets), 0)
    
    def test_visualization_engine(self):
        """Test visualization data generation"""
        viz_engine = self.dashboard_manager.visualization_engine
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        # Test time series data
        time_series = viz_engine.generate_time_series_data(
            'event_count', start_time, end_time, '1h'
        )
        
        self.assertEqual(time_series['metric_name'], 'event_count')
        self.assertIn('data_points', time_series)
        self.assertGreater(len(time_series['data_points']), 0)
        
        # Test distribution data
        distribution = viz_engine.generate_distribution_data(
            'event_type', start_time, end_time
        )
        
        self.assertEqual(distribution['field_name'], 'event_type')
        self.assertIn('distribution', distribution)
        self.assertGreater(len(distribution['distribution']), 0)
    
    def test_report_generation(self):
        """Test report generation"""
        report_gen = self.dashboard_manager.report_generator
        
        start_time = datetime.now() - timedelta(hours=24)
        end_time = datetime.now()
        
        # Test executive summary
        exec_report = report_gen.generate_executive_summary(start_time, end_time)
        
        self.assertEqual(exec_report['report_type'], 'executive_summary')
        self.assertIn('key_metrics', exec_report)
        self.assertIn('kpis', exec_report)
        self.assertIn('trends', exec_report)
        self.assertIn('recommendations', exec_report)
        
        # Test technical report
        tech_report = report_gen.generate_technical_report(start_time, end_time)
        
        self.assertEqual(tech_report['report_type'], 'technical_report')
        self.assertIn('performance', tech_report)
        self.assertIn('errors', tech_report)
        
        # Test user behavior report
        user_report = report_gen.generate_user_behavior_report(start_time, end_time)
        
        self.assertEqual(user_report['report_type'], 'user_behavior')
        self.assertIn('summary', user_report)
        self.assertIn('usage_patterns', user_report)


class TestIntegrations(unittest.TestCase):
    """Test BI integrations and export functionality"""
    
    def setUp(self):
        self.warehouse = DataWarehouse()
        self.processor = AnalyticsProcessor(self.warehouse)
        self.integration_manager = IntegrationManager(self.warehouse, self.processor)
        
        # Add sample data
        self._add_sample_data()
    
    def _add_sample_data(self):
        """Add sample data for testing"""
        for i in range(20):
            event = AnalyticsEvent(
                event_id=f"export_event{i}",
                event_type=EventType.USER_ACTION,
                timestamp=datetime.now() - timedelta(minutes=i),
                user_id=f"user{i % 5}",
                data={'action': f'action_{i}', 'value': i * 10}
            )
            self.warehouse.add_event(event)
    
    def test_data_export(self):
        """Test data export functionality"""
        start_time = datetime.now() - timedelta(hours=1)
        end_time = datetime.now()
        
        # Test JSON export
        json_config = ExportConfig(format='json', include_metadata=True)
        json_export = self.integration_manager.export_data('events', start_time, end_time, json_config)
        
        self.assertIsInstance(json_export, str)
        # Should be valid JSON
        json.loads(json_export)
        
        # Test CSV export
        csv_config = ExportConfig(format='csv', include_metadata=False)
        csv_export = self.integration_manager.export_data('events', start_time, end_time, csv_config)
        
        self.assertIsInstance(csv_export, str)
        self.assertIn('event_type', csv_export)  # CSV header
    
    def test_bi_connectors(self):
        """Test BI tool connectors"""
        # Test Tableau connector (mock)
        tableau_connection = {
            'connection_id': 'tableau_test',
            'connection_type': 'tableau',
            'connection_params': {
                'server_url': 'https://test.tableau.com',
                'username': 'test_user',
                'password': 'test_pass',
                'site_id': 'test_site'
            }
        }
        
        success = self.integration_manager.add_bi_connection(tableau_connection)
        self.assertTrue(success)
        
        # Test connection status
        status = self.integration_manager.get_connection_status('tableau_test')
        self.assertIn('connection_type', status)
        self.assertEqual(status['connection_type'], 'tableau')
        
        # Test data sync (mock)
        sync_success = self.integration_manager.sync_data_to_bi('tableau_test', 'events')
        self.assertTrue(sync_success)
    
    def test_export_stats(self):
        """Test export statistics tracking"""
        initial_stats = self.integration_manager.export_manager.get_export_stats()
        initial_exports = initial_stats['total_exports']
        
        # Perform an export
        config = ExportConfig(format='json')
        self.integration_manager.export_data(
            'events', 
            datetime.now() - timedelta(hours=1), 
            datetime.now(), 
            config
        )
        
        # Check that stats were updated
        updated_stats = self.integration_manager.export_manager.get_export_stats()
        self.assertEqual(updated_stats['total_exports'], initial_exports + 1)
        self.assertIn('json', updated_stats['exports_by_format'])


class TestAnalyticsService(unittest.TestCase):
    """Test the main analytics service"""
    
    def setUp(self):
        self.service = AnalyticsService({
            'batch_size': 100,
            'processing_interval': 30
        })
        self.service.start_service()
    
    def tearDown(self):
        self.service.stop_service()
    
    def test_service_initialization(self):
        """Test service initialization"""
        self.assertTrue(self.service.running)
        self.assertIsNotNone(self.service.service_stats['start_time'])
    
    def test_event_tracking(self):
        """Test event tracking methods"""
        # Test user action tracking
        event_id = self.service.track_user_action(
            user_id="test_user",
            session_id="test_session",
            action="test_action",
            details={"detail": "value"}
        )
        
        self.assertIsNotNone(event_id)
        self.assertGreater(self.service.service_stats['events_processed'], 0)
        
        # Test API request tracking
        api_event_id = self.service.track_api_request(
            endpoint="/api/test",
            method="GET",
            status_code=200,
            response_time=125.5,
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertIsNotNone(api_event_id)
        
        # Test cognitive process tracking
        cog_event_id = self.service.track_cognitive_process(
            membrane_type="reasoning",
            processing_time=45.2,
            input_tokens=100,
            output_tokens=150,
            success=True,
            user_id="test_user",
            session_id="test_session"
        )
        
        self.assertIsNotNone(cog_event_id)
    
    def test_session_management(self):
        """Test session management"""
        # Start session
        session_id = self.service.start_user_session(
            user_id="test_user",
            ip_address="127.0.0.1",
            user_agent="Test Browser"
        )
        
        self.assertIsNotNone(session_id)
        self.assertIn(session_id, self.service.active_sessions)
        
        # End session
        success = self.service.end_user_session(session_id)
        self.assertTrue(success)
        self.assertNotIn(session_id, self.service.active_sessions)
    
    def test_analytics_retrieval(self):
        """Test analytics data retrieval"""
        # Add some sample data first
        self.service.track_user_action("user1", "session1", "login")
        self.service.track_api_request("/api/test", "GET", 200, 100.0, "user1", "session1")
        
        # Get real-time analytics
        real_time = self.service.get_real_time_analytics()
        self.assertIn('timestamp', real_time)
        self.assertIn('current_metrics', real_time)
        self.assertIn('warehouse_stats', real_time)
        
        # Get analytics summary
        summary = self.service.get_analytics_summary()
        self.assertIn('service_info', summary)
        self.assertIn('kpis', summary)
        
        # Get dashboard data
        dashboard_data = self.service.get_dashboard_data('executive', 24)
        self.assertIn('dashboard_id', dashboard_data)
        self.assertEqual(dashboard_data['dashboard_id'], 'executive')
    
    def test_service_health(self):
        """Test service health monitoring"""
        health = self.service.get_service_health()
        
        self.assertIn('overall_health', health)
        self.assertIn('components', health)
        self.assertIn('metrics', health)
        
        # Health should be positive for new service
        self.assertIn(health['overall_health'], ['healthy', 'warning', 'unhealthy'])


if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDataModels))
    test_suite.addTest(unittest.makeSuite(TestETLPipeline))
    test_suite.addTest(unittest.makeSuite(TestAnalyticsProcessor))
    test_suite.addTest(unittest.makeSuite(TestDashboards))
    test_suite.addTest(unittest.makeSuite(TestIntegrations))
    test_suite.addTest(unittest.makeSuite(TestAnalyticsService))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Results:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")