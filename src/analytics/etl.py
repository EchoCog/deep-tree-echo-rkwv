"""
ETL (Extract, Transform, Load) Pipeline for Analytics
Handles data ingestion, transformation, validation, and loading.
"""

import json
import csv
import os
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import Path

from .data_models import AnalyticsEvent, UserSession, SystemMetrics, CognitiveProcessingMetrics, DataWarehouse, EventType


@dataclass
class ETLConfig:
    """ETL configuration"""
    batch_size: int = 1000
    processing_interval: int = 60  # seconds
    data_directory: str = "data/analytics"
    backup_directory: str = "data/backups"
    max_retries: int = 3
    validation_enabled: bool = True
    compression_enabled: bool = True


class DataValidator:
    """Validates incoming data for quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'required_fields': ['event_id', 'event_type', 'timestamp'],
            'field_types': {
                'event_id': str,
                'timestamp': (str, datetime),
                'user_id': (str, type(None)),
                'session_id': (str, type(None)),
                'data': (dict, type(None))
            },
            'timestamp_format': '%Y-%m-%dT%H:%M:%S'
        }
        self.validation_stats = {
            'total_records': 0,
            'valid_records': 0,
            'invalid_records': 0,
            'validation_errors': []
        }
    
    def validate_event(self, event_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate a single analytics event"""
        errors = []
        
        # Check required fields
        for field in self.validation_rules['required_fields']:
            if field not in event_data:
                errors.append(f"Missing required field: {field}")
        
        # Check field types
        for field, expected_types in self.validation_rules['field_types'].items():
            if field in event_data:
                if not isinstance(event_data[field], expected_types):
                    errors.append(f"Invalid type for {field}: expected {expected_types}, got {type(event_data[field])}")
        
        # Validate timestamp format
        if 'timestamp' in event_data and isinstance(event_data['timestamp'], str):
            try:
                datetime.fromisoformat(event_data['timestamp'].replace('Z', '+00:00'))
            except ValueError:
                errors.append(f"Invalid timestamp format: {event_data['timestamp']}")
        
        # Validate event type
        if 'event_type' in event_data:
            try:
                EventType(event_data['event_type'])
            except ValueError:
                errors.append(f"Invalid event type: {event_data['event_type']}")
        
        is_valid = len(errors) == 0
        self.validation_stats['total_records'] += 1
        if is_valid:
            self.validation_stats['valid_records'] += 1
        else:
            self.validation_stats['invalid_records'] += 1
            self.validation_stats['validation_errors'].extend(errors)
        
        return is_valid, errors
    
    def get_validation_report(self) -> Dict[str, Any]:
        """Get validation statistics report"""
        total = self.validation_stats['total_records']
        valid = self.validation_stats['valid_records']
        invalid = self.validation_stats['invalid_records']
        
        return {
            'total_records': total,
            'valid_records': valid,
            'invalid_records': invalid,
            'validation_rate': (valid / total * 100) if total > 0 else 0,
            'recent_errors': self.validation_stats['validation_errors'][-10:]  # Last 10 errors
        }


class DataTransformer:
    """Transforms raw data into structured analytics events"""
    
    def __init__(self):
        self.transformation_stats = {
            'total_transformations': 0,
            'successful_transformations': 0,
            'failed_transformations': 0
        }
    
    def transform_web_request(self, request_data: Dict[str, Any]) -> AnalyticsEvent:
        """Transform web request data to analytics event"""
        try:
            event = AnalyticsEvent(
                event_id=None,  # Will be auto-generated
                event_type=EventType.API_REQUEST,
                timestamp=datetime.now(),
                user_id=request_data.get('user_id'),
                session_id=request_data.get('session_id'),
                data={
                    'method': request_data.get('method'),
                    'endpoint': request_data.get('endpoint'),
                    'status_code': request_data.get('status_code'),
                    'response_time': request_data.get('response_time'),
                    'ip_address': request_data.get('ip_address'),
                    'user_agent': request_data.get('user_agent')
                }
            )
            self.transformation_stats['successful_transformations'] += 1
            return event
        except Exception as e:
            self.transformation_stats['failed_transformations'] += 1
            raise ValueError(f"Failed to transform web request: {e}")
    
    def transform_cognitive_process(self, process_data: Dict[str, Any]) -> AnalyticsEvent:
        """Transform cognitive processing data to analytics event"""
        try:
            event = AnalyticsEvent(
                event_id=None,
                event_type=EventType.COGNITIVE_PROCESS,
                timestamp=datetime.now(),
                user_id=process_data.get('user_id'),
                session_id=process_data.get('session_id'),
                data={
                    'membrane_type': process_data.get('membrane_type'),
                    'processing_time': process_data.get('processing_time'),
                    'input_size': process_data.get('input_size'),
                    'output_size': process_data.get('output_size'),
                    'success': process_data.get('success', True),
                    'error_message': process_data.get('error_message')
                }
            )
            self.transformation_stats['successful_transformations'] += 1
            return event
        except Exception as e:
            self.transformation_stats['failed_transformations'] += 1
            raise ValueError(f"Failed to transform cognitive process: {e}")
    
    def transform_system_metrics(self, metrics_data: Dict[str, Any]) -> SystemMetrics:
        """Transform system metrics data"""
        try:
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_usage=metrics_data.get('cpu_usage', 0.0),
                memory_usage=metrics_data.get('memory_usage', 0.0),
                disk_usage=metrics_data.get('disk_usage', 0.0),
                network_io=metrics_data.get('network_io', {}),
                response_time=metrics_data.get('response_time', 0.0),
                throughput=metrics_data.get('throughput', 0.0),
                error_rate=metrics_data.get('error_rate', 0.0),
                active_sessions=metrics_data.get('active_sessions', 0),
                cache_hit_rate=metrics_data.get('cache_hit_rate', 0.0)
            )
            self.transformation_stats['successful_transformations'] += 1
            return metrics
        except Exception as e:
            self.transformation_stats['failed_transformations'] += 1
            raise ValueError(f"Failed to transform system metrics: {e}")
    
    def get_transformation_report(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        total = self.transformation_stats['total_transformations']
        success = self.transformation_stats['successful_transformations']
        failed = self.transformation_stats['failed_transformations']
        
        return {
            'total_transformations': total,
            'successful_transformations': success,
            'failed_transformations': failed,
            'success_rate': (success / total * 100) if total > 0 else 0
        }


class DataRetentionManager:
    """Manages data retention and archival policies"""
    
    def __init__(self, config: ETLConfig):
        self.config = config
        self.retention_policies = {
            'events': 365,  # days
            'sessions': 180,
            'system_metrics': 90,
            'cognitive_metrics': 180
        }
        self.archival_stats = {
            'last_cleanup': None,
            'archived_events': 0,
            'deleted_events': 0
        }
    
    def apply_retention_policy(self, warehouse: DataWarehouse) -> Dict[str, int]:
        """Apply retention policies to data warehouse"""
        results = {
            'events_removed': 0,
            'sessions_removed': 0,
            'system_metrics_removed': 0,
            'cognitive_metrics_removed': 0
        }
        
        now = datetime.now()
        
        # Clean events
        events_cutoff = now - timedelta(days=self.retention_policies['events'])
        initial_events = len(warehouse.events)
        warehouse.events = [e for e in warehouse.events if e.timestamp > events_cutoff]
        results['events_removed'] = initial_events - len(warehouse.events)
        
        # Clean sessions
        sessions_cutoff = now - timedelta(days=self.retention_policies['sessions'])
        initial_sessions = len(warehouse.sessions)
        warehouse.sessions = {
            k: v for k, v in warehouse.sessions.items() 
            if v.last_activity > sessions_cutoff
        }
        results['sessions_removed'] = initial_sessions - len(warehouse.sessions)
        
        # Clean system metrics
        metrics_cutoff = now - timedelta(days=self.retention_policies['system_metrics'])
        initial_sys_metrics = len(warehouse.system_metrics)
        warehouse.system_metrics = [m for m in warehouse.system_metrics if m.timestamp > metrics_cutoff]
        results['system_metrics_removed'] = initial_sys_metrics - len(warehouse.system_metrics)
        
        # Clean cognitive metrics
        cog_cutoff = now - timedelta(days=self.retention_policies['cognitive_metrics'])
        initial_cog_metrics = len(warehouse.cognitive_metrics)
        warehouse.cognitive_metrics = [m for m in warehouse.cognitive_metrics if m.timestamp > cog_cutoff]
        results['cognitive_metrics_removed'] = initial_cog_metrics - len(warehouse.cognitive_metrics)
        
        self.archival_stats['last_cleanup'] = now
        self.archival_stats['deleted_events'] += results['events_removed']
        
        return results
    
    def archive_data(self, data: List[Any], archive_type: str) -> str:
        """Archive data to backup storage"""
        os.makedirs(self.config.backup_directory, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{archive_type}_{timestamp}.json"
        filepath = os.path.join(self.config.backup_directory, filename)
        
        with open(filepath, 'w') as f:
            json.dump([item.to_dict() if hasattr(item, 'to_dict') else item for item in data], f)
        
        self.archival_stats['archived_events'] += len(data)
        return filepath


class ETLPipeline:
    """Main ETL pipeline orchestrator"""
    
    def __init__(self, config: ETLConfig, warehouse: DataWarehouse):
        self.config = config
        self.warehouse = warehouse
        self.validator = DataValidator()
        self.transformer = DataTransformer()
        self.retention_manager = DataRetentionManager(config)
        
        self.pipeline_stats = {
            'start_time': datetime.now(),
            'total_batches_processed': 0,
            'total_records_processed': 0,
            'last_processing_time': None,
            'average_processing_time': 0.0,
            'errors': []
        }
        
        self.running = False
        self.processing_thread = None
        
        # Create data directories
        os.makedirs(self.config.data_directory, exist_ok=True)
        os.makedirs(self.config.backup_directory, exist_ok=True)
    
    def start(self):
        """Start the ETL pipeline"""
        if not self.running:
            self.running = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
    
    def stop(self):
        """Stop the ETL pipeline"""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join()
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.running:
            try:
                start_time = time.time()
                
                # Process any pending data files
                self._process_data_files()
                
                # Apply retention policies (daily)
                if self._should_run_retention():
                    self._run_retention_cleanup()
                
                processing_time = time.time() - start_time
                self.pipeline_stats['last_processing_time'] = processing_time
                self._update_average_processing_time(processing_time)
                
                time.sleep(self.config.processing_interval)
                
            except Exception as e:
                error_msg = f"ETL pipeline error: {e}"
                self.pipeline_stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg
                })
                time.sleep(10)  # Brief pause on error
    
    def _process_data_files(self):
        """Process data files in the data directory"""
        data_dir = Path(self.config.data_directory)
        for file_path in data_dir.glob("*.json"):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                self._process_batch(data)
                
                # Move processed file to backup
                backup_path = Path(self.config.backup_directory) / f"processed_{file_path.name}"
                file_path.rename(backup_path)
                
            except Exception as e:
                error_msg = f"Failed to process file {file_path}: {e}"
                self.pipeline_stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg
                })
    
    def _process_batch(self, batch_data: List[Dict[str, Any]]):
        """Process a batch of data"""
        processed_events = []
        
        for record in batch_data:
            try:
                # Validate data
                if self.config.validation_enabled:
                    is_valid, errors = self.validator.validate_event(record)
                    if not is_valid:
                        continue
                
                # Transform data based on type
                if record.get('type') == 'web_request':
                    event = self.transformer.transform_web_request(record)
                    processed_events.append(event)
                elif record.get('type') == 'cognitive_process':
                    event = self.transformer.transform_cognitive_process(record)
                    processed_events.append(event)
                elif record.get('type') == 'system_metrics':
                    metrics = self.transformer.transform_system_metrics(record)
                    self.warehouse.add_system_metrics(metrics)
                
            except Exception as e:
                error_msg = f"Failed to process record: {e}"
                self.pipeline_stats['errors'].append({
                    'timestamp': datetime.now().isoformat(),
                    'error': error_msg
                })
        
        # Load processed events into warehouse
        for event in processed_events:
            self.warehouse.add_event(event)
        
        self.pipeline_stats['total_batches_processed'] += 1
        self.pipeline_stats['total_records_processed'] += len(batch_data)
    
    def _should_run_retention(self) -> bool:
        """Check if retention cleanup should run (daily)"""
        last_cleanup = self.retention_manager.archival_stats.get('last_cleanup')
        if not last_cleanup:
            return True
        
        return (datetime.now() - last_cleanup) > timedelta(days=1)
    
    def _run_retention_cleanup(self):
        """Run retention cleanup"""
        try:
            results = self.retention_manager.apply_retention_policy(self.warehouse)
            # Log cleanup results
            self.pipeline_stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'info': f"Retention cleanup completed: {results}"
            })
        except Exception as e:
            error_msg = f"Retention cleanup failed: {e}"
            self.pipeline_stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
    
    def _update_average_processing_time(self, processing_time: float):
        """Update average processing time"""
        current_avg = self.pipeline_stats['average_processing_time']
        total_batches = self.pipeline_stats['total_batches_processed']
        
        if total_batches == 0:
            self.pipeline_stats['average_processing_time'] = processing_time
        else:
            # Calculate weighted average
            self.pipeline_stats['average_processing_time'] = (
                (current_avg * (total_batches - 1) + processing_time) / total_batches
            )
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get ETL pipeline status and statistics"""
        return {
            'running': self.running,
            'stats': self.pipeline_stats,
            'validation_report': self.validator.get_validation_report(),
            'transformation_report': self.transformer.get_transformation_report(),
            'retention_stats': self.retention_manager.archival_stats
        }
    
    def ingest_real_time_data(self, data: Dict[str, Any]):
        """Ingest real-time data directly into the pipeline"""
        try:
            # Validate if enabled
            if self.config.validation_enabled:
                is_valid, errors = self.validator.validate_event(data)
                if not is_valid:
                    raise ValueError(f"Validation failed: {errors}")
            
            # Transform and load based on data type
            if data.get('type') == 'web_request':
                event = self.transformer.transform_web_request(data)
                self.warehouse.add_event(event)
            elif data.get('type') == 'cognitive_process':
                event = self.transformer.transform_cognitive_process(data)
                self.warehouse.add_event(event)
            elif data.get('type') == 'system_metrics':
                metrics = self.transformer.transform_system_metrics(data)
                self.warehouse.add_system_metrics(metrics)
            
            self.pipeline_stats['total_records_processed'] += 1
            
        except Exception as e:
            error_msg = f"Real-time ingestion failed: {e}"
            self.pipeline_stats['errors'].append({
                'timestamp': datetime.now().isoformat(),
                'error': error_msg
            })
            raise