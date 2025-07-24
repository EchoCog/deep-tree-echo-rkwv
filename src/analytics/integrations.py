"""
Business Intelligence Integrations
Implements connectors for BI tools and data export capabilities.
"""

import json
import csv
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
import io
import base64

from .data_models import DataWarehouse, AnalyticsEvent, EventType
from .processor import AnalyticsProcessor


@dataclass
class ExportConfig:
    """Configuration for data exports"""
    format: str  # json, csv, xlsx, xml, parquet
    include_metadata: bool = True
    compress: bool = False
    date_format: str = "%Y-%m-%d %H:%M:%S"
    encoding: str = "utf-8"


@dataclass
class BIConnection:
    """BI tool connection configuration"""
    connection_id: str
    connection_type: str  # tableau, powerbi, qlik, looker
    connection_params: Dict[str, Any]
    api_key: Optional[str] = None
    last_sync: Optional[datetime] = None
    sync_frequency: str = "daily"  # hourly, daily, weekly


class BIIntegration(ABC):
    """Base class for BI tool integrations"""
    
    def __init__(self, connection: BIConnection):
        self.connection = connection
        self.sync_stats = {
            'total_syncs': 0,
            'successful_syncs': 0,
            'failed_syncs': 0,
            'last_sync_time': None,
            'last_error': None
        }
    
    @abstractmethod
    def authenticate(self) -> bool:
        """Authenticate with the BI tool"""
        pass
    
    @abstractmethod
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to the BI tool"""
        pass
    
    @abstractmethod
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a dashboard in the BI tool"""
        pass
    
    @abstractmethod
    def get_connection_status(self) -> Dict[str, Any]:
        """Get connection status and health"""
        pass


class TableauConnector(BIIntegration):
    """Tableau integration connector"""
    
    def __init__(self, connection: BIConnection):
        super().__init__(connection)
        self.server_url = connection.connection_params.get('server_url')
        self.site_id = connection.connection_params.get('site_id')
        self.username = connection.connection_params.get('username')
        self.password = connection.connection_params.get('password')
        self.project_id = connection.connection_params.get('project_id')
    
    def authenticate(self) -> bool:
        """Authenticate with Tableau Server"""
        try:
            # In a real implementation, this would use Tableau REST API
            # For now, we'll simulate authentication
            if self.server_url and self.username and self.password:
                self.sync_stats['total_syncs'] += 1
                self.sync_stats['successful_syncs'] += 1
                self.sync_stats['last_sync_time'] = datetime.now()
                return True
            return False
        except Exception as e:
            self.sync_stats['failed_syncs'] += 1
            self.sync_stats['last_error'] = str(e)
            return False
    
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to Tableau as a data source"""
        try:
            if not self.authenticate():
                return False
            
            # Convert data to Tableau-friendly format
            tableau_data = self._convert_to_tableau_format(data)
            
            # In real implementation, would use Tableau REST API to:
            # 1. Create/update data source
            # 2. Publish data
            # 3. Refresh extracts
            
            # Simulate successful push
            return True
            
        except Exception as e:
            self.sync_stats['last_error'] = str(e)
            return False
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a dashboard in Tableau"""
        try:
            # In real implementation, would use Tableau REST API to:
            # 1. Create workbook
            # 2. Add sheets/views
            # 3. Configure dashboard layout
            # 4. Set permissions
            
            dashboard_id = f"tableau_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return dashboard_id
            
        except Exception as e:
            self.sync_stats['last_error'] = str(e)
            return ""
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get Tableau connection status"""
        return {
            'connection_type': 'tableau',
            'status': 'connected' if self.authenticate() else 'disconnected',
            'server_url': self.server_url,
            'last_sync': self.sync_stats['last_sync_time'],
            'sync_stats': self.sync_stats
        }
    
    def _convert_to_tableau_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert analytics data to Tableau-compatible format"""
        # Flatten nested data structures
        flattened_data = []
        
        if 'events' in data:
            for event in data['events']:
                row = {
                    'event_id': event.get('event_id'),
                    'event_type': event.get('event_type'),
                    'timestamp': event.get('timestamp'),
                    'user_id': event.get('user_id'),
                    'session_id': event.get('session_id')
                }
                
                # Flatten event data
                if 'data' in event:
                    for key, value in event['data'].items():
                        row[f'data_{key}'] = value
                
                flattened_data.append(row)
        
        return {
            'schema': self._generate_tableau_schema(flattened_data),
            'data': flattened_data
        }
    
    def _generate_tableau_schema(self, data: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Generate Tableau schema from data"""
        schema = []
        if data:
            sample_row = data[0]
            for key, value in sample_row.items():
                if isinstance(value, str):
                    data_type = 'string'
                elif isinstance(value, (int, float)):
                    data_type = 'number'
                elif isinstance(value, bool):
                    data_type = 'boolean'
                else:
                    data_type = 'string'
                
                schema.append({
                    'name': key,
                    'type': data_type
                })
        
        return schema


class PowerBIConnector(BIIntegration):
    """Power BI integration connector"""
    
    def __init__(self, connection: BIConnection):
        super().__init__(connection)
        self.tenant_id = connection.connection_params.get('tenant_id')
        self.app_id = connection.connection_params.get('app_id')
        self.app_secret = connection.connection_params.get('app_secret')
        self.workspace_id = connection.connection_params.get('workspace_id')
    
    def authenticate(self) -> bool:
        """Authenticate with Power BI Service"""
        try:
            # In real implementation, would use Microsoft Graph API
            # to get OAuth token
            if self.tenant_id and self.app_id and self.app_secret:
                self.sync_stats['total_syncs'] += 1
                self.sync_stats['successful_syncs'] += 1
                self.sync_stats['last_sync_time'] = datetime.now()
                return True
            return False
        except Exception as e:
            self.sync_stats['failed_syncs'] += 1
            self.sync_stats['last_error'] = str(e)
            return False
    
    def push_data(self, data: Dict[str, Any]) -> bool:
        """Push data to Power BI dataset"""
        try:
            if not self.authenticate():
                return False
            
            # Convert data to Power BI format
            powerbi_data = self._convert_to_powerbi_format(data)
            
            # In real implementation, would use Power BI REST API to:
            # 1. Create/update dataset
            # 2. Push data rows
            # 3. Refresh dataset
            
            return True
            
        except Exception as e:
            self.sync_stats['last_error'] = str(e)
            return False
    
    def create_dashboard(self, dashboard_config: Dict[str, Any]) -> str:
        """Create a dashboard in Power BI"""
        try:
            # In real implementation, would use Power BI REST API to:
            # 1. Create report
            # 2. Add visualizations
            # 3. Create dashboard from report
            # 4. Set permissions
            
            dashboard_id = f"powerbi_dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            return dashboard_id
            
        except Exception as e:
            self.sync_stats['last_error'] = str(e)
            return ""
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get Power BI connection status"""
        return {
            'connection_type': 'powerbi',
            'status': 'connected' if self.authenticate() else 'disconnected',
            'workspace_id': self.workspace_id,
            'last_sync': self.sync_stats['last_sync_time'],
            'sync_stats': self.sync_stats
        }
    
    def _convert_to_powerbi_format(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert analytics data to Power BI-compatible format"""
        tables = []
        
        # Events table
        if 'events' in data:
            events_table = {
                'name': 'Analytics_Events',
                'columns': [
                    {'name': 'event_id', 'dataType': 'string'},
                    {'name': 'event_type', 'dataType': 'string'},
                    {'name': 'timestamp', 'dataType': 'datetime'},
                    {'name': 'user_id', 'dataType': 'string'},
                    {'name': 'session_id', 'dataType': 'string'},
                    {'name': 'data_json', 'dataType': 'string'}
                ],
                'rows': []
            }
            
            for event in data['events']:
                row = [
                    event.get('event_id'),
                    event.get('event_type'),
                    event.get('timestamp'),
                    event.get('user_id'),
                    event.get('session_id'),
                    json.dumps(event.get('data', {}))
                ]
                events_table['rows'].append(row)
            
            tables.append(events_table)
        
        return {'tables': tables}


class ExportManager:
    """Manages data exports in various formats"""
    
    def __init__(self, warehouse: DataWarehouse, analytics_processor: AnalyticsProcessor):
        self.warehouse = warehouse
        self.analytics_processor = analytics_processor
        self.export_stats = {
            'total_exports': 0,
            'exports_by_format': {},
            'last_export': None
        }
    
    def export_events(self, 
                     start_time: datetime,
                     end_time: datetime,
                     config: ExportConfig) -> Union[str, bytes]:
        """Export events data in specified format"""
        
        events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
        
        # Convert events to dictionary format
        events_data = []
        for event in events:
            event_dict = event.to_dict()
            if not config.include_metadata:
                # Remove internal fields
                event_dict.pop('event_id', None)
            events_data.append(event_dict)
        
        # Export based on format
        if config.format.lower() == 'json':
            return self._export_json(events_data, config)
        elif config.format.lower() == 'csv':
            return self._export_csv(events_data, config)
        elif config.format.lower() == 'xml':
            return self._export_xml(events_data, config)
        else:
            raise ValueError(f"Unsupported export format: {config.format}")
    
    def export_analytics_report(self, 
                               report_type: str,
                               start_time: datetime,
                               end_time: datetime,
                               config: ExportConfig) -> Union[str, bytes]:
        """Export analytics report"""
        
        if report_type == 'executive_summary':
            from .dashboard import ReportGenerator
            report_gen = ReportGenerator(self.warehouse, self.analytics_processor)
            report_data = report_gen.generate_executive_summary(start_time, end_time)
        elif report_type == 'technical_report':
            from .dashboard import ReportGenerator
            report_gen = ReportGenerator(self.warehouse, self.analytics_processor)
            report_data = report_gen.generate_technical_report(start_time, end_time)
        elif report_type == 'user_behavior':
            from .dashboard import ReportGenerator
            report_gen = ReportGenerator(self.warehouse, self.analytics_processor)
            report_data = report_gen.generate_user_behavior_report(start_time, end_time)
        else:
            raise ValueError(f"Unsupported report type: {report_type}")
        
        # Export report based on format
        if config.format.lower() == 'json':
            return self._export_json(report_data, config)
        elif config.format.lower() == 'csv':
            # Flatten report for CSV export
            flattened_data = self._flatten_report_for_csv(report_data)
            return self._export_csv(flattened_data, config)
        else:
            return self._export_json(report_data, config)  # Default to JSON
    
    def export_dashboard_data(self,
                             dashboard_id: str,
                             time_range: Dict[str, datetime],
                             config: ExportConfig) -> Union[str, bytes]:
        """Export dashboard data"""
        
        from .dashboard import DashboardManager
        dashboard_manager = DashboardManager(self.warehouse, self.analytics_processor)
        dashboard_data = dashboard_manager.get_dashboard_data(dashboard_id, time_range)
        
        if config.format.lower() == 'json':
            return self._export_json(dashboard_data, config)
        else:
            # Convert dashboard data to tabular format for CSV
            tabular_data = self._convert_dashboard_to_tabular(dashboard_data)
            return self._export_csv(tabular_data, config)
    
    def _export_json(self, data: Union[List, Dict], config: ExportConfig) -> str:
        """Export data as JSON"""
        self._update_export_stats('json')
        
        json_str = json.dumps(data, indent=2, default=str, ensure_ascii=False)
        
        if config.compress:
            import gzip
            compressed = gzip.compress(json_str.encode(config.encoding))
            return base64.b64encode(compressed).decode('ascii')
        
        return json_str
    
    def _export_csv(self, data: List[Dict], config: ExportConfig) -> str:
        """Export data as CSV"""
        self._update_export_stats('csv')
        
        if not data:
            return ""
        
        # Get all possible field names
        fieldnames = set()
        for row in data:
            fieldnames.update(row.keys())
        fieldnames = sorted(list(fieldnames))
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        
        for row in data:
            # Convert datetime objects to strings
            formatted_row = {}
            for key, value in row.items():
                if isinstance(value, datetime):
                    formatted_row[key] = value.strftime(config.date_format)
                elif isinstance(value, (dict, list)):
                    formatted_row[key] = json.dumps(value)
                else:
                    formatted_row[key] = str(value) if value is not None else ""
            writer.writerow(formatted_row)
        
        csv_content = output.getvalue()
        output.close()
        
        if config.compress:
            import gzip
            compressed = gzip.compress(csv_content.encode(config.encoding))
            return base64.b64encode(compressed).decode('ascii')
        
        return csv_content
    
    def _export_xml(self, data: Union[List, Dict], config: ExportConfig) -> str:
        """Export data as XML"""
        self._update_export_stats('xml')
        
        root = ET.Element("analytics_data")
        
        if isinstance(data, list):
            for i, item in enumerate(data):
                item_elem = ET.SubElement(root, f"item_{i}")
                self._dict_to_xml(item, item_elem, config)
        else:
            self._dict_to_xml(data, root, config)
        
        xml_str = ET.tostring(root, encoding=config.encoding).decode(config.encoding)
        
        if config.compress:
            import gzip
            compressed = gzip.compress(xml_str.encode(config.encoding))
            return base64.b64encode(compressed).decode('ascii')
        
        return xml_str
    
    def _dict_to_xml(self, data: Dict[str, Any], parent: ET.Element, config: ExportConfig):
        """Convert dictionary to XML elements"""
        for key, value in data.items():
            # Clean key name for XML
            clean_key = key.replace(' ', '_').replace('-', '_')
            
            if isinstance(value, dict):
                elem = ET.SubElement(parent, clean_key)
                self._dict_to_xml(value, elem, config)
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    elem = ET.SubElement(parent, f"{clean_key}_{i}")
                    if isinstance(item, dict):
                        self._dict_to_xml(item, elem, config)
                    else:
                        elem.text = str(item)
            elif isinstance(value, datetime):
                elem = ET.SubElement(parent, clean_key)
                elem.text = value.strftime(config.date_format)
            else:
                elem = ET.SubElement(parent, clean_key)
                elem.text = str(value) if value is not None else ""
    
    def _flatten_report_for_csv(self, report_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Flatten report data for CSV export"""
        flattened_rows = []
        
        # Extract key metrics as separate rows
        if 'key_metrics' in report_data:
            for metric_name, value in report_data['key_metrics'].items():
                flattened_rows.append({
                    'category': 'key_metrics',
                    'metric_name': metric_name,
                    'value': value,
                    'timestamp': report_data.get('generated_at', '')
                })
        
        # Extract KPIs
        if 'kpis' in report_data:
            for kpi_name, value in report_data['kpis'].items():
                flattened_rows.append({
                    'category': 'kpis',
                    'metric_name': kpi_name,
                    'value': value,
                    'timestamp': report_data.get('generated_at', '')
                })
        
        # Extract trends
        if 'trends' in report_data:
            for trend_name, trend_value in report_data['trends'].items():
                flattened_rows.append({
                    'category': 'trends',
                    'metric_name': trend_name,
                    'value': trend_value,
                    'timestamp': report_data.get('generated_at', '')
                })
        
        return flattened_rows
    
    def _convert_dashboard_to_tabular(self, dashboard_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dashboard data to tabular format"""
        tabular_data = []
        
        for widget_id, widget_data in dashboard_data.get('widgets', {}).items():
            if 'data_points' in widget_data:
                # Time series data
                for point in widget_data['data_points']:
                    tabular_data.append({
                        'widget_id': widget_id,
                        'metric_name': widget_data.get('metric_name', ''),
                        'timestamp': point.get('timestamp', ''),
                        'value': point.get('value', 0)
                    })
            elif isinstance(widget_data, dict) and 'value' in widget_data:
                # Single value metrics
                tabular_data.append({
                    'widget_id': widget_id,
                    'metric_name': widget_id,
                    'timestamp': dashboard_data.get('generated_at', ''),
                    'value': widget_data['value']
                })
        
        return tabular_data
    
    def _update_export_stats(self, format_type: str):
        """Update export statistics"""
        self.export_stats['total_exports'] += 1
        self.export_stats['exports_by_format'][format_type] = (
            self.export_stats['exports_by_format'].get(format_type, 0) + 1
        )
        self.export_stats['last_export'] = datetime.now()
    
    def get_export_stats(self) -> Dict[str, Any]:
        """Get export statistics"""
        return self.export_stats.copy()


class IntegrationManager:
    """Manages all BI integrations and data exports"""
    
    def __init__(self, warehouse: DataWarehouse, analytics_processor: AnalyticsProcessor):
        self.warehouse = warehouse
        self.analytics_processor = analytics_processor
        self.export_manager = ExportManager(warehouse, analytics_processor)
        
        self.connections = {}
        self.integration_stats = {
            'total_connections': 0,
            'active_connections': 0,
            'sync_operations': 0,
            'last_sync': None
        }
    
    def add_bi_connection(self, connection: BIConnection) -> bool:
        """Add a new BI tool connection"""
        try:
            if connection.connection_type == 'tableau':
                integration = TableauConnector(connection)
            elif connection.connection_type == 'powerbi':
                integration = PowerBIConnector(connection)
            else:
                raise ValueError(f"Unsupported BI tool: {connection.connection_type}")
            
            # Test connection
            if integration.authenticate():
                self.connections[connection.connection_id] = integration
                self.integration_stats['total_connections'] += 1
                self.integration_stats['active_connections'] += 1
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Failed to add BI connection: {e}")
            return False
    
    def sync_data_to_bi(self, connection_id: str, data_type: str = 'events') -> bool:
        """Sync data to a specific BI tool"""
        if connection_id not in self.connections:
            return False
        
        integration = self.connections[connection_id]
        
        try:
            # Prepare data based on type
            if data_type == 'events':
                end_time = datetime.now()
                start_time = end_time - timedelta(days=7)  # Last 7 days
                events = self.warehouse.query_events(start_time=start_time, end_time=end_time)
                data = {'events': [event.to_dict() for event in events]}
            
            elif data_type == 'analytics':
                analytics_summary = self.analytics_processor.get_analytics_summary()
                data = {'analytics': analytics_summary}
            
            else:
                return False
            
            # Push data to BI tool
            success = integration.push_data(data)
            
            if success:
                self.integration_stats['sync_operations'] += 1
                self.integration_stats['last_sync'] = datetime.now()
            
            return success
            
        except Exception as e:
            print(f"Failed to sync data to BI: {e}")
            return False
    
    def get_connection_status(self, connection_id: str) -> Dict[str, Any]:
        """Get status of a specific connection"""
        if connection_id not in self.connections:
            return {'error': 'Connection not found'}
        
        return self.connections[connection_id].get_connection_status()
    
    def get_all_connections_status(self) -> Dict[str, Any]:
        """Get status of all connections"""
        status = {}
        for conn_id, integration in self.connections.items():
            status[conn_id] = integration.get_connection_status()
        
        return {
            'connections': status,
            'summary': self.integration_stats
        }
    
    def create_bi_dashboard(self, connection_id: str, dashboard_config: Dict[str, Any]) -> str:
        """Create a dashboard in a BI tool"""
        if connection_id not in self.connections:
            return ""
        
        integration = self.connections[connection_id]
        return integration.create_dashboard(dashboard_config)
    
    def export_data(self, 
                   export_type: str,
                   start_time: datetime,
                   end_time: datetime,
                   config: ExportConfig) -> Union[str, bytes]:
        """Export data using the export manager"""
        
        if export_type == 'events':
            return self.export_manager.export_events(start_time, end_time, config)
        elif export_type.startswith('report_'):
            report_type = export_type.replace('report_', '')
            return self.export_manager.export_analytics_report(report_type, start_time, end_time, config)
        else:
            raise ValueError(f"Unsupported export type: {export_type}")
    
    def schedule_sync(self, connection_id: str, frequency: str = 'daily'):
        """Schedule regular data synchronization"""
        # In a real implementation, this would set up a scheduler
        # For now, we'll just update the connection config
        if connection_id in self.connections:
            integration = self.connections[connection_id]
            integration.connection.sync_frequency = frequency
            return True
        return False
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get comprehensive integration summary"""
        return {
            'stats': self.integration_stats,
            'connections': {
                conn_id: {
                    'type': integration.connection.connection_type,
                    'status': integration.get_connection_status()['status'],
                    'last_sync': integration.sync_stats['last_sync_time'],
                    'sync_frequency': integration.connection.sync_frequency
                }
                for conn_id, integration in self.connections.items()
            },
            'export_stats': self.export_manager.get_export_stats()
        }