"""
Flask Analytics API Integration
Adds analytics endpoints to the existing Flask application.
"""

import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from flask import Blueprint, request, jsonify, current_app
from functools import wraps

from .service import get_analytics_service, initialize_analytics_service
from .integrations import ExportConfig, BIConnection


def create_analytics_blueprint(config: Optional[Dict[str, Any]] = None) -> Blueprint:
    """Create Flask blueprint for analytics endpoints"""
    
    # Initialize analytics service
    analytics_service = initialize_analytics_service(config)
    
    analytics_bp = Blueprint('analytics', __name__, url_prefix='/api/analytics')
    
    def require_analytics_service(f):
        """Decorator to ensure analytics service is available"""
        @wraps(f)
        def decorated_function(*args, **kwargs):
            service = get_analytics_service()
            if not service:
                return jsonify({'error': 'Analytics service not available'}), 503
            return f(service, *args, **kwargs)
        return decorated_function
    
    @analytics_bp.route('/health', methods=['GET'])
    @require_analytics_service
    def get_health(service):
        """Get analytics service health status"""
        try:
            health = service.get_service_health()
            status_code = 200 if health['overall_health'] == 'healthy' else 503
            return jsonify(health), status_code
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/real-time', methods=['GET'])
    @require_analytics_service
    def get_real_time_analytics(service):
        """Get real-time analytics data"""
        try:
            analytics = service.get_real_time_analytics()
            return jsonify(analytics)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/summary', methods=['GET'])
    @require_analytics_service
    def get_analytics_summary(service):
        """Get comprehensive analytics summary"""
        try:
            summary = service.get_analytics_summary()
            return jsonify(summary)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/dashboards', methods=['GET'])
    @require_analytics_service
    def get_dashboards(service):
        """Get available dashboards"""
        try:
            dashboards = service.get_available_dashboards()
            return jsonify({'dashboards': dashboards})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/dashboard/<dashboard_id>', methods=['GET'])
    @require_analytics_service
    def get_dashboard_data(service, dashboard_id):
        """Get dashboard data"""
        try:
            hours_back = request.args.get('hours', 24, type=int)
            dashboard_data = service.get_dashboard_data(dashboard_id, hours_back)
            return jsonify(dashboard_data)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/dashboard', methods=['POST'])
    @require_analytics_service
    def create_dashboard(service):
        """Create a custom dashboard"""
        try:
            dashboard_config = request.get_json()
            if not dashboard_config:
                return jsonify({'error': 'Dashboard configuration required'}), 400
            
            dashboard_id = service.create_custom_dashboard(dashboard_config)
            return jsonify({'dashboard_id': dashboard_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/reports/<report_type>', methods=['GET'])
    @require_analytics_service
    def generate_report(service, report_type):
        """Generate analytics report"""
        try:
            hours_back = request.args.get('hours', 24, type=int)
            report = service.generate_report(report_type, hours_back)
            return jsonify(report)
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/export/<export_type>', methods=['GET'])
    @require_analytics_service
    def export_data(service, export_type):
        """Export analytics data"""
        try:
            format_type = request.args.get('format', 'json')
            hours_back = request.args.get('hours', 24, type=int)
            include_metadata = request.args.get('metadata', 'true').lower() == 'true'
            
            exported_data = service.export_data(
                export_type, format_type, hours_back, include_metadata
            )
            
            # Set appropriate content type
            if format_type == 'csv':
                content_type = 'text/csv'
            elif format_type == 'xml':
                content_type = 'application/xml'
            else:
                content_type = 'application/json'
            
            response = current_app.response_class(
                exported_data,
                mimetype=content_type
            )
            
            # Add download headers
            filename = f"{export_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_type}"
            response.headers['Content-Disposition'] = f'attachment; filename={filename}'
            
            return response
            
        except ValueError as e:
            return jsonify({'error': str(e)}), 400
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/track/action', methods=['POST'])
    @require_analytics_service
    def track_user_action(service):
        """Track a user action"""
        try:
            data = request.get_json()
            if not data or 'action' not in data:
                return jsonify({'error': 'Action required'}), 400
            
            event_id = service.track_user_action(
                user_id=data.get('user_id'),
                session_id=data.get('session_id'),
                action=data['action'],
                details=data.get('details', {})
            )
            
            return jsonify({'event_id': event_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/track/api-request', methods=['POST'])
    @require_analytics_service
    def track_api_request(service):
        """Track an API request"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Request data required'}), 400
            
            required_fields = ['endpoint', 'method', 'status_code', 'response_time']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'{field} required'}), 400
            
            event_id = service.track_api_request(
                endpoint=data['endpoint'],
                method=data['method'],
                status_code=data['status_code'],
                response_time=data['response_time'],
                user_id=data.get('user_id'),
                session_id=data.get('session_id'),
                ip_address=data.get('ip_address')
            )
            
            return jsonify({'event_id': event_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/track/cognitive-process', methods=['POST'])
    @require_analytics_service
    def track_cognitive_process(service):
        """Track a cognitive processing operation"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Process data required'}), 400
            
            required_fields = ['membrane_type', 'processing_time', 'input_tokens', 'output_tokens', 'success']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'{field} required'}), 400
            
            event_id = service.track_cognitive_process(
                membrane_type=data['membrane_type'],
                processing_time=data['processing_time'],
                input_tokens=data['input_tokens'],
                output_tokens=data['output_tokens'],
                success=data['success'],
                user_id=data.get('user_id'),
                session_id=data.get('session_id'),
                error_message=data.get('error_message')
            )
            
            return jsonify({'event_id': event_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/track/system-metrics', methods=['POST'])
    @require_analytics_service
    def track_system_metrics(service):
        """Track system performance metrics"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Metrics data required'}), 400
            
            required_fields = ['cpu_usage', 'memory_usage', 'disk_usage', 'response_time', 
                             'throughput', 'error_rate', 'active_sessions', 'cache_hit_rate']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'{field} required'}), 400
            
            service.track_system_metrics(
                cpu_usage=data['cpu_usage'],
                memory_usage=data['memory_usage'],
                disk_usage=data['disk_usage'],
                network_io=data.get('network_io', {}),
                response_time=data['response_time'],
                throughput=data['throughput'],
                error_rate=data['error_rate'],
                active_sessions=data['active_sessions'],
                cache_hit_rate=data['cache_hit_rate']
            )
            
            return jsonify({'status': 'recorded'}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/track/error', methods=['POST'])
    @require_analytics_service
    def track_error(service):
        """Track an error occurrence"""
        try:
            data = request.get_json()
            if not data or 'error_type' not in data or 'error_message' not in data:
                return jsonify({'error': 'Error type and message required'}), 400
            
            event_id = service.track_error(
                error_type=data['error_type'],
                error_message=data['error_message'],
                stack_trace=data.get('stack_trace'),
                user_id=data.get('user_id'),
                session_id=data.get('session_id'),
                context=data.get('context', {})
            )
            
            return jsonify({'event_id': event_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/session/start', methods=['POST'])
    @require_analytics_service
    def start_session(service):
        """Start a new user session"""
        try:
            data = request.get_json() or {}
            
            session_id = service.start_user_session(
                user_id=data.get('user_id'),
                ip_address=data.get('ip_address') or request.remote_addr,
                user_agent=data.get('user_agent') or request.headers.get('User-Agent')
            )
            
            return jsonify({'session_id': session_id}), 201
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/session/<session_id>/end', methods=['POST'])
    @require_analytics_service
    def end_session(service, session_id):
        """End a user session"""
        try:
            success = service.end_user_session(session_id)
            if success:
                return jsonify({'status': 'session_ended'}), 200
            else:
                return jsonify({'error': 'Session not found'}), 404
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/bi/connections', methods=['GET'])
    @require_analytics_service
    def get_bi_connections(service):
        """Get BI connections status"""
        try:
            connections = service.integration_manager.get_all_connections_status()
            return jsonify(connections)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/bi/connection', methods=['POST'])
    @require_analytics_service
    def add_bi_connection(service):
        """Add a new BI connection"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({'error': 'Connection configuration required'}), 400
            
            required_fields = ['connection_id', 'connection_type', 'connection_params']
            for field in required_fields:
                if field not in data:
                    return jsonify({'error': f'{field} required'}), 400
            
            success = service.add_bi_connection(data)
            
            if success:
                return jsonify({'status': 'connection_added'}), 201
            else:
                return jsonify({'error': 'Failed to add connection'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/bi/sync/<connection_id>', methods=['POST'])
    @require_analytics_service
    def sync_to_bi(service, connection_id):
        """Sync data to BI tool"""
        try:
            data = request.get_json() or {}
            data_type = data.get('data_type', 'events')
            
            success = service.sync_to_bi(connection_id, data_type)
            
            if success:
                return jsonify({'status': 'sync_completed'}), 200
            else:
                return jsonify({'error': 'Sync failed'}), 400
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @analytics_bp.route('/maintenance/cleanup', methods=['POST'])
    @require_analytics_service
    def cleanup_old_data(service):
        """Clean up old analytics data"""
        try:
            data = request.get_json() or {}
            retention_days = data.get('retention_days', 365)
            
            results = service.cleanup_old_data(retention_days)
            return jsonify({
                'status': 'cleanup_completed',
                'results': results
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    # Error handlers
    @analytics_bp.errorhandler(404)
    def not_found(error):
        return jsonify({'error': 'Analytics endpoint not found'}), 404
    
    @analytics_bp.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({'error': 'Method not allowed for this analytics endpoint'}), 405
    
    @analytics_bp.errorhandler(500)
    def internal_error(error):
        return jsonify({'error': 'Internal analytics service error'}), 500
    
    return analytics_bp


def add_analytics_middleware(app):
    """Add analytics middleware to automatically track requests"""
    
    def track_request():
        """Track incoming requests automatically"""
        service = get_analytics_service()
        if service and request.endpoint and not request.endpoint.startswith('analytics.'):
            # Extract session info from request
            session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
            user_id = request.headers.get('X-User-ID')
            
            # Record request start time
            request.start_time = datetime.now()
    
    def track_response(response):
        """Track response metrics automatically"""
        service = get_analytics_service()
        if service and hasattr(request, 'start_time') and request.endpoint and not request.endpoint.startswith('analytics.'):
            # Calculate response time
            response_time = (datetime.now() - request.start_time).total_seconds() * 1000  # milliseconds
            
            # Extract session info
            session_id = request.headers.get('X-Session-ID') or request.cookies.get('session_id')
            user_id = request.headers.get('X-User-ID')
            
            # Track the API request
            try:
                service.track_api_request(
                    endpoint=request.endpoint or request.path,
                    method=request.method,
                    status_code=response.status_code,
                    response_time=response_time,
                    user_id=user_id,
                    session_id=session_id,
                    ip_address=request.remote_addr
                )
            except Exception:
                # Don't let analytics tracking break the main application
                pass
        
        return response
    
    app.before_request(track_request)
    app.after_request(track_response)


def integrate_analytics_with_app(app, config: Optional[Dict[str, Any]] = None):
    """Complete integration of analytics with Flask app"""
    
    # Register analytics blueprint
    analytics_bp = create_analytics_blueprint(config)
    app.register_blueprint(analytics_bp)
    
    # Add middleware for automatic request tracking
    add_analytics_middleware(app)
    
    # Add shutdown handler
    import atexit
    from .service import shutdown_analytics_service
    atexit.register(shutdown_analytics_service)
    
    return analytics_bp