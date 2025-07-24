"""
Integration Example: Adding Analytics to Existing Flask App
Shows how to integrate the analytics system with the existing Deep Tree Echo application.
"""

from flask import Flask, request, jsonify, render_template
from datetime import datetime
import os
import sys

# Add the parent directory to the path to import analytics
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from analytics.flask_integration import integrate_analytics_with_app
from analytics.service import get_analytics_service


def create_enhanced_app():
    """Create Flask app with integrated analytics"""
    
    app = Flask(__name__)
    
    # Configure analytics
    analytics_config = {
        'batch_size': 1000,
        'processing_interval': 60,
        'data_directory': 'data/analytics',
        'backup_directory': 'data/backups'
    }
    
    # Integrate analytics with the app
    integrate_analytics_with_app(app, analytics_config)
    
    # Add existing application routes (simplified examples)
    
    @app.route('/')
    def index():
        """Main application page"""
        return render_template('analytics_dashboard.html')
    
    @app.route('/api/cognitive_process', methods=['POST'])
    def cognitive_process():
        """Cognitive processing endpoint with analytics tracking"""
        try:
            data = request.get_json()
            
            # Simulate cognitive processing
            membrane_type = data.get('membrane_type', 'reasoning')
            user_input = data.get('input', '')
            
            # Simulate processing time and results
            processing_start = datetime.now()
            
            # Mock cognitive processing logic
            response = {
                'output': f'Processed: {user_input}',
                'membrane_used': membrane_type,
                'confidence': 0.95,
                'tokens_used': len(user_input.split())
            }
            
            processing_end = datetime.now()
            processing_time = (processing_end - processing_start).total_seconds() * 1000  # ms
            
            # Track cognitive processing with analytics
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_service.track_cognitive_process(
                    membrane_type=membrane_type,
                    processing_time=processing_time,
                    input_tokens=len(user_input.split()),
                    output_tokens=len(response['output'].split()),
                    success=True,
                    user_id=request.headers.get('X-User-ID'),
                    session_id=request.headers.get('X-Session-ID')
                )
            
            return jsonify(response)
            
        except Exception as e:
            # Track error with analytics
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_service.track_error(
                    error_type='cognitive_processing_error',
                    error_message=str(e),
                    user_id=request.headers.get('X-User-ID'),
                    session_id=request.headers.get('X-Session-ID'),
                    context={'endpoint': '/api/cognitive_process'}
                )
            
            return jsonify({'error': 'Cognitive processing failed'}), 500
    
    @app.route('/api/session/start', methods=['POST'])
    def start_user_session():
        """Start a user session with analytics tracking"""
        try:
            data = request.get_json() or {}
            
            analytics_service = get_analytics_service()
            if analytics_service:
                session_id = analytics_service.start_user_session(
                    user_id=data.get('user_id'),
                    ip_address=request.remote_addr,
                    user_agent=request.headers.get('User-Agent')
                )
                
                return jsonify({
                    'session_id': session_id,
                    'analytics_enabled': True
                })
            else:
                # Fallback without analytics
                return jsonify({
                    'session_id': 'fallback_session',
                    'analytics_enabled': False
                })
                
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/memory/store', methods=['POST'])
    def store_memory():
        """Memory storage endpoint with analytics"""
        try:
            data = request.get_json()
            
            # Simulate memory storage
            memory_item = {
                'id': f'mem_{datetime.now().timestamp()}',
                'content': data.get('content'),
                'type': data.get('type', 'episodic'),
                'timestamp': datetime.now().isoformat()
            }
            
            # Track user action
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_service.track_user_action(
                    user_id=request.headers.get('X-User-ID'),
                    session_id=request.headers.get('X-Session-ID'),
                    action='memory_store',
                    details={
                        'memory_type': memory_item['type'],
                        'content_length': len(str(memory_item['content']))
                    }
                )
            
            return jsonify(memory_item)
            
        except Exception as e:
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_service.track_error(
                    error_type='memory_storage_error',
                    error_message=str(e),
                    user_id=request.headers.get('X-User-ID'),
                    session_id=request.headers.get('X-Session-ID')
                )
            
            return jsonify({'error': 'Memory storage failed'}), 500
    
    @app.route('/api/status', methods=['GET'])
    def system_status():
        """System status endpoint with metrics tracking"""
        try:
            # Simulate system metrics collection
            import psutil
            import random
            
            # Get actual system metrics where possible
            try:
                cpu_usage = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_usage = memory.percent
                disk = psutil.disk_usage('/')
                disk_usage = disk.percent
            except:
                # Fallback to simulated metrics
                cpu_usage = random.uniform(10, 80)
                memory_usage = random.uniform(30, 90)
                disk_usage = random.uniform(20, 70)
            
            status = {
                'system_health': 'healthy',
                'uptime': '24h 35m',
                'cpu_usage': cpu_usage,
                'memory_usage': memory_usage,
                'disk_usage': disk_usage,
                'active_sessions': len(get_analytics_service().active_sessions if get_analytics_service() else {}),
                'timestamp': datetime.now().isoformat()
            }
            
            # Track system metrics
            analytics_service = get_analytics_service()
            if analytics_service:
                analytics_service.track_system_metrics(
                    cpu_usage=cpu_usage,
                    memory_usage=memory_usage,
                    disk_usage=disk_usage,
                    network_io={'bytes_in': 1024, 'bytes_out': 2048},
                    response_time=50.0,  # Simulated
                    throughput=100.0,    # Simulated
                    error_rate=0.5,      # Simulated
                    active_sessions=status['active_sessions'],
                    cache_hit_rate=85.0  # Simulated
                )
            
            return jsonify(status)
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/dashboard')
    def analytics_dashboard():
        """Analytics dashboard page"""
        return render_template('analytics_dashboard.html')
    
    # Health check endpoint
    @app.route('/health')
    def health_check():
        """Application health check"""
        analytics_service = get_analytics_service()
        
        health_info = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'analytics_enabled': analytics_service is not None
        }
        
        if analytics_service:
            health_info['analytics_health'] = analytics_service.get_service_health()
        
        return jsonify(health_info)
    
    return app


def demonstrate_analytics_features():
    """Demonstrate analytics features with sample data"""
    
    print("🧠 Deep Tree Echo Analytics Integration Demo")
    print("=" * 60)
    
    # Create app with analytics
    app = create_enhanced_app()
    
    with app.app_context():
        analytics_service = get_analytics_service()
        
        if not analytics_service:
            print("❌ Analytics service not available")
            return
        
        print("✅ Analytics service initialized successfully")
        
        # Simulate some activity
        print("\n📊 Generating sample analytics data...")
        
        # Create sample sessions
        session_ids = []
        for i in range(5):
            session_id = analytics_service.start_user_session(
                user_id=f"demo_user_{i}",
                ip_address=f"192.168.1.{100+i}",
                user_agent="Demo Browser 1.0"
            )
            session_ids.append(session_id)
        
        # Generate sample events
        for i, session_id in enumerate(session_ids):
            user_id = f"demo_user_{i}"
            
            # User actions
            analytics_service.track_user_action(
                user_id=user_id,
                session_id=session_id,
                action="login",
                details={"login_method": "demo"}
            )
            
            # Cognitive processes
            analytics_service.track_cognitive_process(
                membrane_type="reasoning",
                processing_time=45.2 + i * 10,
                input_tokens=100 + i * 20,
                output_tokens=150 + i * 25,
                success=True,
                user_id=user_id,
                session_id=session_id
            )
            
            # API requests
            analytics_service.track_api_request(
                endpoint="/api/cognitive_process",
                method="POST",
                status_code=200,
                response_time=120.5 + i * 15,
                user_id=user_id,
                session_id=session_id
            )
        
        # Generate system metrics
        analytics_service.track_system_metrics(
            cpu_usage=25.4,
            memory_usage=67.8,
            disk_usage=45.2,
            network_io={"bytes_in": 1024000, "bytes_out": 2048000},
            response_time=95.3,
            throughput=250.0,
            error_rate=1.2,
            active_sessions=len(session_ids),
            cache_hit_rate=87.5
        )
        
        print(f"✅ Generated data for {len(session_ids)} sessions")
        
        # Show analytics summary
        print("\n📈 Analytics Summary:")
        summary = analytics_service.get_analytics_summary()
        
        print(f"   📊 Total Events: {summary['data_warehouse_stats']['total_events']}")
        print(f"   👥 Active Sessions: {summary['data_warehouse_stats']['active_sessions']}")
        print(f"   🧠 Cognitive Metrics: {summary['data_warehouse_stats']['cognitive_metrics_count']}")
        
        print(f"\n🎯 KPIs:")
        for kpi_name, kpi_value in summary['kpis'].items():
            print(f"   {kpi_name}: {kpi_value:.1f}%")
        
        # Show real-time analytics
        print(f"\n⚡ Real-time Analytics:")
        real_time = analytics_service.get_real_time_analytics()
        for metric_name, value in real_time['current_metrics'].items():
            print(f"   {metric_name}: {value:.2f}")
        
        # Show service health
        print(f"\n💚 Service Health:")
        health = analytics_service.get_service_health()
        print(f"   Overall Health: {health['overall_health']}")
        print(f"   Uptime: {health['metrics']['uptime_hours']:.1f} hours")
        
        # Show dashboards
        print(f"\n📊 Available Dashboards:")
        dashboards = analytics_service.get_available_dashboards()
        for dashboard in dashboards:
            print(f"   - {dashboard['name']}: {dashboard['widget_count']} widgets")
        
        print(f"\n🔗 Analytics Integration Complete!")
        print(f"   • Access dashboard: http://localhost:5000/dashboard")
        print(f"   • Analytics API: http://localhost:5000/api/analytics/")
        print(f"   • Health check: http://localhost:5000/health")


if __name__ == '__main__':
    # Run demonstration
    demonstrate_analytics_features()
    
    # Start the Flask app
    print(f"\n🚀 Starting Flask application with integrated analytics...")
    app = create_enhanced_app()
    app.run(debug=True, host='0.0.0.0', port=5000)