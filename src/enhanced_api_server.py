"""
Enhanced API Server with Comprehensive Ecosystem
Integrates RESTful API, GraphQL, SDK support, integrations, and plugins
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import logging
from datetime import datetime
import asyncio
from typing import Dict, Any, Optional

# Import our API ecosystem components
from api.rest_api import create_enhanced_api_blueprint, setup_flask_limiter
from api.simple_graphql import create_simple_graphql_blueprint
from api.documentation import create_documentation_blueprint
from integrations.base import IntegrationManager, IntegrationEvent, create_webhook_integration
from plugins.base import plugin_registry

# Import existing components
from persistent_memory import PersistentMemorySystem
from rwkv_echo_integration import DeepTreeEchoRWKV, RWKVConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class EnhancedEchoServer:
    """Enhanced Echo API server with full ecosystem support"""
    
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app, origins="*")
        
        # Initialize components
        self.echo_system = None
        self.memory_system = None
        self.integration_manager = IntegrationManager()
        self.limiter = None
        
        # Server statistics
        self.stats = {
            'start_time': datetime.now(),
            'total_requests': 0,
            'api_requests': 0,
            'graphql_requests': 0,
            'plugin_activations': 0,
            'integration_events': 0
        }
        
        # Configuration
        self.config = {
            'model_path': os.environ.get('ECHO_MODELS_PATH', '/models') + '/RWKV-x070-World-1.5B-v2.8.pth',
            'model_size': '1.5B',
            'context_length': 2048,
            'temperature': 0.8,
            'webvm_mode': os.environ.get('ECHO_WEBVM_MODE', 'false').lower() == 'true',
            'memory_limit': int(os.environ.get('ECHO_MEMORY_LIMIT', '600')),
            'enable_integrations': os.environ.get('ECHO_ENABLE_INTEGRATIONS', 'true').lower() == 'true',
            'enable_plugins': os.environ.get('ECHO_ENABLE_PLUGINS', 'true').lower() == 'true',
            'plugin_paths': os.environ.get('ECHO_PLUGIN_PATHS', 'plugins').split(',')
        }
    
    async def initialize_echo_system(self):
        """Initialize the Deep Tree Echo cognitive system"""
        try:
            logger.info("Initializing Deep Tree Echo system...")
            
            # Initialize memory system
            self.memory_system = PersistentMemorySystem()
            
            # Try to initialize RWKV system
            try:
                config = RWKVConfig(
                    model_path=self.config['model_path'],
                    model_size=self.config['model_size'],
                    context_length=self.config['context_length'],
                    temperature=self.config['temperature']
                )
                self.echo_system = DeepTreeEchoRWKV(config)
                logger.info("RWKV Echo system initialized successfully")
            except Exception as e:
                logger.warning(f"RWKV system unavailable, using mock: {e}")
                self.echo_system = MockEchoSystem()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Echo system: {e}")
            return False
    
    async def initialize_integrations(self):
        """Initialize third-party integrations"""
        if not self.config['enable_integrations']:
            logger.info("Integrations disabled by configuration")
            return
        
        try:
            logger.info("Initializing integrations...")
            
            # Example webhook integration (would be configured via environment)
            webhook_url = os.environ.get('ECHO_WEBHOOK_URL')
            if webhook_url:
                webhook = create_webhook_integration(
                    "main_webhook",
                    webhook_url,
                    os.environ.get('ECHO_WEBHOOK_SECRET')
                )
                self.integration_manager.register_integration(webhook)
            
            # Initialize all registered integrations
            results = await self.integration_manager.initialize_all()
            logger.info(f"Integration initialization results: {results}")
            
        except Exception as e:
            logger.error(f"Error initializing integrations: {e}")
    
    async def initialize_plugins(self):
        """Initialize plugin system"""
        if not self.config['enable_plugins']:
            logger.info("Plugins disabled by configuration")
            return
        
        try:
            logger.info("Initializing plugin system...")
            
            # Add plugin paths
            for path in self.config['plugin_paths']:
                plugin_registry.add_plugin_path(path.strip())
            
            # Load plugins from paths
            for path in plugin_registry.plugin_paths:
                results = await plugin_registry.load_plugins_from_path(path)
                logger.info(f"Plugin loading results from {path}: {results}")
            
            # Initialize all plugins
            if self.echo_system:
                results = await plugin_registry.initialize_all_plugins(self.echo_system)
                logger.info(f"Plugin initialization results: {results}")
            
        except Exception as e:
            logger.error(f"Error initializing plugins: {e}")
    
    def setup_routes(self):
        """Setup all routes and blueprints"""
        
        # Setup rate limiting
        self.limiter = setup_flask_limiter(self.app)
        
        # Register enhanced REST API blueprints
        api_v1, api_v2 = create_enhanced_api_blueprint(self.echo_system)
        self.app.register_blueprint(api_v1)
        self.app.register_blueprint(api_v2)
        
        # Register GraphQL blueprint
        graphql_bp = create_simple_graphql_blueprint(self.echo_system)
        self.app.register_blueprint(graphql_bp)
        
        # Register documentation blueprint
        docs_bp = create_documentation_blueprint()
        self.app.register_blueprint(docs_bp)
        
        # Add middleware for request tracking
        @self.app.before_request
        def before_request():
            self.stats['total_requests'] += 1
            
            if request.path.startswith('/api/'):
                self.stats['api_requests'] += 1
            elif request.path.startswith('/graphql'):
                self.stats['graphql_requests'] += 1
        
        # Enhanced status endpoint
        @self.app.route('/api/ecosystem/status')
        def ecosystem_status():
            """Get comprehensive ecosystem status"""
            try:
                status = {
                    'timestamp': datetime.now().isoformat(),
                    'server': {
                        'uptime': (datetime.now() - self.stats['start_time']).total_seconds(),
                        'stats': self.stats,
                        'config': {
                            'webvm_mode': self.config['webvm_mode'],
                            'integrations_enabled': self.config['enable_integrations'],
                            'plugins_enabled': self.config['enable_plugins']
                        }
                    },
                    'echo_system': {
                        'initialized': self.echo_system is not None,
                        'type': 'RWKV' if hasattr(self.echo_system, 'model') else 'Mock'
                    },
                    'memory_system': {
                        'initialized': self.memory_system is not None,
                        'type': 'Persistent'
                    },
                    'integrations': {
                        'manager_active': self.integration_manager is not None,
                        'integrations': self.integration_manager.get_integration_status() if self.integration_manager else {}
                    },
                    'plugins': {
                        'registry_active': True,
                        'total_plugins': len(plugin_registry.plugins),
                        'active_plugins': len(plugin_registry.get_active_plugins()),
                        'plugin_info': plugin_registry.get_plugin_info()
                    },
                    'api_features': {
                        'rest_api': True,
                        'graphql_api': True,
                        'rate_limiting': True,
                        'versioning': True,
                        'documentation': True,
                        'sdk_support': True
                    }
                }
                
                return jsonify({
                    'success': True,
                    'data': status,
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                })
                
            except Exception as e:
                logger.error(f"Error getting ecosystem status: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }), 500
        
        # Plugin management endpoints
        @self.app.route('/api/ecosystem/plugins', methods=['GET'])
        def list_plugins():
            """List all plugins"""
            try:
                return jsonify({
                    'success': True,
                    'data': plugin_registry.get_plugin_info(),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }), 500
        
        @self.app.route('/api/ecosystem/plugins/<plugin_name>/activate', methods=['POST'])
        async def activate_plugin(plugin_name):
            """Activate a plugin"""
            try:
                success = await plugin_registry.activate_plugin(plugin_name)
                if success:
                    self.stats['plugin_activations'] += 1
                
                return jsonify({
                    'success': success,
                    'message': f"Plugin '{plugin_name}' {'activated' if success else 'failed to activate'}",
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }), 500
        
        # Integration management endpoints
        @self.app.route('/api/ecosystem/integrations', methods=['GET'])
        def list_integrations():
            """List all integrations"""
            try:
                return jsonify({
                    'success': True,
                    'data': self.integration_manager.get_integration_status(),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                })
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }), 500
        
        @self.app.route('/api/ecosystem/integrations/event', methods=['POST'])
        async def send_integration_event():
            """Send event to all integrations"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({
                        'success': False,
                        'error': 'Request data is required',
                        'timestamp': datetime.now().isoformat(),
                        'version': 'v1'
                    }), 400
                
                event = IntegrationEvent(
                    event_type=data.get('event_type', 'custom'),
                    data=data.get('data', {}),
                    source='api',
                    correlation_id=data.get('correlation_id')
                )
                
                results = await self.integration_manager.send_event_to_all(event)
                self.stats['integration_events'] += 1
                
                return jsonify({
                    'success': True,
                    'data': {
                        'event_sent': True,
                        'integration_results': results
                    },
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                })
                
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat(),
                    'version': 'v1'
                }), 500
        
        # SDK download endpoints
        @self.app.route('/api/sdks')
        def sdk_downloads():
            """SDK downloads page"""
            sdk_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo SDK Downloads</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .sdk-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 30px;
        }
        .sdk-card {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .sdk-card h3 {
            color: #4CAF50;
            margin-bottom: 15px;
            font-size: 1.5em;
        }
        .code-block {
            background: rgba(0,0,0,0.3);
            padding: 15px;
            border-radius: 8px;
            font-family: monospace;
            margin: 15px 0;
            overflow-x: auto;
        }
        .download-btn {
            display: inline-block;
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin: 10px 10px 10px 0;
            transition: background 0.2s ease;
        }
        .download-btn:hover {
            background: #45a049;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛠️ Deep Tree Echo SDKs</h1>
            <p>Official software development kits for rapid integration</p>
        </div>
        
        <div class="sdk-grid">
            <div class="sdk-card">
                <h3>🐍 Python SDK</h3>
                <p>Full-featured Python client with async support, type hints, and comprehensive error handling.</p>
                
                <h4>Installation:</h4>
                <div class="code-block">pip install deep-tree-echo-sdk</div>
                
                <h4>Quick Start:</h4>
                <div class="code-block">from echo_sdk import EchoClient

client = EchoClient(api_key="your_key")
result = client.process_cognitive_input("Hello, Echo!")</div>
                
                <a href="/api/sdk/python/download" class="download-btn">Download Source</a>
                <a href="/api/docs" class="download-btn">View Docs</a>
            </div>
            
            <div class="sdk-card">
                <h3>🟨 JavaScript SDK</h3>
                <p>Modern TypeScript/JavaScript SDK with Promise-based API and WebSocket support.</p>
                
                <h4>Installation:</h4>
                <div class="code-block">npm install @deeptreeecho/sdk</div>
                
                <h4>Quick Start:</h4>
                <div class="code-block">import { EchoClient } from '@deeptreeecho/sdk';

const client = new EchoClient({ apiKey: 'your_key' });
await client.processCognitiveInput('Hello, Echo!');</div>
                
                <a href="/api/sdk/javascript/download" class="download-btn">Download Source</a>
                <a href="/api/docs" class="download-btn">View Docs</a>
            </div>
            
            <div class="sdk-card">
                <h3>⚡ CLI Tool</h3>
                <p>Command-line interface for system management, testing, and automation.</p>
                
                <h4>Installation:</h4>
                <div class="code-block">pip install deep-tree-echo-cli</div>
                
                <h4>Quick Start:</h4>
                <div class="code-block"># Set your API key
export ECHO_API_KEY="your_key"

# Process cognitive input
echo cognitive process "What is consciousness?"

# Check system status
echo system status</div>
                
                <a href="/api/sdk/cli/download" class="download-btn">Download Source</a>
                <a href="/api/docs" class="download-btn">View Docs</a>
            </div>
        </div>
    </div>
</body>
</html>
            """
            return sdk_html
        
        # Main dashboard (enhanced)
        @self.app.route('/')
        def enhanced_dashboard():
            """Enhanced main dashboard"""
            dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo - API Ecosystem</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        .header {
            text-align: center;
            margin-bottom: 40px;
        }
        .header h1 {
            font-size: 3.5em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .features-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .feature-card {
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
        }
        .feature-card h3 {
            color: #4CAF50;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        .feature-card a {
            display: inline-block;
            padding: 10px 20px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            margin-top: 15px;
            transition: background 0.2s ease;
        }
        .feature-card a:hover {
            background: #45a049;
        }
        .quick-links {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
        }
        .link-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
        }
        .quick-link {
            padding: 15px;
            background: rgba(255,255,255,0.1);
            border-radius: 8px;
            text-decoration: none;
            color: white;
            font-weight: bold;
            transition: background 0.2s ease;
        }
        .quick-link:hover {
            background: rgba(255,255,255,0.2);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Deep Tree Echo</h1>
            <p>Comprehensive API Ecosystem for Cognitive Architecture</p>
        </div>
        
        <div class="features-grid">
            <div class="feature-card">
                <h3>📖 RESTful API</h3>
                <p>Complete REST API with versioning, rate limiting, and comprehensive endpoints for all cognitive operations.</p>
                <a href="/api/docs/swagger">Explore API</a>
            </div>
            
            <div class="feature-card">
                <h3>🔍 GraphQL API</h3>
                <p>Flexible GraphQL interface for complex queries and real-time data exploration.</p>
                <a href="/graphql/playground">GraphQL Playground</a>
            </div>
            
            <div class="feature-card">
                <h3>🛠️ Official SDKs</h3>
                <p>Python, JavaScript, and CLI tools for rapid development and integration.</p>
                <a href="/api/sdks">Download SDKs</a>
            </div>
            
            <div class="feature-card">
                <h3>🔌 Integrations</h3>
                <p>Built-in support for webhooks, databases, cloud services, and third-party platforms.</p>
                <a href="/api/ecosystem/integrations">View Integrations</a>
            </div>
            
            <div class="feature-card">
                <h3>🧩 Plugin System</h3>
                <p>Extensible plugin architecture for custom cognitive processors and memory providers.</p>
                <a href="/api/ecosystem/plugins">Manage Plugins</a>
            </div>
            
            <div class="feature-card">
                <h3>📊 Analytics</h3>
                <p>Comprehensive usage analytics, performance metrics, and system monitoring.</p>
                <a href="/api/v2/analytics/usage">View Analytics</a>
            </div>
        </div>
        
        <div class="quick-links">
            <h3>🚀 Quick Links</h3>
            <div class="link-grid">
                <a href="/api/docs" class="quick-link">📚 Documentation</a>
                <a href="/api/ecosystem/status" class="quick-link">📊 System Status</a>
                <a href="/health" class="quick-link">❤️ Health Check</a>
                <a href="/graphql" class="quick-link">🔍 GraphQL</a>
            </div>
        </div>
    </div>
</body>
</html>
            """
            return dashboard_html
    
    async def start_background_tasks(self):
        """Start background tasks for integrations and plugins"""
        try:
            # Health check task for integrations
            if self.integration_manager:
                async def integration_health_check():
                    while True:
                        await asyncio.sleep(300)  # Check every 5 minutes
                        await self.integration_manager.health_check_all()
                
                asyncio.create_task(integration_health_check())
            
            logger.info("Background tasks started")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def initialize_all(self):
        """Initialize all components"""
        try:
            # Initialize Echo system
            await self.initialize_echo_system()
            
            # Initialize integrations
            await self.initialize_integrations()
            
            # Initialize plugins
            await self.initialize_plugins()
            
            # Setup routes
            self.setup_routes()
            
            # Start background tasks
            await self.start_background_tasks()
            
            logger.info("Enhanced Echo server initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing enhanced server: {e}")
            return False

class MockEchoSystem:
    """Mock Echo system for testing"""
    
    def process_cognitive_input(self, input_text: str) -> Dict[str, Any]:
        """Mock cognitive processing"""
        return {
            'input_text': input_text,
            'integrated_response': f"Enhanced mock response for: {input_text}",
            'processing_time': 0.045,
            'session_id': 'mock-session',
            'confidence': 0.92
        }

# Create enhanced server instance
enhanced_server = EnhancedEchoServer()

if __name__ == '__main__':
    # Initialize server
    async def init_server():
        success = await enhanced_server.initialize_all()
        if not success:
            logger.error("Failed to initialize server")
            return
    
    # Run initialization
    asyncio.run(init_server())
    
    # Configure Flask for deployment
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Enhanced Deep Tree Echo API Server on {host}:{port}")
    logger.info(f"WebVM Mode: {enhanced_server.config['webvm_mode']}")
    logger.info(f"Integrations: {'Enabled' if enhanced_server.config['enable_integrations'] else 'Disabled'}")
    logger.info(f"Plugins: {'Enabled' if enhanced_server.config['enable_plugins'] else 'Disabled'}")
    
    enhanced_server.app.run(host=host, port=port, debug=debug, threaded=True)