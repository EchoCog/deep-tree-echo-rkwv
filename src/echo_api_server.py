"""
Deep Tree Echo API Server
Flask-based API server for WebVM deployment with RWKV integration
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import json
import time
import threading
import logging
from typing import Dict, Any, List
from datetime import datetime

# Import our RWKV-Echo integration
try:
    from rwkv_echo_integration import DeepTreeEchoRWKV, RWKVConfig, EchoMemoryState
    INTEGRATION_AVAILABLE = True
except ImportError:
    INTEGRATION_AVAILABLE = False
    print("Warning: RWKV-Echo integration not available")

# Import EchoLisp for echo structure generation
try:
    from echo_lisp import EchoLisp
    ECHO_LISP_AVAILABLE = True
except ImportError:
    ECHO_LISP_AVAILABLE = False
    print("Warning: EchoLisp not available")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")  # Allow all origins for WebVM deployment

# Global variables
echo_system = None
system_status = {
    'initialized': False,
    'model_loaded': False,
    'last_activity': None,
    'total_requests': 0,
    'error_count': 0
}

# Configuration
CONFIG = {
    'model_path': os.environ.get('ECHO_MODELS_PATH', '/models') + '/RWKV-x070-World-1.5B-v2.8.pth',
    'model_size': '1.5B',
    'context_length': 2048,
    'temperature': 0.8,
    'webvm_mode': os.environ.get('ECHO_WEBVM_MODE', 'false').lower() == 'true',
    'memory_limit': int(os.environ.get('ECHO_MEMORY_LIMIT', '600')),
    'state_file': '/tmp/echo_cognitive_state.json'
}

def initialize_echo_system():
    """Initialize the Deep Tree Echo system with RWKV"""
    global echo_system, system_status
    
    try:
        logger.info("Initializing Deep Tree Echo system...")
        
        if INTEGRATION_AVAILABLE:
            config = RWKVConfig(
                model_path=CONFIG['model_path'],
                model_size=CONFIG['model_size'],
                context_length=CONFIG['context_length'],
                temperature=CONFIG['temperature']
            )
            
            echo_system = DeepTreeEchoRWKV(config)
            
            # Try to load previous cognitive state
            if os.path.exists(CONFIG['state_file']):
                echo_system.load_cognitive_state(CONFIG['state_file'])
                logger.info("Previous cognitive state loaded")
            
            system_status['initialized'] = True
            system_status['model_loaded'] = True
            logger.info("Deep Tree Echo system initialized successfully")
        else:
            # Mock system for testing
            echo_system = MockEchoSystem()
            system_status['initialized'] = True
            system_status['model_loaded'] = False
            logger.info("Mock Echo system initialized (RWKV not available)")
            
    except Exception as e:
        logger.error(f"Failed to initialize Echo system: {e}")
        system_status['error_count'] += 1
        echo_system = MockEchoSystem()

class MockEchoSystem:
    """Mock Echo system for testing without RWKV"""
    
    def __init__(self):
        self.cognitive_state = {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'current_goals': []},
            'temporal_context': []
        }
    
    def process_cognitive_input(self, input_text: str) -> Dict[str, Any]:
        """Mock cognitive processing"""
        return {
            'input': input_text,
            'memory_output': {'response': f"Mock memory response for: {input_text}"},
            'reasoning_output': {'response': f"Mock reasoning response for: {input_text}"},
            'grammar_output': {'grammatical_analysis': f"Mock grammar analysis for: {input_text}"},
            'integrated_response': f"Mock integrated response for: {input_text}",
            'processing_time': 0.1,
            'cognitive_state_summary': {
                'declarative_memory_items': 0,
                'procedural_memory_items': 0,
                'episodic_memory_items': 0,
                'temporal_context_length': 0,
                'current_goals': 0
            }
        }
    
    def save_cognitive_state(self, filepath: str) -> None:
        """Mock save state"""
        pass
    
    def _summarize_cognitive_state(self) -> Dict[str, Any]:
        """Mock state summary"""
        return {
            'declarative_memory_items': 0,
            'procedural_memory_items': 0,
            'episodic_memory_items': 0,
            'temporal_context_length': 0,
            'current_goals': 0
        }

# API Routes

@app.route('/')
def index():
    """Main dashboard page"""
    dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo API Dashboard</title>
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
            margin-bottom: 30px;
        }
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .status-card {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .status-card h3 {
            margin: 0 0 15px 0;
            color: #4CAF50;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-online { background: #4CAF50; }
        .status-offline { background: #f44336; }
        .status-warning { background: #ff9800; }
        .chat-container {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chat-messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid rgba(255,255,255,0.3);
            padding: 15px;
            margin-bottom: 15px;
            background: rgba(0,0,0,0.2);
            border-radius: 5px;
        }
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 5px;
        }
        .user-message {
            background: rgba(33, 150, 243, 0.3);
            text-align: right;
        }
        .echo-message {
            background: rgba(76, 175, 80, 0.3);
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        .input-group input {
            flex: 1;
            padding: 10px;
            border: none;
            border-radius: 5px;
            background: rgba(255,255,255,0.9);
            color: #333;
        }
        .input-group button {
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            background: #4CAF50;
            color: white;
            cursor: pointer;
            font-weight: bold;
        }
        .input-group button:hover {
            background: #45a049;
        }
        .api-docs {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            margin-top: 20px;
        }
        .endpoint {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            margin: 10px 0;
            border-radius: 5px;
            font-family: monospace;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Deep Tree Echo</h1>
            <p>Cognitive Architecture API with RWKV Integration</p>
        </div>
        
        <div class="status-grid">
            <div class="status-card">
                <h3>System Status</h3>
                <p><span class="status-indicator status-online"></span>API Server: Online</p>
                <p><span class="status-indicator status-{{ 'online' if status.initialized else 'offline' }}"></span>Echo System: {{ 'Initialized' if status.initialized else 'Not Initialized' }}</p>
                <p><span class="status-indicator status-{{ 'online' if status.model_loaded else 'warning' }}"></span>RWKV Model: {{ 'Loaded' if status.model_loaded else 'Mock Mode' }}</p>
            </div>
            
            <div class="status-card">
                <h3>Performance Metrics</h3>
                <p>Total Requests: {{ status.total_requests }}</p>
                <p>Error Count: {{ status.error_count }}</p>
                <p>Last Activity: {{ status.last_activity or 'None' }}</p>
                <p>Memory Limit: {{ config.memory_limit }}MB</p>
            </div>
            
            <div class="status-card">
                <h3>Configuration</h3>
                <p>Model Size: {{ config.model_size }}</p>
                <p>Context Length: {{ config.context_length }}</p>
                <p>Temperature: {{ config.temperature }}</p>
                <p>WebVM Mode: {{ config.webvm_mode }}</p>
            </div>
        </div>
        
        <div class="chat-container">
            <h3>Interactive Cognitive Interface</h3>
            <div class="chat-messages" id="chatMessages">
                <div class="message echo-message">
                    <strong>Echo System:</strong> Hello! I'm the Deep Tree Echo cognitive architecture. Ask me anything and I'll process it through my memory, reasoning, and grammar membranes.
                </div>
            </div>
            <div class="input-group">
                <input type="text" id="userInput" placeholder="Enter your message..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        
        <div class="api-docs">
            <h3>API Endpoints</h3>
            <div class="endpoint">POST /api/process - Process cognitive input</div>
            <div class="endpoint">GET /api/status - Get system status</div>
            <div class="endpoint">GET /api/state - Get cognitive state summary</div>
            <div class="endpoint">POST /api/save_state - Save cognitive state</div>
            <div class="endpoint">POST /api/load_state - Load cognitive state</div>
        </div>
    </div>
    
    <script>
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const messages = document.getElementById('chatMessages');
            const message = input.value.trim();
            
            if (!message) return;
            
            // Add user message
            const userDiv = document.createElement('div');
            userDiv.className = 'message user-message';
            userDiv.innerHTML = `<strong>You:</strong> ${message}`;
            messages.appendChild(userDiv);
            
            // Clear input
            input.value = '';
            
            // Add loading message
            const loadingDiv = document.createElement('div');
            loadingDiv.className = 'message echo-message';
            loadingDiv.innerHTML = '<strong>Echo System:</strong> Processing...';
            messages.appendChild(loadingDiv);
            messages.scrollTop = messages.scrollHeight;
            
            try {
                // Send to API
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ input: message })
                });
                
                const result = await response.json();
                
                // Remove loading message
                messages.removeChild(loadingDiv);
                
                // Add echo response
                const echoDiv = document.createElement('div');
                echoDiv.className = 'message echo-message';
                echoDiv.innerHTML = `
                    <strong>Echo System:</strong> ${result.integrated_response}<br>
                    <small>Processing time: ${result.processing_time.toFixed(3)}s</small>
                `;
                messages.appendChild(echoDiv);
                
            } catch (error) {
                // Remove loading message
                messages.removeChild(loadingDiv);
                
                // Add error message
                const errorDiv = document.createElement('div');
                errorDiv.className = 'message echo-message';
                errorDiv.innerHTML = `<strong>Echo System:</strong> Error processing request: ${error.message}`;
                messages.appendChild(errorDiv);
            }
            
            messages.scrollTop = messages.scrollHeight;
        }
        
        // Auto-refresh status every 30 seconds
        setInterval(() => {
            location.reload();
        }, 30000);
    </script>
</body>
</html>
    """
    
    return render_template_string(dashboard_html, status=system_status, config=CONFIG)

@app.route('/api/status')
def get_status():
    """Get system status"""
    global system_status
    
    status_info = {
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'system_status': system_status,
        'config': CONFIG,
        'cognitive_state': echo_system._summarize_cognitive_state() if echo_system else {}
    }
    
    return jsonify(status_info)

@app.route('/api/process', methods=['POST'])
def process_input():
    """Process cognitive input through Deep Tree Echo"""
    global echo_system, system_status
    
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'No input provided'}), 400
        
        input_text = data['input']
        
        if not echo_system:
            return jsonify({'error': 'Echo system not initialized'}), 500
        
        # Process through Echo system
        result = echo_system.process_cognitive_input(input_text)
        
        # Update system status
        system_status['total_requests'] += 1
        system_status['last_activity'] = datetime.now().isoformat()
        
        # Save cognitive state periodically
        if system_status['total_requests'] % 10 == 0:
            try:
                echo_system.save_cognitive_state(CONFIG['state_file'])
            except Exception as e:
                logger.warning(f"Failed to save cognitive state: {e}")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        system_status['error_count'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/state')
def get_cognitive_state():
    """Get cognitive state summary"""
    global echo_system
    
    if not echo_system:
        return jsonify({'error': 'Echo system not initialized'}), 500
    
    try:
        state_summary = echo_system._summarize_cognitive_state()
        return jsonify({
            'cognitive_state': state_summary,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error getting cognitive state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_state', methods=['POST'])
def save_cognitive_state():
    """Save cognitive state to file"""
    global echo_system
    
    if not echo_system:
        return jsonify({'error': 'Echo system not initialized'}), 500
    
    try:
        data = request.get_json()
        filepath = data.get('filepath', CONFIG['state_file'])
        
        echo_system.save_cognitive_state(filepath)
        
        return jsonify({
            'message': 'Cognitive state saved successfully',
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error saving cognitive state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/load_state', methods=['POST'])
def load_cognitive_state():
    """Load cognitive state from file"""
    global echo_system
    
    if not echo_system:
        return jsonify({'error': 'Echo system not initialized'}), 500
    
    try:
        data = request.get_json()
        filepath = data.get('filepath', CONFIG['state_file'])
        
        if not os.path.exists(filepath):
            return jsonify({'error': 'State file not found'}), 404
        
        echo_system.load_cognitive_state(filepath)
        
        return jsonify({
            'message': 'Cognitive state loaded successfully',
            'filepath': filepath,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error loading cognitive state: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/config', methods=['GET', 'POST'])
def manage_config():
    """Get or update configuration"""
    global CONFIG
    
    if request.method == 'GET':
        return jsonify(CONFIG)
    
    elif request.method == 'POST':
        try:
            data = request.get_json()
            
            # Update configuration
            for key, value in data.items():
                if key in CONFIG:
                    CONFIG[key] = value
            
            return jsonify({
                'message': 'Configuration updated successfully',
                'config': CONFIG,
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
            return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'echo_system_initialized': system_status['initialized']
    })

# EchoLisp API Endpoints
@app.route('/api/echo_lisp/simulate', methods=['POST'])
def echo_lisp_simulate():
    """Generate echo structure evolution simulation"""
    global system_status
    
    if not ECHO_LISP_AVAILABLE:
        return jsonify({'error': 'EchoLisp not available'}), 500
    
    try:
        data = request.get_json() or {}
        n = data.get('n', 4)  # Default to n=4 as in the problem statement
        
        # Validate input
        if not isinstance(n, int) or n < 1 or n > 20:  # Reasonable upper limit
            return jsonify({'error': 'n must be an integer between 1 and 20'}), 400
        
        # Create EchoLisp instance and run simulation
        echo_lisp = EchoLisp()
        results = echo_lisp.simulate(n)
        
        # Format results
        steps = []
        for step, structure in results:
            steps.append({
                'step': step,
                'structure': structure
            })
        
        response = {
            'n': n,
            'steps': steps,
            'total_structures_tracked': echo_lisp.get_tree_id_count(),
            'tree_id_mappings': {
                echo_lisp.tostr(structure): tree_id 
                for structure, tree_id in echo_lisp.get_tree_ids().items()
            },
            'timestamp': datetime.now().isoformat()
        }
        
        system_status['total_requests'] += 1
        system_status['last_activity'] = datetime.now().isoformat()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in echo_lisp_simulate: {e}")
        system_status['error_count'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/echo_lisp/successors', methods=['POST'])
def echo_lisp_successors():
    """Generate successors for a given echo structure"""
    global system_status
    
    if not ECHO_LISP_AVAILABLE:
        return jsonify({'error': 'EchoLisp not available'}), 500
    
    try:
        data = request.get_json() or {}
        structure_str = data.get('structure', '()')
        
        # Create EchoLisp instance
        echo_lisp = EchoLisp()
        
        # Parse structure string to tuple (simple parser for this demo)
        # For now, we'll support a few common structures
        structure_map = {
            '()': (),
            '(())': ((),),
            '(()())': ((), ()),
            '((()))': (((),),),
            '(()()())': ((), (), ()),
            '(()(()))': ((), ((),)),
            '((()()))': ((((),),),),
            '(((()))': ((((),),),)
        }
        
        if structure_str not in structure_map:
            return jsonify({'error': f'Unsupported structure format: {structure_str}'}), 400
        
        structure = structure_map[structure_str]
        
        # Generate successors
        successors = list(echo_lisp.succ(structure))
        
        response = {
            'input_structure': structure_str,
            'successors': [echo_lisp.tostr(succ) for succ in successors],
            'successor_count': len(successors),
            'timestamp': datetime.now().isoformat()
        }
        
        system_status['total_requests'] += 1
        system_status['last_activity'] = datetime.now().isoformat()
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in echo_lisp_successors: {e}")
        system_status['error_count'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/api/echo_lisp/info')
def echo_lisp_info():
    """Get information about EchoLisp capabilities"""
    return jsonify({
        'available': ECHO_LISP_AVAILABLE,
        'description': 'Echo Structure Generation and Evolution System',
        'capabilities': [
            'Generate echo structure evolution simulations',
            'Calculate successors for echo structures',
            'Track tree IDs for generated structures',
            'Convert structures to Lisp-style string representation'
        ],
        'endpoints': {
            '/api/echo_lisp/simulate': 'POST - Generate echo evolution simulation',
            '/api/echo_lisp/successors': 'POST - Get successors for a structure',
            '/api/echo_lisp/info': 'GET - Get system information'
        },
        'example_usage': {
            'simulate': {
                'method': 'POST',
                'url': '/api/echo_lisp/simulate',
                'body': {'n': 4},
                'description': 'Generate 4-step echo evolution'
            },
            'successors': {
                'method': 'POST', 
                'url': '/api/echo_lisp/successors',
                'body': {'structure': '(())'},
                'description': 'Get successors for structure (())'
            }
        },
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# Initialize system on startup
def startup_initialization():
    """Initialize system in background thread"""
    time.sleep(2)  # Give Flask time to start
    initialize_echo_system()

if __name__ == '__main__':
    # Start initialization in background
    init_thread = threading.Thread(target=startup_initialization)
    init_thread.daemon = True
    init_thread.start()
    
    # Configure Flask for WebVM deployment
    host = '0.0.0.0'  # Allow external access
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Deep Tree Echo API Server on {host}:{port}")
    logger.info(f"WebVM Mode: {CONFIG['webvm_mode']}")
    logger.info(f"Model Path: {CONFIG['model_path']}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

