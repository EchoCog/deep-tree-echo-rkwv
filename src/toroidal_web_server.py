#!/usr/bin/env python3
"""
Toroidal Cognitive System Web Server
Flask-based web interface for the dual-hemisphere architecture
"""

import asyncio
import json
import logging
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from toroidal_integration import create_toroidal_bridge, create_toroidal_api

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)
CORS(app)

# Global bridge and API instances
bridge = None
api = None

# HTML template for the web interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🌳 Toroidal Cognitive System</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 20px 40px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            margin-bottom: 30px;
        }
        .title {
            color: #4a5568;
            font-size: 2.5em;
            margin-bottom: 10px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            background-clip: text;
        }
        .subtitle {
            color: #718096;
            font-size: 1.1em;
            margin-bottom: 20px;
        }
        .input-section {
            margin-bottom: 30px;
        }
        .input-group {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        input[type="text"] {
            flex: 1;
            padding: 15px;
            border: 2px solid #e2e8f0;
            border-radius: 10px;
            font-size: 16px;
            transition: all 0.3s ease;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
            box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
        }
        button {
            padding: 15px 30px;
            background: linear-gradient(45deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.3);
        }
        button:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        .response-container {
            margin-top: 30px;
        }
        .hemisphere-response {
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 10px;
            border-left: 4px solid;
        }
        .echo-response {
            background: linear-gradient(135deg, #ffeef8, #f0f9ff);
            border-left-color: #667eea;
        }
        .marduk-response {
            background: linear-gradient(135deg, #f0fff4, #fafafa);
            border-left-color: #764ba2;
        }
        .reflection-response {
            background: linear-gradient(135deg, #fffbeb, #f7fafc);
            border-left-color: #f59e0b;
        }
        .hemisphere-title {
            font-size: 1.3em;
            font-weight: 600;
            margin-bottom: 10px;
            color: #2d3748;
        }
        .metrics {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
            margin-top: 20px;
            padding: 20px;
            background: #f7fafc;
            border-radius: 10px;
        }
        .metric {
            text-align: center;
        }
        .metric-value {
            font-size: 1.5em;
            font-weight: 700;
            color: #667eea;
        }
        .metric-label {
            font-size: 0.9em;
            color: #718096;
            margin-top: 5px;
        }
        .loading {
            text-align: center;
            padding: 40px;
            color: #718096;
        }
        .spinner {
            border: 3px solid #f3f3f3;
            border-top: 3px solid #667eea;
            border-radius: 50%;
            width: 30px;
            height: 30px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px 15px;
            background: #edf2f7;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9em;
            color: #4a5568;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #48bb78;
        }
        pre {
            background: #1a202c;
            color: #e2e8f0;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.9em;
            line-height: 1.4;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 class="title">🌳 Toroidal Cognitive System</h1>
            <p class="subtitle">
                Dual-hemisphere architecture with Echo (right) and Marduk (left) working in harmony
            </p>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span>System Online</span>
            </div>
            <div id="session-info">
                Session: <span id="session-id">Loading...</span>
            </div>
        </div>
        
        <div class="input-section">
            <div class="input-group">
                <input type="text" id="user-input" placeholder="Enter your query for the toroidal cognitive system..." />
                <button onclick="processQuery()" id="process-btn">Process</button>
            </div>
        </div>
        
        <div id="response-container" class="response-container" style="display: none;">
            <!-- Responses will be populated here -->
        </div>
    </div>

    <script>
        // Generate session ID
        const sessionId = 'web-' + Math.random().toString(36).substr(2, 9);
        document.getElementById('session-id').textContent = sessionId;
        
        // Conversation history
        let conversationHistory = [];
        
        // Handle Enter key in input
        document.getElementById('user-input').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                processQuery();
            }
        });
        
        async function processQuery() {
            const input = document.getElementById('user-input').value.trim();
            if (!input) return;
            
            const processBtn = document.getElementById('process-btn');
            const responseContainer = document.getElementById('response-container');
            
            // Show loading state
            processBtn.disabled = true;
            processBtn.textContent = 'Processing...';
            responseContainer.style.display = 'block';
            responseContainer.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <p>Both hemispheres are processing your query...</p>
                </div>
            `;
            
            try {
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        input: input,
                        session_id: sessionId,
                        conversation_history: conversationHistory,
                        memory_state: { web_interface: true },
                        processing_goals: ['web_demonstration']
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResponse(data.response, input);
                    
                    // Update conversation history
                    conversationHistory.push({ role: 'user', content: input });
                    conversationHistory.push({ role: 'assistant', content: data.response.synchronized_output });
                    
                    // Keep history manageable
                    if (conversationHistory.length > 10) {
                        conversationHistory = conversationHistory.slice(-10);
                    }
                } else {
                    responseContainer.innerHTML = `
                        <div class="hemisphere-response" style="border-left-color: #e53e3e; background: #fed7d7;">
                            <div class="hemisphere-title">Error</div>
                            <p>${data.error}</p>
                        </div>
                    `;
                }
            } catch (error) {
                responseContainer.innerHTML = `
                    <div class="hemisphere-response" style="border-left-color: #e53e3e; background: #fed7d7;">
                        <div class="hemisphere-title">Network Error</div>
                        <p>Failed to communicate with the toroidal system: ${error.message}</p>
                    </div>
                `;
            }
            
            // Reset button
            processBtn.disabled = false;
            processBtn.textContent = 'Process';
            document.getElementById('user-input').value = '';
        }
        
        function displayResponse(response, userInput) {
            const container = document.getElementById('response-container');
            
            container.innerHTML = `
                <div class="hemisphere-response echo-response">
                    <div class="hemisphere-title">🌳 Deep Tree Echo (Right Hemisphere)</div>
                    <pre>${escapeHtml(response.echo_response.response_text)}</pre>
                    <div style="margin-top: 10px; font-size: 0.9em; color: #718096;">
                        Confidence: ${(response.echo_response.confidence * 100).toFixed(1)}% | 
                        Processing: ${(response.echo_response.processing_time * 1000).toFixed(1)}ms
                    </div>
                </div>
                
                <div class="hemisphere-response marduk-response">
                    <div class="hemisphere-title">🧬 Marduk the Mad Scientist (Left Hemisphere)</div>
                    <pre>${escapeHtml(response.marduk_response.response_text)}</pre>
                    <div style="margin-top: 10px; font-size: 0.9em; color: #718096;">
                        Confidence: ${(response.marduk_response.confidence * 100).toFixed(1)}% | 
                        Processing: ${(response.marduk_response.processing_time * 1000).toFixed(1)}ms
                    </div>
                </div>
                
                <div class="hemisphere-response reflection-response">
                    <div class="hemisphere-title">🔄 System Reflection</div>
                    <pre>${escapeHtml(response.reflection)}</pre>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-value">${(response.total_processing_time * 1000).toFixed(1)}ms</div>
                        <div class="metric-label">Total Processing</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(response.convergence_metrics.temporal_sync * 100).toFixed(1)}%</div>
                        <div class="metric-label">Temporal Sync</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(response.convergence_metrics.confidence_alignment * 100).toFixed(1)}%</div>
                        <div class="metric-label">Confidence Alignment</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(response.convergence_metrics.complementarity * 100).toFixed(1)}%</div>
                        <div class="metric-label">Complementarity</div>
                    </div>
                    <div class="metric">
                        <div class="metric-value">${(response.convergence_metrics.coherence * 100).toFixed(1)}%</div>
                        <div class="metric-label">Coherence</div>
                    </div>
                </div>
            `;
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        // Load system status on page load
        window.addEventListener('load', async function() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                if (data.success) {
                    console.log('System Status:', data.status);
                }
            } catch (error) {
                console.error('Failed to load system status:', error);
            }
        });
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Serve the main web interface"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/process', methods=['POST'])
def process_query():
    """Process a query through the Toroidal system"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided", "success": False}), 400
        
        # Run async processing in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(api.process_query(data))
            return jsonify(result)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/api/status', methods=['GET'])
def get_status():
    """Get system status"""
    try:
        # Run async status check in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(api.get_system_status())
            return jsonify(result)
        finally:
            loop.close()
            
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return jsonify({"error": str(e), "success": False}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "system": "toroidal_cognitive_system"
    })

async def initialize_system():
    """Initialize the toroidal system"""
    global bridge, api
    
    logger.info("Initializing Toroidal Cognitive System...")
    
    # Create bridge and API
    bridge = create_toroidal_bridge(buffer_size=1000, use_real_rwkv=False)
    api = create_toroidal_api(bridge)
    
    # Initialize bridge
    success = await bridge.initialize()
    if success:
        logger.info("Toroidal Cognitive System initialized successfully")
    else:
        logger.error("Failed to initialize Toroidal Cognitive System")
        
    return success

def run_server(host='0.0.0.0', port=5000, debug=False):
    """Run the Flask server"""
    # Initialize system before starting server
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        success = loop.run_until_complete(initialize_system())
        if not success:
            logger.error("System initialization failed, exiting...")
            return
    finally:
        loop.close()
    
    logger.info(f"Starting Toroidal Cognitive System Web Server on {host}:{port}")
    logger.info(f"Access the interface at: http://{host}:{port}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Toroidal Cognitive System Web Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    args = parser.parse_args()
    
    run_server(host=args.host, port=args.port, debug=args.debug)