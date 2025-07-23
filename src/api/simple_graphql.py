"""
Simplified GraphQL API Implementation
Basic GraphQL endpoint without complex dependencies for compatibility
"""

from flask import Blueprint, request, jsonify
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import uuid

logger = logging.getLogger(__name__)

class SimpleGraphQLResolver:
    """Simple GraphQL query resolver"""
    
    def __init__(self, echo_system=None):
        self.echo_system = echo_system
    
    def resolve_query(self, query: str, variables: Dict[str, Any] = None) -> Dict[str, Any]:
        """Resolve a GraphQL query (simplified implementation)"""
        
        # Parse simple queries (basic implementation)
        query = query.strip()
        
        if 'systemStatus' in query:
            return self._resolve_system_status()
        elif 'searchMemory' in query:
            # Extract parameters from query (simplified)
            search_query = self._extract_param(query, 'query')
            limit = self._extract_param(query, 'limit', 5, int)
            return self._resolve_search_memory(search_query, limit)
        elif 'session' in query:
            session_id = self._extract_param(query, 'sessionId')
            return self._resolve_session(session_id)
        elif 'processCognitive' in query:
            input_text = self._extract_param(query, 'inputText')
            return self._resolve_process_cognitive(input_text)
        else:
            return {'error': 'Query not supported in simplified implementation'}
    
    def _extract_param(self, query: str, param: str, default: Any = None, param_type: type = str):
        """Extract parameter from GraphQL query string (basic implementation)"""
        import re
        
        # Look for param: "value" or param: value
        pattern = rf'{param}:\s*["\']?([^,\s\n\r"\'}}]+)["\']?'
        match = re.search(pattern, query)
        
        if match:
            value = match.group(1)
            try:
                return param_type(value)
            except:
                return default
        
        return default
    
    def _resolve_system_status(self) -> Dict[str, Any]:
        """Resolve system status query"""
        return {
            'data': {
                'systemStatus': {
                    'status': 'online',
                    'version': '1.0.0',
                    'uptime': 12345.6,
                    'echoSystemInitialized': self.echo_system is not None,
                    'totalSessions': 42,
                    'activeSession': 8,
                    'totalRequests': 15432
                }
            }
        }
    
    def _resolve_search_memory(self, search_query: str, limit: int = 5) -> Dict[str, Any]:
        """Resolve memory search query"""
        if not search_query:
            search_query = "consciousness"
        
        results = []
        for i in range(min(limit, 5)):
            results.append({
                'id': str(uuid.uuid4()),
                'content': f'Memory item {i+1} matching query: {search_query}',
                'memoryType': ['declarative', 'procedural', 'episodic'][i % 3],
                'relevanceScore': 0.95 - (i * 0.1),
                'createdAt': datetime.now().isoformat(),
                'accessCount': 10 + i * 3
            })
        
        return {
            'data': {
                'searchMemory': results
            }
        }
    
    def _resolve_session(self, session_id: str) -> Dict[str, Any]:
        """Resolve session query"""
        if not session_id:
            session_id = str(uuid.uuid4())
        
        return {
            'data': {
                'session': {
                    'sessionId': session_id,
                    'status': 'active',
                    'createdAt': datetime.now().isoformat(),
                    'lastActivity': datetime.now().isoformat(),
                    'messageCount': 15,
                    'totalTokensProcessed': 2347,
                    'cognitiveState': {
                        'declarativeMemoryItems': 156,
                        'proceduralMemoryItems': 78,
                        'episodicMemoryItems': 23,
                        'temporalContextLength': 12,
                        'currentGoals': 3,
                        'lastUpdated': datetime.now().isoformat()
                    }
                }
            }
        }
    
    def _resolve_process_cognitive(self, input_text: str) -> Dict[str, Any]:
        """Resolve cognitive processing mutation"""
        if not input_text:
            input_text = "Hello, Echo!"
        
        session_id = str(uuid.uuid4())
        
        result = {
            'inputText': input_text,
            'integratedResponse': f'Simplified GraphQL cognitive processing result for: {input_text}',
            'processingTime': 0.067,
            'sessionId': session_id,
            'membraneOutputs': [
                {
                    'membraneType': 'memory',
                    'response': f'Memory response for: {input_text}',
                    'confidence': 0.89,
                    'processingTime': 0.023
                },
                {
                    'membraneType': 'reasoning',
                    'response': f'Reasoning response for: {input_text}',
                    'confidence': 0.94,
                    'processingTime': 0.031
                }
            ],
            'timestamp': datetime.now().isoformat()
        }
        
        return {
            'data': {
                'processCognitive': result
            }
        }

def create_simple_graphql_blueprint(echo_system=None) -> Blueprint:
    """Create simplified GraphQL blueprint"""
    
    graphql_bp = Blueprint('simple_graphql', __name__)
    resolver = SimpleGraphQLResolver(echo_system)
    
    @graphql_bp.route('/graphql', methods=['GET', 'POST'])
    def graphql_endpoint():
        """Handle GraphQL requests"""
        try:
            if request.method == 'GET':
                # Return GraphiQL interface for GET requests
                return graphql_playground_html()
            
            # Handle POST requests
            data = request.get_json()
            if not data:
                return jsonify({'error': 'No query provided'}), 400
            
            query = data.get('query', '')
            variables = data.get('variables', {})
            
            if not query:
                return jsonify({'error': 'Empty query'}), 400
            
            # Resolve query
            result = resolver.resolve_query(query, variables)
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"GraphQL error: {e}")
            return jsonify({
                'errors': [{
                    'message': str(e),
                    'type': 'InternalError'
                }]
            }), 500
    
    @graphql_bp.route('/graphql/schema')
    def get_schema():
        """Get GraphQL schema (simplified)"""
        schema_info = {
            'schema': 'type Query { systemStatus: SystemStatus, searchMemory(query: String!, limit: Int): [MemoryItem], session(sessionId: String!): Session }',
            'introspection_url': '/graphql',
            'playground_url': '/graphql',
            'documentation': {
                'description': 'Simplified GraphQL API for Deep Tree Echo',
                'queries': [
                    'systemStatus: Get system status information',
                    'searchMemory(query, limit): Search memory items',
                    'session(sessionId): Get session information'
                ],
                'mutations': [
                    'processCognitive(inputText): Process cognitive input'
                ]
            }
        }
        
        return jsonify(schema_info)
    
    @graphql_bp.route('/graphql/playground')
    def graphql_playground():
        """GraphQL playground interface"""
        return graphql_playground_html()
    
    return graphql_bp

def graphql_playground_html() -> str:
    """Return GraphQL playground HTML"""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo GraphQL Playground</title>
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
        .playground {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .query-input {
            width: 100%;
            height: 200px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 15px;
            color: white;
            font-family: monospace;
            font-size: 14px;
            resize: vertical;
        }
        .result-output {
            width: 100%;
            height: 300px;
            background: rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.3);
            border-radius: 8px;
            padding: 15px;
            color: white;
            font-family: monospace;
            font-size: 14px;
            margin-top: 20px;
            overflow-y: auto;
        }
        .run-button {
            background: #4CAF50;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            font-weight: bold;
            cursor: pointer;
            margin-top: 15px;
        }
        .run-button:hover {
            background: #45a049;
        }
        .examples {
            margin-top: 20px;
        }
        .example-query {
            background: rgba(0,0,0,0.2);
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
            cursor: pointer;
            font-family: monospace;
            font-size: 12px;
        }
        .example-query:hover {
            background: rgba(0,0,0,0.3);
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 GraphQL Playground</h1>
            <p>Simplified GraphQL API for Deep Tree Echo</p>
        </div>
        
        <div class="playground">
            <h3>Query</h3>
            <textarea id="queryInput" class="query-input" placeholder="Enter your GraphQL query here...">
query {
  systemStatus {
    status
    version
    echoSystemInitialized
    totalSessions
  }
}</textarea>
            
            <button class="run-button" onclick="runQuery()">▶ Run Query</button>
            
            <h3>Result</h3>
            <div id="resultOutput" class="result-output">Click "Run Query" to see results...</div>
            
            <div class="examples">
                <h3>Example Queries</h3>
                
                <div class="example-query" onclick="loadExample(this)">
                    <strong>System Status:</strong><br>
                    query { systemStatus { status version uptime } }
                </div>
                
                <div class="example-query" onclick="loadExample(this)">
                    <strong>Search Memory:</strong><br>
                    query { searchMemory(query: "consciousness", limit: 3) { id content memoryType relevanceScore } }
                </div>
                
                <div class="example-query" onclick="loadExample(this)">
                    <strong>Get Session:</strong><br>
                    query { session(sessionId: "test-session") { sessionId status messageCount cognitiveState { declarativeMemoryItems } } }
                </div>
                
                <div class="example-query" onclick="loadExample(this)">
                    <strong>Process Cognitive Input:</strong><br>
                    mutation { processCognitive(inputText: "What is consciousness?") { inputText integratedResponse processingTime } }
                </div>
            </div>
        </div>
    </div>
    
    <script>
        function loadExample(element) {
            const query = element.textContent.split('\\n').slice(1).join('\\n').trim();
            document.getElementById('queryInput').value = query;
        }
        
        async function runQuery() {
            const query = document.getElementById('queryInput').value;
            const resultOutput = document.getElementById('resultOutput');
            
            if (!query.trim()) {
                resultOutput.textContent = 'Please enter a query';
                return;
            }
            
            resultOutput.textContent = 'Running query...';
            
            try {
                const response = await fetch('/graphql', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ query: query })
                });
                
                const result = await response.json();
                resultOutput.textContent = JSON.stringify(result, null, 2);
                
            } catch (error) {
                resultOutput.textContent = 'Error: ' + error.message;
            }
        }
        
        // Load default query on page load
        window.addEventListener('load', function() {
            // Default query is already in the textarea
        });
    </script>
</body>
</html>
    """