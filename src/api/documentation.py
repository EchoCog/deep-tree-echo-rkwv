"""
API Documentation System
OpenAPI/Swagger documentation for the Deep Tree Echo API ecosystem
"""

from flask import Blueprint, jsonify, render_template_string
from flask_swagger_ui import get_swaggerui_blueprint
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# ============================================================================
# OpenAPI Specification
# ============================================================================

def create_api_spec() -> APISpec:
    """Create OpenAPI specification for the API"""
    
    spec = APISpec(
        title="Deep Tree Echo API",
        version="1.0.0",
        openapi_version="3.0.3",
        info={
            "description": """
# Deep Tree Echo Cognitive Architecture API

A comprehensive API ecosystem for the Deep Tree Echo cognitive architecture platform. 
This API provides access to advanced cognitive processing, memory management, and 
real-time interaction capabilities.

## Features

- **RESTful API**: Comprehensive endpoints for all cognitive operations
- **GraphQL API**: Flexible query interface for complex data retrieval
- **Rate Limiting**: Tiered quotas based on API key subscription level
- **API Versioning**: Backward compatible versioning system
- **Real-time WebSocket**: Live cognitive processing streams
- **Third-party Integrations**: Extensible plugin architecture

## Authentication

All API endpoints require authentication via API key. Include your API key in the request:

- **Header**: `X-API-Key: your_api_key_here`
- **Query Parameter**: `?api_key=your_api_key_here`

## Rate Limits

API usage is limited based on your subscription tier:

- **Free Tier**: 100 requests/hour, 1,000 requests/day
- **Basic Tier**: 1,000 requests/hour, 10,000 requests/day  
- **Premium Tier**: 10,000 requests/hour, 100,000 requests/day

## API Versions

The API supports versioning through URL paths or headers:

- **URL**: `/api/v1/...` or `/api/v2/...`
- **Header**: `API-Version: v1` or `API-Version: v2`

## Error Handling

All errors follow a consistent format:

```json
{
  "success": false,
  "error": "Error description",
  "timestamp": "2024-01-01T00:00:00Z",
  "version": "v1"
}
```

## Support

- **Documentation**: [https://docs.deeptreeecho.ai](https://docs.deeptreeecho.ai)
- **Community**: [https://community.deeptreeecho.ai](https://community.deeptreeecho.ai)
- **GitHub**: [https://github.com/EchoCog/deep-tree-echo-rkwv](https://github.com/EchoCog/deep-tree-echo-rkwv)
            """,
            "contact": {
                "name": "Deep Tree Echo Team",
                "url": "https://deeptreeecho.ai",
                "email": "api-support@deeptreeecho.ai"
            },
            "license": {
                "name": "MIT License",
                "url": "https://opensource.org/licenses/MIT"
            }
        },
        servers=[
            {
                "url": "http://localhost:8000",
                "description": "Development server"
            },
            {
                "url": "https://api.deeptreeecho.ai",
                "description": "Production server"
            }
        ],
        plugins=[MarshmallowPlugin()],
        tags=[
            {
                "name": "System",
                "description": "System status and health endpoints"
            },
            {
                "name": "Cognitive Processing",
                "description": "Core cognitive processing operations"
            },
            {
                "name": "Memory Management",
                "description": "Memory storage, retrieval, and search"
            },
            {
                "name": "Sessions",
                "description": "Session creation and management"
            },
            {
                "name": "Analytics",
                "description": "Usage analytics and reporting"
            },
            {
                "name": "Integration",
                "description": "Third-party service integrations"
            }
        ]
    )
    
    # Add security schemes
    spec.components.security_scheme(
        "ApiKeyAuth",
        {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication"
        }
    )
    
    # Add common components
    add_common_schemas(spec)
    add_api_paths(spec)
    
    return spec

def add_common_schemas(spec: APISpec):
    """Add common schema definitions"""
    
    # Standard API Response
    spec.components.schema(
        "APIResponse",
        {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "description": "Request success status"},
                "timestamp": {"type": "string", "format": "date-time", "description": "Response timestamp"},
                "version": {"type": "string", "description": "API version used"},
                "data": {"type": "object", "description": "Response data"},
                "error": {"type": "string", "description": "Error message if success is false"},
                "meta": {"type": "object", "description": "Additional metadata"}
            },
            "required": ["success", "timestamp", "version"]
        }
    )
    
    # Error Response
    spec.components.schema(
        "ErrorResponse",
        {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": False},
                "error": {"type": "string", "description": "Error description"},
                "timestamp": {"type": "string", "format": "date-time"},
                "version": {"type": "string", "example": "v1"}
            },
            "required": ["success", "error", "timestamp", "version"]
        }
    )
    
    # Cognitive Result
    spec.components.schema(
        "CognitiveResult",
        {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Original input text"},
                "output": {"type": "string", "description": "Cognitive processing result"},
                "processing_time": {"type": "number", "description": "Processing time in seconds"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1, "description": "Confidence score"},
                "session_id": {"type": "string", "description": "Session identifier"},
                "membranes_activated": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of cognitive membranes activated"
                }
            }
        }
    )
    
    # System Status
    spec.components.schema(
        "SystemStatus",
        {
            "type": "object",
            "properties": {
                "system": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["online", "offline", "maintenance"]},
                        "version": {"type": "string"},
                        "uptime": {"type": "number"},
                        "echo_system_initialized": {"type": "boolean"}
                    }
                },
                "services": {
                    "type": "object",
                    "properties": {
                        "cognitive_processing": {"type": "boolean"},
                        "memory_system": {"type": "boolean"},
                        "api_server": {"type": "boolean"}
                    }
                },
                "performance": {
                    "type": "object",
                    "properties": {
                        "response_time_ms": {"type": "number"},
                        "throughput_rpm": {"type": "number"},
                        "cache_hit_rate": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                }
            }
        }
    )
    
    # Session Info
    spec.components.schema(
        "SessionInfo",
        {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Unique session identifier"},
                "status": {"type": "string", "enum": ["active", "inactive", "archived"]},
                "created_at": {"type": "string", "format": "date-time"},
                "last_activity": {"type": "string", "format": "date-time"},
                "message_count": {"type": "integer", "minimum": 0},
                "total_tokens_processed": {"type": "integer", "minimum": 0},
                "configuration": {
                    "type": "object",
                    "properties": {
                        "temperature": {"type": "number", "minimum": 0, "maximum": 2},
                        "max_context_length": {"type": "integer", "minimum": 1},
                        "memory_persistence": {"type": "boolean"}
                    }
                }
            }
        }
    )

def add_api_paths(spec: APISpec):
    """Add API path definitions"""
    
    # System Status Endpoint
    spec.path(
        path="/api/v1/status",
        operations={
            "get": {
                "tags": ["System"],
                "summary": "Get system status",
                "description": "Retrieve comprehensive system status information including services and performance metrics",
                "security": [{"ApiKeyAuth": []}],
                "responses": {
                    "200": {
                        "description": "System status retrieved successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "allOf": [
                                        {"$ref": "#/components/schemas/APIResponse"},
                                        {
                                            "properties": {
                                                "data": {"$ref": "#/components/schemas/SystemStatus"}
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Authentication required",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        }
    )
    
    # Cognitive Processing Endpoint
    spec.path(
        path="/api/v1/cognitive/process",
        operations={
            "post": {
                "tags": ["Cognitive Processing"],
                "summary": "Process cognitive input",
                "description": "Process text input through the cognitive architecture and return structured results",
                "security": [{"ApiKeyAuth": []}],
                "requestBody": {
                    "required": True,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "input": {
                                        "type": "string",
                                        "description": "Text input to process",
                                        "example": "What is the meaning of consciousness?"
                                    },
                                    "options": {
                                        "type": "object",
                                        "properties": {
                                            "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.8},
                                            "max_tokens": {"type": "integer", "minimum": 1, "default": 2048},
                                            "session_id": {"type": "string", "description": "Optional session ID"}
                                        }
                                    }
                                },
                                "required": ["input"]
                            }
                        }
                    }
                },
                "responses": {
                    "200": {
                        "description": "Cognitive processing completed successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "allOf": [
                                        {"$ref": "#/components/schemas/APIResponse"},
                                        {
                                            "properties": {
                                                "data": {"$ref": "#/components/schemas/CognitiveResult"}
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "400": {
                        "description": "Invalid request data",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "401": {
                        "description": "Authentication required",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    },
                    "429": {
                        "description": "Rate limit exceeded",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        }
    )
    
    # Session Creation Endpoint
    spec.path(
        path="/api/v1/sessions",
        operations={
            "post": {
                "tags": ["Sessions"],
                "summary": "Create new session",
                "description": "Create a new cognitive processing session with optional configuration",
                "security": [{"ApiKeyAuth": []}],
                "requestBody": {
                    "required": False,
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "temperature": {"type": "number", "minimum": 0, "maximum": 2, "default": 0.8},
                                    "max_context_length": {"type": "integer", "minimum": 1, "default": 2048},
                                    "memory_persistence": {"type": "boolean", "default": True},
                                    "metadata": {"type": "object", "description": "Optional session metadata"}
                                }
                            }
                        }
                    }
                },
                "responses": {
                    "201": {
                        "description": "Session created successfully",
                        "content": {
                            "application/json": {
                                "schema": {
                                    "allOf": [
                                        {"$ref": "#/components/schemas/APIResponse"},
                                        {
                                            "properties": {
                                                "data": {"$ref": "#/components/schemas/SessionInfo"}
                                            }
                                        }
                                    ]
                                }
                            }
                        }
                    },
                    "401": {
                        "description": "Authentication required",
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                            }
                        }
                    }
                }
            }
        }
    )

def create_documentation_blueprint() -> Blueprint:
    """Create documentation blueprint with Swagger UI"""
    
    docs_bp = Blueprint('docs', __name__)
    
    # Create API specification
    spec = create_api_spec()
    
    @docs_bp.route('/api/docs/openapi.json')
    def get_openapi_spec():
        """Get OpenAPI specification as JSON"""
        try:
            return jsonify(spec.to_dict())
        except Exception as e:
            logger.error(f"Error generating OpenAPI spec: {e}")
            return jsonify({'error': str(e)}), 500
    
    @docs_bp.route('/api/docs')
    def api_docs_home():
        """API documentation home page"""
        docs_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo API Documentation</title>
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
        .header h1 {
            font-size: 3em;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .doc-links {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 40px;
        }
        .doc-card {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            text-align: center;
            transition: transform 0.2s ease;
        }
        .doc-card:hover {
            transform: translateY(-5px);
        }
        .doc-card h3 {
            margin: 0 0 15px 0;
            color: #4CAF50;
            font-size: 1.5em;
        }
        .doc-card p {
            margin-bottom: 20px;
            line-height: 1.6;
        }
        .doc-card a {
            display: inline-block;
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 5px;
            font-weight: bold;
            transition: background 0.2s ease;
        }
        .doc-card a:hover {
            background: #45a049;
        }
        .quick-start {
            background: rgba(255,255,255,0.1);
            padding: 30px;
            border-radius: 10px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .code-block {
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 5px;
            font-family: monospace;
            margin: 15px 0;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Deep Tree Echo API</h1>
            <p>Comprehensive API Documentation & Developer Resources</p>
        </div>
        
        <div class="doc-links">
            <div class="doc-card">
                <h3>📖 Interactive API Docs</h3>
                <p>Explore and test API endpoints with Swagger UI. Complete with examples and try-it-out functionality.</p>
                <a href="/api/docs/swagger">Open Swagger UI</a>
            </div>
            
            <div class="doc-card">
                <h3>🔍 GraphQL Playground</h3>
                <p>Query and explore data with our GraphQL API. Flexible queries with real-time schema exploration.</p>
                <a href="/graphql/playground">Open GraphQL Playground</a>
            </div>
            
            <div class="doc-card">
                <h3>📋 OpenAPI Specification</h3>
                <p>Download the complete OpenAPI spec for code generation and integration tools.</p>
                <a href="/api/docs/openapi.json">Download OpenAPI JSON</a>
            </div>
            
            <div class="doc-card">
                <h3>🛠️ SDK Downloads</h3>
                <p>Official SDKs for Python, JavaScript, and CLI tools to accelerate development.</p>
                <a href="/api/sdks">Browse SDKs</a>
            </div>
        </div>
        
        <div class="quick-start">
            <h3>🚀 Quick Start</h3>
            <p>Get started with the Deep Tree Echo API in minutes:</p>
            
            <h4>1. Get Your API Key</h4>
            <div class="code-block">
# Sign up at https://deeptreeecho.ai to get your API key
export ECHO_API_KEY="your_api_key_here"
            </div>
            
            <h4>2. Make Your First Request</h4>
            <div class="code-block">
curl -X POST "http://localhost:8000/api/v1/cognitive/process" \\
  -H "X-API-Key: $ECHO_API_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{"input": "What is consciousness?"}'
            </div>
            
            <h4>3. Try GraphQL</h4>
            <div class="code-block">
query {
  systemStatus {
    status
    version
    echoSystemInitialized
  }
  
  searchMemory(query: "consciousness", limit: 5) {
    id
    content
    relevanceScore
  }
}
            </div>
        </div>
    </div>
</body>
</html>
        """
        return docs_html
    
    # Setup Swagger UI
    swagger_ui_bp = get_swaggerui_blueprint(
        '/api/docs/swagger',
        '/api/docs/openapi.json',
        config={
            'app_name': "Deep Tree Echo API",
            'dom_id': '#swagger-ui',
            'url': '/api/docs/openapi.json',
            'layout': 'StandaloneLayout',
            'deepLinking': True,
            'showExtensions': True,
            'showCommonExtensions': True,
            'defaultModelsExpandDepth': 2,
            'defaultModelExpandDepth': 2
        }
    )
    
    # Register Swagger UI blueprint
    docs_bp.register_blueprint(swagger_ui_bp, url_prefix='/api/docs/swagger')
    
    return docs_bp