"""
GraphQL API Implementation
Flexible query interface for the Deep Tree Echo cognitive architecture
"""

import graphene
from graphene import ObjectType, String, Int, Float, Boolean, List, Field, Schema, DateTime
from graphene import Mutation, Argument
from flask import Blueprint, request, jsonify
from flask_graphql import GraphQLView
import logging
from datetime import datetime
from typing import Optional, Dict, Any
import uuid
import json

logger = logging.getLogger(__name__)

# ============================================================================
# GraphQL Types
# ============================================================================

class CognitiveState(ObjectType):
    """Cognitive state information"""
    declarative_memory_items = Int()
    procedural_memory_items = Int()
    episodic_memory_items = Int()
    temporal_context_length = Int()
    current_goals = Int()
    last_updated = DateTime()

class MembraneOutput(ObjectType):
    """Output from a cognitive membrane"""
    membrane_type = String()
    response = String()
    confidence = Float()
    processing_time = Float()

class CognitiveResult(ObjectType):
    """Result of cognitive processing"""
    input_text = String()
    integrated_response = String()
    processing_time = Float()
    session_id = String()
    membrane_outputs = List(MembraneOutput)
    cognitive_state = Field(CognitiveState)
    timestamp = DateTime()

class MemoryItem(ObjectType):
    """Memory item from the cognitive system"""
    id = String()
    content = String()
    memory_type = String()  # declarative, procedural, episodic
    relevance_score = Float()
    created_at = DateTime()
    last_accessed = DateTime()
    access_count = Int()

class SessionInfo(ObjectType):
    """Cognitive session information"""
    session_id = String()
    status = String()
    created_at = DateTime()
    last_activity = DateTime()
    message_count = Int()
    total_tokens_processed = Int()
    cognitive_state = Field(CognitiveState)

class SystemStatus(ObjectType):
    """System status information"""
    status = String()
    version = String()
    uptime = Float()
    echo_system_initialized = Boolean()
    total_sessions = Int()
    active_sessions = Int()
    total_requests = Int()

class UsageAnalytics(ObjectType):
    """Usage analytics information"""
    total_requests = Int()
    successful_requests = Int()
    error_requests = Int()
    average_response_time = Float()
    api_tier = String()
    quota_remaining = Int()

# ============================================================================
# GraphQL Queries
# ============================================================================

class Query(ObjectType):
    """Root query object"""
    
    # System queries
    system_status = Field(SystemStatus)
    
    # Session queries  
    session = Field(SessionInfo, session_id=String(required=True))
    sessions = List(SessionInfo, limit=Int(default_value=10))
    
    # Memory queries
    search_memory = List(
        MemoryItem,
        query=String(required=True),
        memory_type=String(),
        limit=Int(default_value=10),
        min_relevance=Float(default_value=0.5)
    )
    
    memory_item = Field(MemoryItem, id=String(required=True))
    
    # Analytics queries
    usage_analytics = Field(
        UsageAnalytics,
        period=String(default_value="last_30_days")
    )
    
    def resolve_system_status(self, info):
        """Resolve system status query"""
        try:
            return SystemStatus(
                status="online",
                version="1.0.0",
                uptime=12345.6,
                echo_system_initialized=True,
                total_sessions=42,
                active_sessions=8,
                total_requests=15432
            )
        except Exception as e:
            logger.error(f"Error resolving system_status: {e}")
            raise Exception(f"Failed to get system status: {e}")
    
    def resolve_session(self, info, session_id):
        """Resolve single session query"""
        try:
            # Mock session data - in production, query from database
            return SessionInfo(
                session_id=session_id,
                status="active",
                created_at=datetime.now(),
                last_activity=datetime.now(),
                message_count=15,
                total_tokens_processed=2347,
                cognitive_state=CognitiveState(
                    declarative_memory_items=156,
                    procedural_memory_items=78,
                    episodic_memory_items=23,
                    temporal_context_length=12,
                    current_goals=3,
                    last_updated=datetime.now()
                )
            )
        except Exception as e:
            logger.error(f"Error resolving session {session_id}: {e}")
            raise Exception(f"Failed to get session: {e}")
    
    def resolve_sessions(self, info, limit):
        """Resolve sessions list query"""
        try:
            # Mock sessions data
            sessions = []
            for i in range(min(limit, 5)):
                sessions.append(SessionInfo(
                    session_id=str(uuid.uuid4()),
                    status="active" if i < 3 else "inactive",
                    created_at=datetime.now(),
                    last_activity=datetime.now(),
                    message_count=10 + i * 5,
                    total_tokens_processed=1000 + i * 500,
                    cognitive_state=CognitiveState(
                        declarative_memory_items=100 + i * 20,
                        procedural_memory_items=50 + i * 10,
                        episodic_memory_items=20 + i * 5,
                        temporal_context_length=10 + i,
                        current_goals=2 + i,
                        last_updated=datetime.now()
                    )
                ))
            return sessions
        except Exception as e:
            logger.error(f"Error resolving sessions: {e}")
            raise Exception(f"Failed to get sessions: {e}")
    
    def resolve_search_memory(self, info, query, memory_type=None, limit=10, min_relevance=0.5):
        """Resolve memory search query"""
        try:
            # Mock memory search results
            results = []
            for i in range(min(limit, 8)):
                if memory_type and i % 2 == 0:
                    mem_type = memory_type
                else:
                    mem_type = ['declarative', 'procedural', 'episodic'][i % 3]
                
                relevance = 0.95 - (i * 0.1)
                if relevance >= min_relevance:
                    results.append(MemoryItem(
                        id=str(uuid.uuid4()),
                        content=f"Memory item {i+1} matching query: {query}",
                        memory_type=mem_type,
                        relevance_score=relevance,
                        created_at=datetime.now(),
                        last_accessed=datetime.now(),
                        access_count=10 + i * 3
                    ))
            
            return results
        except Exception as e:
            logger.error(f"Error resolving memory search: {e}")
            raise Exception(f"Failed to search memory: {e}")
    
    def resolve_memory_item(self, info, id):
        """Resolve single memory item query"""
        try:
            return MemoryItem(
                id=id,
                content=f"Detailed memory content for item {id}",
                memory_type="declarative",
                relevance_score=0.92,
                created_at=datetime.now(),
                last_accessed=datetime.now(),
                access_count=25
            )
        except Exception as e:
            logger.error(f"Error resolving memory item {id}: {e}")
            raise Exception(f"Failed to get memory item: {e}")
    
    def resolve_usage_analytics(self, info, period):
        """Resolve usage analytics query"""
        try:
            return UsageAnalytics(
                total_requests=1247,
                successful_requests=1189,
                error_requests=58,
                average_response_time=0.045,
                api_tier="premium",
                quota_remaining=8753
            )
        except Exception as e:
            logger.error(f"Error resolving usage analytics: {e}")
            raise Exception(f"Failed to get usage analytics: {e}")

# ============================================================================
# GraphQL Mutations
# ============================================================================

class ProcessCognitive(Mutation):
    """Mutation for cognitive processing"""
    
    class Arguments:
        input_text = String(required=True)
        session_id = String()
        temperature = Float(default_value=0.8)
        max_tokens = Int(default_value=2048)
    
    Output = CognitiveResult
    
    def mutate(self, info, input_text, session_id=None, temperature=0.8, max_tokens=2048):
        """Process cognitive input"""
        try:
            if not session_id:
                session_id = str(uuid.uuid4())
            
            # Mock cognitive processing result
            result = CognitiveResult(
                input_text=input_text,
                integrated_response=f"GraphQL cognitive processing result for: {input_text}",
                processing_time=0.067,
                session_id=session_id,
                membrane_outputs=[
                    MembraneOutput(
                        membrane_type="memory",
                        response=f"Memory membrane response for: {input_text}",
                        confidence=0.89,
                        processing_time=0.023
                    ),
                    MembraneOutput(
                        membrane_type="reasoning",
                        response=f"Reasoning membrane response for: {input_text}",
                        confidence=0.94,
                        processing_time=0.031
                    ),
                    MembraneOutput(
                        membrane_type="grammar",
                        response=f"Grammar membrane response for: {input_text}",
                        confidence=0.87,
                        processing_time=0.013
                    )
                ],
                cognitive_state=CognitiveState(
                    declarative_memory_items=156,
                    procedural_memory_items=78,
                    episodic_memory_items=23,
                    temporal_context_length=12,
                    current_goals=3,
                    last_updated=datetime.now()
                ),
                timestamp=datetime.now()
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cognitive processing mutation: {e}")
            raise Exception(f"Cognitive processing failed: {e}")

class CreateSession(Mutation):
    """Mutation for creating a new session"""
    
    class Arguments:
        configuration = String()  # JSON string of configuration
        metadata = String()  # JSON string of metadata
    
    Output = SessionInfo
    
    def mutate(self, info, configuration=None, metadata=None):
        """Create a new cognitive session"""
        try:
            session_id = str(uuid.uuid4())
            
            return SessionInfo(
                session_id=session_id,
                status="active",
                created_at=datetime.now(),
                last_activity=datetime.now(),
                message_count=0,
                total_tokens_processed=0,
                cognitive_state=CognitiveState(
                    declarative_memory_items=0,
                    procedural_memory_items=0,
                    episodic_memory_items=0,
                    temporal_context_length=0,
                    current_goals=0,
                    last_updated=datetime.now()
                )
            )
            
        except Exception as e:
            logger.error(f"Error creating session: {e}")
            raise Exception(f"Session creation failed: {e}")

class Mutation(ObjectType):
    """Root mutation object"""
    process_cognitive = ProcessCognitive.Field()
    create_session = CreateSession.Field()

# ============================================================================
# GraphQL Schema
# ============================================================================

schema = Schema(query=Query, mutation=Mutation)

# ============================================================================
# Flask Blueprint for GraphQL
# ============================================================================

def create_graphql_blueprint() -> Blueprint:
    """Create GraphQL blueprint"""
    
    graphql_bp = Blueprint('graphql', __name__)
    
    # Add GraphQL endpoint
    graphql_bp.add_url_rule(
        '/graphql',
        view_func=GraphQLView.as_view(
            'graphql',
            schema=schema,
            graphiql=True  # Enable GraphiQL interface for development
        )
    )
    
    @graphql_bp.route('/graphql/schema')
    def get_schema():
        """Get GraphQL schema definition"""
        try:
            from graphql import print_schema
            schema_string = print_schema(schema.graphql_schema)
            
            return jsonify({
                'schema': schema_string,
                'introspection_url': '/graphql',
                'graphiql_url': '/graphql',
                'documentation': {
                    'queries': [
                        'systemStatus: Get system status information',
                        'session(sessionId): Get specific session information',
                        'sessions(limit): Get list of sessions',
                        'searchMemory(query, memoryType, limit, minRelevance): Search memory items',
                        'memoryItem(id): Get specific memory item',
                        'usageAnalytics(period): Get usage analytics'
                    ],
                    'mutations': [
                        'processCognitive(inputText, sessionId, temperature, maxTokens): Process cognitive input',
                        'createSession(configuration, metadata): Create new session'
                    ]
                }
            })
        except Exception as e:
            logger.error(f"Error getting GraphQL schema: {e}")
            return jsonify({'error': str(e)}), 500
    
    @graphql_bp.route('/graphql/playground')
    def graphql_playground():
        """GraphQL Playground interface"""
        playground_html = """
<!DOCTYPE html>
<html>
<head>
  <meta charset=utf-8/>
  <meta name="viewport" content="user-scalable=no, initial-scale=1.0, minimum-scale=1.0, maximum-scale=1.0, minimal-ui">
  <title>Deep Tree Echo GraphQL Playground</title>
  <link rel="stylesheet" href="//cdn.jsdelivr.net/npm/graphql-playground-react/build/static/css/index.css" />
  <link rel="shortcut icon" href="//cdn.jsdelivr.net/npm/graphql-playground-react/build/favicon.png" />
  <script src="//cdn.jsdelivr.net/npm/graphql-playground-react/build/static/js/middleware.js"></script>
</head>
<body>
  <div id="root">
    <style>
      body {
        background-color: rgb(23, 42, 58);
        font-family: Open Sans, sans-serif;
        height: 90vh;
      }
      #root {
        height: 100%;
        width: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
      }
      .loading {
        font-size: 32px;
        font-weight: 200;
        color: rgba(255, 255, 255, .6);
        margin-left: 20px;
      }
      img {
        width: 78px;
        height: 78px;
      }
      .title {
        font-weight: 400;
      }
    </style>
    <img src="//cdn.jsdelivr.net/npm/graphql-playground-react/build/logo.png" alt="">
    <div class="loading"> Loading
      <span class="title">Deep Tree Echo GraphQL Playground</span>
    </div>
  </div>
  <script>window.GraphQLPlayground.init(document.getElementById('root'), {
    endpoint: '/graphql',
    settings: {
      'general.betaUpdates': false,
      'editor.theme': 'dark',
      'editor.cursorShape': 'line',
      'editor.reuseHeaders': true,
      'tracing.hideTracingResponse': true,
      'queryPlan.hideQueryPlanResponse': true,
      'editor.fontSize': 14
    },
    tabs: [{
      endpoint: '/graphql',
      query: `# Welcome to Deep Tree Echo GraphQL API!
# 
# Example Queries:

# Get system status
query GetSystemStatus {
  systemStatus {
    status
    version
    uptime
    echoSystemInitialized
    totalSessions
    activeSession
    totalRequests
  }
}

# Search memory with filters
query SearchMemory {
  searchMemory(query: "artificial intelligence", memoryType: "declarative", limit: 5) {
    id
    content
    memoryType
    relevanceScore
    createdAt
    accessCount
  }
}

# Get session information
query GetSession {
  session(sessionId: "example-session-id") {
    sessionId
    status
    messageCount
    cognitiveState {
      declarativeMemoryItems
      proceduralMemoryItems
      episodicMemoryItems
    }
  }
}

# Example Mutations:

# Process cognitive input
mutation ProcessInput {
  processCognitive(inputText: "What is consciousness?", temperature: 0.8) {
    inputText
    integratedResponse
    processingTime
    sessionId
    membraneOutputs {
      membraneType
      response
      confidence
    }
  }
}

# Create new session
mutation CreateNewSession {
  createSession(configuration: "{\\"temperature\\": 0.9}") {
    sessionId
    status
    createdAt
  }
}`
    }]
  })</script>
</body>
</html>
        """
        return playground_html
    
    return graphql_bp

# ============================================================================
# GraphQL Error Handling
# ============================================================================

class GraphQLErrorHandler:
    """Handle GraphQL-specific errors"""
    
    @staticmethod
    def format_error(error):
        """Format GraphQL errors consistently"""
        return {
            'message': str(error),
            'type': error.__class__.__name__,
            'timestamp': datetime.now().isoformat()
        }