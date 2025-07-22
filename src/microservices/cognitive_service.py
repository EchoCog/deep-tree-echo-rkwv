"""
Cognitive Processing Microservice
Handles core cognitive operations with distributed capabilities
"""

import os
import uuid
import time
import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SERVICE_CONFIG = {
    'name': 'cognitive-service',
    'version': '1.0.0',
    'port': int(os.getenv('COGNITIVE_SERVICE_PORT', 8001)),
    'max_concurrent_sessions': int(os.getenv('MAX_CONCURRENT_SESSIONS', 100)),
    'session_timeout_minutes': int(os.getenv('SESSION_TIMEOUT_MINUTES', 30)),
    'enable_caching': os.getenv('ENABLE_CACHING', 'true').lower() == 'true',
    'cache_ttl_seconds': int(os.getenv('CACHE_TTL_SECONDS', 300))
}

# Pydantic models
class CognitiveRequest(BaseModel):
    session_id: str
    input_text: str
    processing_options: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=1, ge=1, le=10)

class CognitiveResponse(BaseModel):
    session_id: str
    response_text: str
    processing_time_ms: float
    membrane_outputs: Dict[str, str]
    cognitive_metadata: Dict[str, Any]
    cache_hit: bool = False

class ServiceHealth(BaseModel):
    status: str
    timestamp: datetime
    active_sessions: int
    total_processed: int
    avg_processing_time_ms: float
    memory_usage_mb: float

class SessionStatus(BaseModel):
    session_id: str
    status: str
    created_at: datetime
    last_activity: datetime
    total_interactions: int
    avg_processing_time_ms: float

# Global state
cognitive_sessions = {}
processing_stats = {
    'total_processed': 0,
    'total_processing_time_ms': 0,
    'start_time': datetime.now()
}

# Enhanced cognitive session with distributed capabilities
class DistributedCognitiveSession:
    """Enhanced cognitive session for distributed processing"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.total_interactions = 0
        self.total_processing_time_ms = 0
        
        # Distributed state
        self.node_id = f"cognitive-{uuid.uuid4().hex[:8]}"
        
        # Memory state with enhanced capabilities
        self.memory_state = {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'goals': []},
            'working_memory': {},
            'semantic_cache': {}
        }
        
        # Performance tracking
        self.performance_metrics = {
            'response_times': [],
            'cache_hit_rate': 0.0,
            'error_rate': 0.0,
            'quality_scores': []
        }
    
    async def process_cognitive_request(self, input_text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """Process cognitive request with distributed capabilities"""
        start_time = time.time()
        self.last_activity = datetime.now()
        self.total_interactions += 1
        
        options = options or {}
        
        # Process through cognitive membranes
        try:
            # Parallel processing of membranes for better performance
            tasks = [
                self._process_memory_membrane(input_text, options),
                self._process_reasoning_membrane(input_text, options),
                self._process_grammar_membrane(input_text, options)
            ]
            
            membrane_results = await asyncio.gather(*tasks)
            memory_response, reasoning_response, grammar_response = membrane_results
            
            # Integrate responses with advanced cognitive processing
            integrated_response = await self._integrate_responses_advanced(
                memory_response, reasoning_response, grammar_response, input_text, options
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            self.total_processing_time_ms += processing_time_ms
            
            # Update performance metrics
            self.performance_metrics['response_times'].append(processing_time_ms)
            if len(self.performance_metrics['response_times']) > 100:
                self.performance_metrics['response_times'] = self.performance_metrics['response_times'][-100:]
            
            # Create response object
            response_data = {
                'session_id': self.session_id,
                'response_text': integrated_response,
                'processing_time_ms': processing_time_ms,
                'membrane_outputs': {
                    'memory': memory_response,
                    'reasoning': reasoning_response,
                    'grammar': grammar_response
                },
                'cognitive_metadata': {
                    'complexity_detected': self._assess_input_complexity(input_text),
                    'reasoning_type': self._suggest_reasoning_type(input_text),
                    'memory_integration_level': self._assess_memory_integration(input_text),
                    'processing_node': self.node_id,
                    'cache_enabled': SERVICE_CONFIG['enable_caching']
                },
                'cache_hit': False
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Cognitive processing error: {e}")
            self.performance_metrics['error_rate'] += 1
            raise HTTPException(status_code=500, detail=f"Cognitive processing failed: {str(e)}")
    
    async def _process_memory_membrane(self, input_text: str, options: Dict[str, Any]) -> str:
        """Enhanced memory membrane with distributed capabilities"""
        # Simulate distributed memory access
        await asyncio.sleep(0.01)  # Simulate network latency
        
        keywords = input_text.lower().split()
        
        # Enhanced memory processing with semantic search
        if any(word in keywords for word in ['remember', 'recall', 'memory']):
            return f"Distributed memory access: Processing {input_text} across memory nodes"
        elif any(word in keywords for word in ['learn', 'store', 'save']):
            return f"Distributed storage: Storing {input_text} with replication"
        else:
            return f"Memory membrane: Distributed processing of {input_text}"
    
    async def _process_reasoning_membrane(self, input_text: str, options: Dict[str, Any]) -> str:
        """Enhanced reasoning membrane with distributed capabilities"""
        await asyncio.sleep(0.02)  # Simulate processing time
        
        keywords = input_text.lower().split()
        
        if '?' in input_text:
            return f"Distributed reasoning: Question analysis across reasoning nodes for {input_text}"
        elif any(word in keywords for word in ['because', 'therefore', 'if', 'then']):
            return f"Logical reasoning cluster: Analyzing causal relationships in {input_text}"
        else:
            return f"Reasoning membrane: Distributed inference for {input_text}"
    
    async def _process_grammar_membrane(self, input_text: str, options: Dict[str, Any]) -> str:
        """Enhanced grammar membrane with distributed capabilities"""
        await asyncio.sleep(0.01)  # Simulate processing time
        
        word_count = len(input_text.split())
        complexity = 'simple' if word_count < 10 else 'moderate' if word_count < 20 else 'complex'
        
        return f"Grammar analysis cluster: {word_count} words, {complexity} complexity, distributed parsing"
    
    async def _integrate_responses_advanced(self, memory: str, reasoning: str, grammar: str, 
                                          input_text: str, options: Dict[str, Any]) -> str:
        """Advanced response integration with distributed awareness"""
        # Enhanced integration with distributed context
        complexity = self._assess_input_complexity(input_text)
        reasoning_type = self._suggest_reasoning_type(input_text)
        
        integration_quality = 0.85 + (0.1 if complexity == 'high' else 0.0)
        
        return f"Distributed cognitive integration (node: {self.node_id}): " \
               f"Quality {integration_quality:.2f}, Type: {reasoning_type}, " \
               f"Processing: {memory[:30]}..."
    
    def _assess_input_complexity(self, text: str) -> str:
        """Assess cognitive complexity of input"""
        words = text.split()
        word_count = len(words)
        
        if word_count > 20:
            return 'high'
        elif word_count > 10:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_reasoning_type(self, text: str) -> str:
        """Suggest reasoning type based on input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['why', 'explain', 'because']):
            return 'abductive'
        elif any(word in text_lower for word in ['if', 'then', 'therefore']):
            return 'deductive'
        elif any(word in text_lower for word in ['pattern', 'usually', 'often']):
            return 'inductive'
        else:
            return 'general'
    
    def _assess_memory_integration(self, text: str) -> str:
        """Assess memory integration level needed"""
        memory_keywords = ['remember', 'recall', 'before', 'previous', 'history']
        keyword_count = sum(1 for keyword in memory_keywords if keyword in text.lower())
        
        if keyword_count >= 3:
            return 'high'
        elif keyword_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for this session"""
        avg_response_time = (
            sum(self.performance_metrics['response_times']) / 
            len(self.performance_metrics['response_times'])
            if self.performance_metrics['response_times'] else 0
        )
        
        return {
            'session_id': self.session_id,
            'node_id': self.node_id,
            'total_interactions': self.total_interactions,
            'avg_response_time_ms': avg_response_time,
            'cache_hit_rate': self.performance_metrics['cache_hit_rate'],
            'error_rate': self.performance_metrics['error_rate'],
            'created_at': self.created_at,
            'last_activity': self.last_activity
        }

# FastAPI app with lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info(f"Starting {SERVICE_CONFIG['name']} v{SERVICE_CONFIG['version']}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down cognitive service")

# Create FastAPI app
app = FastAPI(
    title="Cognitive Processing Service",
    description="Distributed cognitive processing microservice for Deep Tree Echo",
    version=SERVICE_CONFIG['version'],
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Utility functions
def get_session(session_id: str) -> DistributedCognitiveSession:
    """Get or create cognitive session"""
    if session_id not in cognitive_sessions:
        if len(cognitive_sessions) >= SERVICE_CONFIG['max_concurrent_sessions']:
            raise HTTPException(status_code=429, detail="Maximum concurrent sessions reached")
        cognitive_sessions[session_id] = DistributedCognitiveSession(session_id)
    
    return cognitive_sessions[session_id]

async def cleanup_expired_sessions():
    """Clean up expired sessions"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session in cognitive_sessions.items():
        time_diff = current_time - session.last_activity
        if time_diff.total_seconds() > SERVICE_CONFIG['session_timeout_minutes'] * 60:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del cognitive_sessions[session_id]
        logger.info(f"Cleaned up expired session: {session_id}")

# API Endpoints

@app.post("/api/cognitive/process", response_model=CognitiveResponse)
async def process_cognitive_request(
    request: CognitiveRequest,
    background_tasks: BackgroundTasks
):
    """Process cognitive request with distributed capabilities"""
    try:
        # Clean up expired sessions in background
        background_tasks.add_task(cleanup_expired_sessions)
        
        session = get_session(request.session_id)
        result = await session.process_cognitive_request(
            request.input_text, 
            request.processing_options
        )
        
        # Update global stats
        processing_stats['total_processed'] += 1
        processing_stats['total_processing_time_ms'] += result['processing_time_ms']
        
        return CognitiveResponse(**result)
        
    except Exception as e:
        logger.error(f"Processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/cognitive/session/{session_id}/status", response_model=SessionStatus)
async def get_session_status(session_id: str):
    """Get status of a cognitive session"""
    if session_id not in cognitive_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = cognitive_sessions[session_id]
    avg_time = (
        session.total_processing_time_ms / session.total_interactions
        if session.total_interactions > 0 else 0
    )
    
    return SessionStatus(
        session_id=session_id,
        status="active",
        created_at=session.created_at,
        last_activity=session.last_activity,
        total_interactions=session.total_interactions,
        avg_processing_time_ms=avg_time
    )

@app.get("/api/cognitive/session/{session_id}/performance")
async def get_session_performance(session_id: str):
    """Get detailed performance metrics for a session"""
    if session_id not in cognitive_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = cognitive_sessions[session_id]
    return session.get_performance_summary()

@app.get("/health", response_model=ServiceHealth)
async def health_check():
    """Health check endpoint for load balancer"""
    avg_processing_time = (
        processing_stats['total_processing_time_ms'] / processing_stats['total_processed']
        if processing_stats['total_processed'] > 0 else 0
    )
    
    # Simulate memory usage calculation
    memory_usage_mb = len(cognitive_sessions) * 5  # Rough estimate
    
    return ServiceHealth(
        status="healthy",
        timestamp=datetime.now(),
        active_sessions=len(cognitive_sessions),
        total_processed=processing_stats['total_processed'],
        avg_processing_time_ms=avg_processing_time,
        memory_usage_mb=memory_usage_mb
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    # Service metrics
    metrics.append(f"cognitive_service_active_sessions {len(cognitive_sessions)}")
    metrics.append(f"cognitive_service_total_processed {processing_stats['total_processed']}")
    
    # Performance metrics
    avg_processing_time = (
        processing_stats['total_processing_time_ms'] / processing_stats['total_processed']
        if processing_stats['total_processed'] > 0 else 0
    )
    metrics.append(f"cognitive_service_avg_processing_time_ms {avg_processing_time}")
    
    # Memory metrics
    memory_usage_mb = len(cognitive_sessions) * 5
    metrics.append(f"cognitive_service_memory_usage_mb {memory_usage_mb}")
    
    return {"metrics": "\n".join(metrics)}

@app.get("/api/cognitive/sessions")
async def list_active_sessions():
    """List all active cognitive sessions"""
    sessions = []
    for session_id, session in cognitive_sessions.items():
        sessions.append({
            'session_id': session_id,
            'node_id': session.node_id,
            'created_at': session.created_at,
            'last_activity': session.last_activity,
            'total_interactions': session.total_interactions
        })
    
    return {
        'active_sessions': sessions,
        'total_count': len(sessions)
    }

# Background tasks
async def periodic_cleanup():
    """Periodic cleanup of expired sessions"""
    while True:
        try:
            await cleanup_expired_sessions()
            await asyncio.sleep(300)  # Clean up every 5 minutes
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(60)  # Retry after 1 minute

if __name__ == "__main__":
    # Run the service
    uvicorn.run(
        "cognitive_service:app",
        host="0.0.0.0",
        port=SERVICE_CONFIG['port'],
        reload=False,
        log_level="info"
    )