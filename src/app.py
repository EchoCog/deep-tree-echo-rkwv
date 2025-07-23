"""
Deep Tree Echo Integrated Application
Complete cognitive architecture deployment with RWKV integration for WebVM
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import time
import logging
import threading
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, Any, List, Optional
import uuid
from collections import defaultdict
from persistent_memory import PersistentMemorySystem, MemoryQuery

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins="*")

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

# Global state
cognitive_sessions = {}
system_metrics = {
    'total_sessions': 0,
    'active_sessions': 0,
    'total_requests': 0,
    'start_time': datetime.now(),
    'last_activity': None,
    'distributed_requests': 0,
    'cache_requests': 0,
    'load_balancer_requests': 0
}

# Initialize persistent memory system
persistent_memory = None

# Configuration
CONFIG = {
    'webvm_mode': os.environ.get('ECHO_WEBVM_MODE', 'true').lower() == 'true',
    'memory_limit_mb': int(os.environ.get('ECHO_MEMORY_LIMIT', '600')),
    'max_sessions': 10,
    'session_timeout_minutes': 30,
    'enable_persistence': True,
    'data_dir': '/tmp/echo_data',
    # Distributed architecture configuration
    'enable_distributed_mode': os.environ.get('ENABLE_DISTRIBUTED_MODE', 'false').lower() == 'true',
    'load_balancer_url': os.environ.get('LOAD_BALANCER_URL', 'http://localhost:8000'),
    'cache_service_url': os.environ.get('CACHE_SERVICE_URL', 'http://localhost:8002'),
    'redis_url': os.environ.get('REDIS_URL', 'redis://localhost:6379'),
}

# Ensure data directory exists and initialize persistent memory
os.makedirs(CONFIG['data_dir'], exist_ok=True)
if CONFIG['enable_persistence']:
    try:
        persistent_memory = PersistentMemorySystem(CONFIG['data_dir'])
        logger.info("Persistent memory system initialized")
    except Exception as e:
        logger.error(f"Failed to initialize persistent memory: {e}")
        CONFIG['enable_persistence'] = False

class MockCognitiveSession:
    """Mock cognitive session for demonstration without full RWKV"""
    
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.memory_state = {
            'declarative': {},
            'procedural': {},
            'episodic': [],
            'intentional': {'goals': []}
        }
        self.conversation_history = []
        self.processing_stats = {
            'total_inputs': 0,
            'avg_processing_time': 0.0,
            'memory_usage_mb': 0
        }
    
    def process_input(self, input_text: str) -> Dict[str, Any]:
        """Process cognitive input through mock membranes with advanced capabilities"""
        start_time = time.time()
        self.last_activity = datetime.now()
        self.processing_stats['total_inputs'] += 1
        
        # Mock membrane processing with enhanced cognitive features
        memory_response = self._process_memory_membrane(input_text)
        reasoning_response = self._process_reasoning_membrane(input_text)
        grammar_response = self._process_grammar_membrane(input_text)
        
        # Integrate responses with advanced cognitive awareness
        integrated_response = self._integrate_responses_advanced(
            memory_response, reasoning_response, grammar_response, input_text
        )
        
        processing_time = time.time() - start_time
        self.processing_stats['avg_processing_time'] = (
            (self.processing_stats['avg_processing_time'] * (self.processing_stats['total_inputs'] - 1) + processing_time) /
            self.processing_stats['total_inputs']
        )
        
        # Store in conversation history with advanced metadata
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'response': integrated_response,
            'processing_time': processing_time,
            'membrane_outputs': {
                'memory': memory_response,
                'reasoning': reasoning_response,
                'grammar': grammar_response
            },
            'cognitive_metadata': {
                'complexity_detected': self._assess_input_complexity(input_text),
                'reasoning_type_suggested': self._suggest_reasoning_type(input_text),
                'memory_integration_level': self._assess_memory_integration(input_text),
                'adaptive_learning_opportunities': self._identify_learning_opportunities(input_text)
            }
        }
        
        self.conversation_history.append(conversation_entry)
        
        # Update episodic memory with enhanced context
        self.memory_state['episodic'].append({
            'event': input_text,
            'response': integrated_response,
            'timestamp': datetime.now().isoformat(),
            'cognitive_context': conversation_entry['cognitive_metadata']
        })
        
        return conversation_entry
    
    def _process_memory_membrane(self, input_text: str) -> str:
        """Mock memory membrane processing with persistent storage"""
        keywords = input_text.lower().split()
        
        # Store significant memories
        if self._is_memory_significant(input_text):
            memory_type = self._classify_memory_type(input_text)
            if persistent_memory:
                try:
                    memory_id = persistent_memory.store_memory(
                        content=input_text,
                        memory_type=memory_type,
                        session_id=self.session_id,
                        metadata={'processing_type': 'memory_membrane'}
                    )
                    if memory_id:
                        logger.debug(f"Stored memory {memory_id} for session {self.session_id}")
                except Exception as e:
                    logger.error(f"Error storing memory: {e}")
        
        # Retrieve relevant memories
        relevant_memories = []
        if persistent_memory:
            try:
                search_results = persistent_memory.search_memories(
                    query_text=input_text,
                    session_id=self.session_id,
                    max_results=3
                )
                relevant_memories = [result.item for result in search_results]
            except Exception as e:
                logger.error(f"Error retrieving memories: {e}")
        
        if any(word in keywords for word in ['remember', 'recall', 'memory']):
            if relevant_memories:
                memory_summary = f"Found {len(relevant_memories)} related memories: " + \
                               "; ".join([mem.content[:50] + "..." for mem in relevant_memories[:2]])
                return f"Accessing memory systems for: {input_text}. {memory_summary}"
            else:
                return f"Accessing memory systems for: {input_text}. Found {len(self.memory_state['episodic'])} local experiences."
        elif any(word in keywords for word in ['learn', 'store', 'save']):
            return f"Storing new information in persistent memory: {input_text}"
        else:
            memory_context = ""
            if relevant_memories:
                memory_context = f" Drawing from {len(relevant_memories)} related memories."
            return f"Memory membrane activated.{memory_context} Processing: {input_text}"
    
    def _process_reasoning_membrane(self, input_text: str) -> str:
        """Mock reasoning membrane processing"""
        keywords = input_text.lower().split()
        
        if '?' in input_text:
            return f"Reasoning membrane engaged for question: {input_text}. Applying deductive and inductive reasoning patterns."
        elif any(word in keywords for word in ['because', 'therefore', 'if', 'then']):
            return f"Logical reasoning detected. Analyzing causal relationships in: {input_text}"
        elif any(word in keywords for word in ['problem', 'solve', 'solution']):
            return f"Problem-solving mode activated. Breaking down: {input_text}"
        else:
            return f"General reasoning applied to: {input_text}"
    
    def _process_grammar_membrane(self, input_text: str) -> str:
        """Mock grammar membrane processing"""
        word_count = len(input_text.split())
        sentence_count = input_text.count('.') + input_text.count('!') + input_text.count('?')
        
        complexity = 'simple' if word_count < 10 else 'moderate' if word_count < 20 else 'complex'
        
        return f"Grammar analysis: {word_count} words, {sentence_count} sentences, {complexity} complexity. Symbolic patterns detected."
    
    def _integrate_responses(self, memory: str, reasoning: str, grammar: str) -> str:
        """Integrate membrane responses into coherent output with memory context"""
        # Get memory statistics if persistent memory is available
        memory_context = ""
        if persistent_memory:
            try:
                stats = persistent_memory.get_system_stats()
                total_memories = stats.get('database_stats', {}).get('total_memories', 0)
                if total_memories > 0:
                    memory_context = f" [Drawing from {total_memories} persistent memories]"
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
        
        return f"Integrated cognitive response{memory_context}: Drawing from memory ({len(self.memory_state['episodic'])} experiences), applying reasoning patterns, and considering linguistic structure. {memory[:50]}..."
    
    def _integrate_responses_advanced(self, memory: str, reasoning: str, grammar: str, input_text: str) -> str:
        """Advanced integration with cognitive awareness"""
        # Get memory statistics if persistent memory is available
        memory_context = ""
        if persistent_memory:
            try:
                stats = persistent_memory.get_system_stats()
                total_memories = stats.get('database_stats', {}).get('total_memories', 0)
                if total_memories > 0:
                    memory_context = f" [Drawing from {total_memories} persistent memories]"
            except Exception as e:
                logger.error(f"Error getting memory stats: {e}")
        
        # Assess cognitive requirements
        complexity = self._assess_input_complexity(input_text)
        reasoning_type = self._suggest_reasoning_type(input_text)
        memory_integration = self._assess_memory_integration(input_text)
        
        # Create sophisticated integration
        integration_aspects = []
        
        if memory_integration == 'high':
            integration_aspects.append(f"memory-guided processing ({len(self.memory_state['episodic'])} experiences)")
        
        if reasoning_type != 'general':
            integration_aspects.append(f"{reasoning_type} reasoning patterns")
            
        if complexity in ['medium', 'high']:
            integration_aspects.append("multi-layered analysis")
        
        integration_description = ", ".join(integration_aspects) if integration_aspects else "standard cognitive processing"
        
        return f"Advanced cognitive integration{memory_context}: Applying {integration_description}. Confidence: {self._calculate_response_confidence(memory, reasoning, grammar):.2f}. {memory[:50]}..."
    
    def _assess_input_complexity(self, text: str) -> str:
        """Assess cognitive complexity of input"""
        words = text.split()
        word_count = len(words)
        
        # Count complexity indicators
        questions = text.count('?')
        logical_words = len([w for w in words if w.lower() in 
                           ['because', 'therefore', 'however', 'although', 'if', 'then', 'analyze', 'explain']])
        
        complexity_score = 0
        if word_count > 20:
            complexity_score += 2
        elif word_count > 10:
            complexity_score += 1
        
        if questions > 1:
            complexity_score += 1
            
        if logical_words > 2:
            complexity_score += 1
        
        if complexity_score >= 3:
            return 'high'
        elif complexity_score >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_reasoning_type(self, text: str) -> str:
        """Suggest appropriate reasoning type for input"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['why', 'explain', 'because', 'cause']):
            return 'abductive'
        elif any(word in text_lower for word in ['if', 'then', 'therefore', 'must', 'all']):
            return 'deductive'
        elif any(word in text_lower for word in ['pattern', 'usually', 'often', 'tend to']):
            return 'inductive'
        elif any(word in text_lower for word in ['like', 'similar', 'analogy', 'compare']):
            return 'analogical'
        else:
            return 'general'
    
    def _assess_memory_integration(self, text: str) -> str:
        """Assess level of memory integration needed"""
        memory_keywords = ['remember', 'recall', 'before', 'earlier', 'previous', 'last time', 'history']
        
        keyword_count = sum(1 for keyword in memory_keywords if keyword in text.lower())
        
        if keyword_count >= 3:
            return 'high'
        elif keyword_count >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _identify_learning_opportunities(self, text: str) -> List[str]:
        """Identify opportunities for adaptive learning"""
        opportunities = []
        
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['prefer', 'like', 'want', 'need']):
            opportunities.append('preference_learning')
        
        if any(word in text_lower for word in ['explain', 'detail', 'more', 'less']):
            opportunities.append('response_style_adaptation')
        
        if any(word in text_lower for word in ['fast', 'slow', 'quick', 'time']):
            opportunities.append('processing_speed_optimization')
        
        if '?' in text:
            opportunities.append('question_handling_improvement')
        
        return opportunities
    
    def _calculate_response_confidence(self, memory: str, reasoning: str, grammar: str) -> float:
        """Calculate confidence score for integrated response"""
        # Simple heuristic based on response characteristics
        base_confidence = 0.7
        
        # Adjust based on response length and complexity
        total_length = len(memory) + len(reasoning) + len(grammar)
        if total_length > 200:
            base_confidence += 0.1
        elif total_length < 50:
            base_confidence -= 0.1
        
        # Adjust based on memory availability
        if len(self.memory_state['episodic']) > 5:
            base_confidence += 0.1
        
        return min(1.0, max(0.0, base_confidence))
    
    def _is_memory_significant(self, text: str) -> bool:
        """Determine if input should be stored in persistent memory"""
        # Store memories that are questions, statements with significant content, or learning-related
        words = text.split()
        return (
            len(words) >= 3 and  # Minimum length
            (len(words) >= 8 or  # Long enough to be significant
             '?' in text or      # Questions are significant
             any(word in text.lower() for word in [
                 'remember', 'learn', 'important', 'fact', 'know', 'understand',
                 'think', 'believe', 'feel', 'experience', 'happened'
             ]))
        )
    
    def _classify_memory_type(self, text: str) -> str:
        """Classify memory type based on content"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['how to', 'step', 'process', 'method', 'way to']):
            return 'procedural'
        elif any(word in text_lower for word in ['i', 'me', 'my', 'happened', 'experience', 'felt', 'did']):
            return 'episodic'  
        elif any(word in text_lower for word in ['is', 'are', 'fact', 'definition', 'means', 'because']):
            return 'declarative'
        else:
            return 'semantic'
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive state"""
        return {
            'session_id': self.session_id,
            'created_at': self.created_at.isoformat(),
            'last_activity': self.last_activity.isoformat(),
            'memory_items': {
                'declarative': len(self.memory_state['declarative']),
                'procedural': len(self.memory_state['procedural']),
                'episodic': len(self.memory_state['episodic']),
                'goals': len(self.memory_state['intentional']['goals'])
            },
            'conversation_length': len(self.conversation_history),
            'processing_stats': self.processing_stats
        }
    
    def _analyze_complexity_distribution(self) -> Dict[str, int]:
        """Analyze distribution of complexity levels in conversations"""
        complexity_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for entry in self.conversation_history:
            if 'cognitive_metadata' in entry:
                complexity = entry['cognitive_metadata'].get('complexity_detected', 'low')
                complexity_counts[complexity] += 1
        
        return complexity_counts
    
    def _analyze_reasoning_types(self) -> Dict[str, int]:
        """Analyze reasoning types used in conversations"""
        reasoning_counts = defaultdict(int)
        
        for entry in self.conversation_history:
            if 'cognitive_metadata' in entry:
                reasoning_type = entry['cognitive_metadata'].get('reasoning_type_suggested', 'general')
                reasoning_counts[reasoning_type] += 1
        
        return dict(reasoning_counts)

# Distributed processing functions
async def call_distributed_cognitive_service(session_id: str, input_text: str, options: Dict[str, Any] = None) -> Dict[str, Any]:
    """Call distributed cognitive processing service"""
    if not CONFIG['enable_distributed_mode']:
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            # Call through load balancer
            url = f"{CONFIG['load_balancer_url']}/api/proxy/cognitive-service"
            
            payload = {
                "method": "POST",
                "path": "/api/cognitive/process",
                "headers": {"Content-Type": "application/json"},
                "body": {
                    "session_id": session_id,
                    "input_text": input_text,
                    "processing_options": options or {},
                    "priority": 1
                }
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    system_metrics['distributed_requests'] += 1
                    return result.get('data', {})
                else:
                    logger.warning(f"Distributed cognitive service failed: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error calling distributed cognitive service: {e}")
        return None

async def get_from_cache(key: str) -> Optional[Any]:
    """Get value from distributed cache service"""
    if not CONFIG['enable_distributed_mode']:
        return None
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{CONFIG['cache_service_url']}/api/cache/{key}"
            
            async with session.get(url) as response:
                if response.status == 200:
                    result = await response.json()
                    system_metrics['cache_requests'] += 1
                    return result.get('value')
                elif response.status == 404:
                    return None
                else:
                    logger.warning(f"Cache service failed: {response.status}")
                    return None
                    
    except Exception as e:
        logger.error(f"Error calling cache service: {e}")
        return None

async def set_in_cache(key: str, value: Any, ttl_seconds: int = 300) -> bool:
    """Set value in distributed cache service"""
    if not CONFIG['enable_distributed_mode']:
        return False
    
    try:
        async with aiohttp.ClientSession() as session:
            url = f"{CONFIG['cache_service_url']}/api/cache/{key}"
            
            payload = {
                "key": key,
                "value": value,
                "ttl_seconds": ttl_seconds,
                "cache_level": "L1"
            }
            
            async with session.post(url, json=payload) as response:
                if response.status == 200:
                    system_metrics['cache_requests'] += 1
                    return True
                else:
                    logger.warning(f"Cache set failed: {response.status}")
                    return False
                    
    except Exception as e:
        logger.error(f"Error setting cache: {e}")
        return False

def run_async_task(coro):
    """Helper to run async tasks in sync context"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)
    
    def _analyze_memory_integration(self) -> Dict[str, int]:
        """Analyze levels of memory integration"""
        integration_counts = {'low': 0, 'medium': 0, 'high': 0}
        
        for entry in self.conversation_history:
            if 'cognitive_metadata' in entry:
                integration_level = entry['cognitive_metadata'].get('memory_integration_level', 'low')
                integration_counts[integration_level] += 1
        
        return integration_counts

def create_cognitive_session() -> str:
    """Create new cognitive session"""
    session_id = str(uuid.uuid4())
    cognitive_sessions[session_id] = MockCognitiveSession(session_id)
    system_metrics['total_sessions'] += 1
    system_metrics['active_sessions'] = len(cognitive_sessions)
    logger.info(f"Created cognitive session: {session_id}")
    return session_id

def get_cognitive_session(session_id: str) -> Optional[MockCognitiveSession]:
    """Get cognitive session by ID"""
    return cognitive_sessions.get(session_id)

def cleanup_expired_sessions():
    """Clean up expired sessions"""
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session in cognitive_sessions.items():
        time_diff = current_time - session.last_activity
        if time_diff.total_seconds() > CONFIG['session_timeout_minutes'] * 60:
            expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        del cognitive_sessions[session_id]
        logger.info(f"Cleaned up expired session: {session_id}")
    
    system_metrics['active_sessions'] = len(cognitive_sessions)

# Routes

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('dashboard.html', config=CONFIG, metrics=system_metrics)

@app.route('/cognitive')
def cognitive_interface():
    """Cognitive interaction interface"""
    return render_template('cognitive.html')

@app.route('/api/session', methods=['POST'])
def create_session():
    """Create new cognitive session"""
    try:
        cleanup_expired_sessions()
        
        if len(cognitive_sessions) >= CONFIG['max_sessions']:
            return jsonify({'error': 'Maximum sessions reached'}), 429
        
        session_id = create_cognitive_session()
        session = get_cognitive_session(session_id)
        
        return jsonify({
            'session_id': session_id,
            'created_at': session.created_at.isoformat(),
            'status': 'active'
        })
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/session/<session_id>', methods=['GET'])
def get_session_info(session_id: str):
    """Get session information"""
    try:
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify(session.get_state_summary())
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/process', methods=['POST'])
def process_cognitive_input():
    """Process cognitive input with distributed capabilities"""
    try:
        data = request.get_json()
        if not data or 'input' not in data:
            return jsonify({'error': 'No input provided'}), 400
        
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'No session ID provided'}), 400
        
        input_text = data['input']
        
        # Try distributed processing first if enabled
        if CONFIG['enable_distributed_mode']:
            # Check cache first
            cache_key = f"cognitive:{session_id}:{hash(input_text)}"
            cached_result = run_async_task(get_from_cache(cache_key))
            
            if cached_result:
                cached_result['cache_hit'] = True
                system_metrics['total_requests'] += 1
                system_metrics['last_activity'] = datetime.now()
                return jsonify(cached_result)
            
            # Try distributed cognitive service
            distributed_result = run_async_task(
                call_distributed_cognitive_service(session_id, input_text, data.get('options', {}))
            )
            
            if distributed_result:
                # Cache the result
                run_async_task(set_in_cache(cache_key, distributed_result, 300))
                
                system_metrics['total_requests'] += 1
                system_metrics['last_activity'] = datetime.now()
                return jsonify(distributed_result)
        
        # Fallback to local processing
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        result = session.process_input(input_text)
        
        # Cache the result if distributed mode is enabled
        if CONFIG['enable_distributed_mode']:
            cache_key = f"cognitive:{session_id}:{hash(input_text)}"
            run_async_task(set_in_cache(cache_key, result, 300))
        
        system_metrics['total_requests'] += 1
        system_metrics['last_activity'] = datetime.now()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error processing input: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversation/<session_id>')
def get_conversation_history(session_id: str):
    """Get conversation history for session"""
    try:
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        return jsonify({
            'session_id': session_id,
            'conversation_history': session.conversation_history,
            'total_entries': len(session.conversation_history)
        })
    except Exception as e:
        logger.error(f"Error getting conversation history: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/status')
def system_status():
    """Get system status with distributed architecture information"""
    cleanup_expired_sessions()
    
    uptime = datetime.now() - system_metrics['start_time']
    
    # Get distributed service status if enabled
    distributed_status = {}
    if CONFIG['enable_distributed_mode']:
        try:
            # Check load balancer status
            lb_status = run_async_task(check_service_health(f"{CONFIG['load_balancer_url']}/health"))
            distributed_status['load_balancer'] = lb_status
            
            # Check cache service status
            cache_status = run_async_task(check_service_health(f"{CONFIG['cache_service_url']}/health"))
            distributed_status['cache_service'] = cache_status
            
        except Exception as e:
            logger.error(f"Error checking distributed service status: {e}")
            distributed_status['error'] = str(e)
    
    return jsonify({
        'status': 'online',
        'timestamp': datetime.now().isoformat(),
        'uptime_seconds': uptime.total_seconds(),
        'config': CONFIG,
        'metrics': system_metrics,
        'active_sessions': len(cognitive_sessions),
        'memory_usage_estimate_mb': len(cognitive_sessions) * 10,  # Rough estimate
        'distributed_mode': CONFIG['enable_distributed_mode'],
        'distributed_services': distributed_status
    })

async def check_service_health(url: str) -> Dict[str, Any]:
    """Check health of a distributed service"""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    data = await response.json()
                    return {'status': 'healthy', 'data': data}
                else:
                    return {'status': 'unhealthy', 'status_code': response.status}
    except Exception as e:
        return {'status': 'unreachable', 'error': str(e)}

@app.route('/api/metrics')
def detailed_metrics():
    """Get detailed system metrics"""
    session_summaries = []
    for session_id, session in cognitive_sessions.items():
        session_summaries.append(session.get_state_summary())
    
    # Add persistent memory statistics
    memory_stats = {}
    if persistent_memory:
        try:
            memory_stats = persistent_memory.get_system_stats()
        except Exception as e:
            logger.error(f"Error getting persistent memory stats: {e}")
    
    return jsonify({
        'system_metrics': system_metrics,
        'session_summaries': session_summaries,
        'persistent_memory': memory_stats,
        'resource_usage': {
            'memory_limit_mb': CONFIG['memory_limit_mb'],
            'estimated_usage_mb': len(cognitive_sessions) * 10,
            'sessions_count': len(cognitive_sessions),
            'max_sessions': CONFIG['max_sessions']
        }
    })

@app.route('/api/memory/search', methods=['POST'])
def search_memories():
    """Search persistent memories"""
    if not persistent_memory:
        return jsonify({'error': 'Persistent memory not available'}), 503
    
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query_text = data['query']
        session_id = data.get('session_id')
        memory_types = data.get('memory_types')
        max_results = data.get('max_results', 10)
        
        search_results = persistent_memory.search_memories(
            query_text=query_text,
            memory_types=memory_types,
            session_id=session_id,
            max_results=max_results
        )
        
        # Convert to JSON-serializable format
        results = []
        for result in search_results:
            results.append({
                'id': result.item.id,
                'content': result.item.content,
                'memory_type': result.item.memory_type,
                'importance': float(result.item.importance),
                'timestamp': result.item.timestamp,
                'session_id': result.item.session_id,
                'relevance_score': float(result.relevance_score),
                'similarity_score': float(result.similarity_score),
                'access_count': result.item.access_count
            })
        
        return jsonify({
            'query': query_text,
            'results': results,
            'total_found': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/consolidate', methods=['POST'])
def consolidate_memories():
    """Consolidate similar memories to reduce redundancy"""
    if not persistent_memory:
        return jsonify({'error': 'Persistent memory not available'}), 503
    
    try:
        data = request.get_json() or {}
        session_id = data.get('session_id')
        
        consolidated_count = persistent_memory.consolidate_memories(session_id)
        
        return jsonify({
            'consolidated_count': consolidated_count,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error consolidating memories: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/memory/stats')
def memory_statistics():
    """Get persistent memory statistics"""
    if not persistent_memory:
        return jsonify({'error': 'Persistent memory not available'}), 503
    
    try:
        stats = persistent_memory.get_system_stats()
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error getting memory statistics: {e}")
        return jsonify({'error': str(e)}), 500

# Advanced Cognitive Processing API Endpoints

@app.route('/api/cognitive/insights/<session_id>')
def get_cognitive_insights(session_id: str):
    """Get comprehensive cognitive insights for a session"""
    try:
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Basic insights from session
        basic_insights = {
            'session_summary': session.get_state_summary(),
            'conversation_analysis': {
                'total_interactions': len(session.conversation_history),
                'avg_processing_time': session.processing_stats['avg_processing_time'],
                'complexity_distribution': session._analyze_complexity_distribution(),
                'reasoning_type_usage': session._analyze_reasoning_types(),
                'memory_integration_levels': session._analyze_memory_integration()
            }
        }
        
        # Try to get advanced insights if available
        advanced_insights = {}
        try:
            # This would integrate with the advanced systems if they were initialized
            # For now, return basic insights with placeholders for advanced features
            advanced_insights = {
                'meta_cognitive_status': 'Available (Mock)',
                'reasoning_chain_analysis': 'Available (Mock)',
                'adaptive_learning_progress': 'Available (Mock)',
                'personalization_level': 'Basic'
            }
        except Exception as e:
            logger.warning(f"Advanced insights not available: {e}")
        
        return jsonify({
            'session_id': session_id,
            'basic_insights': basic_insights,
            'advanced_insights': advanced_insights,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting cognitive insights: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cognitive/reasoning/execute', methods=['POST'])
def execute_complex_reasoning():
    """Execute complex reasoning for a query"""
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({'error': 'No query provided'}), 400
        
        query = data['query']
        session_id = data.get('session_id', 'default')
        reasoning_type = data.get('reasoning_type')  # optional
        
        # Mock complex reasoning execution
        reasoning_result = {
            'success': True,
            'query': query,
            'reasoning_type': reasoning_type or 'auto-selected',
            'chain_id': f"mock_chain_{int(time.time())}",
            'confidence': 0.85,
            'conclusion': f"Through complex reasoning analysis: {query}",
            'explanation': f"Applied {reasoning_type or 'adaptive'} reasoning with multi-step validation",
            'steps': [
                {'step': 1, 'type': 'analysis', 'description': 'Analyzed query components'},
                {'step': 2, 'type': 'reasoning', 'description': 'Applied logical inference'},
                {'step': 3, 'type': 'validation', 'description': 'Validated reasoning chain'},
                {'step': 4, 'type': 'conclusion', 'description': 'Generated final conclusion'}
            ],
            'processing_time': 0.5,
            'validation_score': 0.9
        }
        
        return jsonify(reasoning_result)
        
    except Exception as e:
        logger.error(f"Error in complex reasoning execution: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cognitive/feedback', methods=['POST'])
def submit_cognitive_feedback():
    """Submit feedback for adaptive learning"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No feedback data provided'}), 400
        
        required_fields = ['session_id', 'feedback_type', 'feedback_value']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Missing required feedback fields'}), 400
        
        # Mock feedback processing
        feedback_result = {
            'success': True,
            'feedback_id': f"fb_{int(time.time())}",
            'processing_status': 'processed',
            'learning_updates': {
                'preferences_updated': 1,
                'strategy_adaptations': 1,
                'personalization_improved': True
            },
            'message': 'Feedback processed successfully and integrated into adaptive learning system'
        }
        
        return jsonify(feedback_result)
        
    except Exception as e:
        logger.error(f"Error processing cognitive feedback: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cognitive/personalization/<session_id>')
def get_personalization_status(session_id: str):
    """Get personalization status for a session"""
    try:
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Mock personalization analysis
        personalization_status = {
            'session_id': session_id,
            'personalization_level': 'moderate',
            'learned_preferences': {
                'response_style': 'analytical',
                'detail_level': 'comprehensive',
                'reasoning_preference': 'step_by_step'
            },
            'adaptation_opportunities': [
                'Response timing optimization',
                'Content complexity adjustment',
                'Interaction style refinement'
            ],
            'learning_progress': {
                'total_interactions_analyzed': len(session.conversation_history),
                'preference_confidence': 0.75,
                'adaptation_readiness': True
            }
        }
        
        return jsonify(personalization_status)
        
    except Exception as e:
        logger.error(f"Error getting personalization status: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/cognitive/meta-analysis/<session_id>')
def get_meta_cognitive_analysis(session_id: str):
    """Get meta-cognitive analysis for a session"""
    try:
        session = get_cognitive_session(session_id)
        if not session:
            return jsonify({'error': 'Session not found'}), 404
        
        # Mock meta-cognitive analysis
        meta_analysis = {
            'session_id': session_id,
            'cognitive_strategy_performance': {
                'current_strategy': 'adaptive',
                'strategy_effectiveness': 0.82,
                'strategy_recommendations': [
                    'Continue with adaptive strategy',
                    'Consider memory-first approach for complex queries'
                ]
            },
            'error_detection': {
                'errors_detected': 0,
                'performance_trends': 'stable',
                'confidence_trajectory': 'improving'
            },
            'self_monitoring_insights': {
                'processing_efficiency': 0.85,
                'response_quality_trend': 'improving',
                'adaptation_success_rate': 0.90
            }
        }
        
        return jsonify(meta_analysis)
        
    except Exception as e:
        logger.error(f"Error getting meta-cognitive analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(cognitive_sessions)
    })

@app.route('/api/analytics', methods=['POST'])
def collect_analytics():
    """Collect user interaction analytics"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        session_id = data.get('session_id')
        actions = data.get('actions', [])
        summary = data.get('summary', {})
        
        # Store analytics data (in production, this would go to a database)
        analytics_data = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'actions': actions,
            'summary': summary,
            'user_agent': request.headers.get('User-Agent', ''),
            'ip_address': request.remote_addr
        }
        
        # For now, just log the analytics (in production, store in database)
        logger.info(f"Analytics collected for session {session_id}: {len(actions)} actions")
        
        # Update system metrics
        if 'messages_sent' in summary:
            system_metrics['total_requests'] += summary['messages_sent']
        
        return jsonify({'status': 'success', 'actions_processed': len(actions)})
        
    except Exception as e:
        logger.error(f"Error collecting analytics: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/metrics')
def prometheus_metrics():
    """Prometheus-compatible metrics endpoint"""
    metrics = []
    
    # System metrics
    metrics.extend([
        f"deep_echo_active_sessions {len(cognitive_sessions)}",
        f"deep_echo_total_sessions {system_metrics['total_sessions']}",
        f"deep_echo_total_requests {system_metrics['total_requests']}",
        f"deep_echo_distributed_requests {system_metrics['distributed_requests']}",
        f"deep_echo_cache_requests {system_metrics['cache_requests']}",
        f"deep_echo_load_balancer_requests {system_metrics['load_balancer_requests']}"
    ])
    
    # Uptime
    uptime_seconds = (datetime.now() - system_metrics['start_time']).total_seconds()
    metrics.append(f"deep_echo_uptime_seconds {uptime_seconds}")
    
    # Memory usage estimate
    memory_usage_mb = len(cognitive_sessions) * 10
    metrics.append(f"deep_echo_memory_usage_mb {memory_usage_mb}")
    
    # Distributed mode status
    metrics.append(f"deep_echo_distributed_mode {1 if CONFIG['enable_distributed_mode'] else 0}")
    
    # Configuration metrics
    metrics.extend([
        f"deep_echo_max_sessions {CONFIG['max_sessions']}",
        f"deep_echo_memory_limit_mb {CONFIG['memory_limit_mb']}",
        f"deep_echo_webvm_mode {1 if CONFIG['webvm_mode'] else 0}"
    ])
    
    # Session performance metrics
    if cognitive_sessions:
        total_interactions = sum(
            len(session.conversation_history) for session in cognitive_sessions.values()
        )
        avg_processing_times = [
            session.processing_stats['avg_processing_time'] 
            for session in cognitive_sessions.values()
            if session.processing_stats['avg_processing_time'] > 0
        ]
        avg_response_time = sum(avg_processing_times) / len(avg_processing_times) if avg_processing_times else 0
        
        metrics.extend([
            f"deep_echo_total_interactions {total_interactions}",
            f"deep_echo_avg_response_time_ms {avg_response_time}"
        ])
    
    # Persistent memory metrics
    if persistent_memory:
        try:
            stats = persistent_memory.get_system_stats()
            db_stats = stats.get('database_stats', {})
            metrics.extend([
                f"deep_echo_total_memories {db_stats.get('total_memories', 0)}",
                f"deep_echo_memory_types {len(db_stats.get('memory_types', []))}",
                f"deep_echo_avg_memory_importance {db_stats.get('avg_importance', 0)}"
            ])
        except Exception as e:
            logger.error(f"Error getting memory stats for metrics: {e}")
    
    return '\n'.join(metrics), 200, {'Content-Type': 'text/plain; version=0.0.4; charset=utf-8'}

# Template directory setup
template_dir = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(template_dir, exist_ok=True)

# Create dashboard template
dashboard_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo - Cognitive Architecture</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            min-height: 100vh;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
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
        .header p {
            font-size: 1.3em;
            margin: 10px 0;
            opacity: 0.9;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
            gap: 25px;
            margin-bottom: 30px;
        }
        .card {
            background: rgba(255,255,255,0.1);
            padding: 25px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
            box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        }
        .card h3 {
            margin: 0 0 20px 0;
            color: #4CAF50;
            font-size: 1.4em;
        }
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 10px;
            animation: pulse 2s infinite;
        }
        .status-online { background: #4CAF50; }
        .status-warning { background: #ff9800; }
        .status-offline { background: #f44336; }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 10px 0;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .btn {
            display: inline-block;
            padding: 12px 24px;
            background: #4CAF50;
            color: white;
            text-decoration: none;
            border-radius: 8px;
            font-weight: bold;
            transition: background 0.3s;
            border: none;
            cursor: pointer;
            font-size: 1em;
        }
        .btn:hover {
            background: #45a049;
        }
        .btn-secondary {
            background: #2196F3;
        }
        .btn-secondary:hover {
            background: #1976D2;
        }
        .architecture-diagram {
            text-align: center;
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
        }
        .membrane {
            margin: 5px 0;
            padding: 5px;
        }
        .membrane-root { color: #ff6b6b; }
        .membrane-cognitive { color: #4ecdc4; }
        .membrane-extension { color: #45b7d1; }
        .membrane-security { color: #f9ca24; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🧠 Deep Tree Echo</h1>
            <p>Membrane-Based Cognitive Architecture with RWKV Integration</p>
            <p>WebVM Deployment Ready</p>
        </div>
        
        <div class="grid">
            <div class="card">
                <h3><span class="status-indicator status-online"></span>System Status</h3>
                <div class="metric">
                    <span>Deployment Mode:</span>
                    <span>{{ 'WebVM' if config.webvm_mode else 'Standard' }}</span>
                </div>
                <div class="metric">
                    <span>Memory Limit:</span>
                    <span>{{ config.memory_limit_mb }}MB</span>
                </div>
                <div class="metric">
                    <span>Max Sessions:</span>
                    <span>{{ config.max_sessions }}</span>
                </div>
                <div class="metric">
                    <span>Persistence:</span>
                    <span>{{ 'Enabled' if config.enable_persistence else 'Disabled' }}</span>
                </div>
            </div>
            
            <div class="card">
                <h3>📊 Performance Metrics</h3>
                <div class="metric">
                    <span>Total Sessions:</span>
                    <span>{{ metrics.total_sessions }}</span>
                </div>
                <div class="metric">
                    <span>Active Sessions:</span>
                    <span>{{ metrics.active_sessions }}</span>
                </div>
                <div class="metric">
                    <span>Total Requests:</span>
                    <span>{{ metrics.total_requests }}</span>
                </div>
                <div class="metric">
                    <span>Uptime:</span>
                    <span id="uptime">Calculating...</span>
                </div>
            </div>
            
            <div class="card">
                <h3>🚀 Quick Actions</h3>
                <p>Start interacting with the cognitive architecture:</p>
                <a href="/cognitive" class="btn">Launch Cognitive Interface</a>
                <br><br>
                <a href="/api/status" class="btn btn-secondary">View API Status</a>
                <br><br>
                <a href="/api/metrics" class="btn btn-secondary">Detailed Metrics</a>
            </div>
        </div>
        
        <div class="card">
            <h3>🏗️ Membrane Architecture</h3>
            <div class="architecture-diagram">
                <div class="membrane membrane-root">🎪 Root Membrane (System Boundary)</div>
                <div class="membrane membrane-cognitive">├── 🧠 Cognitive Membrane (Core Processing)</div>
                <div class="membrane">│   ├── 💭 Memory Membrane (Storage & Retrieval)</div>
                <div class="membrane">│   ├── ⚡ Reasoning Membrane (Inference & Logic)</div>
                <div class="membrane">│   └── 🎭 Grammar Membrane (Symbolic Processing)</div>
                <div class="membrane membrane-extension">├── 🔌 Extension Membrane (Plugin Container)</div>
                <div class="membrane">│   ├── 🌍 Browser Membrane</div>
                <div class="membrane">│   ├── 📊 ML Membrane (RWKV Integration)</div>
                <div class="membrane">│   └── 🪞 Introspection Membrane</div>
                <div class="membrane membrane-security">└── 🛡️ Security Membrane (Validation & Control)</div>
            </div>
        </div>
        
        <div class="card">
            <h3>🤖 RWKV Integration Features</h3>
            <div class="metric">
                <span>Model Architecture:</span>
                <span>RWKV-7 (Hybrid RNN-Transformer)</span>
            </div>
            <div class="metric">
                <span>Memory Complexity:</span>
                <span>Linear (O(1) inference)</span>
            </div>
            <div class="metric">
                <span>Context Length:</span>
                <span>Theoretically Infinite</span>
            </div>
            <div class="metric">
                <span>WebVM Optimized:</span>
                <span>Yes (600MB memory limit)</span>
            </div>
        </div>
    </div>
    
    <script>
        // Update uptime
        function updateUptime() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const uptimeSeconds = data.uptime_seconds;
                    const hours = Math.floor(uptimeSeconds / 3600);
                    const minutes = Math.floor((uptimeSeconds % 3600) / 60);
                    const seconds = Math.floor(uptimeSeconds % 60);
                    document.getElementById('uptime').textContent = 
                        `${hours}h ${minutes}m ${seconds}s`;
                })
                .catch(error => {
                    document.getElementById('uptime').textContent = 'Error';
                });
        }
        
        // Update every 5 seconds
        setInterval(updateUptime, 5000);
        updateUptime();
    </script>
</body>
</html>
'''

# Create cognitive interface template
cognitive_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deep Tree Echo - Cognitive Interface</title>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            height: 100vh;
            display: flex;
            flex-direction: column;
        }
        .header {
            background: rgba(0,0,0,0.3);
            padding: 15px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 {
            margin: 0;
            font-size: 1.8em;
        }
        .session-info {
            font-size: 0.9em;
            opacity: 0.8;
        }
        .main-container {
            flex: 1;
            display: flex;
            padding: 20px;
            gap: 20px;
            overflow: hidden;
        }
        .chat-container {
            flex: 2;
            display: flex;
            flex-direction: column;
            background: rgba(255,255,255,0.1);
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .chat-header {
            padding: 20px;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            max-height: calc(100vh - 300px);
        }
        .message {
            margin-bottom: 20px;
            padding: 15px;
            border-radius: 10px;
            animation: fadeIn 0.3s ease-in;
        }
        .user-message {
            background: rgba(33, 150, 243, 0.3);
            margin-left: 20%;
            text-align: right;
        }
        .echo-message {
            background: rgba(76, 175, 80, 0.3);
            margin-right: 20%;
        }
        .system-message {
            background: rgba(255, 193, 7, 0.3);
            text-align: center;
            font-style: italic;
        }
        .message-meta {
            font-size: 0.8em;
            opacity: 0.7;
            margin-top: 8px;
        }
        .chat-input {
            padding: 20px;
            border-top: 1px solid rgba(255,255,255,0.2);
        }
        .input-group {
            display: flex;
            gap: 10px;
        }
        .input-group input {
            flex: 1;
            padding: 12px;
            border: none;
            border-radius: 8px;
            background: rgba(255,255,255,0.9);
            color: #333;
            font-size: 1em;
        }
        .input-group button {
            padding: 12px 24px;
            border: none;
            border-radius: 8px;
            background: #4CAF50;
            color: white;
            cursor: pointer;
            font-weight: bold;
            font-size: 1em;
        }
        .input-group button:hover {
            background: #45a049;
        }
        .input-group button:disabled {
            background: #666;
            cursor: not-allowed;
        }
        .sidebar {
            flex: 1;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }
        .info-panel {
            background: rgba(255,255,255,0.1);
            padding: 20px;
            border-radius: 15px;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255,255,255,0.2);
        }
        .info-panel h3 {
            margin: 0 0 15px 0;
            color: #4CAF50;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            margin: 8px 0;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }
        .metric:last-child {
            border-bottom: none;
        }
        .loading {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,0.3);
            border-radius: 50%;
            border-top-color: #4CAF50;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .membrane-status {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
            margin: 10px 0;
        }
        .membrane-item {
            background: rgba(0,0,0,0.2);
            padding: 8px;
            border-radius: 5px;
            text-align: center;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🧠 Deep Tree Echo - Cognitive Interface</h1>
        <div class="session-info">
            Session: <span id="sessionId">Initializing...</span>
        </div>
    </div>
    
    <div class="main-container">
        <div class="chat-container">
            <div class="chat-header">
                <h3>Cognitive Conversation</h3>
                <p>Interact with the Deep Tree Echo cognitive architecture. Your inputs will be processed through memory, reasoning, and grammar membranes.</p>
            </div>
            
            <div class="chat-messages" id="chatMessages">
                <div class="message system-message">
                    <strong>System:</strong> Initializing cognitive session...
                </div>
            </div>
            
            <div class="chat-input">
                <div class="input-group">
                    <input type="text" id="userInput" placeholder="Enter your message or question..." disabled>
                    <button id="sendButton" onclick="sendMessage()" disabled>Send</button>
                </div>
            </div>
        </div>
        
        <div class="sidebar">
            <div class="info-panel">
                <h3>Session Status</h3>
                <div class="metric">
                    <span>Status:</span>
                    <span id="sessionStatus">Initializing</span>
                </div>
                <div class="metric">
                    <span>Messages:</span>
                    <span id="messageCount">0</span>
                </div>
                <div class="metric">
                    <span>Avg Response Time:</span>
                    <span id="avgResponseTime">0ms</span>
                </div>
                <div class="metric">
                    <span>Memory Items:</span>
                    <span id="memoryItems">0</span>
                </div>
            </div>
            
            <div class="info-panel">
                <h3>Membrane Status</h3>
                <div class="membrane-status">
                    <div class="membrane-item">💭 Memory<br><span id="memoryStatus">Active</span></div>
                    <div class="membrane-item">⚡ Reasoning<br><span id="reasoningStatus">Active</span></div>
                    <div class="membrane-item">🎭 Grammar<br><span id="grammarStatus">Active</span></div>
                    <div class="membrane-item">🤖 RWKV<br><span id="rwkvStatus">Mock</span></div>
                </div>
            </div>
            
            <div class="info-panel">
                <h3>Quick Actions</h3>
                <button class="input-group button" onclick="clearConversation()" style="width: 100%; margin-bottom: 10px;">Clear Conversation</button>
                <button class="input-group button" onclick="exportConversation()" style="width: 100%; margin-bottom: 10px;">Export History</button>
                <button class="input-group button" onclick="window.location.href='/'" style="width: 100%;">Back to Dashboard</button>
            </div>
        </div>
    </div>
    
    <script>
        let currentSessionId = null;
        let messageCount = 0;
        let totalResponseTime = 0;
        
        // Initialize session
        async function initializeSession() {
            try {
                const response = await fetch('/api/session', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    }
                });
                
                const data = await response.json();
                currentSessionId = data.session_id;
                
                document.getElementById('sessionId').textContent = currentSessionId.substring(0, 8) + '...';
                document.getElementById('sessionStatus').textContent = 'Active';
                document.getElementById('userInput').disabled = false;
                document.getElementById('sendButton').disabled = false;
                
                addSystemMessage('Cognitive session initialized. Ready for interaction.');
                
            } catch (error) {
                addSystemMessage('Error initializing session: ' + error.message);
            }
        }
        
        // Send message
        async function sendMessage() {
            const input = document.getElementById('userInput');
            const sendButton = document.getElementById('sendButton');
            const message = input.value.trim();
            
            if (!message || !currentSessionId) return;
            
            // Disable input
            input.disabled = true;
            sendButton.disabled = true;
            sendButton.innerHTML = '<span class="loading"></span>';
            
            // Add user message
            addUserMessage(message);
            input.value = '';
            
            try {
                const startTime = Date.now();
                
                const response = await fetch('/api/process', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        session_id: currentSessionId,
                        input: message
                    })
                });
                
                const result = await response.json();
                const responseTime = Date.now() - startTime;
                
                // Add echo response
                addEchoMessage(result.response, responseTime, result.membrane_outputs);
                
                // Update metrics
                messageCount++;
                totalResponseTime += responseTime;
                updateMetrics();
                
            } catch (error) {
                addSystemMessage('Error processing message: ' + error.message);
            } finally {
                // Re-enable input
                input.disabled = false;
                sendButton.disabled = false;
                sendButton.textContent = 'Send';
                input.focus();
            }
        }
        
        // Add messages to chat
        function addUserMessage(message) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message user-message';
            messageDiv.innerHTML = `
                <strong>You:</strong> ${message}
                <div class="message-meta">${new Date().toLocaleTimeString()}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addEchoMessage(message, responseTime, membraneOutputs) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message echo-message';
            
            let membraneInfo = '';
            if (membraneOutputs) {
                membraneInfo = `
                    <details style="margin-top: 10px;">
                        <summary style="cursor: pointer; opacity: 0.8;">Membrane Details</summary>
                        <div style="margin-top: 8px; font-size: 0.9em;">
                            <div><strong>Memory:</strong> ${membraneOutputs.memory}</div>
                            <div><strong>Reasoning:</strong> ${membraneOutputs.reasoning}</div>
                            <div><strong>Grammar:</strong> ${membraneOutputs.grammar}</div>
                        </div>
                    </details>
                `;
            }
            
            messageDiv.innerHTML = `
                <strong>Echo:</strong> ${message}
                <div class="message-meta">
                    ${new Date().toLocaleTimeString()} • ${responseTime}ms
                </div>
                ${membraneInfo}
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        function addSystemMessage(message) {
            const messagesDiv = document.getElementById('chatMessages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'message system-message';
            messageDiv.innerHTML = `
                <strong>System:</strong> ${message}
                <div class="message-meta">${new Date().toLocaleTimeString()}</div>
            `;
            messagesDiv.appendChild(messageDiv);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }
        
        // Update metrics
        function updateMetrics() {
            document.getElementById('messageCount').textContent = messageCount;
            const avgTime = messageCount > 0 ? Math.round(totalResponseTime / messageCount) : 0;
            document.getElementById('avgResponseTime').textContent = avgTime + 'ms';
        }
        
        // Clear conversation
        function clearConversation() {
            if (confirm('Clear conversation history?')) {
                document.getElementById('chatMessages').innerHTML = '';
                addSystemMessage('Conversation cleared.');
                messageCount = 0;
                totalResponseTime = 0;
                updateMetrics();
            }
        }
        
        // Export conversation
        function exportConversation() {
            const messages = document.getElementById('chatMessages').innerText;
            const blob = new Blob([messages], { type: 'text/plain' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `echo_conversation_${new Date().toISOString().split('T')[0]}.txt`;
            a.click();
            URL.revokeObjectURL(url);
        }
        
        // Handle Enter key
        document.getElementById('userInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendMessage();
            }
        });
        
        // Initialize on load
        window.onload = function() {
            initializeSession();
        };
    </script>
</body>
</html>
'''

# Write templates
with open(os.path.join(template_dir, 'dashboard.html'), 'w') as f:
    f.write(dashboard_template)

with open(os.path.join(template_dir, 'cognitive.html'), 'w') as f:
    f.write(cognitive_template)

if __name__ == '__main__':
    # Configure for deployment
    host = '0.0.0.0'
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Starting Deep Tree Echo Integrated Application")
    logger.info(f"WebVM Mode: {CONFIG['webvm_mode']}")
    logger.info(f"Memory Limit: {CONFIG['memory_limit_mb']}MB")
    logger.info(f"Server: {host}:{port}")
    
    app.run(host=host, port=port, debug=debug, threaded=True)

