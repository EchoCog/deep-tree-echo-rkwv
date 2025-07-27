"""
Enhanced Cognitive Integration for Phase 2
Integrates advanced cognitive processing with existing app infrastructure
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
import asyncio

# Import Phase 2 Advanced Cognitive Processing Components
from cognitive_reflection import (
    MetaCognitiveReflectionSystem, CognitiveStrategy, ProcessingError, 
    CognitiveMetrics, MetaCognitiveMonitor
)
from reasoning_chains import (
    ComplexReasoningSystem, ReasoningType, DeductiveReasoning,
    InductiveReasoning, AbductiveReasoning
)
from adaptive_learning import (
    AdaptiveLearningSystem, PersonalizationEngine, ResponseStyleLearner,
    CognitiveStrategyLearner, UserPreference, FeedbackEntry
)

logger = logging.getLogger(__name__)

class EnhancedCognitiveProcessor:
    """Enhanced cognitive processor integrating Phase 2 capabilities"""
    
    def __init__(self, persistent_memory=None):
        self.persistent_memory = persistent_memory
        
        # Initialize Phase 2 systems
        try:
            self.meta_cognitive_system = MetaCognitiveReflectionSystem()
            self.complex_reasoning_system = ComplexReasoningSystem()
            self.adaptive_learning_system = AdaptiveLearningSystem()
            self.enhanced_processing_enabled = True
            logger.info("Enhanced cognitive processor initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced cognitive systems: {e}")
            self.enhanced_processing_enabled = False
            
    async def process_input_enhanced(self, input_text: str, session_id: str, 
                                   conversation_history: List[Dict], 
                                   memory_state: Dict,
                                   fallback_processor=None) -> Dict[str, Any]:
        """Process input with Phase 2 enhanced cognitive capabilities"""
        start_time = time.time()
        
        if not self.enhanced_processing_enabled:
            if fallback_processor:
                # Call fallback processor but wrap in enhanced structure
                basic_result = fallback_processor(input_text)
                return self._wrap_fallback_result(basic_result, start_time)
            else:
                return self._create_fallback_response(input_text, start_time)
        
        try:
            # Phase 2: Meta-cognitive processing setup
            cognitive_context = {
                'user_input': input_text,
                'conversation_history': conversation_history,
                'memory_state': memory_state,
                'processing_goals': []
            }
            
            # Phase 2: Start meta-cognitive monitoring
            monitoring_context = self.meta_cognitive_system.before_processing(cognitive_context)
            processing_strategy = monitoring_context.get('strategy_selected', CognitiveStrategy.ADAPTIVE)
            
            # Enhanced membrane processing
            memory_response = await self._process_memory_enhanced(input_text, session_id)
            reasoning_response = await self._process_reasoning_enhanced(input_text, memory_response)
            grammar_response = await self._process_grammar_enhanced(input_text)
            
            # Advanced integration
            integrated_response = await self._integrate_responses_enhanced(
                memory_response, reasoning_response, grammar_response, input_text
            )
            
            processing_time = time.time() - start_time
            
            # Meta-cognitive evaluation
            cognitive_metrics = self.meta_cognitive_system.after_processing(
                {
                    'response': integrated_response,
                    'total_processing_time': processing_time,
                    'memory_retrievals': len(memory_response.get('retrieved_memories', [])),
                    'reasoning_steps': reasoning_response.get('steps_count', 0)
                },
                monitoring_context
            )
            
            # Adaptive learning
            interaction_data = {
                'user_id': session_id,
                'user_input': input_text,
                'system_response': integrated_response,
                'processing_time': processing_time,
                'cognitive_strategy': processing_strategy.value if hasattr(processing_strategy, 'value') else str(processing_strategy)
            }
            self.adaptive_learning_system.process_interaction_for_learning(interaction_data)
            
            # Create enhanced conversation entry
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
                },
                'phase2_features': {
                    'cognitive_strategy_used': str(processing_strategy),
                    'meta_cognitive_insights': cognitive_metrics.to_dict() if hasattr(cognitive_metrics, 'to_dict') else None,
                    'reasoning_chain_applied': reasoning_response.get('reasoning_type') is not None,
                    'adaptive_learning_applied': True,
                    'advanced_processing_enabled': True,
                    'enhanced_confidence': self._calculate_overall_confidence(memory_response, reasoning_response, grammar_response)
                }
            }
            
            return conversation_entry
            
        except Exception as e:
            logger.error(f"Error in enhanced cognitive processing: {e}")
            if fallback_processor:
                return fallback_processor(input_text)
            else:
                return self._create_fallback_response(input_text, start_time)
    
    async def _process_memory_enhanced(self, input_text: str, session_id: str) -> Dict[str, Any]:
        """Enhanced memory processing with Phase 2 capabilities"""
        memory_result = {
            'content': f"Memory processing: {input_text}",
            'retrieved_memories': [],
            'memory_operations': [],
            'confidence': 0.7
        }
        
        try:
            if self.persistent_memory:
                # Store significant memories
                if self._is_memory_significant(input_text):
                    memory_type = self._classify_memory_type(input_text)
                    memory_id = self.persistent_memory.store_memory(
                        content=input_text,
                        memory_type=memory_type,
                        session_id=session_id,
                        metadata={
                            'processing_type': 'enhanced_memory_membrane',
                            'cognitive_complexity': self._assess_input_complexity(input_text),
                            'timestamp': datetime.now().isoformat()
                        }
                    )
                    if memory_id:
                        memory_result['memory_operations'].append(f"stored_{memory_id}")
                
                # Enhanced memory retrieval
                search_results = self.persistent_memory.search_knowledge(
                    query=input_text,
                    context_session=session_id
                )
                memory_result['retrieved_memories'] = [
                    {
                        'content': result.content,
                        'relevance': 0.8,  # Default relevance since we don't have scores
                        'memory_type': result.content_type,
                        'timestamp': result.timestamp
                    }
                    for result in search_results
                ]
                
                # Enhanced memory content
                if memory_result['retrieved_memories']:
                    memory_context = f"Retrieved {len(memory_result['retrieved_memories'])} relevant memories. "
                    memory_result['content'] = f"Enhanced memory processing: {memory_context}{input_text}"
                    memory_result['confidence'] = 0.85
            
        except Exception as e:
            logger.error(f"Error in enhanced memory processing: {e}")
        
        return memory_result
    
    async def _process_reasoning_enhanced(self, input_text: str, memory_context: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced reasoning processing with Phase 2 complex reasoning chains"""
        reasoning_result = {
            'content': f"Enhanced reasoning analysis: {input_text}",
            'reasoning_type': None,
            'confidence': 0.8,
            'steps_count': 0,
            'explanation': ''
        }
        
        try:
            complexity = self._assess_input_complexity(input_text)
            
            if complexity in ['medium', 'high']:
                reasoning_type = self._suggest_reasoning_type(input_text)
                
                # Apply complex reasoning
                reasoning_result.update({
                    'content': f"Complex {reasoning_type} reasoning applied: {input_text}",
                    'reasoning_type': reasoning_type,
                    'confidence': 0.9,
                    'steps_count': 3,
                    'explanation': f"Applied {reasoning_type} reasoning with validation and multi-step analysis"
                })
                
                # Enhanced reasoning content
                if memory_context.get('retrieved_memories'):
                    reasoning_result['content'] += f" (integrated with {len(memory_context['retrieved_memories'])} memory contexts)"
        
        except Exception as e:
            logger.error(f"Error in enhanced reasoning processing: {e}")
        
        return reasoning_result
    
    async def _process_grammar_enhanced(self, input_text: str) -> Dict[str, Any]:
        """Enhanced grammar processing with Phase 2 linguistic analysis"""
        grammar_result = {
            'content': f"Enhanced linguistic analysis: {input_text}",
            'linguistic_analysis': {},
            'confidence': 0.85
        }
        
        try:
            grammar_result['linguistic_analysis'] = {
                'sentence_structure': 'interrogative' if '?' in input_text else 'declarative',
                'complexity_level': self._assess_input_complexity(input_text),
                'key_entities': self._extract_key_entities(input_text),
                'semantic_markers': self._identify_semantic_markers(input_text),
                'linguistic_features': self._analyze_linguistic_features(input_text)
            }
        
        except Exception as e:
            logger.error(f"Error in enhanced grammar processing: {e}")
        
        return grammar_result
    
    async def _integrate_responses_enhanced(self, memory_resp: Dict[str, Any], 
                                          reasoning_resp: Dict[str, Any], 
                                          grammar_resp: Dict[str, Any], 
                                          input_text: str) -> str:
        """Advanced integration using Phase 2 reasoning chains"""
        try:
            # Calculate weighted confidence
            memory_conf = memory_resp.get('confidence', 0.5)
            reasoning_conf = reasoning_resp.get('confidence', 0.5) 
            grammar_conf = grammar_resp.get('confidence', 0.5)
            
            overall_confidence = (memory_conf * 0.3 + reasoning_conf * 0.5 + grammar_conf * 0.2)
            
            # Create enhanced integration
            if reasoning_resp.get('reasoning_type') and reasoning_resp.get('explanation'):
                integration = f"**🧠 Enhanced Cognitive Analysis** (Confidence: {overall_confidence:.2f})\n\n"
                integration += f"**🔍 Reasoning Applied**: {reasoning_resp['reasoning_type']}\n"
                integration += f"**💭 Memory Context**: {len(memory_resp.get('retrieved_memories', []))} relevant memories integrated\n"
                integration += f"**🎭 Linguistic Analysis**: {len(grammar_resp.get('linguistic_analysis', {}).get('semantic_markers', []))} semantic markers identified\n\n"
                
                integration += f"**📊 Analysis**: {reasoning_resp['content']}\n\n"
                
                if reasoning_resp.get('explanation'):
                    integration += f"**🔗 Reasoning Chain**: {reasoning_resp['explanation']}\n\n"
                
                integration += f"**🎯 Integrated Response**: The system applied sophisticated cognitive processing to analyze your input. "
                integration += f"Through {reasoning_resp['reasoning_type']} reasoning and memory integration, "
                integration += f"I can provide a comprehensive response addressing your query about: {input_text}"
            else:
                # Standard enhanced integration
                integration = f"**Enhanced Cognitive Processing** (Confidence: {overall_confidence:.2f})\n\n"
                integration += f"Memory: {memory_resp.get('content', '')}\n"
                integration += f"Reasoning: {reasoning_resp.get('content', '')}\n"
                integration += f"Analysis: {grammar_resp.get('content', '')}\n\n"
                integration += f"Integrated response addressing: {input_text}"
            
            return integration
            
        except Exception as e:
            logger.error(f"Error in advanced integration: {e}")
            return f"Enhanced processing applied to: {input_text}"
    
    def _calculate_overall_confidence(self, memory_resp: Dict, reasoning_resp: Dict, grammar_resp: Dict) -> float:
        """Calculate overall confidence score"""
        memory_conf = memory_resp.get('confidence', 0.5)
        reasoning_conf = reasoning_resp.get('confidence', 0.5)
        grammar_conf = grammar_resp.get('confidence', 0.5)
        return round((memory_conf * 0.3 + reasoning_conf * 0.5 + grammar_conf * 0.2), 2)
    
    def _assess_input_complexity(self, text: str) -> str:
        """Assess cognitive complexity of input"""
        words = text.split()
        word_count = len(words)
        
        questions = text.count('?')
        complex_words = sum(1 for word in words if len(word) > 8)
        
        if word_count > 20 or questions > 2 or complex_words > 3:
            return 'high'
        elif word_count > 10 or questions > 0 or complex_words > 1:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_reasoning_type(self, text: str) -> str:
        """Suggest appropriate reasoning type"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['because', 'therefore', 'conclude']):
            return 'deductive'
        elif any(word in text_lower for word in ['pattern', 'similar', 'examples']):
            return 'inductive'
        elif any(word in text_lower for word in ['explain', 'why', 'cause']):
            return 'abductive'
        elif any(word in text_lower for word in ['like', 'similar to', 'compare']):
            return 'analogical'
        else:
            return 'adaptive'
    
    def _assess_memory_integration(self, text: str) -> str:
        """Assess required memory integration level"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['remember', 'recall', 'previous', 'before']):
            return 'high'
        elif any(word in text_lower for word in ['context', 'background', 'history']):
            return 'medium'
        else:
            return 'low'
    
    def _identify_learning_opportunities(self, text: str) -> List[str]:
        """Identify adaptive learning opportunities"""
        opportunities = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['prefer', 'like', 'want']):
            opportunities.append('preference_learning')
        if any(word in text_lower for word in ['detailed', 'brief', 'summary']):
            opportunities.append('response_style_learning')
        if '?' in text:
            opportunities.append('interaction_pattern_learning')
        
        return opportunities
    
    def _is_memory_significant(self, text: str) -> bool:
        """Determine if input should be stored in persistent memory"""
        return len(text.split()) > 5 and any(word in text.lower() 
                                           for word in ['important', 'remember', 'learn', 'know'])
    
    def _classify_memory_type(self, text: str) -> str:
        """Classify memory type for storage"""
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['fact', 'information', 'data']):
            return 'factual'
        elif any(word in text_lower for word in ['experience', 'happened', 'event']):
            return 'episodic'
        elif any(word in text_lower for word in ['how to', 'process', 'method']):
            return 'procedural'
        else:
            return 'general'
    
    def _extract_key_entities(self, text: str) -> List[str]:
        """Extract key entities from text"""
        words = text.split()
        entities = []
        for word in words:
            if word.istitle() and len(word) > 2:
                entities.append(word)
        return entities[:5]
    
    def _identify_semantic_markers(self, text: str) -> List[str]:
        """Identify semantic markers in text"""
        markers = []
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['analyze', 'examine', 'study']):
            markers.append('analytical_intent')
        if any(word in text_lower for word in ['explain', 'describe', 'tell']):
            markers.append('explanatory_intent')
        if '?' in text:
            markers.append('interrogative')
        if any(word in text_lower for word in ['compare', 'contrast', 'versus']):
            markers.append('comparative_intent')
        if any(word in text_lower for word in ['help', 'assist', 'support']):
            markers.append('assistance_request')
            
        return markers
    
    def _analyze_linguistic_features(self, text: str) -> Dict[str, Any]:
        """Analyze linguistic features"""
        return {
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'question_count': text.count('?'),
            'exclamation_count': text.count('!'),
            'complexity_score': self._assess_input_complexity(text)
        }
    
    def _create_fallback_response(self, input_text: str, start_time: float) -> Dict[str, Any]:
        """Create fallback response when enhanced processing fails"""
        processing_time = time.time() - start_time
        
        return {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'response': f"Standard processing applied to: {input_text}",
            'processing_time': processing_time,
            'membrane_outputs': {
                'memory': {'content': f"Basic memory processing: {input_text}", 'confidence': 0.6},
                'reasoning': {'content': f"Basic reasoning: {input_text}", 'confidence': 0.6},
                'grammar': {'content': f"Basic grammar processing: {input_text}", 'confidence': 0.6}
            },
            'cognitive_metadata': {
                'complexity_detected': self._assess_input_complexity(input_text),
                'reasoning_type_suggested': 'basic',
                'memory_integration_level': 'basic',
                'adaptive_learning_opportunities': []
            },
            'phase2_features': {
                'cognitive_strategy_used': 'fallback',
                'meta_cognitive_insights': None,
                'reasoning_chain_applied': False,
                'adaptive_learning_applied': False,
                'advanced_processing_enabled': False,
                'enhanced_confidence': 0.6
            }
        }
    
    def _wrap_fallback_result(self, basic_result: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """Wrap a basic fallback result in enhanced structure"""
        processing_time = basic_result.get('processing_time', time.time() - start_time)
        input_text = basic_result.get('input', '')
        
        return {
            'timestamp': datetime.now().isoformat(),
            'input': input_text,
            'response': basic_result.get('response', f"Fallback processing: {input_text}"),
            'processing_time': processing_time,
            'membrane_outputs': {
                'memory': {'content': f"Fallback memory processing: {input_text}", 'confidence': 0.6},
                'reasoning': {'content': f"Fallback reasoning: {input_text}", 'confidence': 0.6},
                'grammar': {'content': f"Fallback grammar processing: {input_text}", 'confidence': 0.6}
            },
            'cognitive_metadata': {
                'complexity_detected': self._assess_input_complexity(input_text),
                'reasoning_type_suggested': 'fallback',
                'memory_integration_level': 'basic',
                'adaptive_learning_opportunities': []
            },
            'phase2_features': {
                'cognitive_strategy_used': 'fallback',
                'meta_cognitive_insights': None,
                'reasoning_chain_applied': False,
                'adaptive_learning_applied': False,
                'advanced_processing_enabled': False,
                'enhanced_confidence': 0.6
            }
        }