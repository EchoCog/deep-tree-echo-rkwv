"""
Advanced AI Integration System
Multi-model AI integration, comparison, and hybrid cognitive architectures
"""

import time
import json
import logging
import asyncio
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, asdict
from abc import ABC, abstractmethod
from enum import Enum
# Try to import numpy, use fallback if not available
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    # Fallback implementations
    class NumpyFallback:
        @staticmethod
        def random():
            import random
            return random.random()
        
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        
        @staticmethod
        def std(values):
            if not values:
                return 0.0
            mean_val = sum(values) / len(values)
            variance = sum((x - mean_val) ** 2 for x in values) / len(values)
            return variance ** 0.5
        
        @staticmethod
        def linalg_norm(vector):
            return sum(x * x for x in vector) ** 0.5
        
        @staticmethod
        def array(values):
            return values
        
        @staticmethod
        def min(values):
            return min(values) if values else 0
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def sqrt(value):
            return value ** 0.5
        
        @staticmethod
        def randint(low, high):
            import random
            return random.randint(low, high)
        
        @staticmethod
        def choice(items):
            import random
            return random.choice(items)
    
    np = NumpyFallback()
from collections import defaultdict, deque
import uuid

# Import existing components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)

class AIModelType(Enum):
    """Types of AI models that can be integrated"""
    LANGUAGE_MODEL = "language_model"
    VISION_MODEL = "vision_model"
    AUDIO_MODEL = "audio_model"
    MULTIMODAL = "multimodal"
    SPECIALIZED = "specialized"

class IntegrationStrategy(Enum):
    """Strategies for AI model integration"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ENSEMBLE = "ensemble"
    HIERARCHICAL = "hierarchical"
    DYNAMIC_ROUTING = "dynamic_routing"

class OptimizationMethod(Enum):
    """Methods for AI model optimization"""
    FINE_TUNING = "fine_tuning"
    PROMPT_ENGINEERING = "prompt_engineering"
    PARAMETER_ADJUSTMENT = "parameter_adjustment"
    ENSEMBLE_WEIGHTING = "ensemble_weighting"
    DYNAMIC_SELECTION = "dynamic_selection"

@dataclass
class AIModelConfig:
    """Configuration for an AI model"""
    model_id: str
    model_type: AIModelType
    model_name: str
    api_endpoint: Optional[str]
    parameters: Dict[str, Any]
    capabilities: List[str]
    performance_metrics: Dict[str, float]
    cost_per_request: float
    latency_ms: float
    enabled: bool = True

@dataclass
class ModelComparisonResult:
    """Result from comparing AI models"""
    comparison_id: str
    models_compared: List[str]
    test_cases: List[Dict[str, Any]]
    performance_metrics: Dict[str, Dict[str, float]]
    recommendations: List[str]
    best_model_overall: str
    best_model_by_metric: Dict[str, str]
    timestamp: str

@dataclass
class HybridResponse:
    """Response from hybrid cognitive-AI architecture"""
    response_id: str
    primary_response: str
    confidence_score: float
    model_contributions: Dict[str, Any]
    cognitive_processing: Dict[str, Any]
    ai_processing: Dict[str, Any]
    fusion_strategy: str
    processing_time: float
    metadata: Dict[str, Any]

class AIModel(ABC):
    """Abstract base class for AI models"""
    
    def __init__(self, config: AIModelConfig):
        self.config = config
        self.request_count = 0
        self.total_latency = 0.0
        self.error_count = 0
        self.last_request_time = None
    
    @abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input data and return response"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Get list of model capabilities"""
        pass
    
    def update_metrics(self, latency: float, success: bool):
        """Update performance metrics"""
        self.request_count += 1
        self.total_latency += latency
        if not success:
            self.error_count += 1
        self.last_request_time = datetime.now()
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current performance statistics"""
        if self.request_count == 0:
            return {}
        
        return {
            'avg_latency': self.total_latency / self.request_count,
            'error_rate': self.error_count / self.request_count,
            'requests_per_minute': self.request_count / max(1, (datetime.now() - self.last_request_time).seconds / 60) if self.last_request_time else 0,
            'total_requests': self.request_count
        }

class MockLanguageModel(AIModel):
    """Mock language model for testing and demonstration"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        self.response_templates = {
            'question': "Based on the question '{query}', I would say that {response_content}",
            'explanation': "To explain {topic}, I need to consider {explanation_content}",
            'analysis': "My analysis of '{content}' reveals {analysis_content}",
            'default': "I understand you're asking about '{query}'. {default_content}"
        }
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process language input"""
        start_time = time.time()
        
        try:
            query = input_data.get('query', input_data.get('text', ''))
            task_type = input_data.get('task_type', 'default')
            
            # Simulate processing delay
            await asyncio.sleep(0.1 + np.random.random() * 0.2)
            
            # Generate response based on task type
            if task_type in self.response_templates:
                template = self.response_templates[task_type]
            else:
                template = self.response_templates['default']
            
            # Simple content generation (placeholder)
            response_content = f"relevant information about {query[:50]}..."
            explanation_content = f"multiple perspectives and factors"
            analysis_content = f"key patterns and insights"
            default_content = f"Here's my response regarding this topic."
            
            response_text = template.format(
                query=query,
                topic=query,
                content=query,
                response_content=response_content,
                explanation_content=explanation_content,
                analysis_content=analysis_content,
                default_content=default_content
            )
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, True)
            
            return {
                'response': response_text,
                'confidence': 0.7 + np.random.random() * 0.3,
                'model_id': self.config.model_id,
                'processing_time': processing_time,
                'tokens_generated': len(response_text.split()),
                'metadata': {
                    'task_type': task_type,
                    'input_length': len(query)
                }
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            logger.error(f"Language model error: {e}")
            return {
                'error': str(e),
                'model_id': self.config.model_id,
                'processing_time': processing_time
            }
    
    def get_capabilities(self) -> List[str]:
        return ['text_generation', 'question_answering', 'explanation', 'analysis']

class MockVisionModel(AIModel):
    """Mock vision model for testing and demonstration"""
    
    def __init__(self, config: AIModelConfig):
        super().__init__(config)
        self.object_classes = ['person', 'car', 'building', 'tree', 'animal', 'object']
        self.scene_types = ['indoor', 'outdoor', 'urban', 'natural', 'abstract']
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process vision input"""
        start_time = time.time()
        
        try:
            image_data = input_data.get('image', input_data.get('image_url', ''))
            task_type = input_data.get('task_type', 'object_detection')
            
            # Simulate processing delay
            await asyncio.sleep(0.2 + np.random.random() * 0.3)
            
            if task_type == 'object_detection':
                # Simulate object detection
                num_objects = np.random.randint(1, 6)
                objects = []
                for i in range(num_objects):
                    obj = {
                        'class': np.random.choice(self.object_classes),
                        'confidence': 0.6 + np.random.random() * 0.4,
                        'bbox': [np.random.randint(0, 100) for _ in range(4)]
                    }
                    objects.append(obj)
                
                result = {
                    'objects': objects,
                    'object_count': num_objects
                }
                
            elif task_type == 'scene_classification':
                # Simulate scene classification
                scene_scores = {scene: np.random.random() for scene in self.scene_types}
                best_scene = max(scene_scores.keys(), key=lambda k: scene_scores[k])
                
                result = {
                    'scene_type': best_scene,
                    'scene_scores': scene_scores
                }
                
            else:
                result = {
                    'description': f"Image analysis completed for {task_type}",
                    'features': [f"feature_{i}" for i in range(5)]
                }
            
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, True)
            
            return {
                'result': result,
                'confidence': 0.6 + np.random.random() * 0.4,
                'model_id': self.config.model_id,
                'processing_time': processing_time,
                'task_type': task_type
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.update_metrics(processing_time, False)
            logger.error(f"Vision model error: {e}")
            return {
                'error': str(e),
                'model_id': self.config.model_id,
                'processing_time': processing_time
            }
    
    def get_capabilities(self) -> List[str]:
        return ['object_detection', 'scene_classification', 'image_analysis']

class MultiModelAIIntegrator:
    """Integrates multiple AI models for enhanced cognitive processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.models: Dict[str, AIModel] = {}
        self.integration_strategies = {}
        self.request_history = deque(maxlen=1000)
        self.performance_cache = {}
        
    def register_model(self, model: AIModel):
        """Register an AI model for integration"""
        self.models[model.config.model_id] = model
        logger.info(f"Registered AI model: {model.config.model_id} ({model.config.model_type.value})")
    
    def set_integration_strategy(self, task_type: str, strategy: IntegrationStrategy, model_ids: List[str]):
        """Set integration strategy for a specific task type"""
        self.integration_strategies[task_type] = {
            'strategy': strategy,
            'models': model_ids,
            'weights': {model_id: 1.0 / len(model_ids) for model_id in model_ids}
        }
    
    async def process_with_integration(self, 
                                     input_data: Dict[str, Any],
                                     task_type: str = 'default') -> Dict[str, Any]:
        """Process input using integrated AI models"""
        
        start_time = time.time()
        
        # Get integration strategy for task type
        strategy_config = self.integration_strategies.get(task_type, {
            'strategy': IntegrationStrategy.SEQUENTIAL,
            'models': list(self.models.keys())[:2],  # Use first 2 models
            'weights': {}
        })
        
        strategy = strategy_config['strategy']
        model_ids = strategy_config['models']
        
        # Filter available models
        available_models = [mid for mid in model_ids if mid in self.models]
        
        if not available_models:
            return {'error': 'No available models for integration'}
        
        try:
            if strategy == IntegrationStrategy.SEQUENTIAL:
                result = await self._sequential_integration(input_data, available_models)
            elif strategy == IntegrationStrategy.PARALLEL:
                result = await self._parallel_integration(input_data, available_models)
            elif strategy == IntegrationStrategy.ENSEMBLE:
                result = await self._ensemble_integration(input_data, available_models, strategy_config['weights'])
            elif strategy == IntegrationStrategy.HIERARCHICAL:
                result = await self._hierarchical_integration(input_data, available_models)
            elif strategy == IntegrationStrategy.DYNAMIC_ROUTING:
                result = await self._dynamic_routing_integration(input_data, available_models)
            else:
                result = await self._parallel_integration(input_data, available_models)  # Default
            
            processing_time = time.time() - start_time
            
            # Record request
            request_record = {
                'timestamp': datetime.now().isoformat(),
                'task_type': task_type,
                'strategy': strategy.value,
                'models_used': available_models,
                'processing_time': processing_time,
                'success': 'error' not in result
            }
            self.request_history.append(request_record)
            
            result['integration_metadata'] = {
                'strategy_used': strategy.value,
                'models_involved': available_models,
                'total_processing_time': processing_time
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Integration processing error: {e}")
            return {
                'error': str(e),
                'integration_metadata': {
                    'strategy_attempted': strategy.value,
                    'models_attempted': available_models
                }
            }
    
    async def _sequential_integration(self, input_data: Dict[str, Any], model_ids: List[str]) -> Dict[str, Any]:
        """Sequential processing through models"""
        
        current_data = input_data.copy()
        results = []
        
        for model_id in model_ids:
            model = self.models[model_id]
            model_result = await model.process(current_data)
            results.append({
                'model_id': model_id,
                'result': model_result
            })
            
            # Pass result to next model as additional context
            if 'response' in model_result:
                current_data['previous_response'] = model_result['response']
            
            # Stop if error occurred
            if 'error' in model_result:
                break
        
        # Combine results
        final_response = results[-1]['result'].get('response', 'No final response available')
        confidence = np.mean([r['result'].get('confidence', 0.5) for r in results])
        
        return {
            'response': final_response,
            'confidence': confidence,
            'sequential_results': results,
            'models_chain': model_ids
        }
    
    async def _parallel_integration(self, input_data: Dict[str, Any], model_ids: List[str]) -> Dict[str, Any]:
        """Parallel processing through models"""
        
        tasks = []
        for model_id in model_ids:
            model = self.models[model_id]
            task = asyncio.create_task(model.process(input_data))
            tasks.append((model_id, task))
        
        # Wait for all tasks to complete
        results = []
        for model_id, task in tasks:
            try:
                model_result = await task
                results.append({
                    'model_id': model_id,
                    'result': model_result
                })
            except Exception as e:
                results.append({
                    'model_id': model_id,
                    'result': {'error': str(e)}
                })
        
        # Combine results
        successful_results = [r for r in results if 'error' not in r['result']]
        
        if not successful_results:
            return {'error': 'All parallel models failed'}
        
        # Take the highest confidence response
        best_result = max(successful_results, key=lambda r: r['result'].get('confidence', 0))
        
        return {
            'response': best_result['result'].get('response', 'No response available'),
            'confidence': best_result['result'].get('confidence', 0.5),
            'best_model': best_result['model_id'],
            'parallel_results': results
        }
    
    async def _ensemble_integration(self, input_data: Dict[str, Any], model_ids: List[str], weights: Dict[str, float]) -> Dict[str, Any]:
        """Ensemble integration with weighted combination"""
        
        # Run parallel processing first
        parallel_result = await self._parallel_integration(input_data, model_ids)
        
        if 'error' in parallel_result:
            return parallel_result
        
        # Weight and combine results
        model_results = parallel_result['parallel_results']
        successful_results = [r for r in model_results if 'error' not in r['result']]
        
        if not successful_results:
            return {'error': 'No successful ensemble members'}
        
        # Calculate weighted confidence
        total_weight = 0
        weighted_confidence = 0
        
        for result in successful_results:
            model_id = result['model_id']
            weight = weights.get(model_id, 1.0)
            confidence = result['result'].get('confidence', 0.5)
            
            weighted_confidence += weight * confidence
            total_weight += weight
        
        ensemble_confidence = weighted_confidence / total_weight if total_weight > 0 else 0.5
        
        # For simplicity, use the best individual response
        # In practice, could combine text responses more sophisticated
        best_response = max(successful_results, key=lambda r: r['result'].get('confidence', 0))
        
        return {
            'response': best_response['result'].get('response', 'No ensemble response'),
            'confidence': ensemble_confidence,
            'ensemble_metadata': {
                'member_count': len(successful_results),
                'weights_used': weights,
                'individual_confidences': {r['model_id']: r['result'].get('confidence', 0) for r in successful_results}
            }
        }
    
    async def _hierarchical_integration(self, input_data: Dict[str, Any], model_ids: List[str]) -> Dict[str, Any]:
        """Hierarchical integration with primary and fallback models"""
        
        if not model_ids:
            return {'error': 'No models for hierarchical integration'}
        
        # Use first model as primary, others as fallbacks
        primary_model_id = model_ids[0]
        fallback_model_ids = model_ids[1:]
        
        # Try primary model first
        primary_model = self.models[primary_model_id]
        primary_result = await primary_model.process(input_data)
        
        # Check if primary result is satisfactory
        primary_confidence = primary_result.get('confidence', 0)
        confidence_threshold = self.config.get('hierarchical_threshold', 0.7)
        
        if 'error' not in primary_result and primary_confidence >= confidence_threshold:
            return {
                'response': primary_result.get('response', 'No primary response'),
                'confidence': primary_confidence,
                'hierarchical_metadata': {
                    'primary_model': primary_model_id,
                    'used_fallback': False
                }
            }
        
        # Try fallback models
        for fallback_model_id in fallback_model_ids:
            fallback_model = self.models[fallback_model_id]
            fallback_result = await fallback_model.process(input_data)
            
            fallback_confidence = fallback_result.get('confidence', 0)
            
            if 'error' not in fallback_result and fallback_confidence >= confidence_threshold:
                return {
                    'response': fallback_result.get('response', 'No fallback response'),
                    'confidence': fallback_confidence,
                    'hierarchical_metadata': {
                        'primary_model': primary_model_id,
                        'fallback_model': fallback_model_id,
                        'used_fallback': True,
                        'primary_confidence': primary_confidence
                    }
                }
        
        # If all failed, return best available result
        return {
            'response': primary_result.get('response', 'No hierarchical response available'),
            'confidence': primary_confidence,
            'hierarchical_metadata': {
                'primary_model': primary_model_id,
                'all_fallbacks_failed': True
            },
            'warning': 'All models below confidence threshold'
        }
    
    async def _dynamic_routing_integration(self, input_data: Dict[str, Any], model_ids: List[str]) -> Dict[str, Any]:
        """Dynamic routing based on input characteristics and model capabilities"""
        
        # Analyze input to determine best model
        input_characteristics = self._analyze_input_characteristics(input_data)
        
        # Score models based on suitability
        model_scores = {}
        for model_id in model_ids:
            model = self.models[model_id]
            capabilities = model.get_capabilities()
            performance_stats = model.get_performance_stats()
            
            # Simple scoring based on capabilities and performance
            capability_score = len(set(capabilities) & set(input_characteristics.get('required_capabilities', [])))
            performance_score = 1.0 - performance_stats.get('error_rate', 0.5)
            latency_score = 1.0 / (1.0 + performance_stats.get('avg_latency', 1.0))
            
            total_score = capability_score * 0.5 + performance_score * 0.3 + latency_score * 0.2
            model_scores[model_id] = total_score
        
        # Select best model
        if not model_scores:
            best_model_id = model_ids[0]  # Fallback
        else:
            best_model_id = max(model_scores.keys(), key=lambda k: model_scores[k])
        
        # Process with selected model
        selected_model = self.models[best_model_id]
        result = await selected_model.process(input_data)
        
        return {
            'response': result.get('response', 'No dynamic routing response'),
            'confidence': result.get('confidence', 0.5),
            'dynamic_routing_metadata': {
                'selected_model': best_model_id,
                'model_scores': model_scores,
                'input_characteristics': input_characteristics
            }
        }
    
    def _analyze_input_characteristics(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input to determine required capabilities"""
        
        characteristics = {
            'required_capabilities': [],
            'complexity': 'medium',
            'input_type': 'text'
        }
        
        # Determine input type
        if 'image' in input_data or 'image_url' in input_data:
            characteristics['input_type'] = 'image'
            characteristics['required_capabilities'].append('image_analysis')
        elif 'audio' in input_data or 'audio_url' in input_data:
            characteristics['input_type'] = 'audio'
            characteristics['required_capabilities'].append('audio_processing')
        else:
            characteristics['input_type'] = 'text'
            characteristics['required_capabilities'].append('text_generation')
        
        # Determine complexity
        text_content = str(input_data.get('query', input_data.get('text', '')))
        if len(text_content) > 500:
            characteristics['complexity'] = 'high'
        elif len(text_content) < 50:
            characteristics['complexity'] = 'low'
        
        # Add task-specific capabilities
        task_type = input_data.get('task_type', '')
        if 'question' in task_type.lower():
            characteristics['required_capabilities'].append('question_answering')
        elif 'analysis' in task_type.lower():
            characteristics['required_capabilities'].append('analysis')
        elif 'explanation' in task_type.lower():
            characteristics['required_capabilities'].append('explanation')
        
        return characteristics
    
    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get statistics about AI integration performance"""
        
        if not self.request_history:
            return {'message': 'No integration requests recorded'}
        
        recent_requests = list(self.request_history)
        
        # Basic statistics
        total_requests = len(recent_requests)
        successful_requests = sum(1 for r in recent_requests if r['success'])
        
        # Group by strategy
        strategy_stats = defaultdict(list)
        for request in recent_requests:
            strategy_stats[request['strategy']].append(request)
        
        # Calculate performance metrics
        strategy_performance = {}
        for strategy, requests in strategy_stats.items():
            if requests:
                strategy_performance[strategy] = {
                    'request_count': len(requests),
                    'success_rate': sum(1 for r in requests if r['success']) / len(requests),
                    'avg_processing_time': np.mean([r['processing_time'] for r in requests]),
                    'models_used': list(set(model for r in requests for model in r['models_used']))
                }
        
        return {
            'total_integration_requests': total_requests,
            'success_rate': successful_requests / total_requests,
            'registered_models': len(self.models),
            'configured_strategies': len(self.integration_strategies),
            'strategy_performance': strategy_performance,
            'model_performance': {model_id: model.get_performance_stats() 
                                for model_id, model in self.models.items()}
        }

class ModelComparator:
    """Compares performance of different AI models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.comparison_results = []
        self.test_suites = {}
        self._initialize_standard_tests()
    
    def _initialize_standard_tests(self):
        """Initialize standard test suites for model comparison"""
        
        self.test_suites['language_understanding'] = [
            {'query': 'What is artificial intelligence?', 'expected_keywords': ['AI', 'machine', 'learning', 'computer']},
            {'query': 'Explain quantum computing', 'expected_keywords': ['quantum', 'qubit', 'superposition']},
            {'query': 'How does machine learning work?', 'expected_keywords': ['algorithm', 'data', 'training']}
        ]
        
        self.test_suites['reasoning'] = [
            {'query': 'If all cats are animals, and Fluffy is a cat, what is Fluffy?', 'expected_answer': 'animal'},
            {'query': 'What comes next: 2, 4, 6, 8, ?', 'expected_answer': '10'},
            {'query': 'If it rains, the ground gets wet. The ground is wet. Did it rain?', 'expected_answer': 'maybe'}
        ]
        
        self.test_suites['creativity'] = [
            {'query': 'Write a short poem about technology', 'evaluation': 'creativity'},
            {'query': 'Invent a new app idea', 'evaluation': 'originality'},
            {'query': 'Tell a story in exactly 50 words', 'evaluation': 'constraint_satisfaction'}
        ]
    
    async def compare_models(self, 
                           model_list: List[AIModel],
                           test_suite_name: str = 'language_understanding',
                           custom_tests: List[Dict[str, Any]] = None) -> ModelComparisonResult:
        """Compare multiple AI models on a test suite"""
        
        if not model_list:
            raise ValueError("No models provided for comparison")
        
        test_cases = custom_tests or self.test_suites.get(test_suite_name, [])
        if not test_cases:
            raise ValueError(f"No test cases found for suite: {test_suite_name}")
        
        model_results = {}
        
        # Test each model
        for model in model_list:
            model_id = model.config.model_id
            model_results[model_id] = {
                'responses': [],
                'metrics': {
                    'avg_latency': 0.0,
                    'avg_confidence': 0.0,
                    'success_rate': 0.0,
                    'accuracy_score': 0.0
                }
            }
            
            latencies = []
            confidences = []
            successes = 0
            accuracy_scores = []
            
            for test_case in test_cases:
                try:
                    start_time = time.time()
                    result = await model.process(test_case)
                    latency = time.time() - start_time
                    
                    latencies.append(latency)
                    
                    if 'error' not in result:
                        successes += 1
                        confidence = result.get('confidence', 0.5)
                        confidences.append(confidence)
                        
                        # Calculate accuracy if expected answer provided
                        if 'expected_answer' in test_case:
                            accuracy = self._calculate_accuracy(result, test_case)
                            accuracy_scores.append(accuracy)
                    
                    model_results[model_id]['responses'].append({
                        'test_case': test_case,
                        'result': result,
                        'latency': latency
                    })
                    
                except Exception as e:
                    logger.error(f"Error testing model {model_id}: {e}")
                    model_results[model_id]['responses'].append({
                        'test_case': test_case,
                        'result': {'error': str(e)},
                        'latency': 0.0
                    })
            
            # Calculate metrics
            if latencies:
                model_results[model_id]['metrics']['avg_latency'] = np.mean(latencies)
            if confidences:
                model_results[model_id]['metrics']['avg_confidence'] = np.mean(confidences)
            if accuracy_scores:
                model_results[model_id]['metrics']['accuracy_score'] = np.mean(accuracy_scores)
            
            model_results[model_id]['metrics']['success_rate'] = successes / len(test_cases)
        
        # Generate comparison analysis
        best_model_overall = self._determine_best_model_overall(model_results)
        best_model_by_metric = self._determine_best_model_by_metric(model_results)
        recommendations = self._generate_comparison_recommendations(model_results)
        
        comparison_result = ModelComparisonResult(
            comparison_id=str(uuid.uuid4()),
            models_compared=[model.config.model_id for model in model_list],
            test_cases=test_cases,
            performance_metrics={model_id: results['metrics'] for model_id, results in model_results.items()},
            recommendations=recommendations,
            best_model_overall=best_model_overall,
            best_model_by_metric=best_model_by_metric,
            timestamp=datetime.now().isoformat()
        )
        
        self.comparison_results.append(comparison_result)
        return comparison_result
    
    def _calculate_accuracy(self, result: Dict[str, Any], test_case: Dict[str, Any]) -> float:
        """Calculate accuracy of model response"""
        
        expected_answer = test_case.get('expected_answer', '').lower()
        response_text = result.get('response', '').lower()
        
        if not expected_answer or not response_text:
            return 0.0
        
        # Simple accuracy based on keyword presence
        if expected_answer in response_text:
            return 1.0
        
        # Check for keyword overlap
        expected_keywords = test_case.get('expected_keywords', [])
        if expected_keywords:
            keyword_matches = sum(1 for keyword in expected_keywords 
                                if keyword.lower() in response_text)
            return keyword_matches / len(expected_keywords)
        
        return 0.0
    
    def _determine_best_model_overall(self, model_results: Dict[str, Any]) -> str:
        """Determine best overall model based on combined metrics"""
        
        model_scores = {}
        
        for model_id, results in model_results.items():
            metrics = results['metrics']
            
            # Weighted score calculation
            score = (
                metrics['success_rate'] * 0.3 +
                metrics['avg_confidence'] * 0.3 +
                metrics['accuracy_score'] * 0.3 +
                (1.0 / (1.0 + metrics['avg_latency'])) * 0.1  # Lower latency is better
            )
            
            model_scores[model_id] = score
        
        return max(model_scores.keys(), key=lambda k: model_scores[k]) if model_scores else "none"
    
    def _determine_best_model_by_metric(self, model_results: Dict[str, Any]) -> Dict[str, str]:
        """Determine best model for each individual metric"""
        
        best_by_metric = {}
        metric_names = ['success_rate', 'avg_confidence', 'accuracy_score']
        
        for metric in metric_names:
            best_model = max(model_results.keys(), 
                           key=lambda k: model_results[k]['metrics'][metric],
                           default="none")
            best_by_metric[metric] = best_model
        
        # For latency, lower is better
        best_latency_model = min(model_results.keys(), 
                               key=lambda k: model_results[k]['metrics']['avg_latency'],
                               default="none")
        best_by_metric['best_latency'] = best_latency_model
        
        return best_by_metric
    
    def _generate_comparison_recommendations(self, model_results: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results"""
        
        recommendations = []
        
        for model_id, results in model_results.items():
            metrics = results['metrics']
            
            if metrics['success_rate'] < 0.8:
                recommendations.append(f"Model {model_id} has low success rate ({metrics['success_rate']:.2%})")
            
            if metrics['avg_latency'] > 2.0:
                recommendations.append(f"Model {model_id} has high latency ({metrics['avg_latency']:.2f}s)")
            
            if metrics['avg_confidence'] < 0.6:
                recommendations.append(f"Model {model_id} shows low confidence ({metrics['avg_confidence']:.2f})")
            
            if metrics['accuracy_score'] < 0.7:
                recommendations.append(f"Model {model_id} has low accuracy ({metrics['accuracy_score']:.2%})")
        
        return recommendations
    
    def get_comparison_history(self) -> List[ModelComparisonResult]:
        """Get history of model comparisons"""
        return self.comparison_results

class HybridCognitiveArchitecture:
    """Hybrid architecture combining cognitive processing with AI models"""
    
    def __init__(self, 
                 cognitive_system: Any,
                 ai_integrator: MultiModelAIIntegrator,
                 config: Dict[str, Any]):
        self.cognitive_system = cognitive_system
        self.ai_integrator = ai_integrator
        self.config = config
        self.fusion_strategies = {
            'cognitive_first': self._cognitive_first_fusion,
            'ai_first': self._ai_first_fusion,
            'parallel_fusion': self._parallel_fusion,
            'adaptive_fusion': self._adaptive_fusion
        }
        self.response_history = deque(maxlen=100)
    
    async def process_hybrid(self, 
                           input_data: Dict[str, Any],
                           fusion_strategy: str = 'adaptive_fusion') -> HybridResponse:
        """Process input using hybrid cognitive-AI architecture"""
        
        start_time = time.time()
        
        if fusion_strategy not in self.fusion_strategies:
            fusion_strategy = 'adaptive_fusion'
        
        try:
            fusion_function = self.fusion_strategies[fusion_strategy]
            result = await fusion_function(input_data)
            
            processing_time = time.time() - start_time
            
            hybrid_response = HybridResponse(
                response_id=str(uuid.uuid4()),
                primary_response=result['response'],
                confidence_score=result['confidence'],
                model_contributions=result.get('ai_results', {}),
                cognitive_processing=result.get('cognitive_results', {}),
                ai_processing=result.get('ai_processing', {}),
                fusion_strategy=fusion_strategy,
                processing_time=processing_time,
                metadata=result.get('metadata', {})
            )
            
            self.response_history.append(hybrid_response)
            return hybrid_response
            
        except Exception as e:
            logger.error(f"Hybrid processing error: {e}")
            return HybridResponse(
                response_id=str(uuid.uuid4()),
                primary_response=f"Error in hybrid processing: {e}",
                confidence_score=0.0,
                model_contributions={},
                cognitive_processing={},
                ai_processing={},
                fusion_strategy=fusion_strategy,
                processing_time=time.time() - start_time,
                metadata={'error': str(e)}
            )
    
    async def _cognitive_first_fusion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Cognitive processing first, then AI enhancement"""
        
        # Process with cognitive system first
        if hasattr(self.cognitive_system, 'process'):
            cognitive_result = self.cognitive_system.process(input_data)
        else:
            cognitive_result = {'response': 'Cognitive processing not available', 'confidence': 0.5}
        
        # Use cognitive result as context for AI processing
        ai_input = input_data.copy()
        ai_input['cognitive_context'] = cognitive_result
        
        ai_result = await self.ai_integrator.process_with_integration(ai_input, 'enhancement')
        
        # Combine results (cognitive as primary, AI as enhancement)
        final_response = cognitive_result.get('response', '')
        if 'response' in ai_result and ai_result['response']:
            final_response += f"\n\nAI Enhancement: {ai_result['response']}"
        
        combined_confidence = (cognitive_result.get('confidence', 0.5) * 0.7 + 
                             ai_result.get('confidence', 0.5) * 0.3)
        
        return {
            'response': final_response,
            'confidence': combined_confidence,
            'cognitive_results': cognitive_result,
            'ai_results': ai_result,
            'fusion_method': 'cognitive_first'
        }
    
    async def _ai_first_fusion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """AI processing first, then cognitive refinement"""
        
        # Process with AI first
        ai_result = await self.ai_integrator.process_with_integration(input_data, 'primary')
        
        # Use AI result as input for cognitive processing
        if hasattr(self.cognitive_system, 'process'):
            cognitive_input = {
                'query': ai_result.get('response', ''),
                'ai_context': ai_result,
                'original_query': input_data.get('query', '')
            }
            cognitive_result = self.cognitive_system.process(cognitive_input)
        else:
            cognitive_result = {'response': 'Cognitive refinement not available', 'confidence': 0.5}
        
        # Combine results (AI as primary, cognitive as refinement)
        final_response = ai_result.get('response', '')
        if 'response' in cognitive_result and cognitive_result['response']:
            final_response += f"\n\nCognitive Refinement: {cognitive_result['response']}"
        
        combined_confidence = (ai_result.get('confidence', 0.5) * 0.7 + 
                             cognitive_result.get('confidence', 0.5) * 0.3)
        
        return {
            'response': final_response,
            'confidence': combined_confidence,
            'cognitive_results': cognitive_result,
            'ai_results': ai_result,
            'fusion_method': 'ai_first'
        }
    
    async def _parallel_fusion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Parallel processing with result fusion"""
        
        # Start both processes in parallel
        ai_task = asyncio.create_task(
            self.ai_integrator.process_with_integration(input_data, 'parallel')
        )
        
        # Process cognitive system (assuming it's synchronous for now)
        if hasattr(self.cognitive_system, 'process'):
            cognitive_result = self.cognitive_system.process(input_data)
        else:
            cognitive_result = {'response': 'Cognitive processing not available', 'confidence': 0.5}
        
        # Wait for AI processing
        ai_result = await ai_task
        
        # Fuse results
        cognitive_response = cognitive_result.get('response', '')
        ai_response = ai_result.get('response', '')
        
        # Simple fusion: combine responses
        if cognitive_response and ai_response:
            final_response = f"Cognitive Analysis: {cognitive_response}\n\nAI Analysis: {ai_response}"
        elif cognitive_response:
            final_response = cognitive_response
        elif ai_response:
            final_response = ai_response
        else:
            final_response = "No response available from either system"
        
        # Weight confidences equally for parallel fusion
        cognitive_confidence = cognitive_result.get('confidence', 0.5)
        ai_confidence = ai_result.get('confidence', 0.5)
        combined_confidence = (cognitive_confidence + ai_confidence) / 2
        
        return {
            'response': final_response,
            'confidence': combined_confidence,
            'cognitive_results': cognitive_result,
            'ai_results': ai_result,
            'fusion_method': 'parallel'
        }
    
    async def _adaptive_fusion(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adaptive fusion based on input characteristics and system performance"""
        
        # Analyze input to determine best fusion strategy
        input_complexity = len(str(input_data.get('query', '')))
        requires_reasoning = 'why' in str(input_data.get('query', '')).lower() or 'how' in str(input_data.get('query', '')).lower()
        requires_knowledge = 'what' in str(input_data.get('query', '')).lower() or 'who' in str(input_data.get('query', '')).lower()
        
        # Decision logic for adaptive fusion
        if requires_reasoning and input_complexity > 100:
            # Complex reasoning - use cognitive first
            return await self._cognitive_first_fusion(input_data)
        elif requires_knowledge and input_complexity < 50:
            # Simple knowledge query - use AI first
            return await self._ai_first_fusion(input_data)
        else:
            # Default to parallel fusion
            return await self._parallel_fusion(input_data)
    
    def get_hybrid_statistics(self) -> Dict[str, Any]:
        """Get statistics about hybrid processing performance"""
        
        if not self.response_history:
            return {'message': 'No hybrid responses recorded'}
        
        responses = list(self.response_history)
        
        # Basic statistics
        total_responses = len(responses)
        avg_processing_time = np.mean([r.processing_time for r in responses])
        avg_confidence = np.mean([r.confidence_score for r in responses])
        
        # Group by fusion strategy
        strategy_stats = defaultdict(list)
        for response in responses:
            strategy_stats[response.fusion_strategy].append(response)
        
        strategy_performance = {}
        for strategy, strategy_responses in strategy_stats.items():
            strategy_performance[strategy] = {
                'count': len(strategy_responses),
                'avg_confidence': np.mean([r.confidence_score for r in strategy_responses]),
                'avg_processing_time': np.mean([r.processing_time for r in strategy_responses])
            }
        
        return {
            'total_hybrid_responses': total_responses,
            'avg_processing_time': avg_processing_time,
            'avg_confidence': avg_confidence,
            'strategy_performance': strategy_performance,
            'cognitive_system_available': hasattr(self.cognitive_system, 'process'),
            'ai_models_available': len(self.ai_integrator.models)
        }

class AIOptimizer:
    """Optimizes AI model performance and selection"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.optimization_history = []
        self.performance_baselines = {}
        self.optimization_methods = {
            OptimizationMethod.FINE_TUNING: self._simulate_fine_tuning,
            OptimizationMethod.PROMPT_ENGINEERING: self._optimize_prompts,
            OptimizationMethod.PARAMETER_ADJUSTMENT: self._adjust_parameters,
            OptimizationMethod.ENSEMBLE_WEIGHTING: self._optimize_ensemble_weights,
            OptimizationMethod.DYNAMIC_SELECTION: self._optimize_dynamic_selection
        }
    
    async def optimize_model(self, 
                           model: AIModel,
                           optimization_method: OptimizationMethod,
                           target_metric: str = 'confidence',
                           optimization_data: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Optimize a specific AI model"""
        
        start_time = time.time()
        
        # Get baseline performance
        baseline_performance = await self._measure_baseline_performance(model, optimization_data)
        
        # Apply optimization method
        optimization_function = self.optimization_methods.get(optimization_method)
        if not optimization_function:
            return {'error': f'Unknown optimization method: {optimization_method}'}
        
        try:
            optimization_result = await optimization_function(model, target_metric, optimization_data)
            
            # Measure optimized performance
            optimized_performance = await self._measure_baseline_performance(model, optimization_data)
            
            # Calculate improvement
            improvement = self._calculate_improvement(baseline_performance, optimized_performance, target_metric)
            
            optimization_time = time.time() - start_time
            
            result = {
                'optimization_id': str(uuid.uuid4()),
                'model_id': model.config.model_id,
                'method': optimization_method.value,
                'target_metric': target_metric,
                'baseline_performance': baseline_performance,
                'optimized_performance': optimized_performance,
                'improvement': improvement,
                'optimization_time': optimization_time,
                'optimization_details': optimization_result,
                'timestamp': datetime.now().isoformat()
            }
            
            self.optimization_history.append(result)
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            return {
                'error': str(e),
                'model_id': model.config.model_id,
                'method': optimization_method.value
            }
    
    async def _measure_baseline_performance(self, 
                                          model: AIModel,
                                          test_data: List[Dict[str, Any]] = None) -> Dict[str, float]:
        """Measure baseline performance of a model"""
        
        if not test_data:
            test_data = [
                {'query': 'What is AI?'},
                {'query': 'Explain machine learning'},
                {'query': 'How does deep learning work?'}
            ]
        
        latencies = []
        confidences = []
        successes = 0
        
        for test_case in test_data:
            try:
                start_time = time.time()
                result = await model.process(test_case)
                latency = time.time() - start_time
                
                latencies.append(latency)
                
                if 'error' not in result:
                    successes += 1
                    confidences.append(result.get('confidence', 0.5))
                
            except Exception as e:
                logger.warning(f"Test case failed during baseline measurement: {e}")
        
        return {
            'avg_latency': np.mean(latencies) if latencies else 0.0,
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'success_rate': successes / len(test_data) if test_data else 0.0,
            'total_requests': len(test_data)
        }
    
    def _calculate_improvement(self, 
                             baseline: Dict[str, float],
                             optimized: Dict[str, float],
                             target_metric: str) -> Dict[str, float]:
        """Calculate improvement metrics"""
        
        improvement = {}
        
        for metric in ['avg_latency', 'avg_confidence', 'success_rate']:
            baseline_value = baseline.get(metric, 0.0)
            optimized_value = optimized.get(metric, 0.0)
            
            if metric == 'avg_latency':
                # For latency, lower is better
                if baseline_value > 0:
                    improvement[f'{metric}_improvement'] = (baseline_value - optimized_value) / baseline_value
                else:
                    improvement[f'{metric}_improvement'] = 0.0
            else:
                # For other metrics, higher is better
                if baseline_value > 0:
                    improvement[f'{metric}_improvement'] = (optimized_value - baseline_value) / baseline_value
                else:
                    improvement[f'{metric}_improvement'] = 0.0
        
        return improvement
    
    async def _simulate_fine_tuning(self, 
                                  model: AIModel,
                                  target_metric: str,
                                  training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Simulate fine-tuning optimization (placeholder implementation)"""
        
        # In a real implementation, this would perform actual fine-tuning
        # For now, we simulate by adjusting model parameters
        
        original_config = model.config.parameters.copy()
        
        # Simulate parameter adjustments
        if target_metric == 'confidence':
            # Adjust parameters to improve confidence
            model.config.parameters['confidence_boost'] = 0.1
        elif target_metric == 'latency':
            # Adjust parameters to improve speed
            model.config.parameters['speed_mode'] = True
        elif target_metric == 'accuracy':
            # Adjust parameters to improve accuracy
            model.config.parameters['accuracy_mode'] = True
        
        return {
            'method': 'simulated_fine_tuning',
            'original_parameters': original_config,
            'adjusted_parameters': model.config.parameters,
            'training_data_size': len(training_data) if training_data else 0,
            'simulation_note': 'This is a simulated optimization for demonstration'
        }
    
    async def _optimize_prompts(self, 
                              model: AIModel,
                              target_metric: str,
                              test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize prompts for better performance"""
        
        prompt_variations = [
            "Please provide a detailed and accurate response to: {query}",
            "Based on your knowledge, carefully answer: {query}",
            "Think step by step and respond to: {query}",
            "Provide a comprehensive answer to: {query}"
        ]
        
        best_prompt = None
        best_score = 0.0
        prompt_results = {}
        
        if not test_data:
            test_data = [{'query': 'What is machine learning?'}]
        
        for prompt_template in prompt_variations:
            scores = []
            
            for test_case in test_data[:3]:  # Limit for simulation
                try:
                    # Modify the query with the prompt template
                    modified_query = prompt_template.format(query=test_case.get('query', ''))
                    modified_test = test_case.copy()
                    modified_test['query'] = modified_query
                    
                    result = await model.process(modified_test)
                    
                    if 'error' not in result:
                        if target_metric == 'confidence':
                            scores.append(result.get('confidence', 0.0))
                        elif target_metric == 'latency':
                            scores.append(1.0 / (1.0 + result.get('processing_time', 1.0)))
                        else:
                            scores.append(result.get('confidence', 0.0))  # Default
                    
                except Exception as e:
                    logger.warning(f"Prompt optimization test failed: {e}")
            
            avg_score = np.mean(scores) if scores else 0.0
            prompt_results[prompt_template] = avg_score
            
            if avg_score > best_score:
                best_score = avg_score
                best_prompt = prompt_template
        
        return {
            'method': 'prompt_optimization',
            'best_prompt': best_prompt,
            'best_score': best_score,
            'all_prompt_scores': prompt_results,
            'target_metric': target_metric
        }
    
    async def _adjust_parameters(self, 
                               model: AIModel,
                               target_metric: str,
                               test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Adjust model parameters for optimization"""
        
        original_params = model.config.parameters.copy()
        parameter_adjustments = {}
        
        # Simulate parameter adjustments based on target metric
        if target_metric == 'confidence':
            parameter_adjustments = {
                'temperature': 0.3,  # Lower temperature for more confident responses
                'top_p': 0.8,
                'confidence_threshold': 0.7
            }
        elif target_metric == 'latency':
            parameter_adjustments = {
                'max_tokens': 100,  # Shorter responses for faster processing
                'timeout': 5.0,
                'batch_size': 1
            }
        elif target_metric == 'accuracy':
            parameter_adjustments = {
                'temperature': 0.1,  # Very low temperature for accuracy
                'top_p': 0.9,
                'repetition_penalty': 1.1
            }
        
        # Apply adjustments
        model.config.parameters.update(parameter_adjustments)
        
        return {
            'method': 'parameter_adjustment',
            'original_parameters': original_params,
            'parameter_adjustments': parameter_adjustments,
            'target_metric': target_metric
        }
    
    async def _optimize_ensemble_weights(self, 
                                       model: AIModel,
                                       target_metric: str,
                                       test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize ensemble weights (applicable for ensemble models)"""
        
        # This is a placeholder for ensemble weight optimization
        # In practice, this would adjust weights of ensemble members
        
        weights = {
            'member_1': 0.4,
            'member_2': 0.3,
            'member_3': 0.3
        }
        
        optimized_weights = {
            'member_1': 0.5,
            'member_2': 0.3,
            'member_3': 0.2
        }
        
        return {
            'method': 'ensemble_weight_optimization',
            'original_weights': weights,
            'optimized_weights': optimized_weights,
            'target_metric': target_metric,
            'note': 'This is a simulated ensemble optimization'
        }
    
    async def _optimize_dynamic_selection(self, 
                                        model: AIModel,
                                        target_metric: str,
                                        test_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize dynamic model selection criteria"""
        
        # Simulate optimization of selection criteria
        original_criteria = {
            'latency_threshold': 1.0,
            'confidence_threshold': 0.5,
            'complexity_threshold': 100
        }
        
        optimized_criteria = {
            'latency_threshold': 0.8,
            'confidence_threshold': 0.6,
            'complexity_threshold': 80
        }
        
        return {
            'method': 'dynamic_selection_optimization',
            'original_criteria': original_criteria,
            'optimized_criteria': optimized_criteria,
            'target_metric': target_metric
        }
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate optimization performance report"""
        
        if not self.optimization_history:
            return {'message': 'No optimizations performed'}
        
        # Group by optimization method
        method_stats = defaultdict(list)
        for optimization in self.optimization_history:
            method_stats[optimization['method']].append(optimization)
        
        method_performance = {}
        for method, optimizations in method_stats.items():
            improvements = []
            for opt in optimizations:
                target_metric = opt['target_metric']
                improvement_key = f'{target_metric}_improvement'
                if improvement_key in opt['improvement']:
                    improvements.append(opt['improvement'][improvement_key])
            
            method_performance[method] = {
                'optimization_count': len(optimizations),
                'avg_improvement': np.mean(improvements) if improvements else 0.0,
                'max_improvement': max(improvements) if improvements else 0.0,
                'success_rate': len([opt for opt in optimizations if 'error' not in opt]) / len(optimizations)
            }
        
        return {
            'total_optimizations': len(self.optimization_history),
            'method_performance': method_performance,
            'optimization_methods_available': list(self.optimization_methods.keys()),
            'latest_optimization': self.optimization_history[-1] if self.optimization_history else None
        }