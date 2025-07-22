"""
Meta-Cognitive Reflection System for Deep Tree Echo
Implements self-monitoring, strategy selection, and cognitive performance evaluation
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import numpy as np
from enum import Enum

logger = logging.getLogger(__name__)

class CognitiveStrategy(Enum):
    """Available cognitive processing strategies"""
    SEQUENTIAL = "sequential"  # Process membranes sequentially
    PARALLEL = "parallel"     # Process membranes in parallel
    MEMORY_FIRST = "memory_first"  # Prioritize memory processing
    REASONING_FIRST = "reasoning_first"  # Prioritize reasoning processing
    ADAPTIVE = "adaptive"     # Dynamically select best strategy

class ProcessingError(Enum):
    """Types of cognitive processing errors"""
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    INCONSISTENT_RESPONSES = "inconsistent_responses"
    MEMORY_RETRIEVAL_FAILURE = "memory_retrieval_failure"
    REASONING_FAILURE = "reasoning_failure"
    INTEGRATION_FAILURE = "integration_failure"

@dataclass
class CognitiveMetrics:
    """Metrics for cognitive processing performance"""
    session_id: str
    timestamp: str
    strategy_used: str
    total_processing_time: float
    memory_time: float
    reasoning_time: float
    grammar_time: float
    integration_time: float
    confidence_score: float
    error_count: int
    errors: List[str]
    memory_retrievals: int
    reasoning_complexity: str
    user_satisfaction: Optional[float] = None
    
@dataclass 
class StrategyPerformance:
    """Performance metrics for a cognitive strategy"""
    strategy: str
    usage_count: int
    avg_processing_time: float
    avg_confidence: float
    success_rate: float
    user_satisfaction_avg: float
    error_frequencies: Dict[str, int]
    last_used: str

@dataclass
class CognitiveState:
    """Current state of cognitive processing system"""
    current_strategy: str
    strategies_performance: Dict[str, StrategyPerformance]
    recent_metrics: deque  # Last 100 processing metrics
    error_patterns: Dict[str, int]
    adaptation_triggers: List[str]
    meta_confidence: float  # Confidence in meta-cognitive decisions

class MetaCognitiveMonitor:
    """Monitors and analyzes cognitive processing performance"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.processing_history: deque = deque(maxlen=max_history)
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        self.error_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.performance_trends: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.lock = threading.RLock()
        
        # Initialize strategy performances
        for strategy in CognitiveStrategy:
            self.strategy_performances[strategy.value] = StrategyPerformance(
                strategy=strategy.value,
                usage_count=0,
                avg_processing_time=0.0,
                avg_confidence=0.0,
                success_rate=1.0,
                user_satisfaction_avg=0.0,
                error_frequencies={},
                last_used=datetime.now().isoformat()
            )
    
    def record_processing(self, metrics: CognitiveMetrics) -> None:
        """Record cognitive processing metrics"""
        with self.lock:
            self.processing_history.append(metrics)
            
            # Update strategy performance
            strategy_perf = self.strategy_performances[metrics.strategy_used]
            strategy_perf.usage_count += 1
            
            # Update averages using exponential moving average
            alpha = 0.1  # Learning rate
            strategy_perf.avg_processing_time = (
                (1 - alpha) * strategy_perf.avg_processing_time + 
                alpha * metrics.total_processing_time
            )
            strategy_perf.avg_confidence = (
                (1 - alpha) * strategy_perf.avg_confidence + 
                alpha * metrics.confidence_score
            )
            
            # Update success rate
            success = 1.0 if metrics.error_count == 0 else 0.0
            strategy_perf.success_rate = (
                (1 - alpha) * strategy_perf.success_rate + alpha * success
            )
            
            # Update user satisfaction if provided
            if metrics.user_satisfaction is not None:
                if strategy_perf.user_satisfaction_avg == 0:
                    strategy_perf.user_satisfaction_avg = metrics.user_satisfaction
                else:
                    strategy_perf.user_satisfaction_avg = (
                        (1 - alpha) * strategy_perf.user_satisfaction_avg + 
                        alpha * metrics.user_satisfaction
                    )
            
            # Update error frequencies
            for error in metrics.errors:
                strategy_perf.error_frequencies[error] = (
                    strategy_perf.error_frequencies.get(error, 0) + 1
                )
                self.error_patterns[error].append(datetime.now())
            
            strategy_perf.last_used = metrics.timestamp
            
            # Update performance trends
            self.performance_trends[metrics.strategy_used].append({
                'timestamp': metrics.timestamp,
                'processing_time': metrics.total_processing_time,
                'confidence': metrics.confidence_score,
                'success': success
            })
            
            logger.debug(f"Recorded processing metrics for strategy {metrics.strategy_used}")
    
    def detect_performance_degradation(self, strategy: str, window_size: int = 10) -> bool:
        """Detect if performance is degrading for a strategy"""
        with self.lock:
            recent_metrics = list(self.processing_history)[-window_size:]
            strategy_metrics = [m for m in recent_metrics if m.strategy_used == strategy]
            
            if len(strategy_metrics) < window_size // 2:
                return False  # Not enough data
            
            # Check for increasing processing times
            times = [m.total_processing_time for m in strategy_metrics]
            if len(times) >= 5:
                recent_avg = np.mean(times[-3:])
                earlier_avg = np.mean(times[:3])
                if recent_avg > earlier_avg * 1.5:  # 50% increase
                    return True
            
            # Check for decreasing confidence
            confidences = [m.confidence_score for m in strategy_metrics]
            if len(confidences) >= 5:
                recent_conf = np.mean(confidences[-3:])
                earlier_conf = np.mean(confidences[:3])
                if recent_conf < earlier_conf * 0.8:  # 20% decrease
                    return True
            
            # Check for increasing errors
            error_counts = [m.error_count for m in strategy_metrics]
            if len(error_counts) >= 3:
                if all(count > 0 for count in error_counts[-3:]):
                    return True
            
            return False
    
    def identify_error_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Identify patterns in cognitive processing errors"""
        patterns = {}
        
        with self.lock:
            for error_type, timestamps in self.error_patterns.items():
                if len(timestamps) < 2:
                    continue
                
                # Remove old timestamps (older than 24 hours)
                recent_timestamps = [
                    ts for ts in timestamps 
                    if datetime.now() - ts < timedelta(hours=24)
                ]
                
                if len(recent_timestamps) < 2:
                    continue
                
                # Calculate frequency and timing patterns
                time_diffs = []
                for i in range(1, len(recent_timestamps)):
                    diff = (recent_timestamps[i] - recent_timestamps[i-1]).total_seconds()
                    time_diffs.append(diff)
                
                patterns[error_type] = {
                    'frequency_24h': len(recent_timestamps),
                    'avg_interval_seconds': np.mean(time_diffs) if time_diffs else 0,
                    'is_periodic': self._is_periodic_pattern(time_diffs),
                    'severity': self._calculate_error_severity(error_type, recent_timestamps)
                }
        
        return patterns
    
    def _is_periodic_pattern(self, intervals: List[float]) -> bool:
        """Check if error intervals show periodic pattern"""
        if len(intervals) < 3:
            return False
        
        # Simple periodicity check - coefficient of variation
        if np.std(intervals) / np.mean(intervals) < 0.5:
            return True
        return False
    
    def _calculate_error_severity(self, error_type: str, timestamps: List[datetime]) -> str:
        """Calculate severity of error pattern"""
        if len(timestamps) >= 10:
            return "high"
        elif len(timestamps) >= 5:
            return "medium"
        else:
            return "low"
    
    def get_performance_summary(self, time_window_hours: int = 24) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        with self.lock:
            cutoff_time = datetime.now() - timedelta(hours=time_window_hours)
            
            recent_metrics = [
                m for m in self.processing_history
                if datetime.fromisoformat(m.timestamp) >= cutoff_time
            ]
            
            if not recent_metrics:
                return {"error": "No data available for the specified time window"}
            
            # Overall statistics
            total_processing_time = sum(m.total_processing_time for m in recent_metrics)
            avg_confidence = np.mean([m.confidence_score for m in recent_metrics])
            total_errors = sum(m.error_count for m in recent_metrics)
            
            # Strategy breakdown
            strategy_stats = defaultdict(list)
            for m in recent_metrics:
                strategy_stats[m.strategy_used].append(m)
            
            strategy_summary = {}
            for strategy, metrics in strategy_stats.items():
                strategy_summary[strategy] = {
                    'usage_count': len(metrics),
                    'avg_processing_time': np.mean([m.total_processing_time for m in metrics]),
                    'avg_confidence': np.mean([m.confidence_score for m in metrics]),
                    'error_rate': sum(m.error_count for m in metrics) / len(metrics),
                    'success_rate': len([m for m in metrics if m.error_count == 0]) / len(metrics)
                }
            
            return {
                'time_window_hours': time_window_hours,
                'total_requests': len(recent_metrics),
                'total_processing_time': total_processing_time,
                'avg_processing_time': total_processing_time / len(recent_metrics),
                'avg_confidence': avg_confidence,
                'total_errors': total_errors,
                'error_rate': total_errors / len(recent_metrics),
                'strategy_performance': strategy_summary,
                'error_patterns': self.identify_error_patterns()
            }

class CognitiveStrategySelector:
    """Selects optimal cognitive processing strategies based on context and performance"""
    
    def __init__(self, monitor: MetaCognitiveMonitor):
        self.monitor = monitor
        self.strategy_weights = {
            CognitiveStrategy.SEQUENTIAL.value: 1.0,
            CognitiveStrategy.PARALLEL.value: 1.0,
            CognitiveStrategy.MEMORY_FIRST.value: 1.0,
            CognitiveStrategy.REASONING_FIRST.value: 1.0,
            CognitiveStrategy.ADAPTIVE.value: 1.0
        }
        self.context_preferences = defaultdict(dict)
    
    def select_strategy(self, context: Dict[str, Any]) -> str:
        """Select optimal processing strategy based on context and performance history"""
        
        # Extract context features
        input_complexity = self._assess_input_complexity(context.get('user_input', ''))
        session_history = context.get('conversation_history', [])
        memory_load = len(context.get('memory_state', {}))
        processing_goals = context.get('processing_goals', [])
        
        # Get recent performance data
        recent_performance = self.monitor.get_performance_summary(time_window_hours=6)
        
        if 'strategy_performance' not in recent_performance:
            # No recent data, use adaptive strategy
            return CognitiveStrategy.ADAPTIVE.value
        
        # Score strategies based on multiple criteria
        strategy_scores = {}
        
        for strategy in CognitiveStrategy:
            strategy_name = strategy.value
            base_score = 0.5  # Base score
            
            # Performance-based scoring
            if strategy_name in recent_performance['strategy_performance']:
                perf = recent_performance['strategy_performance'][strategy_name]
                
                # Reward high confidence and low error rates
                confidence_score = perf['avg_confidence'] * 0.3
                error_penalty = perf['error_rate'] * 0.4
                success_bonus = perf['success_rate'] * 0.3
                
                base_score += confidence_score + success_bonus - error_penalty
            
            # Context-based adjustments
            if input_complexity == 'high':
                if strategy_name == CognitiveStrategy.PARALLEL.value:
                    base_score += 0.2
                elif strategy_name == CognitiveStrategy.REASONING_FIRST.value:
                    base_score += 0.15
            
            if len(session_history) > 10:  # Long conversation
                if strategy_name == CognitiveStrategy.MEMORY_FIRST.value:
                    base_score += 0.1
            
            if memory_load > 50:  # High memory load
                if strategy_name == CognitiveStrategy.SEQUENTIAL.value:
                    base_score += 0.1  # Avoid overwhelming parallel processing
            
            # Check for performance degradation
            if self.monitor.detect_performance_degradation(strategy_name):
                base_score -= 0.3  # Penalize degrading strategies
            
            strategy_scores[strategy_name] = max(0.1, base_score)  # Minimum score
        
        # Select strategy with highest score
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        
        logger.debug(f"Selected strategy: {best_strategy}, scores: {strategy_scores}")
        return best_strategy
    
    def _assess_input_complexity(self, input_text: str) -> str:
        """Assess complexity of user input"""
        if not input_text:
            return 'low'
        
        words = input_text.split()
        word_count = len(words)
        
        # Count complex features
        questions = input_text.count('?')
        logical_words = len([w for w in words if w.lower() in 
                           ['because', 'therefore', 'however', 'although', 'if', 'then']])
        
        # Simple complexity scoring
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
    
    def adapt_strategy_weights(self, feedback: Dict[str, Any]) -> None:
        """Adapt strategy selection based on user feedback and performance"""
        strategy = feedback.get('strategy_used')
        satisfaction = feedback.get('user_satisfaction', 0.5)
        
        if not strategy or strategy not in self.strategy_weights:
            return
        
        # Adjust weights based on satisfaction
        adjustment = (satisfaction - 0.5) * 0.1  # Scale adjustment
        self.strategy_weights[strategy] = max(0.1, 
                                            min(2.0, self.strategy_weights[strategy] + adjustment))
        
        logger.debug(f"Adjusted weight for {strategy}: {self.strategy_weights[strategy]}")

class CognitiveErrorDetector:
    """Detects and analyzes cognitive processing errors"""
    
    def __init__(self):
        self.error_history = deque(maxlen=500)
        self.correction_strategies = {
            ProcessingError.TIMEOUT: self._handle_timeout_error,
            ProcessingError.LOW_CONFIDENCE: self._handle_low_confidence_error,
            ProcessingError.INCONSISTENT_RESPONSES: self._handle_inconsistent_responses,
            ProcessingError.MEMORY_RETRIEVAL_FAILURE: self._handle_memory_error,
            ProcessingError.REASONING_FAILURE: self._handle_reasoning_error,
            ProcessingError.INTEGRATION_FAILURE: self._handle_integration_error
        }
    
    def detect_errors(self, processing_context: Dict[str, Any]) -> List[ProcessingError]:
        """Detect errors in cognitive processing"""
        errors = []
        
        # Check for timeout
        if processing_context.get('processing_time', 0) > processing_context.get('timeout_threshold', 10):
            errors.append(ProcessingError.TIMEOUT)
        
        # Check for low confidence
        if processing_context.get('confidence_score', 1.0) < 0.3:
            errors.append(ProcessingError.LOW_CONFIDENCE)
        
        # Check for inconsistent membrane responses
        membrane_responses = processing_context.get('membrane_responses', {})
        if self._detect_response_inconsistency(membrane_responses):
            errors.append(ProcessingError.INCONSISTENT_RESPONSES)
        
        # Check for memory retrieval issues
        if processing_context.get('memory_retrievals', 0) == 0 and processing_context.get('query_needs_memory', False):
            errors.append(ProcessingError.MEMORY_RETRIEVAL_FAILURE)
        
        # Check for reasoning failures
        if processing_context.get('reasoning_complexity', 'low') == 'high' and \
           processing_context.get('reasoning_confidence', 1.0) < 0.5:
            errors.append(ProcessingError.REASONING_FAILURE)
        
        return errors
    
    def _detect_response_inconsistency(self, membrane_responses: Dict[str, Any]) -> bool:
        """Detect inconsistencies between membrane responses"""
        if not membrane_responses:
            return False
        
        # Simple inconsistency detection based on confidence variance
        confidences = []
        for response in membrane_responses.values():
            if hasattr(response, 'confidence'):
                confidences.append(response.confidence)
        
        if len(confidences) >= 2:
            confidence_variance = np.var(confidences)
            return confidence_variance > 0.3  # High variance indicates inconsistency
        
        return False
    
    def suggest_corrections(self, errors: List[ProcessingError], 
                          context: Dict[str, Any]) -> Dict[ProcessingError, str]:
        """Suggest corrections for detected errors"""
        suggestions = {}
        
        for error in errors:
            if error in self.correction_strategies:
                suggestion = self.correction_strategies[error](context)
                suggestions[error] = suggestion
        
        return suggestions
    
    def _handle_timeout_error(self, context: Dict[str, Any]) -> str:
        """Handle processing timeout errors"""
        return "Consider using parallel processing or reducing context complexity"
    
    def _handle_low_confidence_error(self, context: Dict[str, Any]) -> str:
        """Handle low confidence errors"""
        return "Retrieve more relevant memories or use reasoning-first strategy"
    
    def _handle_inconsistent_responses(self, context: Dict[str, Any]) -> str:
        """Handle inconsistent membrane responses"""
        return "Apply additional integration validation or use sequential processing"
    
    def _handle_memory_error(self, context: Dict[str, Any]) -> str:
        """Handle memory retrieval failures"""
        return "Check memory encoding quality or expand search parameters"
    
    def _handle_reasoning_error(self, context: Dict[str, Any]) -> str:
        """Handle reasoning failures"""
        return "Break down complex reasoning into simpler steps"
    
    def _handle_integration_error(self, context: Dict[str, Any]) -> str:
        """Handle integration failures"""
        return "Review membrane response compatibility and integration logic"

class MetaCognitiveReflectionSystem:
    """Main system coordinating meta-cognitive reflection capabilities"""
    
    def __init__(self):
        self.monitor = MetaCognitiveMonitor()
        self.strategy_selector = CognitiveStrategySelector(self.monitor)
        self.error_detector = CognitiveErrorDetector()
        self.cognitive_state = CognitiveState(
            current_strategy=CognitiveStrategy.ADAPTIVE.value,
            strategies_performance={},
            recent_metrics=deque(maxlen=100),
            error_patterns={},
            adaptation_triggers=[],
            meta_confidence=0.7
        )
        
        logger.info("Meta-cognitive reflection system initialized")
    
    def before_processing(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Called before cognitive processing to select strategy and prepare monitoring"""
        
        # Select optimal strategy
        selected_strategy = self.strategy_selector.select_strategy(context)
        self.cognitive_state.current_strategy = selected_strategy
        
        # Prepare monitoring context
        monitoring_context = {
            'strategy_selected': selected_strategy,
            'selection_timestamp': datetime.now().isoformat(),
            'context_complexity': self.strategy_selector._assess_input_complexity(
                context.get('user_input', '')
            ),
            'expected_processing_time': self._estimate_processing_time(selected_strategy, context)
        }
        
        logger.debug(f"Meta-cognitive pre-processing: strategy={selected_strategy}")
        return monitoring_context
    
    def after_processing(self, processing_results: Dict[str, Any], 
                        monitoring_context: Dict[str, Any]) -> Dict[str, Any]:
        """Called after cognitive processing to analyze performance and detect errors"""
        
        # Create metrics record
        metrics = CognitiveMetrics(
            session_id=processing_results.get('session_id', 'unknown'),
            timestamp=datetime.now().isoformat(),
            strategy_used=monitoring_context['strategy_selected'],
            total_processing_time=processing_results.get('total_processing_time', 0),
            memory_time=processing_results.get('memory_processing_time', 0),
            reasoning_time=processing_results.get('reasoning_processing_time', 0),
            grammar_time=processing_results.get('grammar_processing_time', 0),
            integration_time=processing_results.get('integration_time', 0),
            confidence_score=processing_results.get('confidence_score', 0),
            error_count=0,  # Will be updated after error detection
            errors=[],
            memory_retrievals=processing_results.get('memory_retrievals', 0),
            reasoning_complexity=processing_results.get('reasoning_complexity', 'low')
        )
        
        # Detect errors
        processing_context = {
            **processing_results,
            **monitoring_context,
            'query_needs_memory': self._requires_memory_retrieval(
                processing_results.get('user_input', '')
            )
        }
        
        detected_errors = self.error_detector.detect_errors(processing_context)
        metrics.error_count = len(detected_errors)
        metrics.errors = [error.value for error in detected_errors]
        
        # Record metrics
        self.monitor.record_processing(metrics)
        self.cognitive_state.recent_metrics.append(metrics)
        
        # Get error corrections if needed
        corrections = {}
        if detected_errors:
            corrections = self.error_detector.suggest_corrections(detected_errors, processing_context)
            logger.warning(f"Detected {len(detected_errors)} cognitive errors with corrections")
        
        # Update meta-cognitive confidence
        self._update_meta_confidence(metrics, detected_errors)
        
        # Prepare reflection results
        reflection_results = {
            'metrics': asdict(metrics),
            'detected_errors': [error.value for error in detected_errors],
            'error_corrections': {error.value: correction for error, correction in corrections.items()},
            'meta_confidence': self.cognitive_state.meta_confidence,
            'performance_degradation_detected': self.monitor.detect_performance_degradation(
                monitoring_context['strategy_selected']
            ),
            'adaptation_recommended': self._should_adapt_strategy(metrics, detected_errors)
        }
        
        logger.debug(f"Meta-cognitive post-processing completed: {len(detected_errors)} errors detected")
        return reflection_results
    
    def get_cognitive_insights(self) -> Dict[str, Any]:
        """Get comprehensive cognitive insights for user interface"""
        performance_summary = self.monitor.get_performance_summary(24)
        
        return {
            'current_state': {
                'active_strategy': self.cognitive_state.current_strategy,
                'meta_confidence': self.cognitive_state.meta_confidence,
                'recent_performance': performance_summary
            },
            'strategy_recommendations': self._get_strategy_recommendations(),
            'error_analysis': {
                'patterns': self.monitor.identify_error_patterns(),
                'recent_errors': [asdict(m) for m in list(self.cognitive_state.recent_metrics)[-10:]]
            },
            'adaptation_status': {
                'triggers': self.cognitive_state.adaptation_triggers,
                'last_adaptation': self._get_last_adaptation_time()
            }
        }
    
    def adapt_from_feedback(self, feedback: Dict[str, Any]) -> None:
        """Adapt system based on user feedback"""
        self.strategy_selector.adapt_strategy_weights(feedback)
        
        # Update metrics with user satisfaction
        if self.cognitive_state.recent_metrics:
            latest_metrics = self.cognitive_state.recent_metrics[-1]
            latest_metrics.user_satisfaction = feedback.get('user_satisfaction', 0.5)
            self.monitor.record_processing(latest_metrics)  # Re-record with satisfaction
        
        logger.info(f"System adapted based on user feedback: satisfaction={feedback.get('user_satisfaction')}")
    
    def _estimate_processing_time(self, strategy: str, context: Dict[str, Any]) -> float:
        """Estimate expected processing time for strategy"""
        base_times = {
            CognitiveStrategy.SEQUENTIAL.value: 0.5,
            CognitiveStrategy.PARALLEL.value: 0.3,
            CognitiveStrategy.MEMORY_FIRST.value: 0.4,
            CognitiveStrategy.REASONING_FIRST.value: 0.6,
            CognitiveStrategy.ADAPTIVE.value: 0.4
        }
        
        base_time = base_times.get(strategy, 0.4)
        
        # Adjust based on context complexity
        complexity = self.strategy_selector._assess_input_complexity(
            context.get('user_input', '')
        )
        
        complexity_multipliers = {'low': 0.8, 'medium': 1.0, 'high': 1.5}
        return base_time * complexity_multipliers.get(complexity, 1.0)
    
    def _requires_memory_retrieval(self, user_input: str) -> bool:
        """Determine if input requires memory retrieval"""
        memory_keywords = ['remember', 'recall', 'what did', 'before', 'earlier', 'previous']
        return any(keyword in user_input.lower() for keyword in memory_keywords)
    
    def _update_meta_confidence(self, metrics: CognitiveMetrics, errors: List[ProcessingError]) -> None:
        """Update confidence in meta-cognitive decisions"""
        if not errors and metrics.confidence_score > 0.7:
            self.cognitive_state.meta_confidence = min(1.0, self.cognitive_state.meta_confidence + 0.05)
        elif errors:
            self.cognitive_state.meta_confidence = max(0.1, self.cognitive_state.meta_confidence - 0.1)
    
    def _should_adapt_strategy(self, metrics: CognitiveMetrics, errors: List[ProcessingError]) -> bool:
        """Determine if strategy adaptation is recommended"""
        return len(errors) > 2 or metrics.confidence_score < 0.3
    
    def _get_strategy_recommendations(self) -> List[str]:
        """Get strategy recommendations based on current performance"""
        recommendations = []
        
        performance = self.monitor.get_performance_summary(6)
        if 'strategy_performance' in performance:
            for strategy, perf in performance['strategy_performance'].items():
                if perf['error_rate'] > 0.3:
                    recommendations.append(f"Consider reducing use of {strategy} strategy due to high error rate")
                elif perf['success_rate'] > 0.9 and perf['avg_confidence'] > 0.8:
                    recommendations.append(f"Strategy {strategy} is performing well, consider increasing usage")
        
        return recommendations
    
    def _get_last_adaptation_time(self) -> str:
        """Get timestamp of last system adaptation"""
        # This would be tracked in a real implementation
        return datetime.now().isoformat()