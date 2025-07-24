"""
Innovation Testing Environment
A/B testing, feature flagging, and experimental framework for cognitive features
"""

import time
import json
import logging
import random
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
        def min(values):
            return min(values) if values else 0
        
        @staticmethod
        def max(values):
            return max(values) if values else 0
        
        @staticmethod
        def sqrt(value):
            return value ** 0.5
    
    np = NumpyFallback()
from collections import defaultdict, deque
import uuid
import hashlib

logger = logging.getLogger(__name__)

class ExperimentType(Enum):
    """Types of experiments that can be conducted"""
    AB_TEST = "ab_test"
    MULTIVARIATE = "multivariate"
    FEATURE_FLAG = "feature_flag"
    CANARY_RELEASE = "canary_release"
    GRADUAL_ROLLOUT = "gradual_rollout"

class ExperimentStatus(Enum):
    """Status of experiments"""
    DRAFT = "draft"
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class UserSegment(Enum):
    """User segments for targeted experiments"""
    ALL_USERS = "all_users"
    NEW_USERS = "new_users"
    POWER_USERS = "power_users"
    BETA_USERS = "beta_users"
    DEVELOPERS = "developers"
    RESEARCHERS = "researchers"

class MetricType(Enum):
    """Types of metrics to track"""
    CONVERSION_RATE = "conversion_rate"
    ENGAGEMENT_TIME = "engagement_time"
    ERROR_RATE = "error_rate"
    RESPONSE_TIME = "response_time"
    USER_SATISFACTION = "user_satisfaction"
    FEATURE_ADOPTION = "feature_adoption"
    COGNITIVE_PERFORMANCE = "cognitive_performance"

@dataclass
class ExperimentConfig:
    """Configuration for an experiment"""
    experiment_id: str
    name: str
    description: str
    experiment_type: ExperimentType
    status: ExperimentStatus
    target_segments: List[UserSegment]
    traffic_allocation: Dict[str, float]  # variant_name -> percentage
    primary_metric: MetricType
    secondary_metrics: List[MetricType]
    start_date: str
    end_date: Optional[str]
    minimum_sample_size: int
    statistical_significance: float
    created_by: str
    metadata: Dict[str, Any]

@dataclass
class ExperimentVariant:
    """A variant in an experiment"""
    variant_id: str
    name: str
    description: str
    feature_config: Dict[str, Any]
    traffic_percentage: float
    is_control: bool = False

@dataclass
class ExperimentResult:
    """Result from an experiment"""
    experiment_id: str
    variant_id: str
    user_id: str
    session_id: str
    timestamp: str
    metric_values: Dict[str, float]
    interaction_data: Dict[str, Any]
    conversion_event: bool
    error_occurred: bool

@dataclass
class StatisticalAnalysis:
    """Statistical analysis of experiment results"""
    experiment_id: str
    analysis_date: str
    sample_sizes: Dict[str, int]
    metric_summaries: Dict[str, Dict[str, float]]
    significance_tests: Dict[str, Dict[str, Any]]
    confidence_intervals: Dict[str, Dict[str, Tuple[float, float]]]
    recommendations: List[str]
    winning_variant: Optional[str]
    statistical_power: float

class FeatureFlag:
    """Feature flag for controlling experimental features"""
    
    def __init__(self, 
                 flag_id: str,
                 name: str,
                 description: str,
                 default_value: Any = False,
                 enabled_segments: List[UserSegment] = None,
                 rollout_percentage: float = 0.0):
        self.flag_id = flag_id
        self.name = name
        self.description = description
        self.default_value = default_value
        self.enabled_segments = enabled_segments or []
        self.rollout_percentage = rollout_percentage
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.evaluation_count = 0
        self.true_evaluations = 0
    
    def evaluate(self, user_context: Dict[str, Any]) -> Any:
        """Evaluate feature flag for a user"""
        self.evaluation_count += 1
        
        user_segment = self._determine_user_segment(user_context)
        user_id = user_context.get('user_id', 'anonymous')
        
        # Check segment targeting
        if self.enabled_segments and user_segment not in self.enabled_segments:
            return self.default_value
        
        # Check rollout percentage
        if self.rollout_percentage < 100.0:
            user_hash = self._hash_user_id(user_id)
            if user_hash >= self.rollout_percentage:
                return self.default_value
        
        # Flag is enabled for this user
        self.true_evaluations += 1
        return True if isinstance(self.default_value, bool) else self.default_value
    
    def _determine_user_segment(self, user_context: Dict[str, Any]) -> UserSegment:
        """Determine user segment based on context"""
        # Simple heuristics for demonstration
        user_type = user_context.get('user_type', 'regular')
        
        if user_type == 'developer':
            return UserSegment.DEVELOPERS
        elif user_type == 'researcher':
            return UserSegment.RESEARCHERS
        elif user_type == 'beta':
            return UserSegment.BETA_USERS
        elif user_context.get('is_new_user', False):
            return UserSegment.NEW_USERS
        elif user_context.get('session_count', 0) > 100:
            return UserSegment.POWER_USERS
        else:
            return UserSegment.ALL_USERS
    
    def _hash_user_id(self, user_id: str) -> float:
        """Hash user ID to consistent percentage (0-100)"""
        hash_object = hashlib.md5((user_id + self.flag_id).encode())
        hash_hex = hash_object.hexdigest()
        hash_int = int(hash_hex[:8], 16)
        return (hash_int % 10000) / 100.0  # 0-99.99
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get flag evaluation statistics"""
        return {
            'flag_id': self.flag_id,
            'total_evaluations': self.evaluation_count,
            'true_evaluations': self.true_evaluations,
            'activation_rate': self.true_evaluations / max(1, self.evaluation_count),
            'rollout_percentage': self.rollout_percentage,
            'enabled_segments': [seg.value for seg in self.enabled_segments]
        }

class ABTestingFramework:
    """A/B testing framework for cognitive features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.variants: Dict[str, List[ExperimentVariant]] = {}
        self.results: List[ExperimentResult] = []
        self.user_assignments: Dict[str, Dict[str, str]] = {}  # user_id -> experiment_id -> variant_id
        self.analysis_cache: Dict[str, StatisticalAnalysis] = {}
        self.results_lock = threading.Lock()
    
    def create_experiment(self, 
                         name: str,
                         description: str,
                         variants: List[ExperimentVariant],
                         experiment_type: ExperimentType = ExperimentType.AB_TEST,
                         target_segments: List[UserSegment] = None,
                         primary_metric: MetricType = MetricType.CONVERSION_RATE,
                         secondary_metrics: List[MetricType] = None,
                         duration_days: int = 14) -> str:
        """Create a new experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        # Validate traffic allocation
        total_traffic = sum(variant.traffic_percentage for variant in variants)
        if abs(total_traffic - 100.0) > 0.1:
            raise ValueError(f"Variant traffic percentages must sum to 100%, got {total_traffic}")
        
        # Ensure one control variant
        control_variants = [v for v in variants if v.is_control]
        if len(control_variants) != 1:
            raise ValueError("Exactly one variant must be marked as control")
        
        traffic_allocation = {variant.variant_id: variant.traffic_percentage for variant in variants}
        
        experiment_config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            experiment_type=experiment_type,
            status=ExperimentStatus.DRAFT,
            target_segments=target_segments or [UserSegment.ALL_USERS],
            traffic_allocation=traffic_allocation,
            primary_metric=primary_metric,
            secondary_metrics=secondary_metrics or [],
            start_date=datetime.now().isoformat(),
            end_date=(datetime.now() + timedelta(days=duration_days)).isoformat(),
            minimum_sample_size=self.config.get('minimum_sample_size', 100),
            statistical_significance=self.config.get('statistical_significance', 0.05),
            created_by='system',
            metadata={}
        )
        
        self.experiments[experiment_id] = experiment_config
        self.variants[experiment_id] = variants
        
        logger.info(f"Created experiment '{name}' with ID: {experiment_id}")
        return experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        if experiment.status != ExperimentStatus.DRAFT:
            logger.error(f"Experiment {experiment_id} is not in draft status")
            return False
        
        experiment.status = ExperimentStatus.ACTIVE
        experiment.start_date = datetime.now().isoformat()
        
        logger.info(f"Started experiment: {experiment.name}")
        return True
    
    def stop_experiment(self, experiment_id: str) -> bool:
        """Stop an experiment"""
        if experiment_id not in self.experiments:
            logger.error(f"Experiment {experiment_id} not found")
            return False
        
        experiment = self.experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_date = datetime.now().isoformat()
        
        logger.info(f"Stopped experiment: {experiment.name}")
        return True
    
    def assign_user_to_variant(self, 
                              experiment_id: str,
                              user_context: Dict[str, Any]) -> Optional[str]:
        """Assign a user to an experiment variant"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        
        # Check if experiment is active
        if experiment.status != ExperimentStatus.ACTIVE:
            return None
        
        # Check if experiment has ended
        if experiment.end_date:
            end_date = datetime.fromisoformat(experiment.end_date)
            if datetime.now() > end_date:
                return None
        
        user_id = user_context.get('user_id', 'anonymous')
        
        # Check if user already assigned
        if user_id in self.user_assignments and experiment_id in self.user_assignments[user_id]:
            return self.user_assignments[user_id][experiment_id]
        
        # Check user segment eligibility
        user_segment = self._determine_user_segment(user_context)
        if user_segment not in experiment.target_segments and UserSegment.ALL_USERS not in experiment.target_segments:
            return None
        
        # Assign variant based on hash
        variant_id = self._hash_user_to_variant(user_id, experiment_id, experiment.traffic_allocation)
        
        # Store assignment
        if user_id not in self.user_assignments:
            self.user_assignments[user_id] = {}
        self.user_assignments[user_id][experiment_id] = variant_id
        
        return variant_id
    
    def _determine_user_segment(self, user_context: Dict[str, Any]) -> UserSegment:
        """Determine user segment"""
        # Reuse logic from FeatureFlag
        user_type = user_context.get('user_type', 'regular')
        
        if user_type == 'developer':
            return UserSegment.DEVELOPERS
        elif user_type == 'researcher':
            return UserSegment.RESEARCHERS
        elif user_type == 'beta':
            return UserSegment.BETA_USERS
        elif user_context.get('is_new_user', False):
            return UserSegment.NEW_USERS
        elif user_context.get('session_count', 0) > 100:
            return UserSegment.POWER_USERS
        else:
            return UserSegment.ALL_USERS
    
    def _hash_user_to_variant(self, 
                            user_id: str,
                            experiment_id: str,
                            traffic_allocation: Dict[str, float]) -> str:
        """Hash user to variant based on traffic allocation"""
        
        # Create deterministic hash
        hash_input = f"{user_id}:{experiment_id}"
        hash_object = hashlib.md5(hash_input.encode())
        hash_value = int(hash_object.hexdigest()[:8], 16) % 10000 / 100.0  # 0-99.99
        
        # Assign based on cumulative traffic allocation
        cumulative_percentage = 0.0
        for variant_id, percentage in traffic_allocation.items():
            cumulative_percentage += percentage
            if hash_value < cumulative_percentage:
                return variant_id
        
        # Fallback to last variant (shouldn't happen with proper allocation)
        return list(traffic_allocation.keys())[-1]
    
    def record_result(self, 
                     experiment_id: str,
                     variant_id: str,
                     user_id: str,
                     session_id: str,
                     metric_values: Dict[str, float],
                     interaction_data: Dict[str, Any] = None,
                     conversion_event: bool = False,
                     error_occurred: bool = False):
        """Record an experiment result"""
        
        result = ExperimentResult(
            experiment_id=experiment_id,
            variant_id=variant_id,
            user_id=user_id,
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metric_values=metric_values,
            interaction_data=interaction_data or {},
            conversion_event=conversion_event,
            error_occurred=error_occurred
        )
        
        with self.results_lock:
            self.results.append(result)
    
    def analyze_experiment(self, experiment_id: str) -> Optional[StatisticalAnalysis]:
        """Analyze experiment results"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        if not experiment_results:
            return None
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in experiment_results:
            variant_results[result.variant_id].append(result)
        
        # Calculate sample sizes
        sample_sizes = {variant_id: len(results) for variant_id, results in variant_results.items()}
        
        # Calculate metric summaries
        metric_summaries = {}
        for variant_id, results in variant_results.items():
            metric_summaries[variant_id] = self._calculate_metric_summary(results, experiment.primary_metric)
        
        # Perform significance tests
        significance_tests = self._perform_significance_tests(variant_results, experiment.primary_metric)
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(variant_results, experiment.primary_metric)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            experiment, metric_summaries, significance_tests, sample_sizes
        )
        
        # Determine winning variant
        winning_variant = self._determine_winning_variant(metric_summaries, significance_tests)
        
        # Calculate statistical power (simplified)
        statistical_power = self._calculate_statistical_power(sample_sizes, metric_summaries)
        
        analysis = StatisticalAnalysis(
            experiment_id=experiment_id,
            analysis_date=datetime.now().isoformat(),
            sample_sizes=sample_sizes,
            metric_summaries=metric_summaries,
            significance_tests=significance_tests,
            confidence_intervals=confidence_intervals,
            recommendations=recommendations,
            winning_variant=winning_variant,
            statistical_power=statistical_power
        )
        
        self.analysis_cache[experiment_id] = analysis
        return analysis
    
    def _calculate_metric_summary(self, 
                                results: List[ExperimentResult],
                                primary_metric: MetricType) -> Dict[str, float]:
        """Calculate summary statistics for a metric"""
        
        metric_name = primary_metric.value
        
        if primary_metric == MetricType.CONVERSION_RATE:
            conversions = sum(1 for r in results if r.conversion_event)
            conversion_rate = conversions / len(results) if results else 0.0
            return {
                'mean': conversion_rate,
                'count': len(results),
                'conversions': conversions
            }
        
        elif primary_metric in [MetricType.RESPONSE_TIME, MetricType.ENGAGEMENT_TIME]:
            values = [r.metric_values.get(metric_name, 0.0) for r in results if metric_name in r.metric_values]
            if values:
                return {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'count': len(values),
                    'min': np.min(values),
                    'max': np.max(values)
                }
        
        elif primary_metric == MetricType.ERROR_RATE:
            errors = sum(1 for r in results if r.error_occurred)
            error_rate = errors / len(results) if results else 0.0
            return {
                'mean': error_rate,
                'count': len(results),
                'errors': errors
            }
        
        # Default case
        return {
            'mean': 0.0,
            'count': len(results)
        }
    
    def _perform_significance_tests(self, 
                                  variant_results: Dict[str, List[ExperimentResult]],
                                  primary_metric: MetricType) -> Dict[str, Dict[str, Any]]:
        """Perform statistical significance tests"""
        
        significance_tests = {}
        
        # Find control variant
        control_variant_id = None
        for experiment_id, variants in self.variants.items():
            for variant in variants:
                if variant.is_control:
                    control_variant_id = variant.variant_id
                    break
        
        if not control_variant_id or control_variant_id not in variant_results:
            return significance_tests
        
        control_results = variant_results[control_variant_id]
        control_summary = self._calculate_metric_summary(control_results, primary_metric)
        
        for variant_id, results in variant_results.items():
            if variant_id == control_variant_id:
                continue
            
            variant_summary = self._calculate_metric_summary(results, primary_metric)
            
            # Simplified significance test
            if primary_metric == MetricType.CONVERSION_RATE:
                p_value = self._calculate_conversion_p_value(control_summary, variant_summary)
            else:
                p_value = self._calculate_t_test_p_value(control_summary, variant_summary)
            
            significance_tests[variant_id] = {
                'vs_control': control_variant_id,
                'p_value': p_value,
                'is_significant': p_value < 0.05,
                'effect_size': self._calculate_effect_size(control_summary, variant_summary),
                'improvement': self._calculate_improvement_percentage(control_summary, variant_summary)
            }
        
        return significance_tests
    
    def _calculate_conversion_p_value(self, 
                                    control_summary: Dict[str, float],
                                    variant_summary: Dict[str, float]) -> float:
        """Calculate p-value for conversion rate test (simplified)"""
        
        # This is a simplified implementation
        # In practice, you'd use proper statistical tests like chi-square or z-test
        
        control_rate = control_summary.get('mean', 0.0)
        variant_rate = variant_summary.get('mean', 0.0)
        control_n = control_summary.get('count', 0)
        variant_n = variant_summary.get('count', 0)
        
        if control_n < 10 or variant_n < 10:
            return 1.0  # Insufficient data
        
        # Simplified z-test approximation
        pooled_rate = ((control_rate * control_n) + (variant_rate * variant_n)) / (control_n + variant_n)
        se = np.sqrt(pooled_rate * (1 - pooled_rate) * (1/control_n + 1/variant_n))
        
        if se == 0:
            return 1.0
        
        z_score = abs(control_rate - variant_rate) / se
        
        # Approximate p-value (two-tailed)
        if z_score > 2.58:
            return 0.01
        elif z_score > 1.96:
            return 0.05
        elif z_score > 1.64:
            return 0.1
        else:
            return 0.5
    
    def _calculate_t_test_p_value(self, 
                                control_summary: Dict[str, float],
                                variant_summary: Dict[str, float]) -> float:
        """Calculate p-value for t-test (simplified)"""
        
        # Simplified t-test implementation
        control_mean = control_summary.get('mean', 0.0)
        variant_mean = variant_summary.get('mean', 0.0)
        control_std = control_summary.get('std', 1.0)
        variant_std = variant_summary.get('std', 1.0)
        control_n = control_summary.get('count', 0)
        variant_n = variant_summary.get('count', 0)
        
        if control_n < 10 or variant_n < 10:
            return 1.0
        
        # Welch's t-test approximation
        se = np.sqrt((control_std**2 / control_n) + (variant_std**2 / variant_n))
        
        if se == 0:
            return 1.0
        
        t_score = abs(control_mean - variant_mean) / se
        
        # Approximate p-value
        if t_score > 2.6:
            return 0.01
        elif t_score > 2.0:
            return 0.05
        elif t_score > 1.7:
            return 0.1
        else:
            return 0.5
    
    def _calculate_effect_size(self, 
                             control_summary: Dict[str, float],
                             variant_summary: Dict[str, float]) -> float:
        """Calculate effect size (Cohen's d approximation)"""
        
        control_mean = control_summary.get('mean', 0.0)
        variant_mean = variant_summary.get('mean', 0.0)
        control_std = control_summary.get('std', 1.0)
        variant_std = variant_summary.get('std', 1.0)
        
        pooled_std = np.sqrt((control_std**2 + variant_std**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return abs(variant_mean - control_mean) / pooled_std
    
    def _calculate_improvement_percentage(self, 
                                        control_summary: Dict[str, float],
                                        variant_summary: Dict[str, float]) -> float:
        """Calculate percentage improvement"""
        
        control_mean = control_summary.get('mean', 0.0)
        variant_mean = variant_summary.get('mean', 0.0)
        
        if control_mean == 0:
            return 0.0
        
        return ((variant_mean - control_mean) / control_mean) * 100.0
    
    def _calculate_confidence_intervals(self, 
                                      variant_results: Dict[str, List[ExperimentResult]],
                                      primary_metric: MetricType) -> Dict[str, Dict[str, Tuple[float, float]]]:
        """Calculate confidence intervals"""
        
        confidence_intervals = {}
        
        for variant_id, results in variant_results.items():
            summary = self._calculate_metric_summary(results, primary_metric)
            mean = summary.get('mean', 0.0)
            n = summary.get('count', 0)
            
            if n < 10:
                confidence_intervals[variant_id] = {'95%': (mean, mean)}
                continue
            
            if primary_metric == MetricType.CONVERSION_RATE:
                # Binomial confidence interval
                se = np.sqrt(mean * (1 - mean) / n)
                margin = 1.96 * se  # 95% CI
                confidence_intervals[variant_id] = {
                    '95%': (max(0, mean - margin), min(1, mean + margin))
                }
            else:
                # Normal approximation
                std = summary.get('std', 1.0)
                se = std / np.sqrt(n)
                margin = 1.96 * se
                confidence_intervals[variant_id] = {
                    '95%': (mean - margin, mean + margin)
                }
        
        return confidence_intervals
    
    def _generate_recommendations(self, 
                                experiment: ExperimentConfig,
                                metric_summaries: Dict[str, Dict[str, float]],
                                significance_tests: Dict[str, Dict[str, Any]],
                                sample_sizes: Dict[str, int]) -> List[str]:
        """Generate recommendations based on analysis"""
        
        recommendations = []
        
        # Check sample size adequacy
        min_sample_size = experiment.minimum_sample_size
        for variant_id, sample_size in sample_sizes.items():
            if sample_size < min_sample_size:
                recommendations.append(f"Variant {variant_id} has insufficient sample size ({sample_size} < {min_sample_size})")
        
        # Check for significant results
        significant_variants = [variant_id for variant_id, test in significance_tests.items() 
                              if test['is_significant']]
        
        if significant_variants:
            best_variant = max(significant_variants, 
                             key=lambda v: significance_tests[v]['improvement'])
            improvement = significance_tests[best_variant]['improvement']
            recommendations.append(f"Variant {best_variant} shows significant improvement ({improvement:.1f}%)")
        else:
            recommendations.append("No variants show statistically significant differences")
        
        # Check for practical significance
        for variant_id, test in significance_tests.items():
            effect_size = test['effect_size']
            if effect_size > 0.8:
                recommendations.append(f"Variant {variant_id} shows large effect size ({effect_size:.2f})")
            elif effect_size > 0.5:
                recommendations.append(f"Variant {variant_id} shows medium effect size ({effect_size:.2f})")
        
        return recommendations
    
    def _determine_winning_variant(self, 
                                 metric_summaries: Dict[str, Dict[str, float]],
                                 significance_tests: Dict[str, Dict[str, Any]]) -> Optional[str]:
        """Determine winning variant"""
        
        # Find variants with significant positive improvement
        significant_positive_variants = []
        for variant_id, test in significance_tests.items():
            if test['is_significant'] and test['improvement'] > 0:
                significant_positive_variants.append((variant_id, test['improvement']))
        
        if significant_positive_variants:
            # Return variant with highest improvement
            return max(significant_positive_variants, key=lambda x: x[1])[0]
        
        return None
    
    def _calculate_statistical_power(self, 
                                   sample_sizes: Dict[str, int],
                                   metric_summaries: Dict[str, Dict[str, float]]) -> float:
        """Calculate statistical power (simplified)"""
        
        # Simplified power calculation
        total_sample_size = sum(sample_sizes.values())
        
        if total_sample_size < 100:
            return 0.2
        elif total_sample_size < 500:
            return 0.5
        elif total_sample_size < 1000:
            return 0.8
        else:
            return 0.9
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of an experiment"""
        
        if experiment_id not in self.experiments:
            return None
        
        experiment = self.experiments[experiment_id]
        experiment_results = [r for r in self.results if r.experiment_id == experiment_id]
        
        # Calculate current sample sizes
        variant_counts = defaultdict(int)
        for result in experiment_results:
            variant_counts[result.variant_id] += 1
        
        return {
            'experiment_id': experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'start_date': experiment.start_date,
            'end_date': experiment.end_date,
            'total_participants': len(set(r.user_id for r in experiment_results)),
            'total_results': len(experiment_results),
            'variant_sample_sizes': dict(variant_counts),
            'minimum_sample_size': experiment.minimum_sample_size,
            'ready_for_analysis': all(count >= experiment.minimum_sample_size 
                                    for count in variant_counts.values()),
            'variants': [asdict(variant) for variant in self.variants.get(experiment_id, [])]
        }

class FeatureFlags:
    """Feature flag management system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.flags: Dict[str, FeatureFlag] = {}
        self.evaluation_log = deque(maxlen=10000)
        
    def create_flag(self, 
                   flag_id: str,
                   name: str,
                   description: str,
                   default_value: Any = False,
                   enabled_segments: List[UserSegment] = None,
                   rollout_percentage: float = 0.0) -> FeatureFlag:
        """Create a new feature flag"""
        
        flag = FeatureFlag(
            flag_id=flag_id,
            name=name,
            description=description,
            default_value=default_value,
            enabled_segments=enabled_segments or [],
            rollout_percentage=rollout_percentage
        )
        
        self.flags[flag_id] = flag
        logger.info(f"Created feature flag: {name}")
        return flag
    
    def evaluate_flag(self, flag_id: str, user_context: Dict[str, Any]) -> Any:
        """Evaluate a feature flag for a user"""
        
        if flag_id not in self.flags:
            logger.warning(f"Unknown feature flag: {flag_id}")
            return False
        
        flag = self.flags[flag_id]
        result = flag.evaluate(user_context)
        
        # Log evaluation
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'flag_id': flag_id,
            'user_id': user_context.get('user_id', 'anonymous'),
            'result': result,
            'user_segment': flag._determine_user_segment(user_context).value
        }
        self.evaluation_log.append(evaluation_record)
        
        return result
    
    def update_flag_rollout(self, flag_id: str, rollout_percentage: float) -> bool:
        """Update flag rollout percentage"""
        
        if flag_id not in self.flags:
            return False
        
        self.flags[flag_id].rollout_percentage = rollout_percentage
        self.flags[flag_id].updated_at = datetime.now()
        
        logger.info(f"Updated flag {flag_id} rollout to {rollout_percentage}%")
        return True
    
    def update_flag_segments(self, flag_id: str, enabled_segments: List[UserSegment]) -> bool:
        """Update flag enabled segments"""
        
        if flag_id not in self.flags:
            return False
        
        self.flags[flag_id].enabled_segments = enabled_segments
        self.flags[flag_id].updated_at = datetime.now()
        
        logger.info(f"Updated flag {flag_id} segments to {[s.value for s in enabled_segments]}")
        return True
    
    def get_flag_statistics(self) -> Dict[str, Any]:
        """Get feature flag statistics"""
        
        flag_stats = {}
        for flag_id, flag in self.flags.items():
            flag_stats[flag_id] = flag.get_statistics()
        
        # Overall statistics
        total_evaluations = len(self.evaluation_log)
        segment_breakdown = defaultdict(int)
        
        for evaluation in self.evaluation_log:
            segment_breakdown[evaluation['user_segment']] += 1
        
        return {
            'total_flags': len(self.flags),
            'total_evaluations': total_evaluations,
            'segment_breakdown': dict(segment_breakdown),
            'flag_details': flag_stats
        }

class InnovationMetrics:
    """Innovation metrics and evaluation system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.metrics_history = deque(maxlen=10000)
        self.baseline_metrics = {}
        self.innovation_scores = {}
        
    def record_innovation_metric(self, 
                               innovation_id: str,
                               metric_type: str,
                               value: float,
                               context: Dict[str, Any] = None):
        """Record an innovation metric"""
        
        metric_record = {
            'timestamp': datetime.now().isoformat(),
            'innovation_id': innovation_id,
            'metric_type': metric_type,
            'value': value,
            'context': context or {}
        }
        
        self.metrics_history.append(metric_record)
    
    def calculate_innovation_score(self, 
                                 innovation_id: str,
                                 time_window: timedelta = timedelta(days=7)) -> Dict[str, Any]:
        """Calculate innovation score for a specific innovation"""
        
        current_time = datetime.now()
        window_start = current_time - time_window
        
        # Filter metrics for this innovation within time window
        relevant_metrics = [
            m for m in self.metrics_history
            if (m['innovation_id'] == innovation_id and 
                datetime.fromisoformat(m['timestamp']) >= window_start)
        ]
        
        if not relevant_metrics:
            return {'error': 'No metrics found for innovation'}
        
        # Group metrics by type
        metrics_by_type = defaultdict(list)
        for metric in relevant_metrics:
            metrics_by_type[metric['metric_type']].append(metric['value'])
        
        # Calculate scores for each metric type
        type_scores = {}
        for metric_type, values in metrics_by_type.items():
            baseline = self.baseline_metrics.get(metric_type, np.mean(values))
            
            if baseline > 0:
                improvement = (np.mean(values) - baseline) / baseline
                type_scores[metric_type] = {
                    'current_value': np.mean(values),
                    'baseline_value': baseline,
                    'improvement': improvement,
                    'score': max(0, min(100, 50 + improvement * 50))  # 0-100 scale
                }
        
        # Calculate overall innovation score
        if type_scores:
            overall_score = np.mean([score['score'] for score in type_scores.values()])
        else:
            overall_score = 50.0  # Neutral score
        
        innovation_score = {
            'innovation_id': innovation_id,
            'overall_score': overall_score,
            'metric_scores': type_scores,
            'measurement_period': {
                'start': window_start.isoformat(),
                'end': current_time.isoformat()
            },
            'data_points': len(relevant_metrics)
        }
        
        self.innovation_scores[innovation_id] = innovation_score
        return innovation_score
    
    def set_baseline_metrics(self, baseline_metrics: Dict[str, float]):
        """Set baseline metrics for comparison"""
        self.baseline_metrics = baseline_metrics
        logger.info(f"Set baseline metrics for {len(baseline_metrics)} metric types")
    
    def get_innovation_leaderboard(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top performing innovations"""
        
        # Calculate current scores for all innovations
        innovation_ids = set(m['innovation_id'] for m in self.metrics_history)
        current_scores = []
        
        for innovation_id in innovation_ids:
            score_data = self.calculate_innovation_score(innovation_id)
            if 'error' not in score_data:
                current_scores.append({
                    'innovation_id': innovation_id,
                    'score': score_data['overall_score'],
                    'data_points': score_data['data_points']
                })
        
        # Sort by score and return top performers
        current_scores.sort(key=lambda x: x['score'], reverse=True)
        return current_scores[:limit]
    
    def generate_innovation_report(self) -> Dict[str, Any]:
        """Generate comprehensive innovation metrics report"""
        
        if not self.metrics_history:
            return {'error': 'No innovation metrics recorded'}
        
        # Basic statistics
        total_metrics = len(self.metrics_history)
        unique_innovations = len(set(m['innovation_id'] for m in self.metrics_history))
        metric_types = set(m['metric_type'] for m in self.metrics_history)
        
        # Time-based analysis
        recent_metrics = [m for m in self.metrics_history 
                         if datetime.fromisoformat(m['timestamp']) >= datetime.now() - timedelta(days=7)]
        
        # Innovation performance
        leaderboard = self.get_innovation_leaderboard()
        
        # Metric type analysis
        metric_type_stats = {}
        for metric_type in metric_types:
            type_metrics = [m for m in self.metrics_history if m['metric_type'] == metric_type]
            values = [m['value'] for m in type_metrics]
            
            metric_type_stats[metric_type] = {
                'count': len(type_metrics),
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'total_metrics': total_metrics,
                'unique_innovations': unique_innovations,
                'metric_types': list(metric_types),
                'recent_activity': len(recent_metrics)
            },
            'leaderboard': leaderboard,
            'metric_type_analysis': metric_type_stats,
            'baseline_metrics': self.baseline_metrics
        }

class ExperimentTracker:
    """Tracks and manages all innovation experiments"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ab_testing = ABTestingFramework(config)
        self.feature_flags = FeatureFlags(config)
        self.innovation_metrics = InnovationMetrics(config)
        self.experiment_registry = {}
        
    def register_innovation_experiment(self, 
                                     experiment_name: str,
                                     experiment_type: str,
                                     description: str,
                                     success_criteria: Dict[str, Any]) -> str:
        """Register a new innovation experiment"""
        
        experiment_id = str(uuid.uuid4())
        
        experiment_record = {
            'experiment_id': experiment_id,
            'name': experiment_name,
            'type': experiment_type,
            'description': description,
            'success_criteria': success_criteria,
            'created_at': datetime.now().isoformat(),
            'status': 'registered',
            'results': {}
        }
        
        self.experiment_registry[experiment_id] = experiment_record
        logger.info(f"Registered innovation experiment: {experiment_name}")
        return experiment_id
    
    def create_ab_test_experiment(self, 
                                experiment_name: str,
                                variants: List[Dict[str, Any]],
                                primary_metric: str = 'conversion_rate') -> str:
        """Create an A/B test experiment"""
        
        # Convert to ExperimentVariant objects
        ab_variants = []
        for i, variant_config in enumerate(variants):
            variant = ExperimentVariant(
                variant_id=variant_config.get('id', f"variant_{i}"),
                name=variant_config.get('name', f"Variant {i}"),
                description=variant_config.get('description', ''),
                feature_config=variant_config.get('config', {}),
                traffic_percentage=variant_config.get('traffic', 100.0 / len(variants)),
                is_control=variant_config.get('is_control', i == 0)
            )
            ab_variants.append(variant)
        
        # Create A/B test
        ab_experiment_id = self.ab_testing.create_experiment(
            name=experiment_name,
            description=f"A/B test for {experiment_name}",
            variants=ab_variants,
            primary_metric=MetricType(primary_metric)
        )
        
        # Register in experiment registry
        innovation_experiment_id = self.register_innovation_experiment(
            experiment_name=experiment_name,
            experiment_type='ab_test',
            description=f"A/B test experiment with {len(variants)} variants",
            success_criteria={'primary_metric': primary_metric, 'significance_level': 0.05}
        )
        
        # Link the experiments
        self.experiment_registry[innovation_experiment_id]['ab_experiment_id'] = ab_experiment_id
        
        return innovation_experiment_id
    
    def create_feature_flag_experiment(self, 
                                     experiment_name: str,
                                     flag_config: Dict[str, Any],
                                     rollout_plan: Dict[str, Any]) -> str:
        """Create a feature flag experiment"""
        
        # Create feature flag
        flag = self.feature_flags.create_flag(
            flag_id=flag_config.get('flag_id', str(uuid.uuid4())),
            name=flag_config.get('name', experiment_name),
            description=flag_config.get('description', ''),
            default_value=flag_config.get('default_value', False),
            enabled_segments=[UserSegment(seg) for seg in flag_config.get('segments', ['all_users'])],
            rollout_percentage=rollout_plan.get('initial_percentage', 0.0)
        )
        
        # Register in experiment registry
        innovation_experiment_id = self.register_innovation_experiment(
            experiment_name=experiment_name,
            experiment_type='feature_flag',
            description=f"Feature flag experiment: {flag_config.get('name')}",
            success_criteria=rollout_plan.get('success_criteria', {})
        )
        
        # Link the flag
        self.experiment_registry[innovation_experiment_id]['flag_id'] = flag.flag_id
        self.experiment_registry[innovation_experiment_id]['rollout_plan'] = rollout_plan
        
        return innovation_experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an innovation experiment"""
        
        if experiment_id not in self.experiment_registry:
            return False
        
        experiment = self.experiment_registry[experiment_id]
        experiment_type = experiment['type']
        
        if experiment_type == 'ab_test':
            ab_experiment_id = experiment.get('ab_experiment_id')
            if ab_experiment_id:
                success = self.ab_testing.start_experiment(ab_experiment_id)
                if success:
                    experiment['status'] = 'active'
                return success
        
        elif experiment_type == 'feature_flag':
            flag_id = experiment.get('flag_id')
            rollout_plan = experiment.get('rollout_plan', {})
            
            if flag_id:
                # Start with initial rollout percentage
                initial_percentage = rollout_plan.get('initial_percentage', 10.0)
                success = self.feature_flags.update_flag_rollout(flag_id, initial_percentage)
                if success:
                    experiment['status'] = 'active'
                return success
        
        return False
    
    def record_experiment_result(self, 
                               experiment_id: str,
                               user_context: Dict[str, Any],
                               metrics: Dict[str, float],
                               interaction_data: Dict[str, Any] = None):
        """Record a result for an innovation experiment"""
        
        if experiment_id not in self.experiment_registry:
            logger.warning(f"Unknown experiment: {experiment_id}")
            return
        
        experiment = self.experiment_registry[experiment_id]
        experiment_type = experiment['type']
        
        if experiment_type == 'ab_test':
            ab_experiment_id = experiment.get('ab_experiment_id')
            if ab_experiment_id:
                # Determine user's variant
                variant_id = self.ab_testing.assign_user_to_variant(ab_experiment_id, user_context)
                if variant_id:
                    self.ab_testing.record_result(
                        experiment_id=ab_experiment_id,
                        variant_id=variant_id,
                        user_id=user_context.get('user_id', 'anonymous'),
                        session_id=user_context.get('session_id', 'unknown'),
                        metric_values=metrics,
                        interaction_data=interaction_data,
                        conversion_event=metrics.get('conversion', False),
                        error_occurred=metrics.get('error', False)
                    )
        
        # Record innovation metrics
        for metric_name, value in metrics.items():
            self.innovation_metrics.record_innovation_metric(
                innovation_id=experiment_id,
                metric_type=metric_name,
                value=value,
                context=user_context
            )
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze an innovation experiment"""
        
        if experiment_id not in self.experiment_registry:
            return {'error': 'Experiment not found'}
        
        experiment = self.experiment_registry[experiment_id]
        experiment_type = experiment['type']
        
        analysis_result = {
            'experiment_id': experiment_id,
            'experiment_type': experiment_type,
            'experiment_name': experiment['name'],
            'status': experiment['status']
        }
        
        if experiment_type == 'ab_test':
            ab_experiment_id = experiment.get('ab_experiment_id')
            if ab_experiment_id:
                ab_analysis = self.ab_testing.analyze_experiment(ab_experiment_id)
                if ab_analysis:
                    analysis_result['ab_analysis'] = asdict(ab_analysis)
        
        elif experiment_type == 'feature_flag':
            flag_id = experiment.get('flag_id')
            if flag_id:
                flag_stats = self.feature_flags.flags[flag_id].get_statistics()
                analysis_result['flag_analysis'] = flag_stats
        
        # Add innovation score
        innovation_score = self.innovation_metrics.calculate_innovation_score(experiment_id)
        analysis_result['innovation_score'] = innovation_score
        
        return analysis_result
    
    def get_experiment_dashboard(self) -> Dict[str, Any]:
        """Get experiment dashboard with overview of all experiments"""
        
        dashboard = {
            'dashboard_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'total_experiments': len(self.experiment_registry),
            'experiments_by_status': defaultdict(int),
            'experiments_by_type': defaultdict(int),
            'active_experiments': [],
            'recent_results': []
        }
        
        # Analyze all experiments
        for experiment_id, experiment in self.experiment_registry.items():
            status = experiment['status']
            experiment_type = experiment['type']
            
            dashboard['experiments_by_status'][status] += 1
            dashboard['experiments_by_type'][experiment_type] += 1
            
            if status == 'active':
                experiment_summary = {
                    'experiment_id': experiment_id,
                    'name': experiment['name'],
                    'type': experiment_type,
                    'created_at': experiment['created_at']
                }
                
                # Add quick analysis
                analysis = self.analyze_experiment(experiment_id)
                if 'innovation_score' in analysis and 'error' not in analysis['innovation_score']:
                    experiment_summary['current_score'] = analysis['innovation_score']['overall_score']
                
                dashboard['active_experiments'].append(experiment_summary)
        
        # Get innovation leaderboard
        dashboard['innovation_leaderboard'] = self.innovation_metrics.get_innovation_leaderboard()
        
        # Convert defaultdicts to regular dicts
        dashboard['experiments_by_status'] = dict(dashboard['experiments_by_status'])
        dashboard['experiments_by_type'] = dict(dashboard['experiments_by_type'])
        
        return dashboard