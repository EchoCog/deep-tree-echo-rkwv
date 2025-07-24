"""
Research Data Collection System
Implements anonymized data collection, benchmarking, and analysis capabilities
"""

import time
import json
import logging
import hashlib
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
    
    np = NumpyFallback()
from collections import defaultdict, deque
import uuid
import sqlite3
import os

logger = logging.getLogger(__name__)

class DataPrivacyLevel(Enum):
    """Privacy levels for research data collection"""
    ANONYMOUS = "anonymous"
    PSEUDONYMOUS = "pseudonymous"
    AGGREGATED = "aggregated"
    MINIMAL = "minimal"

class BenchmarkType(Enum):
    """Types of cognitive benchmarks"""
    REASONING_SPEED = "reasoning_speed"
    MEMORY_EFFICIENCY = "memory_efficiency"
    ACCURACY = "accuracy"
    CONSISTENCY = "consistency"
    SCALABILITY = "scalability"
    ROBUSTNESS = "robustness"

class InteractionType(Enum):
    """Types of user interactions to analyze"""
    QUERY_RESPONSE = "query_response"
    MEMORY_ACCESS = "memory_access"
    REASONING_CHAIN = "reasoning_chain"
    ERROR_RECOVERY = "error_recovery"
    CONTEXT_SWITCH = "context_switch"

@dataclass
class AnonymizedDataPoint:
    """Anonymized data point for research collection"""
    data_id: str
    timestamp: str
    data_type: str
    anonymized_content: str
    privacy_level: str
    metadata: Dict[str, Any]
    session_hash: str
    performance_metrics: Dict[str, float]

@dataclass
class BenchmarkResult:
    """Result from cognitive performance benchmark"""
    benchmark_id: str
    benchmark_type: str
    timestamp: str
    performance_score: float
    latency_ms: float
    accuracy_score: float
    consistency_score: float
    resource_usage: Dict[str, float]
    test_parameters: Dict[str, Any]
    error_log: List[str]

@dataclass
class InteractionPattern:
    """Detected pattern in user interactions"""
    pattern_id: str
    pattern_type: str
    frequency: int
    confidence: float
    example_interactions: List[Dict[str, Any]]
    temporal_distribution: Dict[str, int]
    user_segments: List[str]

class ResearchDataCollector:
    """Collects and anonymizes data for research purposes"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.privacy_level = DataPrivacyLevel(config.get('privacy_level', 'anonymous'))
        self.data_store = config.get('data_store', 'memory')
        self.collection_enabled = config.get('enabled', True)
        
        # Initialize storage
        self.collected_data = []
        self.session_anonymizer = {}
        self.data_lock = threading.Lock()
        
        # Database setup for persistent storage
        if self.data_store == 'database':
            self.db_path = config.get('db_path', '/tmp/research_data.db')
            self._initialize_database()
    
    def _initialize_database(self):
        """Initialize SQLite database for research data storage"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS research_data (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    data_type TEXT,
                    anonymized_content TEXT,
                    privacy_level TEXT,
                    metadata TEXT,
                    session_hash TEXT,
                    performance_metrics TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Research database initialized at {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize research database: {e}")
    
    def collect_interaction_data(self, 
                                interaction_type: InteractionType,
                                raw_data: Dict[str, Any],
                                session_id: str,
                                performance_metrics: Dict[str, float] = None) -> str:
        """Collect and anonymize interaction data"""
        
        if not self.collection_enabled:
            return None
        
        try:
            # Anonymize session ID
            session_hash = self._anonymize_session(session_id)
            
            # Anonymize content based on privacy level
            anonymized_content = self._anonymize_content(raw_data, self.privacy_level)
            
            # Create data point
            data_point = AnonymizedDataPoint(
                data_id=str(uuid.uuid4()),
                timestamp=datetime.now().isoformat(),
                data_type=interaction_type.value,
                anonymized_content=json.dumps(anonymized_content),
                privacy_level=self.privacy_level.value,
                metadata={
                    'content_length': len(str(raw_data)),
                    'fields_count': len(raw_data) if isinstance(raw_data, dict) else 1,
                    'collection_version': '1.0'
                },
                session_hash=session_hash,
                performance_metrics=performance_metrics or {}
            )
            
            # Store data
            with self.data_lock:
                if self.data_store == 'memory':
                    self.collected_data.append(data_point)
                elif self.data_store == 'database':
                    self._store_in_database(data_point)
            
            logger.debug(f"Collected research data point: {data_point.data_id}")
            return data_point.data_id
            
        except Exception as e:
            logger.error(f"Failed to collect research data: {e}")
            return None
    
    def _anonymize_session(self, session_id: str) -> str:
        """Anonymize session ID using consistent hashing"""
        if session_id in self.session_anonymizer:
            return self.session_anonymizer[session_id]
        
        # Create anonymized session hash
        session_hash = hashlib.sha256(
            (session_id + self.config.get('salt', 'research_salt')).encode()
        ).hexdigest()[:16]
        
        self.session_anonymizer[session_id] = session_hash
        return session_hash
    
    def _anonymize_content(self, raw_data: Dict[str, Any], privacy_level: DataPrivacyLevel) -> Dict[str, Any]:
        """Anonymize content based on privacy level"""
        
        if privacy_level == DataPrivacyLevel.ANONYMOUS:
            return self._fully_anonymize(raw_data)
        elif privacy_level == DataPrivacyLevel.PSEUDONYMOUS:
            return self._pseudonymize(raw_data)
        elif privacy_level == DataPrivacyLevel.AGGREGATED:
            return self._aggregate_anonymize(raw_data)
        elif privacy_level == DataPrivacyLevel.MINIMAL:
            return self._minimal_anonymize(raw_data)
        else:
            return {"error": "unknown_privacy_level"}
    
    def _fully_anonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Full anonymization - only statistical features"""
        anonymized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                anonymized[f"{key}_length"] = len(value)
                anonymized[f"{key}_word_count"] = len(value.split())
                anonymized[f"{key}_has_question"] = '?' in value
                anonymized[f"{key}_has_numbers"] = any(c.isdigit() for c in value)
            elif isinstance(value, (int, float)):
                # Quantize numerical values
                if isinstance(value, float):
                    anonymized[f"{key}_range"] = "low" if value < 0.3 else "medium" if value < 0.7 else "high"
                else:
                    anonymized[f"{key}_magnitude"] = "small" if value < 10 else "medium" if value < 100 else "large"
            elif isinstance(value, dict):
                anonymized[f"{key}_fields_count"] = len(value)
            elif isinstance(value, list):
                anonymized[f"{key}_items_count"] = len(value)
        
        return anonymized
    
    def _pseudonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Pseudonymization - replace identifiers with pseudonyms"""
        pseudonymized = {}
        
        for key, value in data.items():
            if isinstance(value, str):
                # Hash string content but preserve structure
                if len(value) > 0:
                    content_hash = hashlib.md5(value.encode()).hexdigest()[:8]
                    pseudonymized[key] = f"content_{content_hash}_{len(value)}"
                else:
                    pseudonymized[key] = "empty_string"
            else:
                pseudonymized[key] = value  # Keep non-string values
        
        return pseudonymized
    
    def _aggregate_anonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregated anonymization - combine into categories"""
        categories = {
            'text_fields': 0,
            'numeric_fields': 0,
            'boolean_fields': 0,
            'complex_fields': 0,
            'total_content_length': 0,
            'field_types': []
        }
        
        for key, value in data.items():
            if isinstance(value, str):
                categories['text_fields'] += 1
                categories['total_content_length'] += len(value)
                categories['field_types'].append('text')
            elif isinstance(value, (int, float)):
                categories['numeric_fields'] += 1
                categories['field_types'].append('numeric')
            elif isinstance(value, bool):
                categories['boolean_fields'] += 1
                categories['field_types'].append('boolean')
            else:
                categories['complex_fields'] += 1
                categories['field_types'].append('complex')
        
        return categories
    
    def _minimal_anonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Minimal anonymization - remove only explicit identifiers"""
        sensitive_keys = ['user_id', 'session_id', 'ip_address', 'email', 'name']
        
        anonymized = {}
        for key, value in data.items():
            if key.lower() in sensitive_keys:
                anonymized[f"{key}_anonymized"] = True
            else:
                anonymized[key] = value
        
        return anonymized
    
    def _store_in_database(self, data_point: AnonymizedDataPoint):
        """Store data point in SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO research_data 
                (id, timestamp, data_type, anonymized_content, privacy_level, metadata, session_hash, performance_metrics)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                data_point.data_id,
                data_point.timestamp,
                data_point.data_type,
                data_point.anonymized_content,
                data_point.privacy_level,
                json.dumps(data_point.metadata),
                data_point.session_hash,
                json.dumps(data_point.performance_metrics)
            ))
            
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to store data in database: {e}")
    
    def get_collected_data(self, 
                          data_type: Optional[InteractionType] = None,
                          session_hash: Optional[str] = None,
                          time_range: Optional[Tuple[datetime, datetime]] = None) -> List[AnonymizedDataPoint]:
        """Retrieve collected research data with filters"""
        
        with self.data_lock:
            if self.data_store == 'memory':
                data = self.collected_data
            else:
                data = self._load_from_database(data_type, session_hash, time_range)
        
        # Apply filters
        filtered_data = data
        
        if data_type:
            filtered_data = [d for d in filtered_data if d.data_type == data_type.value]
        
        if session_hash:
            filtered_data = [d for d in filtered_data if d.session_hash == session_hash]
        
        if time_range:
            start_time, end_time = time_range
            filtered_data = [d for d in filtered_data 
                           if start_time <= datetime.fromisoformat(d.timestamp) <= end_time]
        
        return filtered_data
    
    def _load_from_database(self, data_type=None, session_hash=None, time_range=None) -> List[AnonymizedDataPoint]:
        """Load data from SQLite database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM research_data"
            params = []
            conditions = []
            
            if data_type:
                conditions.append("data_type = ?")
                params.append(data_type.value)
            
            if session_hash:
                conditions.append("session_hash = ?")
                params.append(session_hash)
            
            if time_range:
                conditions.append("timestamp BETWEEN ? AND ?")
                params.extend([time_range[0].isoformat(), time_range[1].isoformat()])
            
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            
            data_points = []
            for row in rows:
                data_point = AnonymizedDataPoint(
                    data_id=row[0],
                    timestamp=row[1],
                    data_type=row[2],
                    anonymized_content=row[3],
                    privacy_level=row[4],
                    metadata=json.loads(row[5]),
                    session_hash=row[6],
                    performance_metrics=json.loads(row[7])
                )
                data_points.append(data_point)
            
            conn.close()
            return data_points
            
        except Exception as e:
            logger.error(f"Failed to load data from database: {e}")
            return []
    
    def generate_research_report(self) -> Dict[str, Any]:
        """Generate anonymized research report"""
        all_data = self.get_collected_data()
        
        if not all_data:
            return {"error": "No data collected"}
        
        # Basic statistics
        data_by_type = defaultdict(list)
        session_counts = defaultdict(int)
        temporal_distribution = defaultdict(int)
        
        for data_point in all_data:
            data_by_type[data_point.data_type].append(data_point)
            session_counts[data_point.session_hash] += 1
            
            # Group by hour for temporal analysis
            timestamp = datetime.fromisoformat(data_point.timestamp)
            hour_key = timestamp.strftime("%Y-%m-%d %H:00")
            temporal_distribution[hour_key] += 1
        
        # Performance metrics analysis
        performance_summary = {}
        for data_type, data_points in data_by_type.items():
            metrics = [dp.performance_metrics for dp in data_points if dp.performance_metrics]
            if metrics:
                performance_summary[data_type] = self._analyze_performance_metrics(metrics)
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'privacy_level': self.privacy_level.value,
            'total_data_points': len(all_data),
            'unique_sessions': len(session_counts),
            'data_types': {dtype: len(dpoints) for dtype, dpoints in data_by_type.items()},
            'temporal_distribution': dict(temporal_distribution),
            'session_activity': {
                'avg_interactions_per_session': np.mean(list(session_counts.values())),
                'max_interactions_per_session': max(session_counts.values()),
                'sessions_with_single_interaction': sum(1 for count in session_counts.values() if count == 1)
            },
            'performance_summary': performance_summary,
            'collection_period': {
                'start': min(dp.timestamp for dp in all_data),
                'end': max(dp.timestamp for dp in all_data)
            }
        }
        
        return report
    
    def _analyze_performance_metrics(self, metrics_list: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze performance metrics across data points"""
        if not metrics_list:
            return {}
        
        # Collect all metric names
        all_metrics = set()
        for metrics in metrics_list:
            all_metrics.update(metrics.keys())
        
        analysis = {}
        for metric_name in all_metrics:
            values = [metrics.get(metric_name, 0) for metrics in metrics_list if metric_name in metrics]
            if values:
                analysis[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        return analysis

class CognitiveBenchmark:
    """Cognitive performance benchmarking system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.benchmark_results = []
        self.test_suites = {}
        self._initialize_standard_benchmarks()
    
    def _initialize_standard_benchmarks(self):
        """Initialize standard cognitive benchmarks"""
        
        # Reasoning speed benchmark
        self.test_suites[BenchmarkType.REASONING_SPEED] = {
            'test_cases': [
                {'query': 'What is 2+2?', 'expected_response_time': 0.1},
                {'query': 'Explain the concept of recursion', 'expected_response_time': 2.0},
                {'query': 'What are the implications of quantum computing?', 'expected_response_time': 5.0}
            ],
            'time_limit': 10.0,
            'scoring_function': lambda results: self._score_speed_benchmark(results)
        }
        
        # Memory efficiency benchmark
        self.test_suites[BenchmarkType.MEMORY_EFFICIENCY] = {
            'test_cases': [
                {'operation': 'store', 'data_size': 'small', 'expected_memory': 1024},
                {'operation': 'retrieve', 'data_size': 'medium', 'expected_memory': 2048},
                {'operation': 'associate', 'data_size': 'large', 'expected_memory': 4096}
            ],
            'memory_limit': 10240,  # 10KB
            'scoring_function': lambda results: self._score_memory_benchmark(results)
        }
        
        # Accuracy benchmark
        self.test_suites[BenchmarkType.ACCURACY] = {
            'test_cases': [
                {'query': 'What is the capital of France?', 'expected_answer': 'Paris'},
                {'query': 'What is 15 * 8?', 'expected_answer': '120'},
                {'query': 'Name a primary color', 'expected_answers': ['red', 'blue', 'yellow']}
            ],
            'scoring_function': lambda results: self._score_accuracy_benchmark(results)
        }
        
        # Consistency benchmark
        self.test_suites[BenchmarkType.CONSISTENCY] = {
            'test_cases': [
                {'query': 'What is the meaning of life?', 'repeat_count': 5},
                {'query': 'Explain machine learning', 'repeat_count': 3},
                {'query': 'What is 2+2?', 'repeat_count': 10}
            ],
            'consistency_threshold': 0.8,
            'scoring_function': lambda results: self._score_consistency_benchmark(results)
        }
    
    def run_benchmark(self, 
                     benchmark_type: BenchmarkType,
                     cognitive_system: Any,
                     custom_test_cases: List[Dict[str, Any]] = None) -> BenchmarkResult:
        """Run a specific benchmark on a cognitive system"""
        
        start_time = time.time()
        test_suite = self.test_suites.get(benchmark_type)
        
        if not test_suite:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        
        test_cases = custom_test_cases or test_suite['test_cases']
        results = []
        errors = []
        
        try:
            for test_case in test_cases:
                case_start = time.time()
                
                try:
                    # Run test case on cognitive system
                    if hasattr(cognitive_system, 'process'):
                        response = cognitive_system.process(test_case)
                    elif hasattr(cognitive_system, 'query'):
                        response = cognitive_system.query(test_case.get('query', ''))
                    else:
                        response = {"error": "Unsupported cognitive system interface"}
                    
                    case_time = time.time() - case_start
                    
                    test_result = {
                        'test_case': test_case,
                        'response': response,
                        'execution_time': case_time,
                        'success': 'error' not in response
                    }
                    results.append(test_result)
                    
                except Exception as e:
                    errors.append(f"Test case failed: {e}")
                    results.append({
                        'test_case': test_case,
                        'response': {"error": str(e)},
                        'execution_time': time.time() - case_start,
                        'success': False
                    })
            
            # Score the benchmark
            scoring_function = test_suite['scoring_function']
            scores = scoring_function(results)
            
            total_time = time.time() - start_time
            
            benchmark_result = BenchmarkResult(
                benchmark_id=str(uuid.uuid4()),
                benchmark_type=benchmark_type.value,
                timestamp=datetime.now().isoformat(),
                performance_score=scores.get('overall_score', 0.0),
                latency_ms=total_time * 1000,
                accuracy_score=scores.get('accuracy_score', 0.0),
                consistency_score=scores.get('consistency_score', 0.0),
                resource_usage={
                    'total_execution_time': total_time,
                    'avg_case_time': np.mean([r['execution_time'] for r in results]),
                    'test_cases_count': len(test_cases)
                },
                test_parameters={
                    'benchmark_type': benchmark_type.value,
                    'test_cases_count': len(test_cases),
                    'custom_test_cases': custom_test_cases is not None
                },
                error_log=errors
            )
            
            self.benchmark_results.append(benchmark_result)
            return benchmark_result
            
        except Exception as e:
            logger.error(f"Benchmark execution failed: {e}")
            return BenchmarkResult(
                benchmark_id=str(uuid.uuid4()),
                benchmark_type=benchmark_type.value,
                timestamp=datetime.now().isoformat(),
                performance_score=0.0,
                latency_ms=(time.time() - start_time) * 1000,
                accuracy_score=0.0,
                consistency_score=0.0,
                resource_usage={},
                test_parameters={'error': str(e)},
                error_log=[str(e)]
            )
    
    def _score_speed_benchmark(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score reasoning speed benchmark"""
        if not results:
            return {'overall_score': 0.0}
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'overall_score': 0.0}
        
        # Calculate speed score based on execution times
        execution_times = [r['execution_time'] for r in successful_results]
        avg_time = np.mean(execution_times)
        
        # Score inversely proportional to time (faster = better)
        speed_score = max(0.0, 1.0 - (avg_time / 10.0))  # Normalize to 10 second max
        
        return {
            'overall_score': speed_score,
            'avg_execution_time': avg_time,
            'success_rate': len(successful_results) / len(results)
        }
    
    def _score_memory_benchmark(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score memory efficiency benchmark"""
        if not results:
            return {'overall_score': 0.0}
        
        successful_results = [r for r in results if r['success']]
        
        if not successful_results:
            return {'overall_score': 0.0}
        
        # Memory efficiency score (placeholder - would need actual memory monitoring)
        memory_score = 0.8  # Simulated memory efficiency
        
        return {
            'overall_score': memory_score,
            'success_rate': len(successful_results) / len(results)
        }
    
    def _score_accuracy_benchmark(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score accuracy benchmark"""
        if not results:
            return {'overall_score': 0.0, 'accuracy_score': 0.0}
        
        correct_answers = 0
        total_answers = 0
        
        for result in results:
            if not result['success']:
                continue
            
            test_case = result['test_case']
            response = result['response']
            
            # Extract answer from response
            answer = self._extract_answer(response)
            
            if 'expected_answer' in test_case:
                if answer and answer.lower() == test_case['expected_answer'].lower():
                    correct_answers += 1
            elif 'expected_answers' in test_case:
                if answer and any(answer.lower() == expected.lower() 
                                for expected in test_case['expected_answers']):
                    correct_answers += 1
            
            total_answers += 1
        
        accuracy_score = correct_answers / total_answers if total_answers > 0 else 0.0
        
        return {
            'overall_score': accuracy_score,
            'accuracy_score': accuracy_score,
            'correct_answers': correct_answers,
            'total_answers': total_answers
        }
    
    def _score_consistency_benchmark(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Score consistency benchmark"""
        if not results:
            return {'overall_score': 0.0, 'consistency_score': 0.0}
        
        # Group results by query
        query_groups = defaultdict(list)
        for result in results:
            if result['success']:
                query = result['test_case'].get('query', '')
                query_groups[query].append(result['response'])
        
        consistency_scores = []
        
        for query, responses in query_groups.items():
            if len(responses) < 2:
                continue
            
            # Calculate similarity between responses (simplified)
            similarity_sum = 0
            comparison_count = 0
            
            for i in range(len(responses)):
                for j in range(i + 1, len(responses)):
                    similarity = self._calculate_response_similarity(responses[i], responses[j])
                    similarity_sum += similarity
                    comparison_count += 1
            
            if comparison_count > 0:
                query_consistency = similarity_sum / comparison_count
                consistency_scores.append(query_consistency)
        
        overall_consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return {
            'overall_score': overall_consistency,
            'consistency_score': overall_consistency,
            'queries_tested': len(query_groups)
        }
    
    def _extract_answer(self, response: Dict[str, Any]) -> Optional[str]:
        """Extract answer from cognitive system response"""
        if isinstance(response, dict):
            # Try common response fields
            for field in ['answer', 'result', 'response', 'content', 'text']:
                if field in response:
                    return str(response[field]).strip()
        
        return str(response).strip() if response else None
    
    def _calculate_response_similarity(self, response1: Dict[str, Any], response2: Dict[str, Any]) -> float:
        """Calculate similarity between two responses (simplified)"""
        text1 = self._extract_answer(response1) or ""
        text2 = self._extract_answer(response2) or ""
        
        # Simple similarity based on common words
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def get_benchmark_history(self, benchmark_type: Optional[BenchmarkType] = None) -> List[BenchmarkResult]:
        """Get benchmark history with optional filtering"""
        if benchmark_type:
            return [result for result in self.benchmark_results 
                   if result.benchmark_type == benchmark_type.value]
        return self.benchmark_results
    
    def generate_benchmark_report(self) -> Dict[str, Any]:
        """Generate comprehensive benchmark report"""
        if not self.benchmark_results:
            return {"error": "No benchmark results available"}
        
        # Group results by benchmark type
        results_by_type = defaultdict(list)
        for result in self.benchmark_results:
            results_by_type[result.benchmark_type].append(result)
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'total_benchmarks': len(self.benchmark_results),
            'benchmark_types': list(results_by_type.keys()),
            'summary_by_type': {},
            'overall_performance': self._calculate_overall_performance(),
            'recommendations': self._generate_benchmark_recommendations(results_by_type)
        }
        
        # Generate summary for each benchmark type
        for benchmark_type, results in results_by_type.items():
            report['summary_by_type'][benchmark_type] = {
                'runs_count': len(results),
                'avg_performance_score': np.mean([r.performance_score for r in results]),
                'avg_latency_ms': np.mean([r.latency_ms for r in results]),
                'avg_accuracy': np.mean([r.accuracy_score for r in results]),
                'avg_consistency': np.mean([r.consistency_score for r in results]),
                'latest_result': max(results, key=lambda r: r.timestamp, default=None)
            }
        
        return report
    
    def _calculate_overall_performance(self) -> Dict[str, float]:
        """Calculate overall performance metrics"""
        if not self.benchmark_results:
            return {}
        
        return {
            'overall_score': np.mean([r.performance_score for r in self.benchmark_results]),
            'overall_latency': np.mean([r.latency_ms for r in self.benchmark_results]),
            'overall_accuracy': np.mean([r.accuracy_score for r in self.benchmark_results]),
            'overall_consistency': np.mean([r.consistency_score for r in self.benchmark_results])
        }
    
    def _generate_benchmark_recommendations(self, results_by_type: Dict[str, List[BenchmarkResult]]) -> List[str]:
        """Generate recommendations based on benchmark results"""
        recommendations = []
        
        for benchmark_type, results in results_by_type.items():
            if not results:
                continue
            
            latest_result = max(results, key=lambda r: r.timestamp)
            
            if latest_result.performance_score < 0.5:
                recommendations.append(f"Low performance in {benchmark_type} benchmark - consider optimization")
            
            if latest_result.latency_ms > 5000:
                recommendations.append(f"High latency in {benchmark_type} benchmark - optimize for speed")
            
            if latest_result.accuracy_score < 0.7:
                recommendations.append(f"Low accuracy in {benchmark_type} benchmark - improve response quality")
            
            if latest_result.consistency_score < 0.6:
                recommendations.append(f"Low consistency in {benchmark_type} benchmark - stabilize responses")
        
        return recommendations

class InteractionAnalyzer:
    """Analyzes user interaction patterns for research insights"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.interaction_buffer = deque(maxlen=config.get('buffer_size', 10000))
        self.pattern_cache = {}
        self.analysis_results = []
    
    def record_interaction(self, 
                          interaction_type: InteractionType,
                          interaction_data: Dict[str, Any],
                          timestamp: Optional[datetime] = None):
        """Record an interaction for analysis"""
        
        interaction_record = {
            'id': str(uuid.uuid4()),
            'type': interaction_type.value,
            'timestamp': (timestamp or datetime.now()).isoformat(),
            'data': interaction_data,
            'session_id': interaction_data.get('session_id', 'unknown')
        }
        
        self.interaction_buffer.append(interaction_record)
    
    def analyze_patterns(self, 
                        time_window: timedelta = timedelta(hours=24),
                        min_frequency: int = 2) -> List[InteractionPattern]:
        """Analyze interaction patterns within time window"""
        
        current_time = datetime.now()
        window_start = current_time - time_window
        
        # Filter interactions within time window
        recent_interactions = [
            interaction for interaction in self.interaction_buffer
            if datetime.fromisoformat(interaction['timestamp']) >= window_start
        ]
        
        if not recent_interactions:
            return []
        
        patterns = []
        
        # Analyze query patterns
        query_patterns = self._analyze_query_patterns(recent_interactions, min_frequency)
        patterns.extend(query_patterns)
        
        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(recent_interactions, min_frequency)
        patterns.extend(temporal_patterns)
        
        # Analyze error patterns
        error_patterns = self._analyze_error_patterns(recent_interactions, min_frequency)
        patterns.extend(error_patterns)
        
        # Cache results
        cache_key = f"{window_start.isoformat()}_{time_window.total_seconds()}_{min_frequency}"
        self.pattern_cache[cache_key] = patterns
        
        return patterns
    
    def _analyze_query_patterns(self, interactions: List[Dict[str, Any]], min_frequency: int) -> List[InteractionPattern]:
        """Analyze patterns in query types and content"""
        patterns = []
        
        # Group by query similarity
        query_groups = defaultdict(list)
        
        for interaction in interactions:
            if interaction['type'] == InteractionType.QUERY_RESPONSE.value:
                query_text = interaction['data'].get('query', '')
                
                # Simple clustering by first few words
                query_key = ' '.join(query_text.split()[:3]).lower()
                query_groups[query_key].append(interaction)
        
        # Identify frequent patterns
        for query_key, group_interactions in query_groups.items():
            if len(group_interactions) >= min_frequency:
                
                # Analyze temporal distribution
                temporal_dist = defaultdict(int)
                for interaction in group_interactions:
                    hour = datetime.fromisoformat(interaction['timestamp']).hour
                    temporal_dist[f"hour_{hour}"] += 1
                
                pattern = InteractionPattern(
                    pattern_id=str(uuid.uuid4()),
                    pattern_type="frequent_query",
                    frequency=len(group_interactions),
                    confidence=min(1.0, len(group_interactions) / 10.0),
                    example_interactions=group_interactions[:3],
                    temporal_distribution=dict(temporal_dist),
                    user_segments=[interaction['session_id'] for interaction in group_interactions]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _analyze_temporal_patterns(self, interactions: List[Dict[str, Any]], min_frequency: int) -> List[InteractionPattern]:
        """Analyze temporal patterns in interactions"""
        patterns = []
        
        # Group by hour of day
        hourly_activity = defaultdict(list)
        for interaction in interactions:
            hour = datetime.fromisoformat(interaction['timestamp']).hour
            hourly_activity[hour].append(interaction)
        
        # Find peak activity hours
        peak_hours = [(hour, len(interactions)) for hour, interactions in hourly_activity.items() 
                     if len(interactions) >= min_frequency]
        
        if peak_hours:
            peak_hours.sort(key=lambda x: x[1], reverse=True)
            top_hour = peak_hours[0]
            
            pattern = InteractionPattern(
                pattern_id=str(uuid.uuid4()),
                pattern_type="peak_activity_hour",
                frequency=top_hour[1],
                confidence=top_hour[1] / len(interactions),
                example_interactions=hourly_activity[top_hour[0]][:3],
                temporal_distribution={f"hour_{top_hour[0]}": top_hour[1]},
                user_segments=list(set(i['session_id'] for i in hourly_activity[top_hour[0]]))
            )
            patterns.append(pattern)
        
        return patterns
    
    def _analyze_error_patterns(self, interactions: List[Dict[str, Any]], min_frequency: int) -> List[InteractionPattern]:
        """Analyze error patterns in interactions"""
        patterns = []
        
        # Collect error interactions
        error_interactions = []
        for interaction in interactions:
            data = interaction.get('data', {})
            if 'error' in data or 'error_log' in data:
                error_interactions.append(interaction)
        
        if len(error_interactions) >= min_frequency:
            # Group errors by type
            error_types = defaultdict(list)
            for interaction in error_interactions:
                error_info = interaction['data'].get('error', 'unknown_error')
                error_key = str(error_info)[:50]  # Truncate for grouping
                error_types[error_key].append(interaction)
            
            # Create patterns for frequent error types
            for error_type, error_group in error_types.items():
                if len(error_group) >= min_frequency:
                    temporal_dist = defaultdict(int)
                    for interaction in error_group:
                        hour = datetime.fromisoformat(interaction['timestamp']).hour
                        temporal_dist[f"hour_{hour}"] += 1
                    
                    pattern = InteractionPattern(
                        pattern_id=str(uuid.uuid4()),
                        pattern_type="frequent_error",
                        frequency=len(error_group),
                        confidence=len(error_group) / len(error_interactions),
                        example_interactions=error_group[:3],
                        temporal_distribution=dict(temporal_dist),
                        user_segments=list(set(i['session_id'] for i in error_group))
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def generate_interaction_report(self) -> Dict[str, Any]:
        """Generate comprehensive interaction analysis report"""
        
        # Analyze recent patterns
        recent_patterns = self.analyze_patterns()
        
        # Basic statistics
        total_interactions = len(self.interaction_buffer)
        interaction_types = defaultdict(int)
        session_counts = defaultdict(int)
        
        for interaction in self.interaction_buffer:
            interaction_types[interaction['type']] += 1
            session_counts[interaction['session_id']] += 1
        
        report = {
            'report_id': str(uuid.uuid4()),
            'generated_at': datetime.now().isoformat(),
            'analysis_period': {
                'total_interactions': total_interactions,
                'earliest_interaction': min((i['timestamp'] for i in self.interaction_buffer), default=None),
                'latest_interaction': max((i['timestamp'] for i in self.interaction_buffer), default=None)
            },
            'interaction_types': dict(interaction_types),
            'session_statistics': {
                'unique_sessions': len(session_counts),
                'avg_interactions_per_session': np.mean(list(session_counts.values())) if session_counts else 0,
                'max_interactions_per_session': max(session_counts.values()) if session_counts else 0
            },
            'detected_patterns': len(recent_patterns),
            'pattern_details': [asdict(pattern) for pattern in recent_patterns],
            'recommendations': self._generate_interaction_recommendations(recent_patterns)
        }
        
        return report
    
    def _generate_interaction_recommendations(self, patterns: List[InteractionPattern]) -> List[str]:
        """Generate recommendations based on interaction patterns"""
        recommendations = []
        
        for pattern in patterns:
            if pattern.pattern_type == "frequent_query" and pattern.frequency > 10:
                recommendations.append(f"Consider optimizing for common query pattern: {pattern.pattern_id}")
            
            if pattern.pattern_type == "frequent_error" and pattern.frequency > 5:
                recommendations.append(f"Address recurring error pattern: {pattern.pattern_id}")
            
            if pattern.pattern_type == "peak_activity_hour":
                peak_hour = max(pattern.temporal_distribution.items(), key=lambda x: x[1])[0]
                recommendations.append(f"Scale resources for peak activity during {peak_hour}")
        
        return recommendations

class PatternRecognizer:
    """Advanced pattern recognition for cognitive behavior analysis"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pattern_models = {}
        self.recognized_patterns = []
        self.learning_enabled = config.get('learning_enabled', True)
    
    def train_pattern_model(self, 
                           pattern_name: str,
                           training_data: List[Dict[str, Any]],
                           model_type: str = 'statistical') -> bool:
        """Train a pattern recognition model"""
        
        try:
            if model_type == 'statistical':
                model = self._train_statistical_model(training_data)
            elif model_type == 'sequence':
                model = self._train_sequence_model(training_data)
            elif model_type == 'clustering':
                model = self._train_clustering_model(training_data)
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self.pattern_models[pattern_name] = {
                'model': model,
                'type': model_type,
                'trained_at': datetime.now().isoformat(),
                'training_size': len(training_data)
            }
            
            logger.info(f"Trained pattern model '{pattern_name}' with {len(training_data)} examples")
            return True
            
        except Exception as e:
            logger.error(f"Failed to train pattern model '{pattern_name}': {e}")
            return False
    
    def recognize_patterns(self, data_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Recognize patterns in data stream using trained models"""
        
        recognized = []
        
        for pattern_name, model_info in self.pattern_models.items():
            try:
                model = model_info['model']
                model_type = model_info['type']
                
                if model_type == 'statistical':
                    pattern_instances = self._apply_statistical_model(model, data_stream)
                elif model_type == 'sequence':
                    pattern_instances = self._apply_sequence_model(model, data_stream)
                elif model_type == 'clustering':
                    pattern_instances = self._apply_clustering_model(model, data_stream)
                else:
                    continue
                
                for instance in pattern_instances:
                    recognized_pattern = {
                        'pattern_name': pattern_name,
                        'model_type': model_type,
                        'instance': instance,
                        'confidence': instance.get('confidence', 0.5),
                        'timestamp': datetime.now().isoformat()
                    }
                    recognized.append(recognized_pattern)
                    
            except Exception as e:
                logger.error(f"Error applying pattern model '{pattern_name}': {e}")
        
        self.recognized_patterns.extend(recognized)
        return recognized
    
    def _train_statistical_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train statistical pattern model"""
        
        # Extract features from training data
        features = []
        for data_point in training_data:
            feature_vector = self._extract_features(data_point)
            features.append(feature_vector)
        
        if not features:
            return {'type': 'empty', 'features': []}
        
        # Calculate statistical properties
        features_array = np.array(features)
        
        model = {
            'type': 'statistical',
            'mean': np.mean(features_array, axis=0).tolist(),
            'std': np.std(features_array, axis=0).tolist(),
            'min': np.min(features_array, axis=0).tolist(),
            'max': np.max(features_array, axis=0).tolist(),
            'feature_count': len(features[0]),
            'sample_count': len(features)
        }
        
        return model
    
    def _train_sequence_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train sequence pattern model"""
        
        # Extract sequences from training data
        sequences = []
        for data_point in training_data:
            if 'sequence' in data_point:
                sequences.append(data_point['sequence'])
            elif 'events' in data_point:
                sequences.append(data_point['events'])
        
        # Build n-gram model (simplified)
        ngrams = defaultdict(int)
        for sequence in sequences:
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i + 1])
                ngrams[bigram] += 1
        
        # Convert to probabilities
        total_ngrams = sum(ngrams.values())
        ngram_probs = {k: v / total_ngrams for k, v in ngrams.items()}
        
        model = {
            'type': 'sequence',
            'ngrams': dict(ngrams),
            'probabilities': ngram_probs,
            'sequence_lengths': [len(seq) for seq in sequences],
            'unique_events': list(set(event for seq in sequences for event in seq))
        }
        
        return model
    
    def _train_clustering_model(self, training_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train clustering pattern model"""
        
        # Extract features for clustering
        features = []
        for data_point in training_data:
            feature_vector = self._extract_features(data_point)
            features.append(feature_vector)
        
        if not features:
            return {'type': 'empty_clustering', 'clusters': []}
        
        # Simple k-means clustering (simplified implementation)
        k = min(3, len(features))  # Max 3 clusters
        features_array = np.array(features)
        
        # Random initialization
        centroids = []
        for _ in range(k):
            centroid = features_array[np.random.randint(len(features_array))]
            centroids.append(centroid)
        
        # Simple clustering iterations
        for _ in range(10):  # Fixed iterations
            clusters = [[] for _ in range(k)]
            
            # Assign points to clusters
            for feature in features_array:
                distances = [np.linalg.norm(feature - centroid) for centroid in centroids]
                closest_cluster = np.argmin(distances)
                clusters[closest_cluster].append(feature)
            
            # Update centroids
            for i, cluster in enumerate(clusters):
                if cluster:
                    centroids[i] = np.mean(cluster, axis=0)
        
        model = {
            'type': 'clustering',
            'centroids': [centroid.tolist() for centroid in centroids],
            'cluster_count': k,
            'feature_count': len(features[0])
        }
        
        return model
    
    def _extract_features(self, data_point: Dict[str, Any]) -> List[float]:
        """Extract numerical features from data point"""
        features = []
        
        # Basic features
        if isinstance(data_point, dict):
            features.append(len(data_point))  # Number of fields
            
            for key, value in data_point.items():
                if isinstance(value, str):
                    features.extend([
                        len(value),  # String length
                        len(value.split()),  # Word count
                        value.count(' '),  # Space count
                    ])
                elif isinstance(value, (int, float)):
                    features.append(float(value))
                elif isinstance(value, list):
                    features.append(len(value))
                elif isinstance(value, dict):
                    features.append(len(value))
        
        # Pad or truncate to fixed size
        target_size = 20
        while len(features) < target_size:
            features.append(0.0)
        
        return features[:target_size]
    
    def _apply_statistical_model(self, model: Dict[str, Any], data_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply statistical model to detect patterns"""
        
        if model['type'] == 'empty':
            return []
        
        pattern_instances = []
        
        for data_point in data_stream:
            features = self._extract_features(data_point)
            
            if len(features) != model['feature_count']:
                continue
            
            # Calculate statistical distance from model
            mean = np.array(model['mean'])
            std = np.array(model['std'])
            
            # Z-score calculation
            z_scores = np.abs((np.array(features) - mean) / (std + 1e-6))
            max_z_score = np.max(z_scores)
            
            # Consider it a pattern if within 2 standard deviations
            if max_z_score <= 2.0:
                confidence = 1.0 - (max_z_score / 2.0)
                pattern_instances.append({
                    'data_point': data_point,
                    'confidence': confidence,
                    'z_score': float(max_z_score),
                    'match_type': 'statistical'
                })
        
        return pattern_instances
    
    def _apply_sequence_model(self, model: Dict[str, Any], data_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply sequence model to detect patterns"""
        
        pattern_instances = []
        
        for data_point in data_stream:
            # Extract sequence from data point
            sequence = None
            if 'sequence' in data_point:
                sequence = data_point['sequence']
            elif 'events' in data_point:
                sequence = data_point['events']
            
            if not sequence or len(sequence) < 2:
                continue
            
            # Calculate sequence probability
            total_prob = 0.0
            valid_bigrams = 0
            
            for i in range(len(sequence) - 1):
                bigram = (sequence[i], sequence[i + 1])
                if bigram in model['probabilities']:
                    total_prob += model['probabilities'][bigram]
                    valid_bigrams += 1
            
            if valid_bigrams > 0:
                avg_prob = total_prob / valid_bigrams
                if avg_prob > 0.1:  # Threshold for pattern recognition
                    pattern_instances.append({
                        'data_point': data_point,
                        'confidence': avg_prob,
                        'sequence_length': len(sequence),
                        'match_type': 'sequence'
                    })
        
        return pattern_instances
    
    def _apply_clustering_model(self, model: Dict[str, Any], data_stream: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Apply clustering model to detect patterns"""
        
        if model['type'] == 'empty_clustering':
            return []
        
        pattern_instances = []
        centroids = [np.array(centroid) for centroid in model['centroids']]
        
        for data_point in data_stream:
            features = self._extract_features(data_point)
            
            if len(features) != model['feature_count']:
                continue
            
            feature_vector = np.array(features)
            
            # Find closest centroid
            distances = [np.linalg.norm(feature_vector - centroid) for centroid in centroids]
            min_distance = min(distances)
            closest_cluster = np.argmin(distances)
            
            # Consider it a pattern if close enough to a cluster
            distance_threshold = 5.0  # Adjust based on feature scale
            if min_distance <= distance_threshold:
                confidence = 1.0 - (min_distance / distance_threshold)
                pattern_instances.append({
                    'data_point': data_point,
                    'confidence': confidence,
                    'cluster_id': int(closest_cluster),
                    'distance': float(min_distance),
                    'match_type': 'clustering'
                })
        
        return pattern_instances
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of recognized patterns"""
        
        if not self.recognized_patterns:
            return {"message": "No patterns recognized yet"}
        
        # Group by pattern name
        patterns_by_name = defaultdict(list)
        for pattern in self.recognized_patterns:
            patterns_by_name[pattern['pattern_name']].append(pattern)
        
        summary = {
            'total_patterns_recognized': len(self.recognized_patterns),
            'unique_pattern_types': len(patterns_by_name),
            'pattern_details': {},
            'models_trained': len(self.pattern_models)
        }
        
        for pattern_name, instances in patterns_by_name.items():
            summary['pattern_details'][pattern_name] = {
                'instance_count': len(instances),
                'avg_confidence': np.mean([p['confidence'] for p in instances]),
                'latest_occurrence': max(p['timestamp'] for p in instances)
            }
        
        return summary