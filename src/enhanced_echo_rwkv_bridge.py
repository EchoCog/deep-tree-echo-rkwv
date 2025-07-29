"""
Enhanced Echo RWKV Bridge

Multi-backend architecture with automatic selection and performance monitoring
for the Deep Tree Echo framework.
"""

import logging
import time
import threading
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BackendType(Enum):
    """Available backend types"""
    RWKV_CPP = "rwkv_cpp"
    SIMPLE_RWKV = "simple_rwkv"
    ECHO_RWKV = "echo_rwkv"
    MOCK = "mock"

@dataclass
class PerformanceMetrics:
    """Performance tracking metrics"""
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    total_latency: float = 0.0
    min_latency: float = float('inf')
    max_latency: float = 0.0
    cache_hits: int = 0
    backend_switches: int = 0
    last_request_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.requests_total == 0:
            return 0.0
        return self.requests_successful / self.requests_total
    
    @property
    def average_latency(self) -> float:
        """Calculate average latency"""
        if self.requests_successful == 0:
            return 0.0
        return self.total_latency / self.requests_successful
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary"""
        return {
            'requests_total': self.requests_total,
            'requests_successful': self.requests_successful,
            'requests_failed': self.requests_failed,
            'success_rate': self.success_rate,
            'average_latency': self.average_latency,
            'min_latency': self.min_latency if self.min_latency != float('inf') else 0.0,
            'max_latency': self.max_latency,
            'cache_hits': self.cache_hits,
            'backend_switches': self.backend_switches,
            'last_request_time': self.last_request_time
        }

@dataclass
class BackendConfig:
    """Configuration for a backend"""
    enabled: bool = True
    priority: int = 1
    timeout: float = 30.0
    max_retries: int = 3
    health_check_interval: float = 60.0
    
class RWKVBackend(ABC):
    """Abstract base class for RWKV backends"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the backend"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if backend is available"""
        pass
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get backend information"""
        pass
    
    @abstractmethod
    def get_backend_type(self) -> BackendType:
        """Get the backend type"""
        pass

class RWKVCppBackend(RWKVBackend):
    """RWKV C++ backend wrapper"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._interface = None
        self._bridge = None
        
    def initialize(self) -> bool:
        """Initialize the C++ backend"""
        try:
            from rwkv_cpp_integration import RWKVCppInterface, RWKVCppBridge
            self._interface = RWKVCppInterface()
            self._bridge = RWKVCppBridge()
            return self._interface.initialize() and self._bridge.initialize()
        except ImportError:
            logger.warning("RWKV C++ integration not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RWKV C++ backend: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if C++ backend is available"""
        return (self._interface is not None and 
                self._interface.is_available() and 
                self._bridge is not None and 
                self._bridge.is_available())
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using C++ backend"""
        if not self.is_available():
            return None
        try:
            return self._bridge.process_request(prompt, **kwargs)
        except Exception as e:
            logger.error(f"C++ backend generation error: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get C++ backend info"""
        info = {'backend_type': 'rwkv_cpp', 'available': self.is_available()}
        if self._bridge:
            info.update(self._bridge.get_info())
        return info
    
    def get_backend_type(self) -> BackendType:
        return BackendType.RWKV_CPP

class SimpleRWKVBackend(RWKVBackend):
    """Simple RWKV backend wrapper"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._integration = None
        
    def initialize(self) -> bool:
        """Initialize the simple RWKV backend"""
        try:
            from simple_rwkv_integration import SimpleRWKVIntegration
            self._integration = SimpleRWKVIntegration()
            return self._integration.initialize()
        except ImportError:
            logger.warning("Simple RWKV integration not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Simple RWKV backend: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if simple backend is available"""
        return self._integration is not None and self._integration.is_available()
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using simple backend"""
        if not self.is_available():
            return None
        try:
            return self._integration.generate(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Simple backend generation error: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get simple backend info"""
        info = {'backend_type': 'simple_rwkv', 'available': self.is_available()}
        if self._integration:
            info.update(self._integration.get_info())
        return info
    
    def get_backend_type(self) -> BackendType:
        return BackendType.SIMPLE_RWKV

class EchoRWKVBackend(RWKVBackend):
    """Echo RWKV backend wrapper"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        self._bridge = None
        
    def initialize(self) -> bool:
        """Initialize the Echo RWKV backend"""
        try:
            # Import the existing echo_rwkv_bridge
            import sys
            import os
            sys.path.insert(0, os.path.dirname(__file__))
            from echo_rwkv_bridge import EchoRWKVBridge
            
            self._bridge = EchoRWKVBridge()
            return True
        except ImportError:
            logger.warning("Echo RWKV bridge not available")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Echo RWKV backend: {e}")
            return False
    
    def is_available(self) -> bool:
        """Check if Echo backend is available"""
        return self._bridge is not None
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using Echo backend"""
        if not self.is_available():
            return None
        try:
            # Use the existing bridge method
            result = self._bridge.process_with_rwkv(prompt)
            return result.get('response', '')
        except Exception as e:
            logger.error(f"Echo backend generation error: {e}")
            return None
    
    def get_info(self) -> Dict[str, Any]:
        """Get Echo backend info"""
        return {
            'backend_type': 'echo_rwkv',
            'available': self.is_available(),
            'bridge': 'EchoRWKVBridge'
        }
    
    def get_backend_type(self) -> BackendType:
        return BackendType.ECHO_RWKV

class MockBackend(RWKVBackend):
    """Mock backend for testing and fallback"""
    
    def __init__(self, config: Optional[BackendConfig] = None):
        self.config = config or BackendConfig()
        
    def initialize(self) -> bool:
        """Mock initialization always succeeds"""
        return True
    
    def is_available(self) -> bool:
        """Mock backend is always available"""
        return True
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate mock response"""
        max_tokens = kwargs.get('max_tokens', 50)
        return f"[Mock Response] Echo of: {prompt[:max_tokens-20]}..."
    
    def get_info(self) -> Dict[str, Any]:
        """Get mock backend info"""
        return {
            'backend_type': 'mock',
            'available': True,
            'purpose': 'Testing and fallback'
        }
    
    def get_backend_type(self) -> BackendType:
        return BackendType.MOCK

class EnhancedEchoRWKVBridge:
    """
    Enhanced bridge with multi-backend support, automatic selection,
    and comprehensive performance monitoring.
    """
    
    def __init__(self):
        """Initialize the enhanced bridge"""
        self._backends: Dict[BackendType, RWKVBackend] = {}
        self._backend_configs: Dict[BackendType, BackendConfig] = {}
        self._metrics: Dict[BackendType, PerformanceMetrics] = {}
        self._current_backend: Optional[BackendType] = None
        self._lock = threading.Lock()
        self._cache: Dict[str, Tuple[str, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes
        
        # Initialize backends
        self._setup_backends()
    
    def _setup_backends(self):
        """Set up all available backends"""
        # Configure backend priorities
        self._backend_configs = {
            BackendType.RWKV_CPP: BackendConfig(priority=1, enabled=True),
            BackendType.SIMPLE_RWKV: BackendConfig(priority=2, enabled=True),
            BackendType.ECHO_RWKV: BackendConfig(priority=3, enabled=True),
            BackendType.MOCK: BackendConfig(priority=4, enabled=True)
        }
        
        # Initialize backends
        backend_classes = {
            BackendType.RWKV_CPP: RWKVCppBackend,
            BackendType.SIMPLE_RWKV: SimpleRWKVBackend,
            BackendType.ECHO_RWKV: EchoRWKVBackend,
            BackendType.MOCK: MockBackend
        }
        
        for backend_type, backend_class in backend_classes.items():
            try:
                config = self._backend_configs[backend_type]
                backend = backend_class(config)
                if backend.initialize():
                    self._backends[backend_type] = backend
                    self._metrics[backend_type] = PerformanceMetrics()
                    logger.info(f"Initialized backend: {backend_type.value}")
                else:
                    logger.warning(f"Failed to initialize backend: {backend_type.value}")
            except Exception as e:
                logger.error(f"Error setting up backend {backend_type.value}: {e}")
    
    def _select_best_backend(self) -> Optional[BackendType]:
        """Select the best available backend based on priority and availability"""
        available_backends = [
            (backend_type, config) 
            for backend_type, config in self._backend_configs.items()
            if (config.enabled and 
                backend_type in self._backends and 
                self._backends[backend_type].is_available())
        ]
        
        if not available_backends:
            return None
        
        # Sort by priority (lower number = higher priority)
        available_backends.sort(key=lambda x: x[1].priority)
        return available_backends[0][0]
    
    def _check_cache(self, prompt: str) -> Optional[str]:
        """Check cache for existing response"""
        cache_key = hash(prompt)
        if cache_key in self._cache:
            response, timestamp = self._cache[cache_key]
            if time.time() - timestamp < self._cache_ttl:
                return response
            else:
                # Cache expired
                del self._cache[cache_key]
        return None
    
    def _update_cache(self, prompt: str, response: str):
        """Update cache with new response"""
        cache_key = hash(prompt)
        self._cache[cache_key] = (response, time.time())
        
        # Simple cache cleanup - remove old entries if cache gets too large
        if len(self._cache) > 1000:
            current_time = time.time()
            expired_keys = [
                key for key, (_, timestamp) in self._cache.items()
                if current_time - timestamp > self._cache_ttl
            ]
            for key in expired_keys:
                del self._cache[key]
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using the best available backend"""
        start_time = time.time()
        
        # Check cache first
        cached_response = self._check_cache(prompt)
        if cached_response:
            with self._lock:
                for metrics in self._metrics.values():
                    metrics.cache_hits += 1
            return cached_response
        
        # Select backend
        selected_backend = self._select_best_backend()
        if not selected_backend:
            logger.error("No available backends")
            return None
        
        # Track backend switches
        if self._current_backend != selected_backend:
            if self._current_backend is not None:
                self._metrics[selected_backend].backend_switches += 1
            self._current_backend = selected_backend
        
        # Generate response
        backend = self._backends[selected_backend]
        metrics = self._metrics[selected_backend]
        
        try:
            response = backend.generate(prompt, **kwargs)
            latency = time.time() - start_time
            
            # Update metrics
            with self._lock:
                metrics.requests_total += 1
                metrics.last_request_time = time.time()
                
                if response:
                    metrics.requests_successful += 1
                    metrics.total_latency += latency
                    metrics.min_latency = min(metrics.min_latency, latency)
                    metrics.max_latency = max(metrics.max_latency, latency)
                    
                    # Update cache
                    self._update_cache(prompt, response)
                else:
                    metrics.requests_failed += 1
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating with backend {selected_backend.value}: {e}")
            with self._lock:
                metrics.requests_total += 1
                metrics.requests_failed += 1
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive status information"""
        status = {
            'current_backend': self._current_backend.value if self._current_backend else None,
            'available_backends': [],
            'backend_metrics': {},
            'cache_size': len(self._cache),
            'total_requests': sum(m.requests_total for m in self._metrics.values())
        }
        
        for backend_type, backend in self._backends.items():
            backend_info = {
                'type': backend_type.value,
                'available': backend.is_available(),
                'priority': self._backend_configs[backend_type].priority,
                'enabled': self._backend_configs[backend_type].enabled
            }
            backend_info.update(backend.get_info())
            
            if backend.is_available():
                status['available_backends'].append(backend_type.value)
            
            status['backend_metrics'][backend_type.value] = self._metrics[backend_type].to_dict()
        
        return status
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all backends"""
        total_requests = sum(m.requests_total for m in self._metrics.values())
        total_successful = sum(m.requests_successful for m in self._metrics.values())
        total_cache_hits = sum(m.cache_hits for m in self._metrics.values())
        
        if total_requests == 0:
            return {
                'overall_success_rate': 0.0,
                'overall_cache_hit_rate': 0.0,
                'total_requests': 0,
                'backend_performance': {}
            }
        
        backend_performance = {}
        for backend_type, metrics in self._metrics.items():
            if metrics.requests_total > 0:
                backend_performance[backend_type.value] = {
                    'success_rate': metrics.success_rate,
                    'average_latency': metrics.average_latency,
                    'request_share': metrics.requests_total / total_requests
                }
        
        return {
            'overall_success_rate': total_successful / total_requests,
            'overall_cache_hit_rate': total_cache_hits / total_requests,
            'total_requests': total_requests,
            'backend_performance': backend_performance
        }
    
    def clear_cache(self):
        """Clear the response cache"""
        with self._lock:
            self._cache.clear()
            logger.info("Response cache cleared")
    
    def reset_metrics(self):
        """Reset all performance metrics"""
        with self._lock:
            for backend_type in self._metrics:
                self._metrics[backend_type] = PerformanceMetrics()
            logger.info("Performance metrics reset")

# Global instance for easy access
_enhanced_bridge = None

def get_enhanced_bridge() -> EnhancedEchoRWKVBridge:
    """Get the global enhanced bridge instance"""
    global _enhanced_bridge
    if _enhanced_bridge is None:
        _enhanced_bridge = EnhancedEchoRWKVBridge()
    return _enhanced_bridge