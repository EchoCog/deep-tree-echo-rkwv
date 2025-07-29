"""
RWKV C++ Bridge

Bridge component that connects the RWKV C++ interface with the Deep Tree Echo framework.
"""

import logging
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

from .rwkv_cpp_interface import RWKVCppInterface, RWKVModelConfig, RWKVGenerationConfig

logger = logging.getLogger(__name__)

@dataclass
class BridgeConfig:
    """Configuration for the RWKV C++ bridge"""
    enable_fallback: bool = True
    max_retries: int = 3
    timeout_seconds: float = 30.0
    cache_size: int = 1000

class RWKVCppBridge:
    """
    Bridge that manages RWKV C++ interface operations and provides
    integration with the Deep Tree Echo cognitive architecture.
    """
    
    def __init__(self, config: Optional[BridgeConfig] = None):
        """Initialize the RWKV C++ bridge"""
        self.config = config or BridgeConfig()
        self._interface = None
        self._lock = threading.Lock()
        self._cache = {}
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'cache_hits': 0
        }
    
    def initialize(self, model_config: Optional[RWKVModelConfig] = None) -> bool:
        """Initialize the bridge with optional model configuration"""
        try:
            with self._lock:
                self._interface = RWKVCppInterface(model_config)
                success = self._interface.initialize()
                
                if success:
                    logger.info("RWKV C++ bridge initialized successfully")
                else:
                    logger.warning("RWKV C++ bridge initialization failed")
                    
                return success
                
        except Exception as e:
            logger.error(f"Failed to initialize RWKV C++ bridge: {e}")
            return False
    
    def process_request(self, 
                       prompt: str, 
                       generation_config: Optional[RWKVGenerationConfig] = None) -> Optional[str]:
        """Process a text generation request"""
        self._stats['requests'] += 1
        
        # Check cache first
        cache_key = self._create_cache_key(prompt, generation_config)
        if cache_key in self._cache:
            self._stats['cache_hits'] += 1
            return self._cache[cache_key]
        
        try:
            if not self._interface or not self._interface.is_available():
                logger.warning("RWKV C++ interface not available")
                self._stats['failures'] += 1
                return None
            
            result = self._interface.generate(prompt, generation_config)
            
            if result:
                self._stats['successes'] += 1
                self._update_cache(cache_key, result)
                return result
            else:
                self._stats['failures'] += 1
                return None
                
        except Exception as e:
            logger.error(f"Error processing request: {e}")
            self._stats['failures'] += 1
            return None
    
    def _create_cache_key(self, prompt: str, config: Optional[RWKVGenerationConfig]) -> str:
        """Create a cache key for the request"""
        config_str = ""
        if config:
            config_str = f"_{config.max_tokens}_{config.temperature}_{config.top_p}"
        return f"{hash(prompt)}{config_str}"
    
    def _update_cache(self, key: str, value: str):
        """Update the cache with LRU-style eviction"""
        if len(self._cache) >= self.config.cache_size:
            # Simple FIFO eviction
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[key] = value
    
    def is_available(self) -> bool:
        """Check if the bridge is available for processing"""
        return self._interface is not None and self._interface.is_available()
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the bridge and interface"""
        info = {
            'bridge_available': self.is_available(),
            'cache_size': len(self._cache),
            'stats': self._stats.copy()
        }
        
        if self._interface:
            info.update(self._interface.get_model_info())
            
        return info
    
    def get_version(self) -> str:
        """Get version information"""
        if self._interface:
            return f"Bridge-1.0.0 + {self._interface.get_version()}"
        return "Bridge-1.0.0 (Interface not initialized)"
    
    def clear_cache(self):
        """Clear the request cache"""
        with self._lock:
            self._cache.clear()
            logger.info("Cache cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get detailed statistics"""
        stats = self._stats.copy()
        if stats['requests'] > 0:
            stats['success_rate'] = stats['successes'] / stats['requests']
            stats['cache_hit_rate'] = stats['cache_hits'] / stats['requests']
        else:
            stats['success_rate'] = 0.0
            stats['cache_hit_rate'] = 0.0
            
        return stats