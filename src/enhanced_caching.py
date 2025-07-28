"""
Multi-Level Enhanced Caching System for Phase 3 Scalability
Implements L1/L2/L3 caching with compression and cognitive optimization for P1-002.3
"""

import os
import time
import json
import gzip
import hashlib
import logging
import threading
from typing import Dict, Any, Optional, Union, Tuple, List
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import OrderedDict
import pickle

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Enhanced cache entry with metadata"""
    key: str
    value: Any
    created_at: float
    expires_at: Optional[float]
    access_count: int = 0
    last_accessed: float = None
    size_bytes: int = 0
    cache_level: str = "L1"
    content_type: str = "general"
    compression_ratio: float = 1.0
    cognitive_priority: float = 0.5  # 0-1 scale for cognitive importance

class CacheLevel:
    """Base class for cache levels"""
    
    def __init__(self, name: str, max_size_mb: int, eviction_policy: str = "lru"):
        self.name = name
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def get(self, key: str) -> Optional[CacheEntry]:
        """Get entry from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.expires_at and time.time() > entry.expires_at:
                    self.remove(key)
                    self.misses += 1
                    return None
                
                # Update access statistics
                entry.access_count += 1
                entry.last_accessed = time.time()
                
                # Move to end for LRU
                if self.eviction_policy == "lru":
                    self.cache.move_to_end(key)
                
                self.hits += 1
                return entry
            
            self.misses += 1
            return None
    
    def put(self, entry: CacheEntry) -> bool:
        """Put entry in cache"""
        with self.lock:
            # Check if we need to evict
            required_space = entry.size_bytes
            
            # If key exists, subtract old size
            if entry.key in self.cache:
                old_entry = self.cache[entry.key]
                required_space -= old_entry.size_bytes
            
            # Evict if necessary
            while (self.current_size_bytes + required_space > self.max_size_bytes and 
                   len(self.cache) > 0):
                if not self._evict_one():
                    break
            
            # Check if entry fits
            if required_space > self.max_size_bytes:
                logger.warning(f"Entry too large for {self.name}: {required_space} bytes")
                return False
            
            # Add/update entry
            if entry.key in self.cache:
                old_size = self.cache[entry.key].size_bytes
                self.current_size_bytes -= old_size
            
            entry.cache_level = self.name
            self.cache[entry.key] = entry
            self.current_size_bytes += entry.size_bytes
            
            return True
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache"""
        with self.lock:
            if key in self.cache:
                entry = self.cache.pop(key)
                self.current_size_bytes -= entry.size_bytes
                return True
            return False
    
    def _evict_one(self) -> bool:
        """Evict one entry based on policy"""
        if not self.cache:
            return False
        
        if self.eviction_policy == "lru":
            # Remove least recently used (first item)
            key, entry = self.cache.popitem(last=False)
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
            entry = self.cache.pop(key)
        elif self.eviction_policy == "cognitive_priority":
            # Remove entry with lowest cognitive priority
            key = min(self.cache.keys(), key=lambda k: self.cache[k].cognitive_priority)
            entry = self.cache.pop(key)
        else:
            # Default to FIFO
            key, entry = self.cache.popitem(last=False)
        
        self.current_size_bytes -= entry.size_bytes
        self.evictions += 1
        logger.debug(f"Evicted {key} from {self.name} (policy: {self.eviction_policy})")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = (self.hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "name": self.name,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": hit_rate,
                "evictions": self.evictions,
                "entries": len(self.cache),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "utilization": (self.current_size_bytes / self.max_size_bytes * 100)
            }

class EnhancedMultiLevelCache:
    """
    Enhanced multi-level caching system for Phase 3 scalability
    Implements L1/L2/L3 caching with cognitive optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        default_config = {
            "l1_size_mb": 64,    # Fast in-memory cache
            "l2_size_mb": 256,   # Larger in-memory cache
            "l3_size_mb": 512,   # Compressed cache
            "compression_enabled": True,
            "cognitive_optimization": True,
            "default_ttl_seconds": 300
        }
        
        self.config = {**default_config, **(config or {})}
        
        # Initialize cache levels
        self.l1_cache = CacheLevel("L1", self.config["l1_size_mb"], "lru")
        self.l2_cache = CacheLevel("L2", self.config["l2_size_mb"], "cognitive_priority")
        self.l3_cache = CacheLevel("L3", self.config["l3_size_mb"], "lfu")
        
        # Cognitive cache optimization
        self.cognitive_weights = {
            "conversation": 1.0,
            "memory_retrieval": 0.9,
            "reasoning_result": 0.8,
            "grammar_processing": 0.7,
            "general": 0.5
        }
        
        # Background cleanup thread
        self._cleanup_thread = None
        self._stop_cleanup = threading.Event()
        self.start_cleanup()
        
        logger.info("Enhanced multi-level cache initialized")
    
    def start_cleanup(self):
        """Start background cleanup thread"""
        if self._cleanup_thread is None or not self._cleanup_thread.is_alive():
            self._stop_cleanup.clear()
            self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
            self._cleanup_thread.start()
    
    def stop_cleanup(self):
        """Stop background cleanup thread"""
        self._stop_cleanup.set()
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=5)
    
    def _cleanup_loop(self):
        """Background cleanup of expired entries"""
        while not self._stop_cleanup.wait(60):  # Check every minute
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.error(f"Error in cache cleanup: {e}")
    
    def _cleanup_expired(self):
        """Remove expired entries from all cache levels"""
        current_time = time.time()
        
        for cache_level in [self.l1_cache, self.l2_cache, self.l3_cache]:
            with cache_level.lock:
                expired_keys = []
                for key, entry in cache_level.cache.items():
                    if entry.expires_at and current_time > entry.expires_at:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    cache_level.remove(key)
                    logger.debug(f"Expired entry removed: {key} from {cache_level.name}")
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache (checks all levels)"""
        # Try L1 first
        entry = self.l1_cache.get(key)
        if entry:
            logger.debug(f"Cache hit L1: {key}")
            return self._decompress_value(entry)
        
        # Try L2
        entry = self.l2_cache.get(key)
        if entry:
            logger.debug(f"Cache hit L2: {key}")
            # Promote to L1 if valuable
            if self._should_promote_to_l1(entry):
                self._promote_to_l1(entry)
            return self._decompress_value(entry)
        
        # Try L3
        entry = self.l3_cache.get(key)
        if entry:
            logger.debug(f"Cache hit L3: {key}")
            # Promote to L2 if valuable
            if self._should_promote_to_l2(entry):
                self._promote_to_l2(entry)
            return self._decompress_value(entry)
        
        logger.debug(f"Cache miss: {key}")
        return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[int] = None,
            content_type: str = "general", cognitive_priority: Optional[float] = None) -> bool:
        """Put value in cache"""
        
        # Calculate expiration
        expires_at = None
        if ttl_seconds is not None:
            expires_at = time.time() + ttl_seconds
        elif self.config["default_ttl_seconds"]:
            expires_at = time.time() + self.config["default_ttl_seconds"]
        
        # Calculate cognitive priority
        if cognitive_priority is None:
            cognitive_priority = self.cognitive_weights.get(content_type, 0.5)
        
        # Serialize and potentially compress value
        serialized_value, compression_ratio = self._serialize_and_compress(value)
        size_bytes = len(serialized_value) if isinstance(serialized_value, bytes) else len(str(serialized_value))
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=serialized_value,
            created_at=time.time(),
            expires_at=expires_at,
            size_bytes=size_bytes,
            content_type=content_type,
            compression_ratio=compression_ratio,
            cognitive_priority=cognitive_priority
        )
        
        # Determine optimal cache level
        target_level = self._determine_cache_level(entry)
        
        if target_level == "L1":
            success = self.l1_cache.put(entry)
        elif target_level == "L2":
            success = self.l2_cache.put(entry)
        else:
            success = self.l3_cache.put(entry)
        
        if success:
            logger.debug(f"Cached {key} in {target_level} (priority: {cognitive_priority:.2f})")
        
        return success
    
    def _serialize_and_compress(self, value: Any) -> Tuple[Union[str, bytes], float]:
        """Serialize and optionally compress value"""
        try:
            # Serialize
            if isinstance(value, (dict, list)):
                serialized = json.dumps(value).encode('utf-8')
            elif isinstance(value, str):
                serialized = value.encode('utf-8')
            else:
                serialized = pickle.dumps(value)
            
            original_size = len(serialized)
            
            # Compress if enabled and beneficial
            if (self.config["compression_enabled"] and 
                original_size > 1024):  # Only compress if > 1KB
                
                compressed = gzip.compress(serialized)
                compression_ratio = len(compressed) / original_size
                
                # Use compression if it saves significant space
                if compression_ratio < 0.8:
                    return compressed, compression_ratio
            
            return serialized, 1.0
            
        except Exception as e:
            logger.error(f"Error serializing value: {e}")
            return str(value).encode('utf-8'), 1.0
    
    def _decompress_value(self, entry: CacheEntry) -> Any:
        """Decompress and deserialize cached value"""
        try:
            value = entry.value
            
            # Decompress if needed
            if entry.compression_ratio < 1.0 and isinstance(value, bytes):
                try:
                    value = gzip.decompress(value)
                except:
                    pass  # Not compressed or corruption
            
            # Deserialize
            if isinstance(value, bytes):
                try:
                    # Try JSON first
                    return json.loads(value.decode('utf-8'))
                except:
                    try:
                        # Try pickle
                        return pickle.loads(value)
                    except:
                        # Return as string
                        return value.decode('utf-8')
            
            return value
            
        except Exception as e:
            logger.error(f"Error deserializing cached value: {e}")
            return None
    
    def _determine_cache_level(self, entry: CacheEntry) -> str:
        """Determine optimal cache level for entry"""
        
        # High priority items go to L1
        if entry.cognitive_priority > 0.8:
            return "L1"
        
        # Medium priority or frequently accessed go to L2
        if entry.cognitive_priority > 0.6 or entry.content_type in ["conversation", "memory_retrieval"]:
            return "L2"
        
        # Everything else goes to L3
        return "L3"
    
    def _should_promote_to_l1(self, entry: CacheEntry) -> bool:
        """Check if entry should be promoted to L1"""
        return (entry.access_count > 3 and 
                entry.cognitive_priority > 0.7 and
                entry.size_bytes < 100 * 1024)  # < 100KB
    
    def _should_promote_to_l2(self, entry: CacheEntry) -> bool:
        """Check if entry should be promoted to L2"""
        return (entry.access_count > 2 and 
                entry.cognitive_priority > 0.5)
    
    def _promote_to_l1(self, entry: CacheEntry):
        """Promote entry to L1 cache"""
        self.l1_cache.put(entry)
        logger.debug(f"Promoted {entry.key} to L1")
    
    def _promote_to_l2(self, entry: CacheEntry):
        """Promote entry to L2 cache"""
        self.l2_cache.put(entry)
        logger.debug(f"Promoted {entry.key} to L2")
    
    def invalidate(self, key: str):
        """Invalidate key from all cache levels"""
        self.l1_cache.remove(key)
        self.l2_cache.remove(key)
        self.l3_cache.remove(key)
        logger.debug(f"Invalidated {key} from all cache levels")
    
    def clear_all(self):
        """Clear all cache levels"""
        with self.l1_cache.lock, self.l2_cache.lock, self.l3_cache.lock:
            self.l1_cache.cache.clear()
            self.l1_cache.current_size_bytes = 0
            self.l2_cache.cache.clear()
            self.l2_cache.current_size_bytes = 0
            self.l3_cache.cache.clear()
            self.l3_cache.current_size_bytes = 0
        logger.info("Cleared all cache levels")
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        l1_stats = self.l1_cache.get_stats()
        l2_stats = self.l2_cache.get_stats()
        l3_stats = self.l3_cache.get_stats()
        
        total_hits = l1_stats["hits"] + l2_stats["hits"] + l3_stats["hits"]
        total_misses = l1_stats["misses"] + l2_stats["misses"] + l3_stats["misses"]
        total_requests = total_hits + total_misses
        
        overall_hit_rate = (total_hits / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "overall": {
                "hit_rate": overall_hit_rate,
                "total_requests": total_requests,
                "total_hits": total_hits,
                "total_misses": total_misses
            },
            "levels": {
                "L1": l1_stats,
                "L2": l2_stats,
                "L3": l3_stats
            },
            "config": self.config
        }

# Global cache instance
_cache: Optional[EnhancedMultiLevelCache] = None

def initialize_cache(config: Optional[Dict[str, Any]] = None) -> EnhancedMultiLevelCache:
    """Initialize global cache"""
    global _cache
    _cache = EnhancedMultiLevelCache(config)
    return _cache

def get_cache() -> Optional[EnhancedMultiLevelCache]:
    """Get global cache instance"""
    return _cache