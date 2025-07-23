"""
Multi-Level Caching Service
Provides distributed caching with intelligent cache management
"""

import os
import time
import json
import hashlib
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
CACHE_CONFIG = {
    'name': 'cache-service',
    'version': '1.0.0',
    'port': int(os.getenv('CACHE_SERVICE_PORT', 8002)),
    'max_memory_mb': int(os.getenv('MAX_CACHE_MEMORY_MB', 512)),
    'default_ttl_seconds': int(os.getenv('DEFAULT_TTL_SECONDS', 300)),
    'cleanup_interval_seconds': int(os.getenv('CLEANUP_INTERVAL', 60)),
    'eviction_policy': os.getenv('EVICTION_POLICY', 'lru'),  # lru, lfu, fifo
    'enable_compression': os.getenv('ENABLE_COMPRESSION', 'true').lower() == 'true',
    'max_key_size_bytes': int(os.getenv('MAX_KEY_SIZE_BYTES', 1024)),
    'max_value_size_bytes': int(os.getenv('MAX_VALUE_SIZE_BYTES', 1024 * 1024),)  # 1MB
}

# Data models
@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: datetime = None
    size_bytes: int = 0
    cache_level: str = "L1"  # L1, L2, L3

class CacheStats:
    """Cache statistics"""
    def __init__(self):
        self.total_requests = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.evictions = 0
        self.expired_entries = 0
        self.total_size_bytes = 0
        self.entry_count = 0
        
    @property
    def hit_ratio(self) -> float:
        return self.cache_hits / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def miss_ratio(self) -> float:
        return self.cache_misses / self.total_requests if self.total_requests > 0 else 0.0

# Pydantic models
class CacheRequest(BaseModel):
    key: str
    value: Any
    ttl_seconds: Optional[int] = None
    cache_level: str = Field(default="L1", pattern="^(L1|L2|L3)$")
    tags: List[str] = Field(default_factory=list)

class CacheResponse(BaseModel):
    key: str
    value: Any
    cache_hit: bool
    cache_level: str
    access_count: int
    age_seconds: float

class BulkCacheRequest(BaseModel):
    operations: List[Dict[str, Any]]
    
class CacheHealthResponse(BaseModel):
    status: str
    timestamp: datetime
    cache_stats: Dict[str, Any]
    memory_usage_mb: float
    entry_count: int

# Multi-level cache implementation
class MultiLevelCache:
    """Multi-level cache with L1 (memory), L2 (compressed), L3 (persistent)"""
    
    def __init__(self):
        # L1: Fast in-memory cache
        self.l1_cache: Dict[str, CacheEntry] = {}
        
        # L2: Compressed cache for larger items
        self.l2_cache: Dict[str, CacheEntry] = {}
        
        # L3: Persistent cache (simulated with dict for now)
        self.l3_cache: Dict[str, CacheEntry] = {}
        
        # Statistics per level
        self.l1_stats = CacheStats()
        self.l2_stats = CacheStats()
        self.l3_stats = CacheStats()
        
        # LRU tracking
        self.access_order: List[str] = []
        
        # Tag index for batch operations
        self.tag_index: Dict[str, set] = {}
    
    def _calculate_size(self, value: Any) -> int:
        """Calculate approximate size of value in bytes"""
        try:
            return len(json.dumps(value, default=str).encode('utf-8'))
        except:
            return len(str(value).encode('utf-8'))
    
    def _get_cache_level(self, key: str, preferred_level: str = "L1") -> Dict[str, CacheEntry]:
        """Get appropriate cache level for key"""
        if preferred_level == "L1":
            return self.l1_cache
        elif preferred_level == "L2":
            return self.l2_cache
        else:
            return self.l3_cache
    
    def _get_stats_for_level(self, level: str) -> CacheStats:
        """Get statistics object for cache level"""
        if level == "L1":
            return self.l1_stats
        elif level == "L2":
            return self.l2_stats
        else:
            return self.l3_stats
    
    def _compress_value(self, value: Any) -> Any:
        """Simulate compression (placeholder for actual compression)"""
        if CACHE_CONFIG['enable_compression']:
            # In real implementation, use gzip/lz4/snappy
            return {"compressed": True, "data": value}
        return value
    
    def _decompress_value(self, value: Any) -> Any:
        """Simulate decompression"""
        if isinstance(value, dict) and value.get("compressed"):
            return value["data"]
        return value
    
    def _evict_if_needed(self, target_level: str):
        """Evict entries if memory limit is reached"""
        cache = self._get_cache_level("", target_level)
        stats = self._get_stats_for_level(target_level)
        
        if stats.total_size_bytes > CACHE_CONFIG['max_memory_mb'] * 1024 * 1024:
            self._evict_entries(cache, stats, target_level)
    
    def _evict_entries(self, cache: Dict[str, CacheEntry], stats: CacheStats, level: str):
        """Evict entries based on configured policy"""
        eviction_count = max(1, len(cache) // 10)  # Evict 10% of entries
        
        if CACHE_CONFIG['eviction_policy'] == 'lru':
            # Sort by last access time
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: x[1].last_accessed or x[1].created_at
            )
        elif CACHE_CONFIG['eviction_policy'] == 'lfu':
            # Sort by access count
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: x[1].access_count
            )
        else:  # FIFO
            # Sort by creation time
            sorted_entries = sorted(
                cache.items(),
                key=lambda x: x[1].created_at
            )
        
        # Remove oldest/least accessed entries
        for key, entry in sorted_entries[:eviction_count]:
            self._remove_entry(key, cache, stats)
            logger.debug(f"Evicted entry {key} from {level}")
    
    def _remove_entry(self, key: str, cache: Dict[str, CacheEntry], stats: CacheStats):
        """Remove entry and update stats"""
        if key in cache:
            entry = cache[key]
            stats.total_size_bytes -= entry.size_bytes
            stats.entry_count -= 1
            stats.evictions += 1
            del cache[key]
            
            # Remove from access order
            if key in self.access_order:
                self.access_order.remove(key)
    
    async def get(self, key: str) -> Optional[CacheEntry]:
        """Get value from multi-level cache"""
        # Try L1 first
        if key in self.l1_cache:
            entry = self.l1_cache[key]
            if self._is_valid_entry(entry):
                self._update_access(entry, self.l1_stats)
                return entry
            else:
                self._remove_entry(key, self.l1_cache, self.l1_stats)
        
        # Try L2
        if key in self.l2_cache:
            entry = self.l2_cache[key]
            if self._is_valid_entry(entry):
                # Promote to L1 if there's space
                if self.l1_stats.total_size_bytes < CACHE_CONFIG['max_memory_mb'] * 1024 * 1024 // 2:
                    await self._promote_to_l1(key, entry)
                self._update_access(entry, self.l2_stats)
                return entry
            else:
                self._remove_entry(key, self.l2_cache, self.l2_stats)
        
        # Try L3
        if key in self.l3_cache:
            entry = self.l3_cache[key]
            if self._is_valid_entry(entry):
                # Consider promoting to higher levels
                if entry.access_count > 5:  # Frequently accessed
                    await self._promote_to_l2(key, entry)
                self._update_access(entry, self.l3_stats)
                return entry
            else:
                self._remove_entry(key, self.l3_cache, self.l3_stats)
        
        return None
    
    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
                  cache_level: str = "L1", tags: List[str] = None) -> bool:
        """Set value in cache with specified level"""
        tags = tags or []
        
        # Validate key and value sizes
        if len(key.encode('utf-8')) > CACHE_CONFIG['max_key_size_bytes']:
            raise ValueError("Key too large")
        
        value_size = self._calculate_size(value)
        if value_size > CACHE_CONFIG['max_value_size_bytes']:
            raise ValueError("Value too large")
        
        # Calculate expiration
        expires_at = None
        if ttl_seconds:
            expires_at = datetime.now() + timedelta(seconds=ttl_seconds)
        elif CACHE_CONFIG['default_ttl_seconds'] > 0:
            expires_at = datetime.now() + timedelta(seconds=CACHE_CONFIG['default_ttl_seconds'])
        
        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=self._compress_value(value) if cache_level != "L1" else value,
            created_at=datetime.now(),
            expires_at=expires_at,
            size_bytes=value_size,
            cache_level=cache_level,
            last_accessed=datetime.now()
        )
        
        # Get target cache and stats
        cache = self._get_cache_level("", cache_level)
        stats = self._get_stats_for_level(cache_level)
        
        # Remove existing entry if present
        if key in cache:
            self._remove_entry(key, cache, stats)
        
        # Check if eviction is needed
        self._evict_if_needed(cache_level)
        
        # Add new entry
        cache[key] = entry
        stats.total_size_bytes += value_size
        stats.entry_count += 1
        
        # Update tag index
        for tag in tags:
            if tag not in self.tag_index:
                self.tag_index[tag] = set()
            self.tag_index[tag].add(key)
        
        # Update access order
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
        
        logger.debug(f"Cached {key} in {cache_level} (size: {value_size} bytes)")
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache levels"""
        deleted = False
        
        for cache, stats in [
            (self.l1_cache, self.l1_stats),
            (self.l2_cache, self.l2_stats),
            (self.l3_cache, self.l3_stats)
        ]:
            if key in cache:
                self._remove_entry(key, cache, stats)
                deleted = True
        
        return deleted
    
    async def delete_by_tags(self, tags: List[str]) -> int:
        """Delete all entries with specified tags"""
        keys_to_delete = set()
        
        for tag in tags:
            if tag in self.tag_index:
                keys_to_delete.update(self.tag_index[tag])
                del self.tag_index[tag]
        
        deleted_count = 0
        for key in keys_to_delete:
            if await self.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def clear_level(self, level: str) -> int:
        """Clear all entries from specified cache level"""
        cache = self._get_cache_level("", level)
        stats = self._get_stats_for_level(level)
        
        cleared_count = len(cache)
        cache.clear()
        stats.total_size_bytes = 0
        stats.entry_count = 0
        
        return cleared_count
    
    def _is_valid_entry(self, entry: CacheEntry) -> bool:
        """Check if cache entry is still valid"""
        if entry.expires_at and datetime.now() > entry.expires_at:
            return False
        return True
    
    def _update_access(self, entry: CacheEntry, stats: CacheStats):
        """Update access statistics"""
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        stats.total_requests += 1
        stats.cache_hits += 1
        
        # Update access order
        if entry.key in self.access_order:
            self.access_order.remove(entry.key)
        self.access_order.append(entry.key)
    
    async def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1"""
        decompressed_value = self._decompress_value(entry.value)
        await self.set(key, decompressed_value, cache_level="L1")
        self._remove_entry(key, self.l2_cache, self.l2_stats)
    
    async def _promote_to_l2(self, key: str, entry: CacheEntry):
        """Promote entry from L3 to L2"""
        await self.set(key, entry.value, cache_level="L2")
        self._remove_entry(key, self.l3_cache, self.l3_stats)
    
    async def cleanup_expired(self):
        """Remove expired entries from all levels"""
        current_time = datetime.now()
        
        for cache, stats in [
            (self.l1_cache, self.l1_stats),
            (self.l2_cache, self.l2_stats),
            (self.l3_cache, self.l3_stats)
        ]:
            expired_keys = [
                key for key, entry in cache.items()
                if entry.expires_at and current_time > entry.expires_at
            ]
            
            for key in expired_keys:
                self._remove_entry(key, cache, stats)
                stats.expired_entries += 1
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics"""
        total_requests = (
            self.l1_stats.total_requests + 
            self.l2_stats.total_requests + 
            self.l3_stats.total_requests
        )
        total_hits = (
            self.l1_stats.cache_hits + 
            self.l2_stats.cache_hits + 
            self.l3_stats.cache_hits
        )
        
        return {
            'overall': {
                'total_requests': total_requests,
                'total_hits': total_hits,
                'hit_ratio': total_hits / total_requests if total_requests > 0 else 0.0,
                'total_size_bytes': (
                    self.l1_stats.total_size_bytes + 
                    self.l2_stats.total_size_bytes + 
                    self.l3_stats.total_size_bytes
                ),
                'total_entries': (
                    self.l1_stats.entry_count + 
                    self.l2_stats.entry_count + 
                    self.l3_stats.entry_count
                ),
                'total_evictions': (
                    self.l1_stats.evictions + 
                    self.l2_stats.evictions + 
                    self.l3_stats.evictions
                )
            },
            'l1': {
                'requests': self.l1_stats.total_requests,
                'hits': self.l1_stats.cache_hits,
                'hit_ratio': self.l1_stats.hit_ratio,
                'size_bytes': self.l1_stats.total_size_bytes,
                'entries': self.l1_stats.entry_count,
                'evictions': self.l1_stats.evictions
            },
            'l2': {
                'requests': self.l2_stats.total_requests,
                'hits': self.l2_stats.cache_hits,
                'hit_ratio': self.l2_stats.hit_ratio,
                'size_bytes': self.l2_stats.total_size_bytes,
                'entries': self.l2_stats.entry_count,
                'evictions': self.l2_stats.evictions
            },
            'l3': {
                'requests': self.l3_stats.total_requests,
                'hits': self.l3_stats.cache_hits,
                'hit_ratio': self.l3_stats.hit_ratio,
                'size_bytes': self.l3_stats.total_size_bytes,
                'entries': self.l3_stats.entry_count,
                'evictions': self.l3_stats.evictions
            }
        }

# Initialize cache
cache = MultiLevelCache()

# Background tasks
async def periodic_cleanup():
    """Periodically clean up expired entries"""
    while True:
        try:
            await cache.cleanup_expired()
            await asyncio.sleep(CACHE_CONFIG['cleanup_interval_seconds'])
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")
            await asyncio.sleep(60)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info(f"Starting {CACHE_CONFIG['name']} v{CACHE_CONFIG['version']}")
    
    # Start cleanup task
    cleanup_task = asyncio.create_task(periodic_cleanup())
    
    yield
    
    # Shutdown
    cleanup_task.cancel()
    logger.info("Shutting down cache service")

app = FastAPI(
    title="Multi-Level Cache Service",
    description="Distributed multi-level caching service for Deep Tree Echo",
    version=CACHE_CONFIG['version'],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.get("/api/cache/{key}", response_model=CacheResponse)
async def get_cache_value(key: str):
    """Get value from cache"""
    entry = await cache.get(key)
    
    if entry:
        age_seconds = (datetime.now() - entry.created_at).total_seconds()
        return CacheResponse(
            key=key,
            value=cache._decompress_value(entry.value),
            cache_hit=True,
            cache_level=entry.cache_level,
            access_count=entry.access_count,
            age_seconds=age_seconds
        )
    else:
        # Update miss statistics
        cache.l1_stats.total_requests += 1
        cache.l1_stats.cache_misses += 1
        raise HTTPException(status_code=404, detail="Key not found in cache")

@app.post("/api/cache/{key}")
async def set_cache_value(key: str, request: CacheRequest):
    """Set value in cache"""
    try:
        success = await cache.set(
            key=key,
            value=request.value,
            ttl_seconds=request.ttl_seconds,
            cache_level=request.cache_level,
            tags=request.tags
        )
        
        if success:
            return {
                'key': key,
                'status': 'cached',
                'cache_level': request.cache_level,
                'timestamp': datetime.now()
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to cache value")
            
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.delete("/api/cache/{key}")
async def delete_cache_value(key: str):
    """Delete value from cache"""
    deleted = await cache.delete(key)
    
    if deleted:
        return {
            'key': key,
            'status': 'deleted',
            'timestamp': datetime.now()
        }
    else:
        raise HTTPException(status_code=404, detail="Key not found in cache")

@app.post("/api/cache/bulk")
async def bulk_cache_operations(request: BulkCacheRequest):
    """Perform bulk cache operations"""
    results = []
    
    for operation in request.operations:
        try:
            op_type = operation.get('type')
            key = operation.get('key')
            
            if op_type == 'get':
                entry = await cache.get(key)
                if entry:
                    results.append({
                        'key': key,
                        'operation': 'get',
                        'success': True,
                        'value': cache._decompress_value(entry.value)
                    })
                else:
                    results.append({
                        'key': key,
                        'operation': 'get',
                        'success': False,
                        'error': 'Key not found'
                    })
            
            elif op_type == 'set':
                success = await cache.set(
                    key=key,
                    value=operation.get('value'),
                    ttl_seconds=operation.get('ttl_seconds'),
                    cache_level=operation.get('cache_level', 'L1'),
                    tags=operation.get('tags', [])
                )
                results.append({
                    'key': key,
                    'operation': 'set',
                    'success': success
                })
            
            elif op_type == 'delete':
                deleted = await cache.delete(key)
                results.append({
                    'key': key,
                    'operation': 'delete',
                    'success': deleted
                })
                
        except Exception as e:
            results.append({
                'key': operation.get('key'),
                'operation': operation.get('type'),
                'success': False,
                'error': str(e)
            })
    
    return {'results': results}

@app.delete("/api/cache/tags/{tags}")
async def delete_by_tags(tags: str):
    """Delete cache entries by tags"""
    tag_list = [tag.strip() for tag in tags.split(',')]
    deleted_count = await cache.delete_by_tags(tag_list)
    
    return {
        'tags': tag_list,
        'deleted_count': deleted_count,
        'timestamp': datetime.now()
    }

@app.delete("/api/cache/level/{level}")
async def clear_cache_level(level: str):
    """Clear all entries from specified cache level"""
    if level not in ['L1', 'L2', 'L3']:
        raise HTTPException(status_code=400, detail="Invalid cache level")
    
    cleared_count = await cache.clear_level(level)
    
    return {
        'level': level,
        'cleared_count': cleared_count,
        'timestamp': datetime.now()
    }

@app.get("/api/cache/stats")
async def get_cache_statistics():
    """Get comprehensive cache statistics"""
    return cache.get_statistics()

@app.post("/api/cache/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup of expired entries"""
    await cache.cleanup_expired()
    
    return {
        'status': 'cleanup_completed',
        'timestamp': datetime.now()
    }

@app.get("/health", response_model=CacheHealthResponse)
async def health_check():
    """Cache service health check"""
    stats = cache.get_statistics()
    
    memory_usage_mb = stats['overall']['total_size_bytes'] / (1024 * 1024)
    
    return CacheHealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        cache_stats=stats,
        memory_usage_mb=memory_usage_mb,
        entry_count=stats['overall']['total_entries']
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    stats = cache.get_statistics()
    metrics = []
    
    # Overall metrics
    metrics.extend([
        f"cache_total_requests {stats['overall']['total_requests']}",
        f"cache_total_hits {stats['overall']['total_hits']}",
        f"cache_hit_ratio {stats['overall']['hit_ratio']}",
        f"cache_total_size_bytes {stats['overall']['total_size_bytes']}",
        f"cache_total_entries {stats['overall']['total_entries']}",
        f"cache_total_evictions {stats['overall']['total_evictions']}"
    ])
    
    # Per-level metrics
    for level in ['l1', 'l2', 'l3']:
        level_stats = stats[level]
        metrics.extend([
            f"cache_level_requests{{level=\"{level.upper()}\"}} {level_stats['requests']}",
            f"cache_level_hits{{level=\"{level.upper()}\"}} {level_stats['hits']}",
            f"cache_level_hit_ratio{{level=\"{level.upper()}\"}} {level_stats['hit_ratio']}",
            f"cache_level_size_bytes{{level=\"{level.upper()}\"}} {level_stats['size_bytes']}",
            f"cache_level_entries{{level=\"{level.upper()}\"}} {level_stats['entries']}",
            f"cache_level_evictions{{level=\"{level.upper()}\"}} {level_stats['evictions']}"
        ])
    
    return {"metrics": "\n".join(metrics)}

if __name__ == "__main__":
    uvicorn.run(
        "cache_service:app",
        host="0.0.0.0",
        port=CACHE_CONFIG['port'],
        reload=False,
        log_level="info"
    )