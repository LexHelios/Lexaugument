"""
🚀 Redis Caching System for Performance Optimization 🚀
JAI MAHAKAAL! High-performance caching for model responses and data
"""
import asyncio
import json
import logging
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union, List
import aioredis
from dataclasses import dataclass, asdict
import os

logger = logging.getLogger(__name__)

@dataclass
class CacheEntry:
    """Cache entry with metadata"""
    key: str
    value: Any
    created_at: datetime
    expires_at: Optional[datetime]
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['expires_at'] = self.expires_at.isoformat() if self.expires_at else None
        data['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        return data

class RedisCache:
    """
    🚀 High-Performance Redis Cache System
    
    Features:
    - Model response caching with TTL
    - Request deduplication
    - Cache warming and preloading
    - Intelligent cache invalidation
    - Performance metrics tracking
    - Memory-efficient serialization
    """
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.redis_client: Optional[aioredis.Redis] = None
        self.default_ttl = int(os.getenv('CACHE_DEFAULT_TTL', '3600'))  # 1 hour
        self.max_memory_mb = int(os.getenv('CACHE_MAX_MEMORY_MB', '512'))
        
        # Cache prefixes for different data types
        self.prefixes = {
            'model_response': 'lex:model:',
            'user_context': 'lex:user:',
            'business_analysis': 'lex:business:',
            'vision_result': 'lex:vision:',
            'memory_pattern': 'lex:memory:',
            'session': 'lex:session:',
            'rate_limit': 'lex:rate:',
            'metrics': 'lex:metrics:'
        }
        
        # Performance metrics
        self.metrics = {
            'hits': 0,
            'misses': 0,
            'sets': 0,
            'deletes': 0,
            'errors': 0
        }
        
        logger.info("🚀 Redis Cache System initialized")
    
    async def initialize(self) -> None:
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,  # We'll handle encoding ourselves
                max_connections=20,
                retry_on_timeout=True
            )
            
            # Test connection
            await self.redis_client.ping()
            
            # Set memory policy
            await self.redis_client.config_set('maxmemory-policy', 'allkeys-lru')
            await self.redis_client.config_set('maxmemory', f'{self.max_memory_mb}mb')
            
            logger.info("✅ Redis cache connected and configured")
            
        except Exception as e:
            logger.error(f"❌ Redis cache initialization failed: {e}")
            # Fallback to in-memory cache
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, identifier: str, **kwargs) -> str:
        """Generate cache key with consistent hashing"""
        # Create deterministic key from parameters
        key_data = f"{identifier}:{json.dumps(kwargs, sort_keys=True)}"
        key_hash = hashlib.md5(key_data.encode()).hexdigest()
        return f"{self.prefixes[prefix]}{key_hash}"
    
    async def get_model_response(
        self,
        model_name: str,
        prompt: str,
        parameters: Dict[str, Any] = None
    ) -> Optional[Dict[str, Any]]:
        """Get cached model response"""
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key(
                'model_response',
                f"{model_name}:{prompt}",
                **parameters or {}
            )
            
            cached_data = await self.redis_client.get(cache_key)
            if cached_data:
                self.metrics['hits'] += 1
                
                # Update access metadata
                await self._update_access_metadata(cache_key)
                
                # Deserialize and return
                result = pickle.loads(cached_data)
                logger.debug(f"🎯 Cache HIT for model response: {model_name}")
                return result
            
            self.metrics['misses'] += 1
            logger.debug(f"💔 Cache MISS for model response: {model_name}")
            return None
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ Cache get error: {e}")
            return None
    
    async def set_model_response(
        self,
        model_name: str,
        prompt: str,
        response: Dict[str, Any],
        parameters: Dict[str, Any] = None,
        ttl: Optional[int] = None
    ) -> bool:
        """Cache model response"""
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key(
                'model_response',
                f"{model_name}:{prompt}",
                **parameters or {}
            )
            
            # Add metadata
            cache_entry = CacheEntry(
                key=cache_key,
                value=response,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(seconds=ttl or self.default_ttl),
                tags=[model_name, 'model_response']
            )
            
            # Serialize and store
            serialized_data = pickle.dumps(cache_entry.value)
            await self.redis_client.setex(
                cache_key,
                ttl or self.default_ttl,
                serialized_data
            )
            
            # Store metadata separately
            metadata_key = f"{cache_key}:meta"
            await self.redis_client.setex(
                metadata_key,
                ttl or self.default_ttl,
                json.dumps(cache_entry.to_dict())
            )
            
            self.metrics['sets'] += 1
            logger.debug(f"💾 Cached model response: {model_name}")
            return True
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ Cache set error: {e}")
            return False
    
    async def get_user_context(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get cached user context"""
        try:
            if not self.redis_client:
                return None
            
            cache_key = self._generate_cache_key('user_context', user_id)
            cached_data = await self.redis_client.get(cache_key)
            
            if cached_data:
                self.metrics['hits'] += 1
                return pickle.loads(cached_data)
            
            self.metrics['misses'] += 1
            return None
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ User context cache error: {e}")
            return None
    
    async def set_user_context(
        self,
        user_id: str,
        context: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """Cache user context"""
        try:
            if not self.redis_client:
                return False
            
            cache_key = self._generate_cache_key('user_context', user_id)
            serialized_data = pickle.dumps(context)
            
            await self.redis_client.setex(
                cache_key,
                ttl or self.default_ttl,
                serialized_data
            )
            
            self.metrics['sets'] += 1
            return True
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ User context cache set error: {e}")
            return False
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern"""
        try:
            if not self.redis_client:
                return 0
            
            keys = await self.redis_client.keys(pattern)
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.metrics['deletes'] += deleted
                logger.info(f"🗑️ Invalidated {deleted} cache entries matching: {pattern}")
                return deleted
            
            return 0
            
        except Exception as e:
            self.metrics['errors'] += 1
            logger.error(f"❌ Cache invalidation error: {e}")
            return 0
    
    async def invalidate_user_cache(self, user_id: str) -> int:
        """Invalidate all cache entries for a user"""
        pattern = f"{self.prefixes['user_context']}{user_id}*"
        return await self.invalidate_pattern(pattern)
    
    async def warm_cache(self, warm_data: List[Dict[str, Any]]) -> int:
        """Warm cache with precomputed data"""
        try:
            if not self.redis_client:
                return 0
            
            warmed = 0
            for item in warm_data:
                cache_type = item.get('type')
                if cache_type == 'model_response':
                    success = await self.set_model_response(
                        model_name=item['model_name'],
                        prompt=item['prompt'],
                        response=item['response'],
                        parameters=item.get('parameters'),
                        ttl=item.get('ttl')
                    )
                    if success:
                        warmed += 1
                
                elif cache_type == 'user_context':
                    success = await self.set_user_context(
                        user_id=item['user_id'],
                        context=item['context'],
                        ttl=item.get('ttl')
                    )
                    if success:
                        warmed += 1
            
            logger.info(f"🔥 Cache warmed with {warmed} entries")
            return warmed
            
        except Exception as e:
            logger.error(f"❌ Cache warming error: {e}")
            return 0
    
    async def _update_access_metadata(self, cache_key: str) -> None:
        """Update access metadata for cache entry"""
        try:
            metadata_key = f"{cache_key}:meta"
            metadata_json = await self.redis_client.get(metadata_key)
            
            if metadata_json:
                metadata = json.loads(metadata_json)
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_accessed'] = datetime.now().isoformat()
                
                # Update metadata with same TTL as original key
                ttl = await self.redis_client.ttl(cache_key)
                if ttl > 0:
                    await self.redis_client.setex(
                        metadata_key,
                        ttl,
                        json.dumps(metadata)
                    )
                    
        except Exception as e:
            logger.debug(f"Metadata update error: {e}")
    
    async def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        try:
            if not self.redis_client:
                return {"status": "Redis not available", "metrics": self.metrics}
            
            info = await self.redis_client.info('memory')
            keyspace = await self.redis_client.info('keyspace')
            
            # Calculate hit rate
            total_requests = self.metrics['hits'] + self.metrics['misses']
            hit_rate = (self.metrics['hits'] / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "status": "active",
                "hit_rate_percent": round(hit_rate, 2),
                "metrics": self.metrics,
                "memory_used_mb": round(info.get('used_memory', 0) / 1024 / 1024, 2),
                "memory_peak_mb": round(info.get('used_memory_peak', 0) / 1024 / 1024, 2),
                "total_keys": sum(db.get('keys', 0) for db in keyspace.values() if isinstance(db, dict)),
                "redis_info": {
                    "version": info.get('redis_version'),
                    "uptime_seconds": info.get('uptime_in_seconds')
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Cache stats error: {e}")
            return {"status": "error", "error": str(e), "metrics": self.metrics}
    
    async def cleanup_expired(self) -> int:
        """Clean up expired cache entries"""
        try:
            if not self.redis_client:
                return 0
            
            # Redis handles TTL automatically, but we can clean up orphaned metadata
            pattern = "*:meta"
            meta_keys = await self.redis_client.keys(pattern)
            
            cleaned = 0
            for meta_key in meta_keys:
                original_key = meta_key.replace(':meta', '')
                if not await self.redis_client.exists(original_key):
                    await self.redis_client.delete(meta_key)
                    cleaned += 1
            
            if cleaned > 0:
                logger.info(f"🧹 Cleaned up {cleaned} orphaned metadata entries")
            
            return cleaned
            
        except Exception as e:
            logger.error(f"❌ Cache cleanup error: {e}")
            return 0

# Global cache instance
redis_cache = RedisCache()
