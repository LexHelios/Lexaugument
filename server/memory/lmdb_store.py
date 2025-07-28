"""
LexOS Vibe Coder - LMDB Memory Store
Encrypted persistent storage for experiences and conversations
"""
import asyncio
import json
import logging
import lmdb
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import hashlib
from cryptography.fernet import Fernet
import gzip

from ..settings import settings

logger = logging.getLogger(__name__)

class LMDBStore:
    """
    LMDB-based memory store with encryption and compression
    
    Provides persistent storage for:
    - Conversation histories
    - Agent experiences
    - User interactions
    - System events
    """
    
    def __init__(self):
        self.db_path = Path(settings.LMDB_PATH)
        self.map_size = settings.LMDB_MAP_SIZE
        self.env = None
        self.cipher = None
        
        # Initialize encryption
        if settings.ENCRYPTION_KEY:
            self.cipher = Fernet(settings.ENCRYPTION_KEY.encode() if isinstance(settings.ENCRYPTION_KEY, str) else settings.ENCRYPTION_KEY)
        
        # Database handles
        self.conversations_db = None
        self.experiences_db = None
        self.metadata_db = None
        
        # Performance metrics
        self.total_reads = 0
        self.total_writes = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Simple in-memory cache
        self.cache = {}
        self.cache_max_size = 1000
        
        logger.info(f"💾 LMDB Store initialized at {self.db_path}")
    
    async def initialize(self) -> None:
        """Initialize LMDB environment and databases"""
        try:
            # Create directory if it doesn't exist
            self.db_path.mkdir(parents=True, exist_ok=True)
            
            # Open LMDB environment
            self.env = lmdb.open(
                str(self.db_path),
                map_size=self.map_size,
                max_dbs=10,
                sync=True,
                writemap=False
            )
            
            # Create named databases
            with self.env.begin(write=True) as txn:
                self.conversations_db = self.env.open_db(b'conversations', txn=txn)
                self.experiences_db = self.env.open_db(b'experiences', txn=txn)
                self.metadata_db = self.env.open_db(b'metadata', txn=txn)
            
            # Store initialization metadata
            await self._store_metadata("initialized", {
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0",
                "encryption_enabled": self.cipher is not None
            })
            
            logger.info("✅ LMDB Store initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ LMDB initialization error: {e}")
            raise
    
    async def save_experience(self, conversation_id: str, entry: Dict[str, Any]) -> bool:
        """
        Save an experience entry to the database
        """
        try:
            # Prepare data
            timestamp = datetime.now().isoformat()
            experience_data = {
                "conversation_id": conversation_id,
                "timestamp": timestamp,
                "entry": entry
            }
            
            # Serialize and optionally encrypt
            serialized_data = await self._serialize_data(experience_data)
            
            # Generate key
            key = f"{conversation_id}:{timestamp}".encode()
            
            # Store in database
            with self.env.begin(write=True) as txn:
                success = txn.put(serialized_data, db=self.experiences_db, key=key)
            
            if success:
                self.total_writes += 1
                # Update cache
                cache_key = f"exp:{conversation_id}:{timestamp}"
                self._update_cache(cache_key, experience_data)
                
                logger.debug(f"💾 Experience saved: {conversation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Experience save error: {e}")
            return False
    
    async def load_experiences(
        self, 
        conversation_id: str, 
        limit: int = 100,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Load experiences for a conversation
        """
        try:
            experiences = []
            prefix = f"{conversation_id}:".encode()
            
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.experiences_db)
                
                # Seek to the start of this conversation's entries
                if cursor.set_range(prefix):
                    for key, value in cursor:
                        # Stop if we've moved past this conversation
                        if not key.startswith(prefix):
                            break
                        
                        # Deserialize data
                        experience_data = await self._deserialize_data(value)
                        
                        # Apply time filters
                        if start_time or end_time:
                            exp_time = datetime.fromisoformat(experience_data["timestamp"])
                            if start_time and exp_time < start_time:
                                continue
                            if end_time and exp_time > end_time:
                                continue
                        
                        experiences.append(experience_data)
                        
                        # Apply limit
                        if len(experiences) >= limit:
                            break
            
            self.total_reads += 1
            logger.debug(f"💾 Loaded {len(experiences)} experiences for {conversation_id}")
            
            return experiences
            
        except Exception as e:
            logger.error(f"❌ Experience load error: {e}")
            return []
    
    async def save_conversation(
        self, 
        conversation_id: str, 
        messages: List[Dict[str, Any]]
    ) -> bool:
        """
        Save conversation messages
        """
        try:
            conversation_data = {
                "conversation_id": conversation_id,
                "messages": messages,
                "last_updated": datetime.now().isoformat(),
                "message_count": len(messages)
            }
            
            # Serialize and optionally encrypt
            serialized_data = await self._serialize_data(conversation_data)
            
            # Store in database
            key = conversation_id.encode()
            with self.env.begin(write=True) as txn:
                success = txn.put(serialized_data, db=self.conversations_db, key=key)
            
            if success:
                self.total_writes += 1
                # Update cache
                cache_key = f"conv:{conversation_id}"
                self._update_cache(cache_key, conversation_data)
                
                logger.debug(f"💾 Conversation saved: {conversation_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Conversation save error: {e}")
            return False
    
    async def load_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load conversation by ID
        """
        try:
            # Check cache first
            cache_key = f"conv:{conversation_id}"
            if cache_key in self.cache:
                self.cache_hits += 1
                return self.cache[cache_key]
            
            # Load from database
            key = conversation_id.encode()
            with self.env.begin() as txn:
                value = txn.get(key, db=self.conversations_db)
            
            if value:
                conversation_data = await self._deserialize_data(value)
                self.total_reads += 1
                self.cache_misses += 1
                
                # Update cache
                self._update_cache(cache_key, conversation_data)
                
                logger.debug(f"💾 Conversation loaded: {conversation_id}")
                return conversation_data
            
            return None
            
        except Exception as e:
            logger.error(f"❌ Conversation load error: {e}")
            return None
    
    async def search_experiences(
        self, 
        query: str, 
        conversation_id: Optional[str] = None,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Search experiences by content (simple text search)
        """
        try:
            matching_experiences = []
            query_lower = query.lower()
            
            with self.env.begin() as txn:
                cursor = txn.cursor(db=self.experiences_db)
                
                for key, value in cursor:
                    # Apply conversation filter
                    if conversation_id:
                        key_str = key.decode()
                        if not key_str.startswith(f"{conversation_id}:"):
                            continue
                    
                    # Deserialize and search
                    experience_data = await self._deserialize_data(value)
                    
                    # Simple text search in entry content
                    entry_str = json.dumps(experience_data.get("entry", {})).lower()
                    if query_lower in entry_str:
                        matching_experiences.append(experience_data)
                        
                        if len(matching_experiences) >= limit:
                            break
            
            logger.debug(f"💾 Found {len(matching_experiences)} matching experiences")
            return matching_experiences
            
        except Exception as e:
            logger.error(f"❌ Experience search error: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """
        Get database statistics
        """
        try:
            stats = {}
            
            # Environment stats
            env_stat = self.env.stat()
            stats["environment"] = {
                "page_size": env_stat["psize"],
                "depth": env_stat["depth"],
                "branch_pages": env_stat["branch_pages"],
                "leaf_pages": env_stat["leaf_pages"],
                "overflow_pages": env_stat["overflow_pages"],
                "entries": env_stat["entries"]
            }
            
            # Database-specific stats
            with self.env.begin() as txn:
                conv_stat = txn.stat(db=self.conversations_db)
                exp_stat = txn.stat(db=self.experiences_db)
                
                stats["conversations"] = {
                    "entries": conv_stat["entries"],
                    "pages": conv_stat["branch_pages"] + conv_stat["leaf_pages"]
                }
                
                stats["experiences"] = {
                    "entries": exp_stat["entries"],
                    "pages": exp_stat["branch_pages"] + exp_stat["leaf_pages"]
                }
            
            # Performance stats
            stats["performance"] = {
                "total_reads": self.total_reads,
                "total_writes": self.total_writes,
                "cache_hits": self.cache_hits,
                "cache_misses": self.cache_misses,
                "cache_hit_rate": self.cache_hits / (self.cache_hits + self.cache_misses) if (self.cache_hits + self.cache_misses) > 0 else 0,
                "cache_size": len(self.cache)
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Statistics error: {e}")
            return {}
    
    async def _serialize_data(self, data: Dict[str, Any]) -> bytes:
        """Serialize data with optional encryption and compression"""
        try:
            # Convert to JSON then pickle for efficiency
            json_data = json.dumps(data, default=str)
            pickled_data = pickle.dumps(json_data)
            
            # Compress
            compressed_data = gzip.compress(pickled_data)
            
            # Encrypt if cipher is available
            if self.cipher:
                encrypted_data = self.cipher.encrypt(compressed_data)
                return encrypted_data
            
            return compressed_data
            
        except Exception as e:
            logger.error(f"❌ Data serialization error: {e}")
            raise
    
    async def _deserialize_data(self, data: bytes) -> Dict[str, Any]:
        """Deserialize data with optional decryption and decompression"""
        try:
            # Decrypt if cipher is available
            if self.cipher:
                decrypted_data = self.cipher.decrypt(data)
                data = decrypted_data
            
            # Decompress
            decompressed_data = gzip.decompress(data)
            
            # Unpickle and parse JSON
            json_data = pickle.loads(decompressed_data)
            parsed_data = json.loads(json_data)
            
            return parsed_data
            
        except Exception as e:
            logger.error(f"❌ Data deserialization error: {e}")
            raise
    
    def _update_cache(self, key: str, data: Dict[str, Any]) -> None:
        """Update in-memory cache with size limit"""
        if len(self.cache) >= self.cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[key] = data
    
    async def _store_metadata(self, key: str, data: Dict[str, Any]) -> None:
        """Store metadata in the metadata database"""
        try:
            serialized_data = await self._serialize_data(data)
            
            with self.env.begin(write=True) as txn:
                txn.put(serialized_data, db=self.metadata_db, key=key.encode())
                
        except Exception as e:
            logger.error(f"❌ Metadata store error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on the LMDB store"""
        try:
            # Test read/write operations
            test_key = "health_check"
            test_data = {"timestamp": datetime.now().isoformat(), "test": True}
            
            # Test write
            await self._store_metadata(test_key, test_data)
            
            # Test read
            with self.env.begin() as txn:
                value = txn.get(test_key.encode(), db=self.metadata_db)
            
            if value:
                retrieved_data = await self._deserialize_data(value)
                
                return {
                    "status": "healthy",
                    "read_write_test": "passed",
                    "encryption_enabled": self.cipher is not None,
                    "database_path": str(self.db_path),
                    "statistics": await self.get_statistics()
                }
            
            return {"status": "unhealthy", "error": "Read/write test failed"}
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self) -> None:
        """Close the LMDB environment"""
        try:
            if self.env:
                self.env.close()
                logger.info("💾 LMDB Store closed")
        except Exception as e:
            logger.error(f"❌ LMDB close error: {e}")

# Global memory store instance
memory_store = LMDBStore()
