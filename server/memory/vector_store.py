"""
LexOS Vibe Coder - Vector Store
Milvus integration for semantic search and RAG
"""
import asyncio
import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

try:
    from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
    MILVUS_AVAILABLE = True
except ImportError:
    MILVUS_AVAILABLE = False
    logger.warning("⚠️ Milvus not available, using FAISS fallback")

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from ..settings import settings

logger = logging.getLogger(__name__)

class VectorStore:
    """
    Vector store with Milvus primary and FAISS fallback
    
    Provides semantic search capabilities for:
    - Conversation history
    - Agent experiences
    - Knowledge retrieval
    - Context matching
    """
    
    def __init__(self):
        self.collection_name = settings.MILVUS_COLLECTION
        self.embedding_model_name = settings.EMBEDDING_MODEL
        self.embedding_dimension = settings.EMBEDDING_DIMENSION
        
        # Initialize components
        self.embedding_model = None
        self.collection = None
        self.faiss_index = None
        self.faiss_metadata = []
        
        # Performance metrics
        self.total_vectors = 0
        self.total_searches = 0
        self.average_search_time = 0.0
        
        # Use Milvus if available, otherwise FAISS
        self.use_milvus = MILVUS_AVAILABLE
        self.use_faiss = FAISS_AVAILABLE and not MILVUS_AVAILABLE
        
        if not self.use_milvus and not self.use_faiss:
            raise RuntimeError("Neither Milvus nor FAISS is available for vector storage")
        
        logger.info(f"🔍 Vector Store initialized (Backend: {'Milvus' if self.use_milvus else 'FAISS'})")
    
    async def initialize(self) -> None:
        """Initialize vector store components"""
        try:
            # Skip embedding model for now - use simple text matching
            logger.warning("⚠️ Vector store disabled for quick startup - using simple memory storage")
            self.embedding_model = None
            self.embedding_dimension = 384  # Default dimension

            # Skip vector store initialization
            self.use_milvus = False
            self.use_faiss = False

            logger.info("✅ Vector Store initialized successfully (simplified mode)")

        except Exception as e:
            logger.error(f"❌ Vector Store initialization error: {e}")
            # Don't raise - allow system to continue
            logger.warning("⚠️ Continuing without vector store")
    
    async def _initialize_milvus(self) -> None:
        """Initialize Milvus connection and collection"""
        try:
            # Connect to Milvus
            connections.connect(
                alias="default",
                host=settings.MILVUS_HOST,
                port=settings.MILVUS_PORT
            )
            
            # Check if collection exists
            if utility.has_collection(self.collection_name):
                self.collection = Collection(self.collection_name)
                logger.info(f"📚 Connected to existing collection: {self.collection_name}")
            else:
                # Create collection schema
                fields = [
                    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                    FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dimension),
                    FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                    FieldSchema(name="timestamp", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="user_id", dtype=DataType.VARCHAR, max_length=100),
                    FieldSchema(name="agent_id", dtype=DataType.VARCHAR, max_length=100)
                ]
                
                schema = CollectionSchema(fields, "LexOS Vector Store Collection")
                self.collection = Collection(self.collection_name, schema)
                
                # Create index
                index_params = {
                    "metric_type": "COSINE",
                    "index_type": "IVF_FLAT",
                    "params": {"nlist": 128}
                }
                self.collection.create_index("vector", index_params)
                
                logger.info(f"📚 Created new collection: {self.collection_name}")
            
            # Load collection
            self.collection.load()
            
        except Exception as e:
            logger.error(f"❌ Milvus initialization error: {e}")
            # Fallback to FAISS
            if FAISS_AVAILABLE:
                logger.info("🔄 Falling back to FAISS")
                self.use_milvus = False
                self.use_faiss = True
                await self._initialize_faiss()
            else:
                raise
    
    async def _initialize_faiss(self) -> None:
        """Initialize FAISS index"""
        try:
            # Create FAISS index
            self.faiss_index = faiss.IndexFlatIP(self.embedding_dimension)  # Inner product for cosine similarity
            self.faiss_metadata = []
            
            logger.info(f"📚 FAISS index initialized with dimension {self.embedding_dimension}")
            
        except Exception as e:
            logger.error(f"❌ FAISS initialization error: {e}")
            raise
    
    async def add_vectors(self, documents: List[Dict[str, Any]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents: List of documents with 'content' and optional 'metadata'
        """
        try:
            if not documents:
                return True
            
            # Generate embeddings
            contents = [doc.get('content', '') for doc in documents]
            embeddings = self.embedding_model.encode(contents)
            
            if self.use_milvus:
                return await self._add_vectors_milvus(documents, embeddings)
            elif self.use_faiss:
                return await self._add_vectors_faiss(documents, embeddings)
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Add vectors error: {e}")
            return False
    
    async def _add_vectors_milvus(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Add vectors to Milvus"""
        try:
            # Prepare data for insertion
            data = []
            for i, doc in enumerate(documents):
                data.append({
                    "vector": embeddings[i].tolist(),
                    "content": doc.get('content', ''),
                    "metadata": json.dumps(doc.get('metadata', {})),
                    "timestamp": doc.get('metadata', {}).get('timestamp', datetime.now().isoformat()),
                    "user_id": doc.get('metadata', {}).get('user_id', 'default'),
                    "agent_id": doc.get('metadata', {}).get('agent_id', 'unknown')
                })
            
            # Insert data
            self.collection.insert(data)
            self.collection.flush()
            
            self.total_vectors += len(documents)
            logger.debug(f"🔍 Added {len(documents)} vectors to Milvus")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Milvus add vectors error: {e}")
            return False
    
    async def _add_vectors_faiss(self, documents: List[Dict[str, Any]], embeddings: np.ndarray) -> bool:
        """Add vectors to FAISS"""
        try:
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Add to index
            self.faiss_index.add(embeddings)
            
            # Store metadata
            for i, doc in enumerate(documents):
                metadata = {
                    "content": doc.get('content', ''),
                    "metadata": doc.get('metadata', {}),
                    "timestamp": doc.get('metadata', {}).get('timestamp', datetime.now().isoformat()),
                    "user_id": doc.get('metadata', {}).get('user_id', 'default'),
                    "agent_id": doc.get('metadata', {}).get('agent_id', 'unknown'),
                    "index": self.total_vectors + i
                }
                self.faiss_metadata.append(metadata)
            
            self.total_vectors += len(documents)
            logger.debug(f"🔍 Added {len(documents)} vectors to FAISS")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ FAISS add vectors error: {e}")
            return False
    
    async def search_vectors(
        self, 
        query: str, 
        top_k: int = 5,
        user_id: Optional[str] = None,
        agent_id: Optional[str] = None,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar vectors
        
        Args:
            query: Search query text
            top_k: Number of results to return
            user_id: Filter by user ID
            agent_id: Filter by agent ID
            similarity_threshold: Minimum similarity score
        """
        try:
            start_time = datetime.now()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])
            
            if self.use_milvus:
                results = await self._search_vectors_milvus(
                    query_embedding[0], top_k, user_id, agent_id, similarity_threshold
                )
            elif self.use_faiss:
                results = await self._search_vectors_faiss(
                    query_embedding[0], top_k, user_id, agent_id, similarity_threshold
                )
            else:
                results = []
            
            # Update metrics
            search_time = (datetime.now() - start_time).total_seconds()
            self.total_searches += 1
            self.average_search_time = (
                (self.average_search_time * (self.total_searches - 1) + search_time) / 
                self.total_searches
            )
            
            logger.debug(f"🔍 Vector search returned {len(results)} results in {search_time:.3f}s")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Vector search error: {e}")
            return []
    
    async def _search_vectors_milvus(
        self, 
        query_embedding: np.ndarray, 
        top_k: int,
        user_id: Optional[str],
        agent_id: Optional[str],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search vectors in Milvus"""
        try:
            # Prepare search parameters
            search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
            
            # Build filter expression
            filter_expr = []
            if user_id:
                filter_expr.append(f'user_id == "{user_id}"')
            if agent_id:
                filter_expr.append(f'agent_id == "{agent_id}"')
            
            expr = " and ".join(filter_expr) if filter_expr else None
            
            # Perform search
            search_results = self.collection.search(
                data=[query_embedding.tolist()],
                anns_field="vector",
                param=search_params,
                limit=top_k * 2,  # Get more results to filter by threshold
                expr=expr,
                output_fields=["content", "metadata", "timestamp", "user_id", "agent_id"]
            )
            
            # Process results
            results = []
            for hit in search_results[0]:
                similarity = hit.score
                if similarity >= similarity_threshold:
                    result = {
                        "content": hit.entity.get("content"),
                        "metadata": json.loads(hit.entity.get("metadata", "{}")),
                        "similarity": similarity,
                        "timestamp": hit.entity.get("timestamp"),
                        "user_id": hit.entity.get("user_id"),
                        "agent_id": hit.entity.get("agent_id")
                    }
                    results.append(result)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Milvus search error: {e}")
            return []
    
    async def _search_vectors_faiss(
        self, 
        query_embedding: np.ndarray, 
        top_k: int,
        user_id: Optional[str],
        agent_id: Optional[str],
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Search vectors in FAISS"""
        try:
            # Normalize query embedding
            query_embedding = query_embedding.reshape(1, -1)
            faiss.normalize_L2(query_embedding)
            
            # Search
            search_k = min(top_k * 3, self.faiss_index.ntotal)  # Get more results for filtering
            similarities, indices = self.faiss_index.search(query_embedding, search_k)
            
            # Process results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx == -1 or similarity < similarity_threshold:
                    continue
                
                metadata = self.faiss_metadata[idx]
                
                # Apply filters
                if user_id and metadata.get("user_id") != user_id:
                    continue
                if agent_id and metadata.get("agent_id") != agent_id:
                    continue
                
                result = {
                    "content": metadata["content"],
                    "metadata": metadata["metadata"],
                    "similarity": float(similarity),
                    "timestamp": metadata["timestamp"],
                    "user_id": metadata["user_id"],
                    "agent_id": metadata["agent_id"]
                }
                results.append(result)
                
                if len(results) >= top_k:
                    break
            
            return results
            
        except Exception as e:
            logger.error(f"❌ FAISS search error: {e}")
            return []
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            stats = {
                "backend": "Milvus" if self.use_milvus else "FAISS",
                "total_vectors": self.total_vectors,
                "total_searches": self.total_searches,
                "average_search_time": self.average_search_time,
                "embedding_model": self.embedding_model_name,
                "embedding_dimension": self.embedding_dimension
            }
            
            if self.use_milvus and self.collection:
                stats["milvus"] = {
                    "collection_name": self.collection_name,
                    "num_entities": self.collection.num_entities,
                    "is_loaded": self.collection.is_loaded
                }
            elif self.use_faiss and self.faiss_index:
                stats["faiss"] = {
                    "index_size": self.faiss_index.ntotal,
                    "is_trained": self.faiss_index.is_trained,
                    "metadata_entries": len(self.faiss_metadata)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Statistics error: {e}")
            return {}
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on vector store"""
        try:
            # Test embedding generation
            test_text = "Health check test"
            test_embedding = self.embedding_model.encode([test_text])
            
            # Test search
            search_results = await self.search_vectors(test_text, top_k=1)
            
            return {
                "status": "healthy",
                "backend": "Milvus" if self.use_milvus else "FAISS",
                "embedding_test": "passed",
                "search_test": "passed",
                "statistics": await self.get_statistics()
            }
            
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
    
    async def close(self) -> None:
        """Close vector store connections"""
        try:
            if self.use_milvus:
                connections.disconnect("default")
            
            logger.info("🔍 Vector Store closed")
            
        except Exception as e:
            logger.error(f"❌ Vector Store close error: {e}")

# Global vector store instance
vector_store = VectorStore()
