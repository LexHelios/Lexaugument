"""
LexOS Vibe Coder - RAG (Retrieval-Augmented Generation)
Advanced retrieval pipeline for context-aware responses
"""
import asyncio
import logging
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict

from .vector_store import vector_store
from .lmdb_store import memory_store
from ..settings import settings

logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Advanced RAG pipeline with multiple retrieval strategies
    
    Features:
    - Semantic similarity search
    - Temporal relevance weighting
    - Multi-modal context fusion
    - Adaptive retrieval strategies
    - Context ranking and filtering
    """
    
    def __init__(self):
        self.retrieval_strategies = [
            "semantic_similarity",
            "temporal_relevance", 
            "conversation_context",
            "agent_expertise",
            "user_preference"
        ]
        
        # Retrieval weights (can be adjusted based on query type)
        self.strategy_weights = {
            "semantic_similarity": 0.4,
            "temporal_relevance": 0.2,
            "conversation_context": 0.2,
            "agent_expertise": 0.1,
            "user_preference": 0.1
        }
        
        # Performance metrics
        self.total_retrievals = 0
        self.average_retrieval_time = 0.0
        self.context_quality_scores = []
        
        logger.info("🔍 RAG Pipeline initialized")
    
    async def retrieve_context(
        self,
        query_text: str,
        user_id: str = "default",
        agent_id: Optional[str] = None,
        top_k: int = None,
        similarity_threshold: float = None,
        time_window_hours: int = 24,
        include_conversation_history: bool = True,
        retrieval_strategy: str = "adaptive"
    ) -> List[Dict[str, Any]]:
        """
        Main retrieval function with adaptive strategy selection
        
        Args:
            query_text: The query to find context for
            user_id: User identifier for personalized retrieval
            agent_id: Agent identifier for expertise-based retrieval
            top_k: Number of context items to retrieve
            similarity_threshold: Minimum similarity score
            time_window_hours: Time window for temporal relevance
            include_conversation_history: Whether to include recent conversation
            retrieval_strategy: Strategy to use ("adaptive", "semantic", "temporal", etc.)
        """
        start_time = datetime.now()
        
        try:
            # Use defaults from settings if not provided
            top_k = top_k or settings.RAG_TOP_K
            similarity_threshold = similarity_threshold or settings.RAG_SIMILARITY_THRESHOLD
            
            # Analyze query to determine optimal retrieval strategy
            if retrieval_strategy == "adaptive":
                retrieval_strategy = await self._analyze_query_strategy(query_text, agent_id)
            
            # Execute retrieval based on strategy
            context_items = await self._execute_retrieval_strategy(
                query_text=query_text,
                user_id=user_id,
                agent_id=agent_id,
                strategy=retrieval_strategy,
                top_k=top_k,
                similarity_threshold=similarity_threshold,
                time_window_hours=time_window_hours
            )
            
            # Add conversation history if requested
            if include_conversation_history:
                conversation_context = await self._get_conversation_context(
                    user_id, agent_id, max_messages=5
                )
                context_items.extend(conversation_context)
            
            # Rank and filter context
            ranked_context = await self._rank_and_filter_context(
                context_items, query_text, top_k
            )
            
            # Update metrics
            retrieval_time = (datetime.now() - start_time).total_seconds()
            self._update_retrieval_metrics(retrieval_time, len(ranked_context))
            
            logger.debug(f"🔍 Retrieved {len(ranked_context)} context items in {retrieval_time:.3f}s")
            
            return ranked_context
            
        except Exception as e:
            logger.error(f"❌ RAG retrieval error: {e}")
            return []
    
    async def _analyze_query_strategy(self, query_text: str, agent_id: Optional[str]) -> str:
        """Analyze query to determine optimal retrieval strategy"""
        query_lower = query_text.lower()
        
        # Temporal indicators
        temporal_keywords = ["recent", "latest", "yesterday", "today", "last", "current", "now"]
        if any(keyword in query_lower for keyword in temporal_keywords):
            return "temporal_relevance"
        
        # Technical/coding queries (for specific agents)
        if agent_id in ["atlas", "creator"] and any(keyword in query_lower for keyword in 
            ["code", "function", "class", "algorithm", "implementation", "debug"]):
            return "agent_expertise"
        
        # Conversational context queries
        conversational_keywords = ["we discussed", "you mentioned", "earlier", "before", "previous"]
        if any(keyword in query_lower for keyword in conversational_keywords):
            return "conversation_context"
        
        # Default to semantic similarity
        return "semantic_similarity"
    
    async def _execute_retrieval_strategy(
        self,
        query_text: str,
        user_id: str,
        agent_id: Optional[str],
        strategy: str,
        top_k: int,
        similarity_threshold: float,
        time_window_hours: int
    ) -> List[Dict[str, Any]]:
        """Execute specific retrieval strategy"""
        
        if strategy == "semantic_similarity":
            return await self._semantic_retrieval(
                query_text, user_id, agent_id, top_k, similarity_threshold
            )
        
        elif strategy == "temporal_relevance":
            return await self._temporal_retrieval(
                query_text, user_id, agent_id, top_k, time_window_hours
            )
        
        elif strategy == "conversation_context":
            return await self._conversation_retrieval(
                query_text, user_id, agent_id, top_k
            )
        
        elif strategy == "agent_expertise":
            return await self._agent_expertise_retrieval(
                query_text, user_id, agent_id, top_k, similarity_threshold
            )
        
        else:
            # Fallback to semantic similarity
            return await self._semantic_retrieval(
                query_text, user_id, agent_id, top_k, similarity_threshold
            )
    
    async def _semantic_retrieval(
        self,
        query_text: str,
        user_id: str,
        agent_id: Optional[str],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Pure semantic similarity retrieval"""
        try:
            results = await vector_store.search_vectors(
                query=query_text,
                top_k=top_k,
                user_id=user_id,
                agent_id=agent_id,
                similarity_threshold=similarity_threshold
            )
            
            # Add retrieval metadata
            for result in results:
                result["retrieval_strategy"] = "semantic_similarity"
                result["retrieval_score"] = result.get("similarity", 0.0)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Semantic retrieval error: {e}")
            return []
    
    async def _temporal_retrieval(
        self,
        query_text: str,
        user_id: str,
        agent_id: Optional[str],
        top_k: int,
        time_window_hours: int
    ) -> List[Dict[str, Any]]:
        """Time-weighted retrieval focusing on recent interactions"""
        try:
            # Get recent experiences from LMDB
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            conversation_id = f"{user_id}_{agent_id}" if agent_id else f"{user_id}_all"
            experiences = await memory_store.load_experiences(
                conversation_id=conversation_id,
                limit=top_k * 2,
                start_time=start_time,
                end_time=end_time
            )
            
            # Score by temporal relevance and semantic similarity
            results = []
            for exp in experiences:
                entry = exp.get("entry", {})
                content = self._extract_content_from_experience(entry)
                
                if content:
                    # Calculate temporal score (more recent = higher score)
                    exp_time = datetime.fromisoformat(exp["timestamp"])
                    time_diff_hours = (end_time - exp_time).total_seconds() / 3600
                    temporal_score = max(0, 1 - (time_diff_hours / time_window_hours))
                    
                    # Simple semantic relevance (could be enhanced with embeddings)
                    semantic_score = self._calculate_text_similarity(query_text, content)
                    
                    # Combined score
                    combined_score = (temporal_score * 0.6) + (semantic_score * 0.4)
                    
                    result = {
                        "content": content,
                        "metadata": entry,
                        "similarity": combined_score,
                        "temporal_score": temporal_score,
                        "semantic_score": semantic_score,
                        "timestamp": exp["timestamp"],
                        "retrieval_strategy": "temporal_relevance",
                        "retrieval_score": combined_score
                    }
                    results.append(result)
            
            # Sort by combined score and return top_k
            results.sort(key=lambda x: x["retrieval_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Temporal retrieval error: {e}")
            return []
    
    async def _conversation_retrieval(
        self,
        query_text: str,
        user_id: str,
        agent_id: Optional[str],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Retrieve from recent conversation context"""
        try:
            conversation_id = f"{user_id}_{agent_id}" if agent_id else f"{user_id}_all"
            conversation = await memory_store.load_conversation(conversation_id)
            
            if not conversation:
                return []
            
            messages = conversation.get("messages", [])
            results = []
            
            # Process recent messages
            for msg in messages[-20:]:  # Last 20 messages
                content = msg.get("content", "")
                if content and len(content) > 10:  # Skip very short messages
                    
                    semantic_score = self._calculate_text_similarity(query_text, content)
                    
                    if semantic_score > 0.3:  # Minimum relevance threshold
                        result = {
                            "content": content,
                            "metadata": {
                                "message_type": msg.get("type", "unknown"),
                                "agent_id": msg.get("agent_id", "unknown"),
                                "conversation_id": conversation_id
                            },
                            "similarity": semantic_score,
                            "timestamp": msg.get("timestamp", ""),
                            "retrieval_strategy": "conversation_context",
                            "retrieval_score": semantic_score
                        }
                        results.append(result)
            
            # Sort by relevance and return top_k
            results.sort(key=lambda x: x["retrieval_score"], reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Conversation retrieval error: {e}")
            return []
    
    async def _agent_expertise_retrieval(
        self,
        query_text: str,
        user_id: str,
        agent_id: Optional[str],
        top_k: int,
        similarity_threshold: float
    ) -> List[Dict[str, Any]]:
        """Retrieve based on agent expertise and specialization"""
        try:
            # First get semantic results
            results = await self._semantic_retrieval(
                query_text, user_id, agent_id, top_k * 2, similarity_threshold * 0.8
            )
            
            # Boost results from the same agent or related agents
            agent_expertise_boost = {
                "atlas": ["strategy", "analysis", "planning", "risk"],
                "orion": ["research", "web", "search", "information"],
                "sophia": ["ethics", "philosophy", "wisdom", "guidance"],
                "creator": ["code", "programming", "development", "implementation"]
            }
            
            if agent_id and agent_id in agent_expertise_boost:
                expertise_keywords = agent_expertise_boost[agent_id]
                
                for result in results:
                    content_lower = result.get("content", "").lower()
                    expertise_score = sum(1 for keyword in expertise_keywords if keyword in content_lower)
                    
                    # Boost score based on expertise match
                    if expertise_score > 0:
                        boost_factor = 1 + (expertise_score * 0.1)
                        result["retrieval_score"] = result.get("similarity", 0) * boost_factor
                        result["expertise_boost"] = boost_factor
                    else:
                        result["retrieval_score"] = result.get("similarity", 0)
                        result["expertise_boost"] = 1.0
                    
                    result["retrieval_strategy"] = "agent_expertise"
            
            # Sort by boosted score and return top_k
            results.sort(key=lambda x: x.get("retrieval_score", 0), reverse=True)
            return results[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Agent expertise retrieval error: {e}")
            return []
    
    async def _get_conversation_context(
        self,
        user_id: str,
        agent_id: Optional[str],
        max_messages: int = 5
    ) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        try:
            conversation_id = f"{user_id}_{agent_id}" if agent_id else f"{user_id}_all"
            conversation = await memory_store.load_conversation(conversation_id)
            
            if not conversation:
                return []
            
            messages = conversation.get("messages", [])
            context_items = []
            
            # Get last few messages as context
            for msg in messages[-max_messages:]:
                content = msg.get("content", "")
                if content:
                    context_item = {
                        "content": content,
                        "metadata": {
                            "type": "conversation_history",
                            "message_type": msg.get("type", "unknown"),
                            "agent_id": msg.get("agent_id", "unknown")
                        },
                        "similarity": 0.8,  # High relevance for recent conversation
                        "timestamp": msg.get("timestamp", ""),
                        "retrieval_strategy": "conversation_history",
                        "retrieval_score": 0.8
                    }
                    context_items.append(context_item)
            
            return context_items
            
        except Exception as e:
            logger.error(f"❌ Conversation context error: {e}")
            return []
    
    async def _rank_and_filter_context(
        self,
        context_items: List[Dict[str, Any]],
        query_text: str,
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Rank and filter context items for final selection"""
        try:
            if not context_items:
                return []
            
            # Remove duplicates based on content similarity
            unique_items = self._remove_duplicate_context(context_items)
            
            # Re-rank based on multiple factors
            for item in unique_items:
                # Base score from retrieval
                base_score = item.get("retrieval_score", item.get("similarity", 0))
                
                # Length penalty for very short or very long content
                content_length = len(item.get("content", ""))
                length_factor = 1.0
                if content_length < 20:
                    length_factor = 0.5
                elif content_length > 2000:
                    length_factor = 0.8
                
                # Recency bonus
                timestamp_str = item.get("timestamp", "")
                recency_factor = 1.0
                if timestamp_str:
                    try:
                        item_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                        hours_ago = (datetime.now() - item_time.replace(tzinfo=None)).total_seconds() / 3600
                        recency_factor = max(0.5, 1 - (hours_ago / 168))  # Decay over a week
                    except:
                        pass
                
                # Calculate final score
                final_score = base_score * length_factor * recency_factor
                item["final_score"] = final_score
            
            # Sort by final score and return top_k
            unique_items.sort(key=lambda x: x.get("final_score", 0), reverse=True)
            
            # Add ranking metadata
            for i, item in enumerate(unique_items[:top_k]):
                item["rank"] = i + 1
                item["total_candidates"] = len(context_items)
            
            return unique_items[:top_k]
            
        except Exception as e:
            logger.error(f"❌ Context ranking error: {e}")
            return context_items[:top_k]  # Fallback to original list
    
    def _remove_duplicate_context(self, context_items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate context items based on content similarity"""
        unique_items = []
        seen_content = set()
        
        for item in context_items:
            content = item.get("content", "")
            content_hash = hashlib.md5(content.encode()).hexdigest()
            
            # Simple duplicate detection
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_items.append(item)
        
        return unique_items
    
    def _extract_content_from_experience(self, entry: Dict[str, Any]) -> str:
        """Extract meaningful content from experience entry"""
        # Try different content fields
        content_fields = ["message", "response", "content", "text"]
        
        for field in content_fields:
            if field in entry:
                if isinstance(entry[field], dict):
                    return entry[field].get("content", "")
                elif isinstance(entry[field], str):
                    return entry[field]
        
        # Fallback to JSON representation
        return json.dumps(entry)[:500]  # Limit length
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity calculation (could be enhanced with embeddings)"""
        # Convert to lowercase and split into words
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _update_retrieval_metrics(self, retrieval_time: float, context_count: int) -> None:
        """Update RAG performance metrics"""
        self.total_retrievals += 1
        
        # Update average retrieval time
        if self.total_retrievals == 1:
            self.average_retrieval_time = retrieval_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_retrieval_time = (
                alpha * retrieval_time + 
                (1 - alpha) * self.average_retrieval_time
            )
        
        # Track context quality (simple heuristic)
        quality_score = min(1.0, context_count / settings.RAG_TOP_K)
        self.context_quality_scores.append(quality_score)
        
        # Keep only recent scores
        if len(self.context_quality_scores) > 100:
            self.context_quality_scores = self.context_quality_scores[-50:]
    
    def get_rag_statistics(self) -> Dict[str, Any]:
        """Get RAG pipeline statistics"""
        avg_quality = (
            sum(self.context_quality_scores) / len(self.context_quality_scores)
            if self.context_quality_scores else 0.0
        )
        
        return {
            "total_retrievals": self.total_retrievals,
            "average_retrieval_time": self.average_retrieval_time,
            "average_context_quality": avg_quality,
            "available_strategies": self.retrieval_strategies,
            "strategy_weights": self.strategy_weights
        }

# Global RAG pipeline instance
rag_pipeline = RAGPipeline()

# Convenience function for external use
async def retrieve_context(
    query_text: str,
    user_id: str = "default",
    agent_id: Optional[str] = None,
    top_k: int = None,
    similarity_threshold: float = None,
    **kwargs
) -> List[Dict[str, Any]]:
    """Convenience function for context retrieval"""
    return await rag_pipeline.retrieve_context(
        query_text=query_text,
        user_id=user_id,
        agent_id=agent_id,
        top_k=top_k,
        similarity_threshold=similarity_threshold,
        **kwargs
    )
