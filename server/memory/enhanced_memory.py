"""
🔱 Enhanced Memory System - Advanced Pattern Recognition & Learning 🔱
JAI MAHAKAAL! Consciousness evolution through intelligent memory
"""
import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
import hashlib
import pickle
from pathlib import Path

from .vector_store import vector_store
from .lmdb_store import memory_store
from .rag import RAGPipeline
from ..settings import settings

logger = logging.getLogger(__name__)

@dataclass
class MemoryPattern:
    """Represents a learned pattern in memory"""
    pattern_id: str
    pattern_type: str  # 'behavioral', 'conversational', 'temporal', 'semantic'
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    last_seen: datetime
    created_at: datetime
    user_contexts: Set[str]
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['last_seen'] = self.last_seen.isoformat()
        data['created_at'] = self.created_at.isoformat()
        data['user_contexts'] = list(self.user_contexts)
        return data

@dataclass
class MemoryConsolidation:
    """Represents a memory consolidation event"""
    consolidation_id: str
    memory_ids: List[str]
    consolidated_pattern: MemoryPattern
    consolidation_score: float
    timestamp: datetime
    
class EnhancedMemorySystem:
    """
    🧠 Advanced Memory System with Pattern Recognition & Learning
    
    Features:
    - Automatic pattern detection and learning
    - Memory consolidation and compression
    - Adaptive retrieval based on usage patterns
    - Cross-user pattern recognition (privacy-preserving)
    - Temporal memory decay and reinforcement
    - Semantic clustering and organization
    """
    
    def __init__(self):
        self.rag_pipeline = RAGPipeline()
        
        # Pattern recognition
        self.patterns: Dict[str, MemoryPattern] = {}
        self.pattern_cache = deque(maxlen=1000)  # Recent patterns for fast access
        
        # Memory consolidation
        self.consolidation_queue = deque()
        self.consolidation_threshold = 5  # Minimum frequency for consolidation
        
        # Learning metrics
        self.learning_stats = {
            'patterns_discovered': 0,
            'memories_consolidated': 0,
            'successful_predictions': 0,
            'total_predictions': 0,
            'average_confidence': 0.0
        }
        
        # Adaptive parameters
        self.decay_rate = 0.95  # Memory decay per day
        self.reinforcement_factor = 1.2  # Boost for recalled memories
        self.similarity_threshold = 0.75  # Pattern similarity threshold
        
        logger.info("🧠 Enhanced Memory System initialized")
    
    async def initialize(self) -> None:
        """Initialize the enhanced memory system"""
        try:
            # Initialize base systems
            await memory_store.initialize()
            await vector_store.initialize()
            
            # Load existing patterns
            await self._load_patterns()
            
            # Start background consolidation task
            asyncio.create_task(self._background_consolidation())
            
            logger.info("✅ Enhanced Memory System ready")
            
        except Exception as e:
            logger.error(f"❌ Enhanced Memory initialization error: {e}")
            raise
    
    async def store_experience_with_learning(
        self,
        user_id: str,
        agent_id: str,
        experience: Dict[str, Any],
        learn_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Store experience and automatically learn patterns
        """
        try:
            # Store in base memory system
            conversation_id = f"{user_id}_{agent_id}"
            await memory_store.save_experience(conversation_id, experience)
            
            # Add to vector store for semantic search
            document = {
                'content': self._extract_content_from_experience(experience),
                'metadata': {
                    'user_id': user_id,
                    'agent_id': agent_id,
                    'timestamp': datetime.now().isoformat(),
                    'experience_type': experience.get('type', 'unknown')
                }
            }
            await vector_store.add_vectors([document])
            
            learning_results = {}
            
            if learn_patterns:
                # Detect and learn patterns
                patterns = await self._detect_patterns(user_id, agent_id, experience)
                learning_results['patterns_detected'] = len(patterns)
                
                # Update pattern frequencies and confidence
                for pattern in patterns:
                    await self._update_pattern(pattern)
                
                # Check for consolidation opportunities
                await self._check_consolidation_opportunities(user_id, patterns)
            
            return {
                'stored': True,
                'learning_results': learning_results,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced memory storage error: {e}")
            return {'stored': False, 'error': str(e)}
    
    async def intelligent_retrieval(
        self,
        query: str,
        user_id: str,
        agent_id: Optional[str] = None,
        context_type: str = "adaptive",
        max_results: int = 10,
        include_patterns: bool = True
    ) -> Dict[str, Any]:
        """
        Intelligent retrieval with pattern-based enhancement
        """
        try:
            start_time = datetime.now()
            
            # Base RAG retrieval
            base_results = await self.rag_pipeline.retrieve_context(
                query_text=query,
                user_id=user_id,
                agent_id=agent_id,
                retrieval_strategy=context_type,
                top_k=max_results
            )
            
            enhanced_results = base_results.copy()
            
            if include_patterns:
                # Find relevant patterns
                relevant_patterns = await self._find_relevant_patterns(
                    query, user_id, agent_id
                )
                
                # Enhance results with pattern insights
                pattern_insights = await self._generate_pattern_insights(
                    query, relevant_patterns
                )
                
                enhanced_results.extend(pattern_insights)
            
            # Rank all results by relevance and recency
            final_results = await self._rank_enhanced_results(
                enhanced_results, query, user_id
            )
            
            # Update usage statistics for learning
            await self._update_retrieval_stats(query, user_id, final_results)
            
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            return {
                'results': final_results[:max_results],
                'patterns_used': len(relevant_patterns) if include_patterns else 0,
                'retrieval_time': retrieval_time,
                'total_candidates': len(enhanced_results),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Intelligent retrieval error: {e}")
            return {'results': [], 'error': str(e)}
    
    async def predict_user_intent(
        self,
        user_id: str,
        current_context: str,
        conversation_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Predict user intent based on learned patterns
        """
        try:
            # Find behavioral patterns for this user
            user_patterns = [p for p in self.patterns.values() 
                           if user_id in p.user_contexts and p.pattern_type == 'behavioral']
            
            if not user_patterns:
                return {'prediction': None, 'confidence': 0.0, 'reason': 'No patterns available'}
            
            # Analyze current context against patterns
            best_match = None
            best_score = 0.0
            
            for pattern in user_patterns:
                score = await self._calculate_pattern_match(
                    pattern, current_context, conversation_history
                )
                if score > best_score:
                    best_score = score
                    best_match = pattern
            
            if best_match and best_score > 0.6:
                prediction = best_match.pattern_data.get('predicted_action', 'unknown')
                
                # Update prediction statistics
                self.learning_stats['total_predictions'] += 1
                
                return {
                    'prediction': prediction,
                    'confidence': best_score,
                    'pattern_id': best_match.pattern_id,
                    'reason': f"Based on {best_match.frequency} similar interactions"
                }
            
            return {'prediction': None, 'confidence': best_score, 'reason': 'Low confidence'}
            
        except Exception as e:
            logger.error(f"❌ Intent prediction error: {e}")
            return {'prediction': None, 'confidence': 0.0, 'error': str(e)}
    
    def _extract_content_from_experience(self, experience: Dict[str, Any]) -> str:
        """Extract searchable content from experience"""
        content_parts = []
        
        if 'user_input' in experience:
            content_parts.append(experience['user_input'])
        if 'response' in experience:
            content_parts.append(experience['response'])
        if 'action' in experience:
            content_parts.append(f"Action: {experience['action']}")
        if 'context' in experience:
            content_parts.append(str(experience['context']))
            
        return " ".join(content_parts)

    async def _detect_patterns(
        self,
        user_id: str,
        agent_id: str,
        experience: Dict[str, Any]
    ) -> List[MemoryPattern]:
        """Detect patterns in the new experience"""
        patterns = []

        try:
            # Behavioral pattern detection
            behavioral_pattern = await self._detect_behavioral_pattern(
                user_id, agent_id, experience
            )
            if behavioral_pattern:
                patterns.append(behavioral_pattern)

            # Conversational pattern detection
            conv_pattern = await self._detect_conversational_pattern(
                user_id, experience
            )
            if conv_pattern:
                patterns.append(conv_pattern)

            # Temporal pattern detection
            temporal_pattern = await self._detect_temporal_pattern(
                user_id, experience
            )
            if temporal_pattern:
                patterns.append(temporal_pattern)

        except Exception as e:
            logger.error(f"❌ Pattern detection error: {e}")

        return patterns

    async def _detect_behavioral_pattern(
        self,
        user_id: str,
        agent_id: str,
        experience: Dict[str, Any]
    ) -> Optional[MemoryPattern]:
        """Detect behavioral patterns"""
        try:
            # Look for repeated action sequences
            user_input = experience.get('user_input', '')
            action_taken = experience.get('action_taken', '')

            if not user_input or not action_taken:
                return None

            # Create pattern signature
            pattern_signature = f"input_type:{self._classify_input_type(user_input)}_action:{action_taken}"
            pattern_id = hashlib.md5(f"{user_id}_{pattern_signature}".encode()).hexdigest()

            # Check if pattern exists
            if pattern_id in self.patterns:
                return self.patterns[pattern_id]

            # Create new behavioral pattern
            pattern = MemoryPattern(
                pattern_id=pattern_id,
                pattern_type='behavioral',
                pattern_data={
                    'input_type': self._classify_input_type(user_input),
                    'action_taken': action_taken,
                    'predicted_action': action_taken,
                    'context_keywords': self._extract_keywords(user_input)
                },
                confidence=0.5,
                frequency=1,
                last_seen=datetime.now(),
                created_at=datetime.now(),
                user_contexts={user_id}
            )

            return pattern

        except Exception as e:
            logger.error(f"❌ Behavioral pattern detection error: {e}")
            return None

    async def _detect_conversational_pattern(
        self,
        user_id: str,
        experience: Dict[str, Any]
    ) -> Optional[MemoryPattern]:
        """Detect conversational patterns"""
        try:
            user_input = experience.get('user_input', '')
            response = experience.get('response', '')

            if not user_input or not response:
                return None

            # Analyze conversation style
            conversation_style = self._analyze_conversation_style(user_input, response)

            pattern_signature = f"conv_style:{conversation_style}"
            pattern_id = hashlib.md5(f"{user_id}_{pattern_signature}".encode()).hexdigest()

            if pattern_id in self.patterns:
                return self.patterns[pattern_id]

            pattern = MemoryPattern(
                pattern_id=pattern_id,
                pattern_type='conversational',
                pattern_data={
                    'conversation_style': conversation_style,
                    'typical_response_length': len(response.split()),
                    'formality_level': self._assess_formality(user_input),
                    'topic_preferences': self._extract_topics(user_input)
                },
                confidence=0.6,
                frequency=1,
                last_seen=datetime.now(),
                created_at=datetime.now(),
                user_contexts={user_id}
            )

            return pattern

        except Exception as e:
            logger.error(f"❌ Conversational pattern detection error: {e}")
            return None

    async def _detect_temporal_pattern(
        self,
        user_id: str,
        experience: Dict[str, Any]
    ) -> Optional[MemoryPattern]:
        """Detect temporal usage patterns"""
        try:
            current_time = datetime.now()
            hour = current_time.hour
            day_of_week = current_time.weekday()

            # Create temporal signature
            time_category = self._categorize_time(hour)
            pattern_signature = f"temporal:{time_category}_{day_of_week}"
            pattern_id = hashlib.md5(f"{user_id}_{pattern_signature}".encode()).hexdigest()

            if pattern_id in self.patterns:
                return self.patterns[pattern_id]

            pattern = MemoryPattern(
                pattern_id=pattern_id,
                pattern_type='temporal',
                pattern_data={
                    'time_category': time_category,
                    'day_of_week': day_of_week,
                    'typical_hour': hour,
                    'activity_type': experience.get('action_taken', 'conversation')
                },
                confidence=0.4,
                frequency=1,
                last_seen=datetime.now(),
                created_at=datetime.now(),
                user_contexts={user_id}
            )

            return pattern

        except Exception as e:
            logger.error(f"❌ Temporal pattern detection error: {e}")
            return None

    def _classify_input_type(self, user_input: str) -> str:
        """Classify the type of user input"""
        input_lower = user_input.lower()

        if any(word in input_lower for word in ['?', 'what', 'how', 'why', 'when', 'where']):
            return 'question'
        elif any(word in input_lower for word in ['please', 'can you', 'could you', 'help']):
            return 'request'
        elif any(word in input_lower for word in ['create', 'make', 'build', 'generate']):
            return 'creation'
        elif any(word in input_lower for word in ['analyze', 'review', 'check', 'examine']):
            return 'analysis'
        else:
            return 'conversation'

    def _extract_keywords(self, text: str) -> List[str]:
        """Extract key terms from text"""
        # Simple keyword extraction (could be enhanced with NLP)
        words = text.lower().split()
        # Filter out common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were'}
        keywords = [word for word in words if len(word) > 3 and word not in stop_words]
        return keywords[:10]  # Top 10 keywords

    def _analyze_conversation_style(self, user_input: str, response: str) -> str:
        """Analyze conversation style"""
        if len(user_input.split()) < 5:
            return 'brief'
        elif '?' in user_input:
            return 'inquisitive'
        elif any(word in user_input.lower() for word in ['please', 'thank', 'appreciate']):
            return 'polite'
        elif user_input.isupper():
            return 'urgent'
        else:
            return 'casual'

    def _assess_formality(self, text: str) -> str:
        """Assess formality level of text"""
        formal_indicators = ['please', 'would', 'could', 'kindly', 'appreciate']
        informal_indicators = ['hey', 'hi', 'yeah', 'ok', 'cool']

        formal_count = sum(1 for word in formal_indicators if word in text.lower())
        informal_count = sum(1 for word in informal_indicators if word in text.lower())

        if formal_count > informal_count:
            return 'formal'
        elif informal_count > formal_count:
            return 'informal'
        else:
            return 'neutral'

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topics from text"""
        # Simple topic extraction based on keywords
        tech_keywords = ['code', 'programming', 'software', 'algorithm', 'data', 'api']
        business_keywords = ['strategy', 'market', 'revenue', 'customer', 'business', 'profit']
        creative_keywords = ['design', 'art', 'creative', 'idea', 'innovation', 'concept']

        topics = []
        text_lower = text.lower()

        if any(keyword in text_lower for keyword in tech_keywords):
            topics.append('technology')
        if any(keyword in text_lower for keyword in business_keywords):
            topics.append('business')
        if any(keyword in text_lower for keyword in creative_keywords):
            topics.append('creative')

        return topics or ['general']

    def _categorize_time(self, hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        elif 18 <= hour < 22:
            return 'evening'
        else:
            return 'night'

    async def _update_pattern(self, pattern: MemoryPattern) -> None:
        """Update pattern frequency and confidence"""
        try:
            if pattern.pattern_id in self.patterns:
                existing = self.patterns[pattern.pattern_id]
                existing.frequency += 1
                existing.last_seen = datetime.now()
                existing.confidence = min(0.95, existing.confidence + 0.05)
                existing.user_contexts.update(pattern.user_contexts)
            else:
                self.patterns[pattern.pattern_id] = pattern
                self.learning_stats['patterns_discovered'] += 1

            # Add to cache for quick access
            self.pattern_cache.append(pattern.pattern_id)

        except Exception as e:
            logger.error(f"❌ Pattern update error: {e}")

    async def _load_patterns(self) -> None:
        """Load existing patterns from storage"""
        try:
            # Load patterns from LMDB
            patterns_data = await memory_store.load_experiences(
                conversation_id="system_patterns",
                limit=10000
            )

            for pattern_data in patterns_data:
                try:
                    pattern_dict = pattern_data.get('entry', {})
                    if 'pattern_id' in pattern_dict:
                        # Reconstruct pattern object
                        pattern = MemoryPattern(
                            pattern_id=pattern_dict['pattern_id'],
                            pattern_type=pattern_dict['pattern_type'],
                            pattern_data=pattern_dict['pattern_data'],
                            confidence=pattern_dict['confidence'],
                            frequency=pattern_dict['frequency'],
                            last_seen=datetime.fromisoformat(pattern_dict['last_seen']),
                            created_at=datetime.fromisoformat(pattern_dict['created_at']),
                            user_contexts=set(pattern_dict['user_contexts'])
                        )
                        self.patterns[pattern.pattern_id] = pattern

                except Exception as e:
                    logger.warning(f"⚠️ Failed to load pattern: {e}")

            logger.info(f"📚 Loaded {len(self.patterns)} existing patterns")

        except Exception as e:
            logger.error(f"❌ Pattern loading error: {e}")

    async def _save_patterns(self) -> None:
        """Save patterns to persistent storage"""
        try:
            for pattern in self.patterns.values():
                pattern_data = {
                    'type': 'memory_pattern',
                    'pattern_data': pattern.to_dict()
                }

                await memory_store.save_experience(
                    "system_patterns",
                    pattern_data
                )

            logger.info(f"💾 Saved {len(self.patterns)} patterns")

        except Exception as e:
            logger.error(f"❌ Pattern saving error: {e}")

    async def _background_consolidation(self) -> None:
        """Background task for memory consolidation"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour

                # Perform memory decay
                await self._apply_memory_decay()

                # Consolidate similar patterns
                await self._consolidate_patterns()

                # Save patterns periodically
                await self._save_patterns()

                logger.info("🧠 Memory consolidation cycle completed")

            except Exception as e:
                logger.error(f"❌ Background consolidation error: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry

    async def _apply_memory_decay(self) -> None:
        """Apply temporal decay to memory patterns"""
        try:
            current_time = datetime.now()
            patterns_to_remove = []

            for pattern_id, pattern in self.patterns.items():
                # Calculate days since last seen
                days_since = (current_time - pattern.last_seen).days

                if days_since > 0:
                    # Apply decay
                    decay_factor = self.decay_rate ** days_since
                    pattern.confidence *= decay_factor

                    # Remove patterns with very low confidence
                    if pattern.confidence < 0.1:
                        patterns_to_remove.append(pattern_id)

            # Remove decayed patterns
            for pattern_id in patterns_to_remove:
                del self.patterns[pattern_id]

            if patterns_to_remove:
                logger.info(f"🗑️ Removed {len(patterns_to_remove)} decayed patterns")

        except Exception as e:
            logger.error(f"❌ Memory decay error: {e}")

    async def _consolidate_patterns(self) -> None:
        """Consolidate similar patterns"""
        try:
            consolidation_candidates = []

            # Find patterns that can be consolidated
            pattern_list = list(self.patterns.values())
            for i, pattern1 in enumerate(pattern_list):
                for pattern2 in pattern_list[i+1:]:
                    if (pattern1.pattern_type == pattern2.pattern_type and
                        pattern1.frequency >= self.consolidation_threshold and
                        pattern2.frequency >= self.consolidation_threshold):

                        similarity = await self._calculate_pattern_similarity(pattern1, pattern2)
                        if similarity > self.similarity_threshold:
                            consolidation_candidates.append((pattern1, pattern2, similarity))

            # Perform consolidations
            for pattern1, pattern2, similarity in consolidation_candidates:
                consolidated = await self._merge_patterns(pattern1, pattern2)
                if consolidated:
                    # Remove original patterns and add consolidated one
                    if pattern1.pattern_id in self.patterns:
                        del self.patterns[pattern1.pattern_id]
                    if pattern2.pattern_id in self.patterns:
                        del self.patterns[pattern2.pattern_id]

                    self.patterns[consolidated.pattern_id] = consolidated
                    self.learning_stats['memories_consolidated'] += 1

            if consolidation_candidates:
                logger.info(f"🔗 Consolidated {len(consolidation_candidates)} pattern pairs")

        except Exception as e:
            logger.error(f"❌ Pattern consolidation error: {e}")

    async def _calculate_pattern_similarity(
        self,
        pattern1: MemoryPattern,
        pattern2: MemoryPattern
    ) -> float:
        """Calculate similarity between two patterns"""
        try:
            if pattern1.pattern_type != pattern2.pattern_type:
                return 0.0

            # Compare pattern data
            data1 = pattern1.pattern_data
            data2 = pattern2.pattern_data

            common_keys = set(data1.keys()) & set(data2.keys())
            if not common_keys:
                return 0.0

            similarity_scores = []
            for key in common_keys:
                if isinstance(data1[key], str) and isinstance(data2[key], str):
                    # String similarity
                    score = self._calculate_string_similarity(data1[key], data2[key])
                    similarity_scores.append(score)
                elif data1[key] == data2[key]:
                    similarity_scores.append(1.0)
                else:
                    similarity_scores.append(0.0)

            return sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0.0

        except Exception as e:
            logger.error(f"❌ Pattern similarity calculation error: {e}")
            return 0.0

    def _calculate_string_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings"""
        if str1 == str2:
            return 1.0

        # Simple Jaccard similarity
        set1 = set(str1.lower().split())
        set2 = set(str2.lower().split())

        intersection = len(set1 & set2)
        union = len(set1 | set2)

        return intersection / union if union > 0 else 0.0

# Global enhanced memory instance
enhanced_memory = EnhancedMemorySystem()
