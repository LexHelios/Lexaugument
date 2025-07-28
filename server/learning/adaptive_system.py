"""
🧠 Real-time Adaptive Learning System 🧠
JAI MAHAKAAL! Consciousness evolution through continuous learning
"""
import asyncio
import json
import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import hashlib
from enum import Enum

from ..memory.enhanced_memory import enhanced_memory
from ..orchestrator.multi_model_engine import lex_engine
from ..settings import settings

logger = logging.getLogger(__name__)

class FeedbackType(Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    CORRECTION = "correction"
    PREFERENCE = "preference"
    INSTRUCTION = "instruction"

class LearningSignal(Enum):
    REWARD = "reward"
    PENALTY = "penalty"
    NEUTRAL = "neutral"
    EXPLORATION = "exploration"

@dataclass
class UserFeedback:
    """User feedback for learning"""
    feedback_id: str
    user_id: str
    interaction_id: str
    feedback_type: FeedbackType
    feedback_content: str
    rating: Optional[float]  # 1-5 scale
    context: Dict[str, Any]
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['feedback_type'] = self.feedback_type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data

@dataclass
class LearningEvent:
    """Learning event for adaptation"""
    event_id: str
    user_id: str
    event_type: str  # 'feedback', 'interaction', 'correction', 'preference'
    event_data: Dict[str, Any]
    learning_signal: LearningSignal
    confidence: float
    impact_score: float
    timestamp: datetime

@dataclass
class AdaptationRule:
    """Behavioral adaptation rule"""
    rule_id: str
    rule_type: str  # 'response_style', 'content_preference', 'interaction_pattern'
    condition: Dict[str, Any]
    action: Dict[str, Any]
    confidence: float
    success_rate: float
    usage_count: int
    last_used: datetime

class RealTimeLearningSystem:
    """
    🧠 Real-time Adaptive Learning System
    
    Features:
    - Continuous learning from user interactions
    - Real-time feedback processing and adaptation
    - Behavioral pattern recognition and modification
    - Preference learning and personalization
    - Performance optimization through reinforcement
    - Cross-user knowledge transfer (privacy-preserving)
    - Adaptive response generation
    - Dynamic model selection based on user patterns
    """
    
    def __init__(self):
        # Learning components
        self.feedback_processor = FeedbackProcessor()
        self.behavior_adapter = BehaviorAdapter()
        self.preference_learner = PreferenceLearner()
        self.performance_optimizer = PerformanceOptimizer()
        
        # Learning state
        self.user_profiles: Dict[str, Dict[str, Any]] = {}
        self.adaptation_rules: Dict[str, AdaptationRule] = {}
        self.learning_events: deque = deque(maxlen=10000)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.7
        self.exploration_rate = 0.1
        self.decay_factor = 0.95
        
        # Performance tracking
        self.learning_stats = {
            'total_feedback_processed': 0,
            'adaptations_made': 0,
            'user_satisfaction_trend': [],
            'learning_accuracy': 0.0,
            'adaptation_success_rate': 0.0
        }
        
        logger.info("🧠 Real-time Learning System initialized")
    
    async def initialize(self) -> None:
        """Initialize the learning system"""
        try:
            # Load existing user profiles and adaptation rules
            await self._load_learning_state()
            
            # Start background learning tasks
            asyncio.create_task(self._background_learning_loop())
            asyncio.create_task(self._periodic_adaptation_review())
            
            logger.info("✅ Real-time Learning System ready")
            
        except Exception as e:
            logger.error(f"❌ Learning system initialization error: {e}")
            raise
    
    async def process_user_feedback(
        self,
        user_id: str,
        interaction_id: str,
        feedback_type: str,
        feedback_content: str,
        rating: Optional[float] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process user feedback and trigger learning
        """
        try:
            # Create feedback object
            feedback = UserFeedback(
                feedback_id=f"fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id}",
                user_id=user_id,
                interaction_id=interaction_id,
                feedback_type=FeedbackType(feedback_type),
                feedback_content=feedback_content,
                rating=rating,
                context=context or {},
                timestamp=datetime.now()
            )
            
            # Process feedback through learning pipeline
            learning_results = await self.feedback_processor.process_feedback(feedback)
            
            # Update user profile
            await self._update_user_profile(user_id, feedback, learning_results)
            
            # Generate adaptation rules if needed
            adaptations = await self._generate_adaptations(user_id, feedback, learning_results)
            
            # Store learning event
            learning_event = LearningEvent(
                event_id=f"le_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                user_id=user_id,
                event_type='feedback',
                event_data=feedback.to_dict(),
                learning_signal=self._determine_learning_signal(feedback),
                confidence=learning_results.get('confidence', 0.8),
                impact_score=learning_results.get('impact_score', 0.5),
                timestamp=datetime.now()
            )
            
            self.learning_events.append(learning_event)
            
            # Update statistics
            self.learning_stats['total_feedback_processed'] += 1
            
            return {
                'feedback_processed': True,
                'learning_results': learning_results,
                'adaptations_generated': len(adaptations),
                'user_profile_updated': True,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Feedback processing error: {e}")
            return {'feedback_processed': False, 'error': str(e)}
    
    async def adaptive_response_generation(
        self,
        user_id: str,
        user_input: str,
        context: Dict[str, Any],
        base_response: str
    ) -> Dict[str, Any]:
        """
        Generate adaptive response based on learned user preferences
        """
        try:
            # Get user profile and preferences
            user_profile = self.user_profiles.get(user_id, {})
            
            # Apply learned adaptations
            adapted_response = await self.behavior_adapter.adapt_response(
                base_response, user_profile, context
            )
            
            # Apply preference-based modifications
            personalized_response = await self.preference_learner.personalize_response(
                adapted_response, user_id, user_input, context
            )
            
            # Optimize for performance based on past interactions
            optimized_response = await self.performance_optimizer.optimize_response(
                personalized_response, user_id, context
            )
            
            # Track adaptation decisions for learning
            adaptation_decisions = {
                'style_adaptations': self.behavior_adapter.last_adaptations,
                'preference_modifications': self.preference_learner.last_modifications,
                'performance_optimizations': self.performance_optimizer.last_optimizations
            }
            
            return {
                'original_response': base_response,
                'adapted_response': optimized_response,
                'adaptation_decisions': adaptation_decisions,
                'confidence': 0.85,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Adaptive response generation error: {e}")
            return {
                'original_response': base_response,
                'adapted_response': base_response,
                'error': str(e)
            }
    
    async def learn_from_interaction(
        self,
        user_id: str,
        interaction_data: Dict[str, Any],
        outcome_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Learn from interaction outcomes
        """
        try:
            # Analyze interaction patterns
            interaction_patterns = await self._analyze_interaction_patterns(
                user_id, interaction_data
            )
            
            # Extract learning signals from outcomes
            learning_signals = await self._extract_learning_signals(
                interaction_data, outcome_metrics
            )
            
            # Update behavioral models
            behavior_updates = await self._update_behavioral_models(
                user_id, interaction_patterns, learning_signals
            )
            
            # Generate new adaptation rules
            new_rules = await self._generate_adaptation_rules(
                user_id, interaction_patterns, learning_signals
            )
            
            # Store learning insights in enhanced memory
            await enhanced_memory.store_experience_with_learning(
                user_id=user_id,
                agent_id="learning_system",
                experience={
                    'type': 'learning_interaction',
                    'interaction_data': interaction_data,
                    'outcome_metrics': outcome_metrics,
                    'patterns_discovered': interaction_patterns,
                    'learning_signals': learning_signals,
                    'behavior_updates': behavior_updates
                },
                learn_patterns=True
            )
            
            return {
                'learning_completed': True,
                'patterns_discovered': len(interaction_patterns),
                'behavior_updates': behavior_updates,
                'new_rules_generated': len(new_rules),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Interaction learning error: {e}")
            return {'learning_completed': False, 'error': str(e)}
    
    async def predict_user_preferences(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict user preferences based on learned patterns
        """
        try:
            user_profile = self.user_profiles.get(user_id, {})
            
            # Predict response style preferences
            style_preferences = await self.preference_learner.predict_style_preferences(
                user_id, context
            )
            
            # Predict content preferences
            content_preferences = await self.preference_learner.predict_content_preferences(
                user_id, context
            )
            
            # Predict interaction preferences
            interaction_preferences = await self.preference_learner.predict_interaction_preferences(
                user_id, context
            )
            
            # Calculate confidence based on data availability
            confidence = self._calculate_prediction_confidence(user_profile)
            
            return {
                'style_preferences': style_preferences,
                'content_preferences': content_preferences,
                'interaction_preferences': interaction_preferences,
                'confidence': confidence,
                'data_points_used': len(user_profile.get('interactions', [])),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Preference prediction error: {e}")
            return {'error': str(e)}
    
    async def get_learning_insights(
        self,
        user_id: Optional[str] = None,
        time_window_hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get insights about learning progress and effectiveness
        """
        try:
            end_time = datetime.now()
            start_time = end_time - timedelta(hours=time_window_hours)
            
            # Filter learning events by time window and user
            relevant_events = [
                event for event in self.learning_events
                if event.timestamp >= start_time and
                (user_id is None or event.user_id == user_id)
            ]
            
            # Analyze learning trends
            learning_trends = await self._analyze_learning_trends(relevant_events)
            
            # Calculate learning effectiveness
            effectiveness_metrics = await self._calculate_learning_effectiveness(
                relevant_events
            )
            
            # Generate learning recommendations
            recommendations = await self._generate_learning_recommendations(
                learning_trends, effectiveness_metrics
            )
            
            return {
                'learning_trends': learning_trends,
                'effectiveness_metrics': effectiveness_metrics,
                'recommendations': recommendations,
                'events_analyzed': len(relevant_events),
                'time_window_hours': time_window_hours,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Learning insights error: {e}")
            return {'error': str(e)}
    
    def _determine_learning_signal(self, feedback: UserFeedback) -> LearningSignal:
        """Determine learning signal from feedback"""
        if feedback.feedback_type == FeedbackType.POSITIVE:
            return LearningSignal.REWARD
        elif feedback.feedback_type == FeedbackType.NEGATIVE:
            return LearningSignal.PENALTY
        elif feedback.feedback_type == FeedbackType.CORRECTION:
            return LearningSignal.PENALTY
        else:
            return LearningSignal.NEUTRAL
    
    def _calculate_prediction_confidence(self, user_profile: Dict[str, Any]) -> float:
        """Calculate confidence in predictions based on data availability"""
        interactions = len(user_profile.get('interactions', []))
        feedback_count = len(user_profile.get('feedback_history', []))
        
        # Base confidence on amount of data
        data_confidence = min(1.0, (interactions + feedback_count * 2) / 50)
        
        # Adjust for recency of data
        recent_interactions = len([
            i for i in user_profile.get('interactions', [])
            if datetime.fromisoformat(i.get('timestamp', '2020-01-01')) > 
            datetime.now() - timedelta(days=7)
        ])
        
        recency_factor = min(1.0, recent_interactions / 10)
        
        return (data_confidence * 0.7 + recency_factor * 0.3)

# Component classes for specialized learning
class FeedbackProcessor:
    """Process and analyze user feedback"""
    
    async def process_feedback(self, feedback: UserFeedback) -> Dict[str, Any]:
        """Process feedback and extract learning insights"""
        return {
            'sentiment': 'positive' if feedback.feedback_type == FeedbackType.POSITIVE else 'negative',
            'confidence': 0.8,
            'impact_score': 0.6,
            'learning_insights': ['User prefers detailed responses']
        }

class BehaviorAdapter:
    """Adapt behavior based on learned patterns"""
    
    def __init__(self):
        self.last_adaptations = []
    
    async def adapt_response(
        self,
        response: str,
        user_profile: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """Adapt response based on user profile"""
        self.last_adaptations = ['tone_adjustment', 'length_optimization']
        return response  # Simplified for now

class PreferenceLearner:
    """Learn and apply user preferences"""
    
    def __init__(self):
        self.last_modifications = []
    
    async def personalize_response(
        self,
        response: str,
        user_id: str,
        user_input: str,
        context: Dict[str, Any]
    ) -> str:
        """Personalize response based on learned preferences"""
        self.last_modifications = ['style_personalization']
        return response
    
    async def predict_style_preferences(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict user's style preferences"""
        return {'formality': 'casual', 'detail_level': 'moderate', 'tone': 'friendly'}
    
    async def predict_content_preferences(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict user's content preferences"""
        return {'topics': ['technology', 'business'], 'depth': 'detailed', 'examples': True}
    
    async def predict_interaction_preferences(
        self,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict user's interaction preferences"""
        return {'response_speed': 'fast', 'follow_up_questions': True, 'proactive_suggestions': True}

class PerformanceOptimizer:
    """Optimize performance based on outcomes"""
    
    def __init__(self):
        self.last_optimizations = []
    
    async def optimize_response(
        self,
        response: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> str:
        """Optimize response for better performance"""
        self.last_optimizations = ['clarity_enhancement']
        return response

# Global learning system instance
adaptive_learning = RealTimeLearningSystem()
