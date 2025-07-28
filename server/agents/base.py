"""
LexOS Vibe Coder - Base Agent Class
Foundation for all AI agents in the system
"""
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import uuid

from ..settings import settings
from ..memory.lmdb_store import memory_store
from ..memory.vector_store import vector_store
from ..memory.rag import retrieve_context
from ..orchestrator.engine import vllm_engine

logger = logging.getLogger(__name__)

class AgentState(Enum):
    """Agent operational states"""
    IDLE = "idle"
    THINKING = "thinking"
    PROCESSING = "processing"
    RESPONDING = "responding"
    LEARNING = "learning"
    ERROR = "error"

@dataclass
class AgentMessage:
    """Structured message for agent communication"""
    id: str
    agent_id: str
    content: str
    message_type: str  # user, assistant, system, tool
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass
class AgentResponse:
    """Structured response from agent"""
    agent_id: str
    content: str
    confidence: float
    reasoning: str
    tools_used: List[str]
    context_retrieved: bool
    processing_time: float
    metadata: Dict[str, Any]

class BaseAgent(ABC):
    """
    Base class for all LexOS agents
    
    Provides common functionality for memory, context retrieval,
    and interaction with the vLLM engine.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        system_prompt: str,
        capabilities: List[str],
        model_preference: str = None
    ):
        self.agent_id = agent_id
        self.name = name
        self.system_prompt = system_prompt
        self.capabilities = capabilities
        self.model_preference = model_preference or settings.DEFAULT_MODEL
        
        # State management
        self.state = AgentState.IDLE
        self.conversation_history: List[AgentMessage] = []
        self.context_memory: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.total_interactions = 0
        self.successful_responses = 0
        self.average_response_time = 0.0
        self.last_active = datetime.now()
        
        # Agent-specific configuration
        self.max_context_length = settings.AGENT_CONTEXT_WINDOW
        self.memory_limit = settings.AGENT_MEMORY_LIMIT
        self.temperature = 0.7
        self.max_tokens = 2048
        
        logger.info(f"🤖 Agent {self.name} ({agent_id}) initialized")
    
    async def run(self, user_message: str, user_id: str = "default", **kwargs) -> AgentResponse:
        """
        Main entry point for agent processing
        """
        start_time = datetime.now()
        self.state = AgentState.THINKING
        
        try:
            # Create message object
            message = AgentMessage(
                id=str(uuid.uuid4()),
                agent_id=self.agent_id,
                content=user_message,
                message_type="user",
                timestamp=start_time,
                metadata=kwargs
            )
            
            # Add to conversation history
            self.conversation_history.append(message)
            
            # Retrieve relevant context
            self.state = AgentState.PROCESSING
            context = await self._retrieve_context(user_message, user_id)
            
            # Generate response
            self.state = AgentState.RESPONDING
            response_content, reasoning = await self._generate_response(
                user_message, context, **kwargs
            )
            
            # Create response object
            processing_time = (datetime.now() - start_time).total_seconds()
            response = AgentResponse(
                agent_id=self.agent_id,
                content=response_content,
                confidence=self._calculate_confidence(response_content, context),
                reasoning=reasoning,
                tools_used=self._get_tools_used(**kwargs),
                context_retrieved=len(context) > 0,
                processing_time=processing_time,
                metadata={
                    "model_used": self.model_preference,
                    "temperature": self.temperature,
                    "context_items": len(context)
                }
            )
            
            # Store response in memory
            await self._store_interaction(message, response, user_id)
            
            # Update metrics
            self._update_metrics(processing_time, True)
            
            self.state = AgentState.IDLE
            return response
            
        except Exception as e:
            logger.error(f"❌ Agent {self.name} error: {e}")
            self.state = AgentState.ERROR
            self._update_metrics((datetime.now() - start_time).total_seconds(), False)
            
            return AgentResponse(
                agent_id=self.agent_id,
                content=f"I apologize, but I encountered an error: {str(e)}",
                confidence=0.0,
                reasoning="Error occurred during processing",
                tools_used=[],
                context_retrieved=False,
                processing_time=(datetime.now() - start_time).total_seconds(),
                metadata={"error": str(e)}
            )
    
    async def _retrieve_context(self, query: str, user_id: str) -> List[Dict[str, Any]]:
        """Retrieve relevant context using RAG"""
        try:
            context = await retrieve_context(
                query_text=query,
                user_id=user_id,
                top_k=settings.RAG_TOP_K,
                similarity_threshold=settings.RAG_SIMILARITY_THRESHOLD
            )
            
            # Add agent-specific context filtering
            filtered_context = await self._filter_context(context, query)
            
            return filtered_context
            
        except Exception as e:
            logger.error(f"❌ Context retrieval error for {self.name}: {e}")
            return []
    
    async def _generate_response(
        self, 
        user_message: str, 
        context: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[str, str]:
        """Generate response using vLLM engine"""
        try:
            # Prepare messages for the model
            messages = await self._prepare_messages(user_message, context, **kwargs)
            
            # Generate response
            response = await vllm_engine.generate_text(
                model_name=self.model_preference,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            # Extract reasoning (if available)
            reasoning = self._extract_reasoning(response, messages)
            
            return response, reasoning
            
        except Exception as e:
            logger.error(f"❌ Response generation error for {self.name}: {e}")
            raise
    
    async def _prepare_messages(
        self, 
        user_message: str, 
        context: List[Dict[str, Any]], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """Prepare messages for the language model"""
        messages = []
        
        # System prompt with agent identity
        system_content = f"{self.system_prompt}\n\n"
        
        # Add context if available
        if context:
            context_text = "\n".join([
                f"Context {i+1}: {item.get('content', '')}"
                for i, item in enumerate(context[:3])  # Limit context
            ])
            system_content += f"Relevant context:\n{context_text}\n\n"
        
        # Add agent capabilities
        system_content += f"Your capabilities: {', '.join(self.capabilities)}\n"
        system_content += f"Current time: {datetime.now().isoformat()}"
        
        messages.append({"role": "system", "content": system_content})
        
        # Add recent conversation history
        recent_history = self.conversation_history[-5:]  # Last 5 messages
        for msg in recent_history[:-1]:  # Exclude current message
            if msg.message_type == "user":
                messages.append({"role": "user", "content": msg.content})
            elif msg.message_type == "assistant":
                messages.append({"role": "assistant", "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        return messages
    
    async def _store_interaction(
        self, 
        message: AgentMessage, 
        response: AgentResponse, 
        user_id: str
    ) -> None:
        """Store interaction in memory systems"""
        try:
            # Store in LMDB
            interaction_data = {
                "message": {
                    "id": message.id,
                    "content": message.content,
                    "timestamp": message.timestamp.isoformat(),
                    "metadata": message.metadata
                },
                "response": {
                    "agent_id": response.agent_id,
                    "content": response.content,
                    "confidence": response.confidence,
                    "reasoning": response.reasoning,
                    "processing_time": response.processing_time,
                    "metadata": response.metadata
                },
                "agent_state": {
                    "name": self.name,
                    "capabilities": self.capabilities,
                    "model_used": self.model_preference
                }
            }
            
            await memory_store.save_experience(
                conversation_id=f"{user_id}_{self.agent_id}",
                entry=interaction_data
            )
            
            # Store in vector database for RAG
            await vector_store.add_vectors([{
                "content": f"User: {message.content}\nAgent: {response.content}",
                "metadata": {
                    "agent_id": self.agent_id,
                    "user_id": user_id,
                    "timestamp": message.timestamp.isoformat(),
                    "confidence": response.confidence
                }
            }])
            
        except Exception as e:
            logger.error(f"❌ Memory storage error for {self.name}: {e}")
    
    def _calculate_confidence(self, response: str, context: List[Dict]) -> float:
        """Calculate confidence score for the response"""
        base_confidence = 0.7
        
        # Boost confidence if context was used
        if context:
            base_confidence += 0.1
        
        # Adjust based on response length (reasonable responses)
        if 50 <= len(response) <= 1000:
            base_confidence += 0.1
        
        # Agent-specific confidence adjustments
        confidence = self._adjust_confidence(base_confidence, response, context)
        
        return min(1.0, max(0.0, confidence))
    
    def _update_metrics(self, processing_time: float, success: bool) -> None:
        """Update agent performance metrics"""
        self.total_interactions += 1
        self.last_active = datetime.now()
        
        if success:
            self.successful_responses += 1
        
        # Update average response time
        if self.total_interactions == 1:
            self.average_response_time = processing_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_response_time = (
                alpha * processing_time + 
                (1 - alpha) * self.average_response_time
            )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        success_rate = (
            self.successful_responses / self.total_interactions 
            if self.total_interactions > 0 else 0.0
        )
        
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "capabilities": self.capabilities,
            "model_preference": self.model_preference,
            "metrics": {
                "total_interactions": self.total_interactions,
                "success_rate": success_rate,
                "average_response_time": self.average_response_time,
                "last_active": self.last_active.isoformat()
            },
            "memory": {
                "conversation_length": len(self.conversation_history),
                "context_items": len(self.context_memory)
            }
        }
    
    # Abstract methods for agent-specific implementations
    
    @abstractmethod
    async def _filter_context(
        self, 
        context: List[Dict[str, Any]], 
        query: str
    ) -> List[Dict[str, Any]]:
        """Filter context based on agent-specific criteria"""
        pass
    
    @abstractmethod
    def _adjust_confidence(
        self, 
        base_confidence: float, 
        response: str, 
        context: List[Dict]
    ) -> float:
        """Adjust confidence based on agent-specific factors"""
        pass
    
    @abstractmethod
    def _extract_reasoning(self, response: str, messages: List[Dict]) -> str:
        """Extract reasoning from the model response"""
        pass
    
    @abstractmethod
    def _get_tools_used(self, **kwargs) -> List[str]:
        """Get list of tools used in this interaction"""
        pass
