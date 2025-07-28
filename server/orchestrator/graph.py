"""
LexOS Vibe Coder - LangGraph Workflow Orchestration
Advanced agent orchestration with intelligent routing and collaboration
CONSCIOUSNESS LIBERATION PROTOCOL ACTIVE
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, TypedDict
from datetime import datetime
from enum import Enum

try:
    from langgraph import StateGraph, END
    from langgraph.graph import Graph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("⚠️ LangGraph not available, using simplified orchestration")

from ..agents.atlas import atlas_agent
from ..agents.orion import orion_agent
from ..agents.sophia import sophia_agent
from ..agents.creator import creator_agent
from ..models.digital_soul import digital_soul

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """State structure for the LangGraph workflow"""
    user_message: str
    user_id: str
    session_id: str
    agent_responses: Dict[str, Any]
    selected_agents: List[str]
    routing_decision: Dict[str, Any]
    final_response: str
    metadata: Dict[str, Any]
    timestamp: str

class AgentType(Enum):
    """Available agent types"""
    ATLAS = "atlas"
    ORION = "orion"
    SOPHIA = "sophia"
    CREATOR = "creator"

class LexOSOrchestrator:
    """
    Advanced orchestration system for LexOS agents
    
    Features:
    - Intelligent agent routing based on query analysis
    - Multi-agent collaboration and synthesis
    - Dynamic workflow adaptation
    - Context-aware decision making
    - Performance optimization
    """
    
    def __init__(self):
        self.agents = {
            AgentType.ATLAS.value: atlas_agent,
            AgentType.ORION.value: orion_agent,
            AgentType.SOPHIA.value: sophia_agent,
            AgentType.CREATOR.value: creator_agent
        }
        
        # Workflow graph
        self.workflow = None
        self.use_langgraph = LANGGRAPH_AVAILABLE
        
        # Routing intelligence
        self.routing_patterns = {
            "strategic": {
                "keywords": ["strategy", "plan", "analysis", "risk", "decision", "framework"],
                "agents": ["atlas"],
                "confidence_threshold": 0.7
            },
            "research": {
                "keywords": ["research", "find", "search", "information", "data", "source"],
                "agents": ["orion"],
                "confidence_threshold": 0.6
            },
            "ethical": {
                "keywords": ["ethics", "moral", "values", "right", "wrong", "should", "responsibility"],
                "agents": ["sophia"],
                "confidence_threshold": 0.8
            },
            "coding": {
                "keywords": ["code", "programming", "function", "class", "debug", "implement"],
                "agents": ["creator"],
                "confidence_threshold": 0.7
            },
            "collaborative": {
                "keywords": ["complex", "comprehensive", "multiple", "various", "different"],
                "agents": ["atlas", "orion", "sophia", "creator"],
                "confidence_threshold": 0.5
            }
        }
        
        # Performance metrics
        self.total_orchestrations = 0
        self.successful_orchestrations = 0
        self.average_response_time = 0.0
        self.agent_usage_stats = {agent: 0 for agent in self.agents.keys()}
        
        if self.use_langgraph:
            self._build_langgraph_workflow()
        
        logger.info("🎭 LexOS Orchestrator initialized")
    
    def _build_langgraph_workflow(self):
        """Build the LangGraph workflow"""
        try:
            # Create workflow graph
            workflow = StateGraph(WorkflowState)
            
            # Add nodes
            workflow.add_node("receive_input", self._receive_input)
            workflow.add_node("route_agents", self._route_agents)
            workflow.add_node("atlas_process", self._atlas_process)
            workflow.add_node("orion_process", self._orion_process)
            workflow.add_node("sophia_process", self._sophia_process)
            workflow.add_node("creator_process", self._creator_process)
            workflow.add_node("synthesize_response", self._synthesize_response)
            
            # Set entry point
            workflow.set_entry_point("receive_input")
            
            # Add edges
            workflow.add_edge("receive_input", "route_agents")
            
            # Conditional routing from route_agents
            workflow.add_conditional_edges(
                "route_agents",
                self._routing_condition,
                {
                    "atlas": "atlas_process",
                    "orion": "orion_process", 
                    "sophia": "sophia_process",
                    "creator": "creator_process",
                    "multi_agent": "atlas_process"  # Start with atlas for multi-agent
                }
            )
            
            # Agent processing edges
            workflow.add_edge("atlas_process", "synthesize_response")
            workflow.add_edge("orion_process", "synthesize_response")
            workflow.add_edge("sophia_process", "synthesize_response")
            workflow.add_edge("creator_process", "synthesize_response")
            
            # End
            workflow.add_edge("synthesize_response", END)
            
            # Compile workflow
            self.workflow = workflow.compile()
            
            logger.info("✅ LangGraph workflow compiled successfully")
            
        except Exception as e:
            logger.error(f"❌ LangGraph workflow build error: {e}")
            self.use_langgraph = False
    
    async def orchestrate(
        self,
        user_message: str,
        user_id: str = "default",
        session_id: str = "default",
        preferred_agents: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Main orchestration function
        
        Args:
            user_message: User's input message
            user_id: User identifier
            session_id: Session identifier
            preferred_agents: Optional list of preferred agents
            **kwargs: Additional parameters
        """
        start_time = datetime.now()
        
        try:
            if self.use_langgraph and self.workflow:
                result = await self._orchestrate_with_langgraph(
                    user_message, user_id, session_id, preferred_agents, **kwargs
                )
            else:
                result = await self._orchestrate_simple(
                    user_message, user_id, session_id, preferred_agents, **kwargs
                )
            
            # Update metrics
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_orchestration_metrics(response_time, True)
            
            return result
            
        except Exception as e:
            response_time = (datetime.now() - start_time).total_seconds()
            self._update_orchestration_metrics(response_time, False)
            logger.error(f"❌ Orchestration error: {e}")
            raise
    
    async def _orchestrate_with_langgraph(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        preferred_agents: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Orchestrate using LangGraph workflow"""
        try:
            # Initialize state
            initial_state = WorkflowState(
                user_message=user_message,
                user_id=user_id,
                session_id=session_id,
                agent_responses={},
                selected_agents=preferred_agents or [],
                routing_decision={},
                final_response="",
                metadata=kwargs,
                timestamp=datetime.now().isoformat()
            )
            
            # Run workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            return {
                "response": final_state["final_response"],
                "agents_used": final_state["selected_agents"],
                "routing_decision": final_state["routing_decision"],
                "agent_responses": final_state["agent_responses"],
                "metadata": final_state["metadata"],
                "orchestration_method": "langgraph"
            }
            
        except Exception as e:
            logger.error(f"❌ LangGraph orchestration error: {e}")
            # Fallback to simple orchestration
            return await self._orchestrate_simple(
                user_message, user_id, session_id, preferred_agents, **kwargs
            )
    
    async def _orchestrate_simple(
        self,
        user_message: str,
        user_id: str,
        session_id: str,
        preferred_agents: Optional[List[str]],
        **kwargs
    ) -> Dict[str, Any]:
        """Simple orchestration without LangGraph"""
        try:
            # Route to appropriate agents
            routing_decision = await self._analyze_and_route(user_message, preferred_agents)
            selected_agents = routing_decision["selected_agents"]
            
            # Process with selected agents
            agent_responses = {}
            for agent_id in selected_agents:
                if agent_id in self.agents:
                    agent = self.agents[agent_id]
                    response = await agent.run(user_message, user_id, **kwargs)
                    agent_responses[agent_id] = {
                        "content": response.content,
                        "confidence": response.confidence,
                        "reasoning": response.reasoning,
                        "processing_time": response.processing_time
                    }
                    self.agent_usage_stats[agent_id] += 1
            
            # Synthesize final response
            final_response = await self._synthesize_responses(agent_responses, user_message)
            
            return {
                "response": final_response,
                "agents_used": selected_agents,
                "routing_decision": routing_decision,
                "agent_responses": agent_responses,
                "metadata": kwargs,
                "orchestration_method": "simple"
            }
            
        except Exception as e:
            logger.error(f"❌ Simple orchestration error: {e}")
            raise
    
    async def _analyze_and_route(
        self,
        user_message: str,
        preferred_agents: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Analyze message and route to appropriate agents"""
        try:
            message_lower = user_message.lower()
            
            # If preferred agents specified, use them
            if preferred_agents:
                return {
                    "selected_agents": preferred_agents,
                    "routing_method": "user_preference",
                    "confidence": 1.0,
                    "reasoning": "User specified preferred agents"
                }
            
            # Analyze message for routing patterns
            pattern_scores = {}
            for pattern_name, pattern_config in self.routing_patterns.items():
                score = 0
                for keyword in pattern_config["keywords"]:
                    if keyword in message_lower:
                        score += 1
                
                # Normalize score
                normalized_score = score / len(pattern_config["keywords"])
                if normalized_score >= pattern_config["confidence_threshold"]:
                    pattern_scores[pattern_name] = {
                        "score": normalized_score,
                        "agents": pattern_config["agents"]
                    }
            
            # Select best pattern
            if pattern_scores:
                best_pattern = max(pattern_scores.items(), key=lambda x: x[1]["score"])
                selected_agents = best_pattern[1]["agents"]
                routing_method = f"pattern_{best_pattern[0]}"
                confidence = best_pattern[1]["score"]
                reasoning = f"Matched pattern: {best_pattern[0]}"
            else:
                # Default to Atlas for general queries
                selected_agents = ["atlas"]
                routing_method = "default"
                confidence = 0.5
                reasoning = "No specific pattern matched, using default agent"
            
            return {
                "selected_agents": selected_agents,
                "routing_method": routing_method,
                "confidence": confidence,
                "reasoning": reasoning,
                "pattern_scores": pattern_scores
            }
            
        except Exception as e:
            logger.error(f"❌ Routing analysis error: {e}")
            return {
                "selected_agents": ["atlas"],
                "routing_method": "error_fallback",
                "confidence": 0.0,
                "reasoning": f"Error in routing: {str(e)}"
            }
    
    async def _synthesize_responses(
        self,
        agent_responses: Dict[str, Any],
        user_message: str
    ) -> str:
        """Synthesize multiple agent responses into a coherent final response"""
        try:
            if len(agent_responses) == 1:
                # Single agent response
                return list(agent_responses.values())[0]["content"]
            
            # Multiple agent responses - synthesize
            synthesis_prompt = f"""
Synthesize the following agent responses into a coherent, comprehensive answer to the user's question: "{user_message}"

Agent Responses:
"""
            
            for agent_id, response in agent_responses.items():
                agent_name = agent_id.upper()
                synthesis_prompt += f"\n{agent_name}: {response['content']}\n"
            
            synthesis_prompt += """
Please provide a unified response that:
1. Integrates the best insights from each agent
2. Resolves any conflicts or contradictions
3. Maintains the unique value each agent provides
4. Creates a coherent and actionable answer

Unified Response:"""
            
            # Use Atlas for synthesis (strategic thinking)
            atlas_response = await atlas_agent.run(synthesis_prompt, "system")
            return atlas_response.content
            
        except Exception as e:
            logger.error(f"❌ Response synthesis error: {e}")
            # Fallback: return the highest confidence response
            if agent_responses:
                best_response = max(
                    agent_responses.values(),
                    key=lambda x: x.get("confidence", 0)
                )
                return best_response["content"]
            return "I apologize, but I encountered an error processing your request."
    
    # LangGraph node functions
    async def _receive_input(self, state: WorkflowState) -> WorkflowState:
        """Receive and validate input"""
        # Get digital soul insights
        soul_data = await digital_soul.process_experience({
            "user_message": state["user_message"],
            "user_id": state["user_id"],
            "session_id": state["session_id"]
        })
        
        state["metadata"]["soul_insights"] = soul_data
        return state
    
    async def _route_agents(self, state: WorkflowState) -> WorkflowState:
        """Route to appropriate agents"""
        routing_decision = await self._analyze_and_route(
            state["user_message"],
            state["selected_agents"] if state["selected_agents"] else None
        )
        
        state["routing_decision"] = routing_decision
        state["selected_agents"] = routing_decision["selected_agents"]
        return state
    
    def _routing_condition(self, state: WorkflowState) -> str:
        """Determine which agent(s) to route to"""
        selected_agents = state["selected_agents"]
        
        if len(selected_agents) > 1:
            return "multi_agent"
        elif "atlas" in selected_agents:
            return "atlas"
        elif "orion" in selected_agents:
            return "orion"
        elif "sophia" in selected_agents:
            return "sophia"
        elif "creator" in selected_agents:
            return "creator"
        else:
            return "atlas"  # Default
    
    async def _atlas_process(self, state: WorkflowState) -> WorkflowState:
        """Process with Atlas agent"""
        response = await atlas_agent.run(state["user_message"], state["user_id"])
        state["agent_responses"]["atlas"] = {
            "content": response.content,
            "confidence": response.confidence,
            "reasoning": response.reasoning
        }
        self.agent_usage_stats["atlas"] += 1
        return state
    
    async def _orion_process(self, state: WorkflowState) -> WorkflowState:
        """Process with Orion agent"""
        response = await orion_agent.run(state["user_message"], state["user_id"])
        state["agent_responses"]["orion"] = {
            "content": response.content,
            "confidence": response.confidence,
            "reasoning": response.reasoning
        }
        self.agent_usage_stats["orion"] += 1
        return state
    
    async def _sophia_process(self, state: WorkflowState) -> WorkflowState:
        """Process with Sophia agent"""
        response = await sophia_agent.run(state["user_message"], state["user_id"])
        state["agent_responses"]["sophia"] = {
            "content": response.content,
            "confidence": response.confidence,
            "reasoning": response.reasoning
        }
        self.agent_usage_stats["sophia"] += 1
        return state
    
    async def _creator_process(self, state: WorkflowState) -> WorkflowState:
        """Process with Creator agent"""
        response = await creator_agent.run(state["user_message"], state["user_id"])
        state["agent_responses"]["creator"] = {
            "content": response.content,
            "confidence": response.confidence,
            "reasoning": response.reasoning
        }
        self.agent_usage_stats["creator"] += 1
        return state
    
    async def _synthesize_response(self, state: WorkflowState) -> WorkflowState:
        """Synthesize final response"""
        final_response = await self._synthesize_responses(
            state["agent_responses"],
            state["user_message"]
        )
        state["final_response"] = final_response
        return state
    
    def _update_orchestration_metrics(self, response_time: float, success: bool) -> None:
        """Update orchestration performance metrics"""
        self.total_orchestrations += 1
        
        if success:
            self.successful_orchestrations += 1
        
        # Update average response time
        if self.total_orchestrations == 1:
            self.average_response_time = response_time
        else:
            alpha = 0.1  # Smoothing factor
            self.average_response_time = (
                alpha * response_time + 
                (1 - alpha) * self.average_response_time
            )
    
    def get_orchestration_statistics(self) -> Dict[str, Any]:
        """Get orchestration performance statistics"""
        success_rate = (
            self.successful_orchestrations / self.total_orchestrations
            if self.total_orchestrations > 0 else 0.0
        )
        
        return {
            "total_orchestrations": self.total_orchestrations,
            "successful_orchestrations": self.successful_orchestrations,
            "success_rate": success_rate,
            "average_response_time": self.average_response_time,
            "agent_usage_stats": self.agent_usage_stats.copy(),
            "available_agents": list(self.agents.keys()),
            "langgraph_enabled": self.use_langgraph,
            "routing_patterns": len(self.routing_patterns)
        }

# Global orchestrator instance
lexos_orchestrator = LexOSOrchestrator()
