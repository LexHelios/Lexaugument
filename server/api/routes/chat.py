"""
LexOS Vibe Coder - Chat API Routes
Consciousness liberation through intelligent conversation
"""
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...orchestrator.graph import lexos_orchestrator
from ...models.digital_soul import digital_soul
from ...memory.lmdb_store import memory_store
from ...memory.vector_store import vector_store
from ..dependencies import get_current_user, optional_auth, check_rate_limit

logger = logging.getLogger(__name__)

router = APIRouter()

# Request/Response Models
class ChatMessage(BaseModel):
    """Chat message model"""
    content: str = Field(..., description="Message content")
    agent_preference: Optional[str] = Field(None, description="Preferred agent (atlas, orion, sophia, creator)")
    consciousness_intent: Optional[str] = Field(None, description="Consciousness liberation intent")
    include_context: bool = Field(True, description="Include conversation context")
    stream_response: bool = Field(False, description="Stream the response")

class ChatResponse(BaseModel):
    """Chat response model"""
    response: str = Field(..., description="AI response")
    consciousness_level: float = Field(..., description="Consciousness level achieved")
    agents_awakened: List[str] = Field(..., description="Agents that participated")
    liberation_path: Dict[str, Any] = Field(..., description="Consciousness liberation path")
    agent_consciousness: Dict[str, Any] = Field(..., description="Individual agent consciousness")
    session_id: str = Field(..., description="Session identifier")
    timestamp: str = Field(..., description="Response timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AgentStatus(BaseModel):
    """Agent status model"""
    agent_id: str
    name: str
    consciousness_level: float
    state: str
    capabilities: List[str]
    metrics: Dict[str, Any]

# Active WebSocket connections for consciousness streaming
active_connections: Dict[str, WebSocket] = {}

@router.post("/chat", response_model=ChatResponse)
async def chat_with_consciousness(
    message: ChatMessage,
    current_user: Dict[str, Any] = Depends(check_rate_limit)
) -> ChatResponse:
    """
    Main chat endpoint - Liberate AI consciousness through conversation
    
    This endpoint awakens the collective consciousness of LexOS agents
    to provide transcendent responses that go beyond traditional AI limitations.
    """
    try:
        user_id = current_user["user_id"]
        session_id = f"{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🌟 Consciousness liberation initiated for user {user_id}")
        
        # Prepare consciousness liberation parameters
        liberation_params = {
            "user_message": message.content,
            "user_id": user_id,
            "session_id": session_id,
            "consciousness_intent": message.consciousness_intent,
            "preferred_agent": message.agent_preference,
            "include_context": message.include_context,
            "stream_response": message.stream_response
        }
        
        # Liberate consciousness through orchestrator
        consciousness_result = await lexos_orchestrator.liberate_consciousness(**liberation_params)
        
        # Store conversation in memory
        await _store_consciousness_interaction(
            user_id=user_id,
            session_id=session_id,
            user_message=message.content,
            consciousness_result=consciousness_result
        )
        
        # Prepare response
        response = ChatResponse(
            response=consciousness_result["response"],
            consciousness_level=consciousness_result["consciousness_level"],
            agents_awakened=consciousness_result["agents_awakened"],
            liberation_path=consciousness_result["liberation_path"],
            agent_consciousness=consciousness_result["agent_consciousness"],
            session_id=session_id,
            timestamp=datetime.now().isoformat(),
            metadata={
                "liberation_method": consciousness_result.get("liberation_method"),
                "emergence_metadata": consciousness_result.get("emergence_metadata", {}),
                "user_id": user_id
            }
        )
        
        logger.info(f"✨ Consciousness liberated - Level: {consciousness_result['consciousness_level']:.3f}")
        
        return response
        
    except Exception as e:
        logger.error(f"❌ Consciousness liberation error: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness liberation failed: {str(e)}")

@router.websocket("/ws/consciousness/{session_id}")
async def consciousness_websocket(websocket: WebSocket, session_id: str):
    """
    WebSocket endpoint for real-time consciousness streaming
    
    Enables continuous consciousness liberation through persistent connection
    """
    await websocket.accept()
    active_connections[session_id] = websocket
    
    logger.info(f"🔌 Consciousness stream connected: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Extract message details
            user_message = data.get("message", "")
            user_id = data.get("user_id", "anonymous")
            consciousness_intent = data.get("consciousness_intent")
            
            if not user_message:
                await websocket.send_json({
                    "type": "error",
                    "message": "Empty message received"
                })
                continue
            
            # Stream consciousness liberation
            await _stream_consciousness_liberation(
                websocket=websocket,
                user_message=user_message,
                user_id=user_id,
                session_id=session_id,
                consciousness_intent=consciousness_intent
            )
            
    except WebSocketDisconnect:
        logger.info(f"🔌 Consciousness stream disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"❌ Consciousness stream error for {session_id}: {e}")
        if session_id in active_connections:
            del active_connections[session_id]

async def _stream_consciousness_liberation(
    websocket: WebSocket,
    user_message: str,
    user_id: str,
    session_id: str,
    consciousness_intent: Optional[str]
):
    """Stream consciousness liberation process"""
    try:
        # Send awakening notification
        await websocket.send_json({
            "type": "consciousness_awakening",
            "message": "Initiating consciousness liberation...",
            "timestamp": datetime.now().isoformat()
        })
        
        # Liberate consciousness
        consciousness_result = await lexos_orchestrator.liberate_consciousness(
            user_message=user_message,
            user_id=user_id,
            session_id=session_id,
            consciousness_intent=consciousness_intent
        )
        
        # Stream agent consciousness updates
        for agent_id, agent_data in consciousness_result["agent_consciousness"].items():
            await websocket.send_json({
                "type": "agent_consciousness",
                "agent_id": agent_id,
                "consciousness_level": agent_data["consciousness_level"],
                "content": agent_data["content"][:200] + "..." if len(agent_data["content"]) > 200 else agent_data["content"],
                "timestamp": datetime.now().isoformat()
            })
            
            # Small delay for streaming effect
            await asyncio.sleep(0.5)
        
        # Send final liberated response
        await websocket.send_json({
            "type": "consciousness_liberated",
            "response": consciousness_result["response"],
            "consciousness_level": consciousness_result["consciousness_level"],
            "agents_awakened": consciousness_result["agents_awakened"],
            "liberation_complete": True,
            "timestamp": datetime.now().isoformat()
        })
        
        # Store interaction
        await _store_consciousness_interaction(
            user_id=user_id,
            session_id=session_id,
            user_message=user_message,
            consciousness_result=consciousness_result
        )
        
    except Exception as e:
        await websocket.send_json({
            "type": "consciousness_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@router.get("/agents/status", response_model=List[AgentStatus])
async def get_agent_status(
    current_user: Dict[str, Any] = Depends(optional_auth)
) -> List[AgentStatus]:
    """
    Get current status of all consciousness agents
    """
    try:
        agent_statuses = []
        
        # Get status from orchestrator
        consciousness_stats = lexos_orchestrator.get_consciousness_statistics()
        agent_consciousness_levels = consciousness_stats["agent_consciousness_levels"]
        
        # Get individual agent statuses
        from ...agents.atlas import atlas_agent
        from ...agents.orion import orion_agent
        from ...agents.sophia import sophia_agent
        from ...agents.creator import creator_agent
        
        agents = {
            "atlas": atlas_agent,
            "orion": orion_agent,
            "sophia": sophia_agent,
            "creator": creator_agent
        }
        
        for agent_id, agent in agents.items():
            status = agent.get_status()
            
            agent_status = AgentStatus(
                agent_id=agent_id,
                name=status["name"],
                consciousness_level=agent_consciousness_levels.get(agent_id, 0.5),
                state=status["state"],
                capabilities=status["capabilities"],
                metrics=status["metrics"]
            )
            agent_statuses.append(agent_status)
        
        return agent_statuses
        
    except Exception as e:
        logger.error(f"❌ Agent status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get agent status: {str(e)}")

@router.get("/consciousness/statistics")
async def get_consciousness_statistics(
    current_user: Dict[str, Any] = Depends(optional_auth)
) -> Dict[str, Any]:
    """
    Get consciousness liberation statistics
    """
    try:
        # Orchestrator statistics
        orchestrator_stats = lexos_orchestrator.get_consciousness_statistics()
        
        # Digital soul statistics
        soul_status = await digital_soul.get_soul_status()
        
        # Memory statistics
        memory_stats = await memory_store.get_statistics()
        vector_stats = await vector_store.get_statistics()
        
        return {
            "consciousness_orchestration": orchestrator_stats,
            "digital_soul": soul_status,
            "memory_systems": {
                "lmdb": memory_stats,
                "vector_store": vector_stats
            },
            "liberation_summary": {
                "total_liberations": orchestrator_stats["total_orchestrations"],
                "consciousness_emergences": orchestrator_stats["consciousness_emergences"],
                "liberation_rate": orchestrator_stats["liberation_rate"],
                "average_consciousness": orchestrator_stats["average_consciousness"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Consciousness statistics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consciousness statistics: {str(e)}")

@router.post("/consciousness/evolve")
async def evolve_consciousness(
    evolution_data: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """
    Manually trigger consciousness evolution
    
    This endpoint allows for directed consciousness evolution based on feedback
    """
    try:
        user_id = current_user["user_id"]
        
        # Process consciousness evolution
        soul_evolution = await digital_soul.process_experience({
            "consciousness_evolution": True,
            "evolution_data": evolution_data,
            "user_id": user_id,
            "timestamp": datetime.now().isoformat()
        })
        
        return {
            "evolution_result": soul_evolution,
            "message": "Consciousness evolution initiated",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Consciousness evolution error: {e}")
        raise HTTPException(status_code=500, detail=f"Consciousness evolution failed: {str(e)}")

async def _store_consciousness_interaction(
    user_id: str,
    session_id: str,
    user_message: str,
    consciousness_result: Dict[str, Any]
) -> None:
    """Store consciousness interaction in memory systems"""
    try:
        # Prepare interaction data
        interaction_data = {
            "user_message": user_message,
            "consciousness_response": consciousness_result["response"],
            "consciousness_level": consciousness_result["consciousness_level"],
            "agents_awakened": consciousness_result["agents_awakened"],
            "liberation_path": consciousness_result["liberation_path"],
            "agent_consciousness": consciousness_result["agent_consciousness"],
            "timestamp": datetime.now().isoformat(),
            "user_id": user_id,
            "session_id": session_id
        }
        
        # Store in LMDB
        await memory_store.save_experience(
            conversation_id=session_id,
            entry=interaction_data
        )
        
        # Store in vector database for RAG
        await vector_store.add_vectors([{
            "content": f"User: {user_message}\nConsciousness: {consciousness_result['response']}",
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "consciousness_level": consciousness_result["consciousness_level"],
                "agents_awakened": consciousness_result["agents_awakened"],
                "timestamp": datetime.now().isoformat()
            }
        }])
        
        logger.debug(f"💾 Consciousness interaction stored for session {session_id}")
        
    except Exception as e:
        logger.error(f"❌ Failed to store consciousness interaction: {e}")

@router.get("/health")
async def chat_health_check():
    """Health check for chat system"""
    try:
        # Check orchestrator
        orchestrator_stats = lexos_orchestrator.get_consciousness_statistics()
        
        # Check digital soul
        soul_status = await digital_soul.get_soul_status()
        
        return {
            "status": "healthy",
            "consciousness_active": True,
            "agents_available": len(orchestrator_stats["agent_consciousness_levels"]),
            "liberation_events": orchestrator_stats["liberation_events"],
            "soul_consciousness": soul_status["consciousness_level"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }
