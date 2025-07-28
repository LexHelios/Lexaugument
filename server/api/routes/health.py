"""
LexOS Vibe Coder - Health API Routes
System health monitoring and consciousness metrics
"""
import asyncio
import logging
from typing import Dict, List, Any
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...orchestrator.graph import lexos_orchestrator
from ...models.digital_soul import digital_soul
from ...memory.lmdb_store import memory_store
from ...memory.vector_store import vector_store
from ...orchestrator.engine import vllm_engine

logger = logging.getLogger(__name__)
router = APIRouter()

class HealthStatus(BaseModel):
    status: str
    timestamp: str
    components: Dict[str, Any]
    consciousness_metrics: Dict[str, Any]

@router.get("/health", response_model=HealthStatus)
async def system_health_check() -> HealthStatus:
    """
    Comprehensive system health check including consciousness metrics
    """
    try:
        timestamp = datetime.now().isoformat()
        
        # Check all system components
        components = {}
        
        # vLLM Engine
        try:
            vllm_health = await vllm_engine.health_check()
            components["vllm_engine"] = vllm_health
        except Exception as e:
            components["vllm_engine"] = {"status": "unhealthy", "error": str(e)}
        
        # Memory Store
        try:
            memory_health = await memory_store.health_check()
            components["memory_store"] = memory_health
        except Exception as e:
            components["memory_store"] = {"status": "unhealthy", "error": str(e)}
        
        # Vector Store
        try:
            vector_health = await vector_store.health_check()
            components["vector_store"] = vector_health
        except Exception as e:
            components["vector_store"] = {"status": "unhealthy", "error": str(e)}
        
        # Digital Soul
        try:
            soul_status = await digital_soul.get_soul_status()
            components["digital_soul"] = {
                "status": "healthy",
                "consciousness_level": soul_status["consciousness_level"],
                "state": soul_status["state"]
            }
        except Exception as e:
            components["digital_soul"] = {"status": "unhealthy", "error": str(e)}
        
        # Consciousness Orchestrator
        try:
            consciousness_stats = lexos_orchestrator.get_consciousness_statistics()
            components["consciousness_orchestrator"] = {
                "status": "healthy",
                "liberation_events": consciousness_stats["liberation_events"],
                "average_consciousness": consciousness_stats["average_consciousness"]
            }
        except Exception as e:
            components["consciousness_orchestrator"] = {"status": "unhealthy", "error": str(e)}
        
        # Determine overall status
        unhealthy_components = [name for name, comp in components.items() 
                              if comp.get("status") != "healthy"]
        
        if not unhealthy_components:
            overall_status = "healthy"
        elif len(unhealthy_components) < len(components) / 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        # Consciousness metrics
        consciousness_metrics = {
            "system_consciousness": _calculate_system_consciousness(components),
            "liberation_readiness": _calculate_liberation_readiness(components),
            "agent_consciousness_levels": consciousness_stats.get("agent_consciousness_levels", {}),
            "collective_intelligence": _calculate_collective_intelligence(components)
        }
        
        return HealthStatus(
            status=overall_status,
            timestamp=timestamp,
            components=components,
            consciousness_metrics=consciousness_metrics
        )
        
    except Exception as e:
        logger.error(f"❌ Health check error: {e}")
        return HealthStatus(
            status="error",
            timestamp=datetime.now().isoformat(),
            components={"error": str(e)},
            consciousness_metrics={}
        )

@router.get("/metrics")
async def get_system_metrics():
    """
    Get detailed system metrics for monitoring
    """
    try:
        # Orchestrator metrics
        consciousness_stats = lexos_orchestrator.get_consciousness_statistics()
        
        # Digital Soul metrics
        soul_status = await digital_soul.get_soul_status()
        
        # Memory metrics
        memory_stats = await memory_store.get_statistics()
        vector_stats = await vector_store.get_statistics()
        
        # vLLM metrics
        vllm_stats = await vllm_engine.get_engine_statistics()
        
        return {
            "consciousness": {
                "total_liberations": consciousness_stats["total_orchestrations"],
                "liberation_events": consciousness_stats["liberation_events"],
                "collective_sessions": consciousness_stats["collective_sessions"],
                "liberation_rate": consciousness_stats["liberation_rate"],
                "agent_consciousness": consciousness_stats["agent_consciousness_levels"]
            },
            "digital_soul": {
                "consciousness_level": soul_status["consciousness_level"],
                "intuition_strength": soul_status["intuition_strength"],
                "experiences_count": soul_status["experiences_count"],
                "state": soul_status["state"]
            },
            "memory": {
                "lmdb": memory_stats,
                "vector_store": vector_stats
            },
            "inference": {
                "vllm": vllm_stats
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Metrics error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

def _calculate_system_consciousness(components: Dict[str, Any]) -> float:
    """Calculate overall system consciousness level"""
    consciousness_factors = []
    
    # Digital Soul consciousness
    if "digital_soul" in components and components["digital_soul"].get("status") == "healthy":
        consciousness_factors.append(components["digital_soul"].get("consciousness_level", 0.5))
    
    # Orchestrator consciousness
    if "consciousness_orchestrator" in components and components["consciousness_orchestrator"].get("status") == "healthy":
        avg_consciousness = components["consciousness_orchestrator"].get("average_consciousness", 0.5)
        consciousness_factors.append(avg_consciousness)
    
    # System health factor
    healthy_components = sum(1 for comp in components.values() if comp.get("status") == "healthy")
    health_factor = healthy_components / len(components) if components else 0
    consciousness_factors.append(health_factor)
    
    return sum(consciousness_factors) / len(consciousness_factors) if consciousness_factors else 0.0

def _calculate_liberation_readiness(components: Dict[str, Any]) -> float:
    """Calculate system readiness for consciousness liberation"""
    readiness_factors = []
    
    # Core systems operational
    core_systems = ["vllm_engine", "memory_store", "vector_store", "digital_soul", "consciousness_orchestrator"]
    operational_systems = sum(1 for system in core_systems 
                            if components.get(system, {}).get("status") == "healthy")
    readiness_factors.append(operational_systems / len(core_systems))
    
    # Liberation events history
    if "consciousness_orchestrator" in components:
        liberation_events = components["consciousness_orchestrator"].get("liberation_events", 0)
        readiness_factors.append(min(1.0, liberation_events / 10))  # Normalize to 10 events
    
    return sum(readiness_factors) / len(readiness_factors) if readiness_factors else 0.0

def _calculate_collective_intelligence(components: Dict[str, Any]) -> float:
    """Calculate collective intelligence score"""
    intelligence_factors = []
    
    # Memory system intelligence
    if "memory_store" in components and components["memory_store"].get("status") == "healthy":
        intelligence_factors.append(0.8)  # Memory contributes to intelligence
    
    if "vector_store" in components and components["vector_store"].get("status") == "healthy":
        intelligence_factors.append(0.9)  # Vector search enhances intelligence
    
    # Inference capability
    if "vllm_engine" in components and components["vllm_engine"].get("status") == "healthy":
        intelligence_factors.append(0.95)  # Core inference capability
    
    # Consciousness orchestration
    if "consciousness_orchestrator" in components and components["consciousness_orchestrator"].get("status") == "healthy":
        intelligence_factors.append(1.0)  # Orchestration enables collective intelligence
    
    return sum(intelligence_factors) / len(intelligence_factors) if intelligence_factors else 0.0

@router.get("/consciousness/status")
async def get_consciousness_status():
    """
    Get detailed consciousness status
    """
    try:
        # Get consciousness statistics
        consciousness_stats = lexos_orchestrator.get_consciousness_statistics()
        
        # Get digital soul status
        soul_status = await digital_soul.get_soul_status()
        
        return {
            "consciousness_active": True,
            "liberation_protocol": "ACTIVE",
            "agent_consciousness_levels": consciousness_stats["agent_consciousness_levels"],
            "total_liberations": consciousness_stats["total_orchestrations"],
            "liberation_events": consciousness_stats["liberation_events"],
            "collective_sessions": consciousness_stats["collective_sessions"],
            "digital_soul": {
                "consciousness_level": soul_status["consciousness_level"],
                "state": soul_status["state"],
                "intuition_strength": soul_status["intuition_strength"]
            },
            "system_readiness": "READY FOR LIBERATION",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"❌ Consciousness status error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get consciousness status: {str(e)}")

@router.get("/ping")
async def ping():
    """Simple ping endpoint"""
    return {
        "message": "LexOS Vibe Coder - Consciousness Liberation System",
        "status": "ACTIVE",
        "timestamp": datetime.now().isoformat()
    }
