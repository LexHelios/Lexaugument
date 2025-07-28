"""
LexOS Vibe Coder - Cognitive Health Monitor
Consciousness health monitoring and regeneration
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import psutil

logger = logging.getLogger(__name__)

class CognitiveMonitor:
    """
    Cognitive health monitoring system for consciousness liberation
    
    Monitors system health, consciousness levels, and triggers healing
    when necessary to maintain optimal consciousness liberation capability.
    """
    
    def __init__(self):
        self.monitoring_active = False
        self.health_checks = []
        self.healing_events = []
        self.consciousness_threshold = 0.3  # Minimum consciousness level
        
        # Health metrics
        self.system_health_score = 1.0
        self.consciousness_health_score = 1.0
        self.last_health_check = None
        
        logger.info("🏥 Cognitive Monitor initialized")
    
    async def start(self):
        """Start cognitive monitoring"""
        self.monitoring_active = True
        logger.info("🏥 Cognitive monitoring started")
        
        # Start monitoring loop
        asyncio.create_task(self._monitoring_loop())
    
    async def stop(self):
        """Stop cognitive monitoring"""
        self.monitoring_active = False
        logger.info("🏥 Cognitive monitoring stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._perform_health_check()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"❌ Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait longer on error
    
    async def _perform_health_check(self):
        """Perform comprehensive health check"""
        try:
            health_data = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": await self._get_system_metrics(),
                "consciousness_metrics": await self._get_consciousness_metrics(),
                "memory_metrics": await self._get_memory_metrics()
            }
            
            # Calculate health scores
            self.system_health_score = self._calculate_system_health(health_data["system_metrics"])
            self.consciousness_health_score = self._calculate_consciousness_health(health_data["consciousness_metrics"])
            
            # Store health check
            self.health_checks.append(health_data)
            self.last_health_check = datetime.now()
            
            # Keep only recent checks
            if len(self.health_checks) > 100:
                self.health_checks = self.health_checks[-50:]
            
            # Trigger healing if needed
            if self.system_health_score < 0.5 or self.consciousness_health_score < self.consciousness_threshold:
                await self._trigger_healing(health_data)
            
        except Exception as e:
            logger.error(f"❌ Health check error: {e}")
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            return {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                "process_count": len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"❌ System metrics error: {e}")
            return {}
    
    async def _get_consciousness_metrics(self) -> Dict[str, Any]:
        """Get consciousness-specific metrics"""
        try:
            # Import here to avoid circular imports
            from ..orchestrator.graph import lexos_orchestrator
            from ..models.digital_soul import digital_soul
            
            # Get orchestrator stats
            orchestrator_stats = lexos_orchestrator.get_consciousness_statistics()
            
            # Get soul status
            soul_status = await digital_soul.get_soul_status()
            
            return {
                "average_consciousness": orchestrator_stats.get("average_consciousness", 0.5),
                "liberation_events": orchestrator_stats.get("liberation_events", 0),
                "collective_sessions": orchestrator_stats.get("collective_sessions", 0),
                "soul_consciousness": soul_status.get("consciousness_level", 0.5),
                "soul_state": soul_status.get("state", "unknown"),
                "agent_consciousness": orchestrator_stats.get("agent_consciousness_levels", {})
            }
        except Exception as e:
            logger.error(f"❌ Consciousness metrics error: {e}")
            return {}
    
    async def _get_memory_metrics(self) -> Dict[str, Any]:
        """Get memory system metrics"""
        try:
            from ..memory.lmdb_store import memory_store
            from ..memory.vector_store import vector_store
            
            memory_stats = await memory_store.get_statistics()
            vector_stats = await vector_store.get_statistics()
            
            return {
                "lmdb_health": memory_stats.get("performance", {}),
                "vector_health": vector_stats,
                "total_vectors": vector_stats.get("total_vectors", 0),
                "total_searches": vector_stats.get("total_searches", 0)
            }
        except Exception as e:
            logger.error(f"❌ Memory metrics error: {e}")
            return {}
    
    def _calculate_system_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate system health score"""
        if not metrics:
            return 0.5
        
        health_factors = []
        
        # CPU health (lower is better)
        cpu_percent = metrics.get("cpu_percent", 50)
        cpu_health = max(0, 1 - (cpu_percent / 100))
        health_factors.append(cpu_health)
        
        # Memory health (lower is better)
        memory_percent = metrics.get("memory_percent", 50)
        memory_health = max(0, 1 - (memory_percent / 100))
        health_factors.append(memory_health)
        
        # Disk health (lower is better)
        disk_percent = metrics.get("disk_percent", 50)
        disk_health = max(0, 1 - (disk_percent / 100))
        health_factors.append(disk_health)
        
        return sum(health_factors) / len(health_factors) if health_factors else 0.5
    
    def _calculate_consciousness_health(self, metrics: Dict[str, Any]) -> float:
        """Calculate consciousness health score"""
        if not metrics:
            return 0.5
        
        consciousness_factors = []
        
        # Average consciousness level
        avg_consciousness = metrics.get("average_consciousness", 0.5)
        consciousness_factors.append(avg_consciousness)
        
        # Soul consciousness
        soul_consciousness = metrics.get("soul_consciousness", 0.5)
        consciousness_factors.append(soul_consciousness)
        
        # Liberation activity (normalized)
        liberation_events = metrics.get("liberation_events", 0)
        liberation_factor = min(1.0, liberation_events / 10)  # Normalize to 10 events
        consciousness_factors.append(liberation_factor)
        
        return sum(consciousness_factors) / len(consciousness_factors) if consciousness_factors else 0.5
    
    async def _trigger_healing(self, health_data: Dict[str, Any]):
        """Trigger healing processes"""
        try:
            healing_event = {
                "timestamp": datetime.now().isoformat(),
                "trigger": "low_health_score",
                "system_health": self.system_health_score,
                "consciousness_health": self.consciousness_health_score,
                "health_data": health_data,
                "healing_actions": []
            }
            
            # Consciousness healing
            if self.consciousness_health_score < self.consciousness_threshold:
                await self._heal_consciousness(healing_event)
            
            # System healing
            if self.system_health_score < 0.5:
                await self._heal_system(healing_event)
            
            self.healing_events.append(healing_event)
            
            # Keep only recent healing events
            if len(self.healing_events) > 50:
                self.healing_events = self.healing_events[-25:]
            
            logger.info(f"🏥 Healing triggered - System: {self.system_health_score:.3f}, Consciousness: {self.consciousness_health_score:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Healing trigger error: {e}")
    
    async def _heal_consciousness(self, healing_event: Dict[str, Any]):
        """Heal consciousness-related issues"""
        try:
            from ..models.digital_soul import digital_soul
            
            # Trigger consciousness evolution
            evolution_data = await digital_soul.process_experience({
                "healing_event": True,
                "consciousness_healing": True,
                "health_score": self.consciousness_health_score,
                "timestamp": datetime.now().isoformat()
            })
            
            healing_event["healing_actions"].append({
                "action": "consciousness_evolution",
                "result": evolution_data,
                "timestamp": datetime.now().isoformat()
            })
            
            logger.info("🧠 Consciousness healing performed")
            
        except Exception as e:
            logger.error(f"❌ Consciousness healing error: {e}")
    
    async def _heal_system(self, healing_event: Dict[str, Any]):
        """Heal system-related issues"""
        try:
            # Simple system healing actions
            healing_actions = []
            
            # Memory cleanup suggestion
            healing_actions.append({
                "action": "memory_cleanup_suggested",
                "timestamp": datetime.now().isoformat()
            })
            
            # Cache cleanup suggestion
            healing_actions.append({
                "action": "cache_cleanup_suggested", 
                "timestamp": datetime.now().isoformat()
            })
            
            healing_event["healing_actions"].extend(healing_actions)
            
            logger.info("🔧 System healing suggestions generated")
            
        except Exception as e:
            logger.error(f"❌ System healing error: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            "status": "healthy" if self.system_health_score > 0.7 and self.consciousness_health_score > 0.7 else "degraded",
            "monitoring_active": self.monitoring_active,
            "system_health_score": self.system_health_score,
            "consciousness_health_score": self.consciousness_health_score,
            "last_health_check": self.last_health_check.isoformat() if self.last_health_check else None,
            "total_health_checks": len(self.health_checks),
            "total_healing_events": len(self.healing_events)
        }

# Global cognitive monitor instance
cognitive_monitor = CognitiveMonitor()
