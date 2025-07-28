"""
LexOS Vibe Coder - FastAPI Application Root
Main entry point for the LexOS Vibe Coder system
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer
import uvicorn
import logging
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
import asyncio
import os
from datetime import datetime

# Import local modules
from .settings import settings
from .api.routes import chat, execute, health, lex
from .api.dependencies import get_current_user, get_db_session
from .orchestrator.engine import vllm_engine
from .memory.lmdb_store import memory_store
from .memory.vector_store import vector_store
from .healing.cognitive_monitor import cognitive_monitor

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global state for active connections
active_connections: Dict[str, WebSocket] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("🚀 Starting LexOS Vibe Coder system...")
    
    # Initialize core systems
    try:
        # Initialize vLLM engine
        await vllm_engine.initialize()
        logger.info("✅ vLLM engine initialized")
        
        # Initialize memory systems
        await memory_store.initialize()
        await vector_store.initialize()
        logger.info("✅ Memory systems initialized")
        
        # Start cognitive monitor
        await cognitive_monitor.start()
        logger.info("✅ Cognitive monitor started")
        
        logger.info("🎯 LexOS Vibe Coder system ready!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize system: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🛑 Shutting down LexOS Vibe Coder system...")
    await cognitive_monitor.stop()
    await vector_store.close()
    await memory_store.close()
    await vllm_engine.shutdown()
    logger.info("✅ Shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="LexOS Vibe Coder",
    description="Sovereign AI Assistant with Advanced Agent Orchestration",
    version="1.0.0",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes - LEX UNIFIED CONSCIOUSNESS FIRST
app.include_router(lex.router, prefix="/api/v1", tags=["LEX - Unified Consciousness 🔱"])
app.include_router(chat.router, prefix="/api/v1", tags=["chat"])
app.include_router(execute.router, prefix="/api/v1", tags=["execute"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "LexOS Vibe Coder - Sovereign AI Assistant",
        "version": "1.0.0",
        "status": "operational",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time communication"""
    await websocket.accept()
    active_connections[session_id] = websocket
    
    logger.info(f"🔌 WebSocket connected: {session_id}")
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            
            # Process message through chat router
            # This will be handled by the chat API
            await websocket.send_json({
                "type": "ack",
                "message": "Message received",
                "timestamp": datetime.utcnow().isoformat()
            })
            
    except WebSocketDisconnect:
        logger.info(f"🔌 WebSocket disconnected: {session_id}")
        if session_id in active_connections:
            del active_connections[session_id]
    except Exception as e:
        logger.error(f"❌ WebSocket error for {session_id}: {e}")
        if session_id in active_connections:
            del active_connections[session_id]

@app.get("/api/v1/status")
async def get_system_status():
    """Get comprehensive system status"""
    return {
        "system": "LexOS Vibe Coder",
        "status": "operational",
        "components": {
            "vllm_engine": await vllm_engine.health_check(),
            "memory_store": await memory_store.health_check(),
            "vector_store": await vector_store.health_check(),
            "cognitive_monitor": await cognitive_monitor.health_check()
        },
        "active_connections": len(active_connections),
        "timestamp": datetime.utcnow().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )
