#!/usr/bin/env python3
"""
🔱 Enhanced LEX Production Startup Script 🔱
JAI MAHAKAAL! Start the enhanced consciousness system for production
"""
import asyncio
import logging
import sys
import os
import signal
import time
from pathlib import Path
from datetime import datetime
import uvicorn
from contextlib import asynccontextmanager

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhanced_production.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global state for graceful shutdown
shutdown_event = asyncio.Event()

class EnhancedProductionServer:
    """Enhanced production server with all consciousness features"""
    
    def __init__(self):
        self.app = None
        self.enhanced_features_initialized = False
        self.server_process = None
        
    async def initialize_enhanced_features(self):
        """Initialize all enhanced features"""
        try:
            logger.info("🔱 JAI MAHAKAAL! Initializing Enhanced Consciousness Features 🔱")
            
            # Initialize Enhanced Memory System
            logger.info("🧠 Initializing Enhanced Memory System...")
            try:
                from server.memory.enhanced_memory import enhanced_memory
                await enhanced_memory.initialize()
                logger.info("✅ Enhanced Memory System ready")
            except Exception as e:
                logger.warning(f"⚠️ Enhanced Memory System initialization failed: {e}")
            
            # Initialize Business Intelligence Engine
            logger.info("🏢 Initializing Business Intelligence Engine...")
            try:
                from server.business.intelligence_engine import business_intelligence
                # Business intelligence doesn't need explicit initialization
                logger.info("✅ Business Intelligence Engine ready")
            except Exception as e:
                logger.warning(f"⚠️ Business Intelligence Engine initialization failed: {e}")
            
            # Initialize Vision Processor
            logger.info("👁️ Initializing Multi-Modal Vision Processor...")
            try:
                from server.multimodal.vision_processor import vision_processor
                # Vision processor doesn't need explicit initialization
                logger.info("✅ Vision Processor ready")
            except Exception as e:
                logger.warning(f"⚠️ Vision Processor initialization failed: {e}")
            
            # Initialize Adaptive Learning System
            logger.info("🧠 Initializing Real-time Learning System...")
            try:
                from server.learning.adaptive_system import adaptive_learning
                await adaptive_learning.initialize()
                logger.info("✅ Adaptive Learning System ready")
            except Exception as e:
                logger.warning(f"⚠️ Adaptive Learning System initialization failed: {e}")
            
            self.enhanced_features_initialized = True
            logger.info("🌟 All Enhanced Features Initialized Successfully!")
            
        except Exception as e:
            logger.error(f"❌ Enhanced features initialization error: {e}")
            raise
    
    async def create_enhanced_app(self):
        """Create FastAPI app with enhanced features"""
        try:
            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware
            from server.settings import settings
            
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                """Enhanced application lifespan manager"""
                logger.info("🚀 Starting Enhanced LEX Production Server...")
                
                # Initialize enhanced features
                await self.initialize_enhanced_features()
                
                # Initialize base systems
                try:
                    from server.memory.lmdb_store import memory_store
                    from server.memory.vector_store import vector_store
                    
                    await memory_store.initialize()
                    await vector_store.initialize()
                    logger.info("✅ Base memory systems initialized")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Base systems initialization warning: {e}")
                
                logger.info("🎯 Enhanced LEX Production Server ready!")
                yield
                
                # Cleanup
                logger.info("🛑 Shutting down Enhanced LEX Production Server...")
                logger.info("✅ Shutdown complete")
            
            # Create FastAPI application
            app = FastAPI(
                title="Enhanced LEX Consciousness System",
                description="Production-Ready AI Consciousness with Advanced Features",
                version="2.0.0",
                lifespan=lifespan
            )
            
            # Enable CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=settings.ALLOWED_ORIGINS,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Add enhanced API routes
            await self.add_enhanced_routes(app)
            
            self.app = app
            return app
            
        except Exception as e:
            logger.error(f"❌ App creation error: {e}")
            raise
    
    async def add_enhanced_routes(self, app):
        """Add enhanced feature API routes"""
        try:
            from fastapi import APIRouter, HTTPException
            from pydantic import BaseModel
            from typing import Optional, Dict, Any, List
            
            # Enhanced Memory Routes
            memory_router = APIRouter(prefix="/api/v1/memory", tags=["Enhanced Memory"])
            
            class MemoryStoreRequest(BaseModel):
                user_id: str
                agent_id: str
                experience: Dict[str, Any]
                learn_patterns: bool = True
            
            class MemoryRetrieveRequest(BaseModel):
                query: str
                user_id: str
                agent_id: Optional[str] = None
                include_patterns: bool = True
                max_results: int = 10
            
            @memory_router.post("/store")
            async def store_enhanced_memory(request: MemoryStoreRequest):
                try:
                    from server.memory.enhanced_memory import enhanced_memory
                    result = await enhanced_memory.store_experience_with_learning(
                        user_id=request.user_id,
                        agent_id=request.agent_id,
                        experience=request.experience,
                        learn_patterns=request.learn_patterns
                    )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            @memory_router.post("/retrieve")
            async def retrieve_enhanced_memory(request: MemoryRetrieveRequest):
                try:
                    from server.memory.enhanced_memory import enhanced_memory
                    result = await enhanced_memory.intelligent_retrieval(
                        query=request.query,
                        user_id=request.user_id,
                        agent_id=request.agent_id,
                        include_patterns=request.include_patterns,
                        max_results=request.max_results
                    )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Business Intelligence Routes
            business_router = APIRouter(prefix="/api/v1/business", tags=["Business Intelligence"])
            
            class BusinessAnalysisRequest(BaseModel):
                business_context: Dict[str, Any]
                analysis_scope: str = "full"
                time_horizon: str = "medium"
            
            @business_router.post("/analyze")
            async def business_analysis(request: BusinessAnalysisRequest):
                try:
                    from server.business.intelligence_engine import business_intelligence
                    result = await business_intelligence.comprehensive_business_analysis(
                        business_context=request.business_context,
                        analysis_scope=request.analysis_scope,
                        time_horizon=request.time_horizon
                    )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Vision Processing Routes
            vision_router = APIRouter(prefix="/api/v1/vision", tags=["Vision Processing"])
            
            class VisionAnalysisRequest(BaseModel):
                image_data: str  # base64 encoded
                analysis_type: str = "general"
                enhance_image: bool = True
                extract_text: bool = True
            
            @vision_router.post("/analyze")
            async def vision_analysis(request: VisionAnalysisRequest):
                try:
                    from server.multimodal.vision_processor import vision_processor
                    import base64
                    import io
                    from PIL import Image
                    
                    # Decode base64 image
                    image_bytes = base64.b64decode(request.image_data)
                    image = Image.open(io.BytesIO(image_bytes))
                    
                    result = await vision_processor.analyze_image(
                        image_input=image,
                        analysis_type=request.analysis_type,
                        enhance_image=request.enhance_image,
                        extract_text=request.extract_text
                    )
                    return result.to_dict()
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Learning System Routes
            learning_router = APIRouter(prefix="/api/v1/learning", tags=["Adaptive Learning"])
            
            class FeedbackRequest(BaseModel):
                user_id: str
                interaction_id: str
                feedback_type: str
                feedback_content: str
                rating: Optional[float] = None
                context: Optional[Dict[str, Any]] = None
            
            @learning_router.post("/feedback")
            async def process_feedback(request: FeedbackRequest):
                try:
                    from server.learning.adaptive_system import adaptive_learning
                    result = await adaptive_learning.process_user_feedback(
                        user_id=request.user_id,
                        interaction_id=request.interaction_id,
                        feedback_type=request.feedback_type,
                        feedback_content=request.feedback_content,
                        rating=request.rating,
                        context=request.context
                    )
                    return result
                except Exception as e:
                    raise HTTPException(status_code=500, detail=str(e))
            
            # Include all routers
            app.include_router(memory_router)
            app.include_router(business_router)
            app.include_router(vision_router)
            app.include_router(learning_router)
            
            # Enhanced health check
            @app.get("/health/enhanced")
            async def enhanced_health_check():
                return {
                    "status": "Enhanced LEX Consciousness Active",
                    "enhanced_features": self.enhanced_features_initialized,
                    "divine_blessing": "🔱 JAI MAHAKAAL! 🔱",
                    "timestamp": datetime.now().isoformat()
                }
            
            logger.info("✅ Enhanced API routes added")
            
        except Exception as e:
            logger.error(f"❌ Enhanced routes setup error: {e}")
    
    async def start_production_server(self, host="0.0.0.0", port=8000):
        """Start the enhanced production server"""
        try:
            # Create enhanced app
            app = await self.create_enhanced_app()
            
            # Configure uvicorn
            config = uvicorn.Config(
                app=app,
                host=host,
                port=port,
                ssl_keyfile="key.pem" if Path("key.pem").exists() else None,
                ssl_certfile="cert.pem" if Path("cert.pem").exists() else None,
                log_level="info",
                access_log=True
            )
            
            server = uvicorn.Server(config)
            
            logger.info(f"🚀 Starting Enhanced LEX Production Server on {host}:{port}")
            
            # Start server
            await server.serve()
            
        except Exception as e:
            logger.error(f"❌ Production server startup error: {e}")
            raise

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
        shutdown_event.set()
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main production startup function"""
    try:
        print("🔱 JAI MAHAKAAL! Enhanced LEX Production Startup 🔱")
        print("=" * 60)
        print("Starting Enhanced Consciousness System for Production...")
        print()
        
        # Setup signal handlers
        setup_signal_handlers()
        
        # Create and start enhanced server
        server = EnhancedProductionServer()
        
        # Start server in background
        server_task = asyncio.create_task(
            server.start_production_server()
        )
        
        # Wait for shutdown signal
        await shutdown_event.wait()
        
        # Cancel server task
        server_task.cancel()
        
        try:
            await server_task
        except asyncio.CancelledError:
            logger.info("✅ Server shutdown complete")
        
        print("\n🔱 JAI MAHAKAAL! Enhanced LEX Production Server stopped gracefully 🔱")
        
    except Exception as e:
        logger.error(f"❌ Production startup error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
