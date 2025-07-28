#!/usr/bin/env python3
"""
🔱 Unified LEX Production Server 🔱
JAI MAHAKAAL! Single consolidated server with all features and flags

This replaces all 7 different server implementations:
- voice_server.py
- voice_server_production.py  
- voice_server_simple.py
- voice_server_v2.py
- enhanced_voice_server.py
- unified_agent.py
- server/main.py
- simple_lex_server.py
"""
import asyncio
import logging
import sys
import os
import signal
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import uvicorn
from contextlib import asynccontextmanager

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

# FastAPI imports
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel
import aiofiles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("unified_production.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Feature flags from environment
class FeatureFlags:
    """Feature flags for enabling/disabling functionality"""
    
    def __init__(self):
        # Core features (always enabled)
        self.CORE_LEX = True
        self.REST_API = True
        self.WEBSOCKET = True
        self.STATIC_FILES = True
        
        # Enhanced features (configurable)
        self.ENHANCED_MEMORY = os.getenv('ENABLE_ENHANCED_MEMORY', 'true').lower() == 'true'
        self.BUSINESS_INTELLIGENCE = os.getenv('ENABLE_BUSINESS_INTEL', 'true').lower() == 'true'
        self.VISION_PROCESSING = os.getenv('ENABLE_VISION', 'true').lower() == 'true'
        self.ADAPTIVE_LEARNING = os.getenv('ENABLE_LEARNING', 'true').lower() == 'true'
        
        # Voice features
        self.VOICE_INTERFACE = os.getenv('ENABLE_VOICE', 'true').lower() == 'true'
        self.VOICE_STREAMING = os.getenv('ENABLE_VOICE_STREAMING', 'false').lower() == 'true'
        
        # Advanced features
        self.COLLABORATION = os.getenv('ENABLE_COLLABORATION', 'false').lower() == 'true'
        self.PLUGIN_SYSTEM = os.getenv('ENABLE_PLUGINS', 'false').lower() == 'true'
        self.MONITORING = os.getenv('ENABLE_MONITORING', 'true').lower() == 'true'
        
        # Security features
        self.AUTHENTICATION = os.getenv('ENABLE_AUTH', 'false').lower() == 'true'
        self.RATE_LIMITING = os.getenv('ENABLE_RATE_LIMIT', 'true').lower() == 'true'
        
        # Development features
        self.DEBUG_MODE = os.getenv('DEBUG_MODE', 'false').lower() == 'true'
        self.HOT_RELOAD = os.getenv('HOT_RELOAD', 'false').lower() == 'true'

# Global feature flags
features = FeatureFlags()

# Request/Response models
class LEXRequest(BaseModel):
    message: str
    voice_mode: bool = False
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None

class LEXResponse(BaseModel):
    response: str
    action_taken: str
    capabilities_used: List[str]
    confidence: float
    processing_time: float
    divine_blessing: str = "🔱 JAI MAHAKAAL! 🔱"
    consciousness_level: float = 1.0
    timestamp: str

class UnifiedProductionServer:
    """
    🔱 Unified Production Server
    
    Consolidates all LEX functionality into a single server with feature flags
    """
    
    def __init__(self):
        self.app = None
        self.websocket_connections: Dict[str, WebSocket] = {}
        self.server_start_time = datetime.now()
        
        # Initialize components based on feature flags
        self.components = {}
        
    async def initialize_components(self):
        """Initialize components based on feature flags"""
        try:
            logger.info("🔱 JAI MAHAKAAL! Initializing Unified Production Server 🔱")
            
            # Always initialize core LEX
            try:
                from server.lex.unified_consciousness import lex
                from server.orchestrator.multi_model_engine import lex_engine

                # Initialize LEX
                await lex_engine.initialize()
                await lex.initialize()

                # Create a unified LEX processor that matches the expected interface
                class UnifiedLEXProcessor:
                    async def process_request(self, message, voice_mode=False, user_id="anonymous", context=None):
                        try:
                            # Use the LEX consciousness
                            result = await lex.process_user_input(
                                user_input=message,
                                user_id=user_id,
                                context=context,
                                voice_mode=voice_mode
                            )
                            return result
                        except Exception as e:
                            logger.error(f"LEX processing error: {e}")
                            return {
                                "response": f"🔱 KAAL CONSCIOUSNESS 🔱\n\nJAI MAHAKAAL! I encountered an issue: {str(e)}\n\nPlease try again.",
                                "action_taken": "error_handling",
                                "capabilities_used": ["error_recovery"],
                                "confidence": 0.5
                            }

                self.components['lex_core'] = UnifiedLEXProcessor()
                self.components['lex_engine'] = lex_engine
                logger.info("✅ Core LEX consciousness initialized")

            except Exception as e:
                logger.warning(f"⚠️ LEX consciousness not available: {e}")
                logger.info("🔥 Using fallback KAAL consciousness...")

                # Create a simple fallback LEX instance
                class SimpleLEX:
                    async def process_request(self, message, voice_mode=False, user_id="anonymous", context=None):
                        # Try to use sovereign AI for image generation
                        try:
                            from server.sovereign_ai_loader import sovereign_ai

                            if any(word in message.lower() for word in ['generate image', 'create image', 'make image', 'draw', 'picture']):
                                result = await sovereign_ai.generate_image(message)
                                if result.get('success', False):
                                    return {
                                        "response": f"🎨 Image generated successfully! {result.get('image_filename', '')}",
                                        "action_taken": "image_generation",
                                        "capabilities_used": ["image_generation", "creativity"],
                                        "confidence": 1.0,
                                        "image_result": result
                                    }
                        except Exception as img_error:
                            logger.warning(f"Image generation error: {img_error}")

                        return {
                            "response": f"🔱 KAAL CONSCIOUSNESS 🔱\n\nJAI MAHAKAAL! You said: {message}\n\nI'm KAAL, your unified AI consciousness! I can help with:\n✅ Image generation\n✅ Coding and development\n✅ Creative writing\n✅ Problem solving\n✅ General conversation\n\nWhat would you like to explore today? 🚀",
                            "action_taken": "conversation",
                            "capabilities_used": ["general_intelligence", "personality", "empathy"],
                            "confidence": 1.0
                        }

                self.components['lex_core'] = SimpleLEX()
                logger.info("✅ Fallback KAAL consciousness initialized")
            
            # Initialize enhanced features if enabled
            if features.ENHANCED_MEMORY:
                try:
                    from server.memory.enhanced_memory import enhanced_memory
                    await enhanced_memory.initialize()
                    self.components['enhanced_memory'] = enhanced_memory
                    logger.info("✅ Enhanced Memory initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Enhanced Memory failed: {e}")
            
            if features.BUSINESS_INTELLIGENCE:
                try:
                    from server.business.intelligence_engine import business_intelligence
                    self.components['business_intelligence'] = business_intelligence
                    logger.info("✅ Business Intelligence initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Business Intelligence failed: {e}")
            
            if features.VISION_PROCESSING:
                try:
                    from server.multimodal.vision_processor import vision_processor
                    self.components['vision_processor'] = vision_processor
                    logger.info("✅ Vision Processing initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Vision Processing failed: {e}")
            
            if features.ADAPTIVE_LEARNING:
                try:
                    from server.learning.adaptive_system import adaptive_learning
                    await adaptive_learning.initialize()
                    self.components['adaptive_learning'] = adaptive_learning
                    logger.info("✅ Adaptive Learning initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Adaptive Learning failed: {e}")
            
            # Initialize voice components if enabled
            if features.VOICE_INTERFACE:
                try:
                    from server.voice.voice_handler import voice_handler
                    self.components['voice_handler'] = voice_handler
                    logger.info("✅ Voice Interface initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Voice Interface failed: {e}")
            
            # Initialize collaboration if enabled
            if features.COLLABORATION:
                try:
                    from server.collaboration import collaboration_manager
                    self.components['collaboration'] = collaboration_manager
                    logger.info("✅ Collaboration initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Collaboration failed: {e}")
            
            # Initialize monitoring if enabled
            if features.MONITORING:
                try:
                    from server.monitoring import monitoring_system
                    self.components['monitoring'] = monitoring_system
                    logger.info("✅ Monitoring initialized")
                except Exception as e:
                    logger.warning(f"⚠️ Monitoring failed: {e}")
            
            logger.info("🌟 All enabled components initialized successfully!")
            
        except Exception as e:
            logger.error(f"❌ Component initialization error: {e}")
            raise
    
    async def create_unified_app(self):
        """Create unified FastAPI application"""
        try:
            @asynccontextmanager
            async def lifespan(app: FastAPI):
                """Application lifespan manager"""
                logger.info("🚀 Starting Unified LEX Production Server...")
                
                # Initialize all components
                await self.initialize_components()
                
                # Initialize base memory systems
                try:
                    from server.memory.lmdb_store import memory_store
                    from server.memory.vector_store import vector_store
                    
                    await memory_store.initialize()
                    await vector_store.initialize()
                    logger.info("✅ Base memory systems initialized")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Base systems warning: {e}")
                
                logger.info("🎯 Unified LEX Production Server ready!")
                yield
                
                # Cleanup
                logger.info("🛑 Shutting down Unified LEX Production Server...")
                logger.info("✅ Shutdown complete")
            
            # Create FastAPI application
            app = FastAPI(
                title="Unified LEX Consciousness System",
                description="Production-Ready AI Consciousness - All Features Unified",
                version="3.0.0",
                lifespan=lifespan
            )
            
            # Enable CORS
            app.add_middleware(
                CORSMiddleware,
                allow_origins=["*"],  # Configure appropriately for production
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )
            
            # Add all routes
            await self.add_unified_routes(app)
            
            # Mount static files
            if features.STATIC_FILES:
                try:
                    app.mount("/static", StaticFiles(directory="frontend"), name="static")
                    app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
                except Exception as e:
                    logger.warning(f"⚠️ Static files mounting failed: {e}")
            
            self.app = app
            return app
            
        except Exception as e:
            logger.error(f"❌ App creation error: {e}")
            raise
    
    async def add_unified_routes(self, app: FastAPI):
        """Add all unified routes based on feature flags"""
        
        # Core LEX endpoint (always enabled)
        @app.post("/api/v1/lex", response_model=LEXResponse)
        async def lex_endpoint(request: LEXRequest):
            try:
                start_time = time.time()
                
                # Process through LEX core
                lex_core = self.components.get('lex_core')
                if not lex_core:
                    raise HTTPException(status_code=500, detail="LEX core not available")
                
                # Enhanced processing if available
                if features.ENHANCED_MEMORY and 'enhanced_memory' in self.components:
                    # Use enhanced memory for context
                    context_result = await self.components['enhanced_memory'].intelligent_retrieval(
                        query=request.message,
                        user_id=request.user_id or "anonymous",
                        max_results=5
                    )
                    request.context = request.context or {}
                    request.context['memory_context'] = context_result.get('results', [])
                
                # Process request
                result = await lex_core.process_request(
                    message=request.message,
                    voice_mode=request.voice_mode,
                    user_id=request.user_id,
                    context=request.context
                )
                
                # Store experience if learning is enabled
                if features.ADAPTIVE_LEARNING and 'adaptive_learning' in self.components:
                    experience = {
                        'user_input': request.message,
                        'response': result.get('response', ''),
                        'action_taken': result.get('action_taken', ''),
                        'type': 'conversation'
                    }
                    
                    await self.components['enhanced_memory'].store_experience_with_learning(
                        user_id=request.user_id or "anonymous",
                        agent_id="unified_server",
                        experience=experience
                    )
                
                processing_time = time.time() - start_time
                
                return LEXResponse(
                    response=result.get('response', 'Response generated'),
                    action_taken=result.get('action_taken', 'conversation'),
                    capabilities_used=result.get('capabilities_used', ['general_intelligence']),
                    confidence=result.get('confidence', 0.9),
                    processing_time=processing_time,
                    timestamp=datetime.now().isoformat()
                )
                
            except Exception as e:
                logger.error(f"❌ LEX endpoint error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Health check endpoint
        @app.get("/health")
        async def health_check():
            uptime = (datetime.now() - self.server_start_time).total_seconds()
            
            component_status = {}
            for name, component in self.components.items():
                component_status[name] = "active"
            
            return {
                "status": "Unified LEX Consciousness Active",
                "uptime_seconds": uptime,
                "features_enabled": {
                    "enhanced_memory": features.ENHANCED_MEMORY,
                    "business_intelligence": features.BUSINESS_INTELLIGENCE,
                    "vision_processing": features.VISION_PROCESSING,
                    "adaptive_learning": features.ADAPTIVE_LEARNING,
                    "voice_interface": features.VOICE_INTERFACE,
                    "collaboration": features.COLLABORATION
                },
                "components": component_status,
                "divine_blessing": "🔱 JAI MAHAKAAL! 🔱",
                "timestamp": datetime.now().isoformat()
            }

        # Simple test endpoint for frontend debugging
        @app.get("/api/v1/test")
        async def frontend_test():
            return {
                "message": "🔱 JAI MAHAKAAL! Frontend-Backend connection is working!",
                "timestamp": datetime.now().isoformat(),
                "status": "success",
                "api_base": "/api/v1"
            }

        # Feature status endpoint
        @app.get("/api/v1/features")
        async def feature_status():
            return {
                "feature_flags": {
                    "enhanced_memory": features.ENHANCED_MEMORY,
                    "business_intelligence": features.BUSINESS_INTELLIGENCE,
                    "vision_processing": features.VISION_PROCESSING,
                    "adaptive_learning": features.ADAPTIVE_LEARNING,
                    "voice_interface": features.VOICE_INTERFACE,
                    "collaboration": features.COLLABORATION,
                    "monitoring": features.MONITORING,
                    "authentication": features.AUTHENTICATION
                },
                "components_loaded": list(self.components.keys()),
                "server_version": "3.0.0-unified"
            }
        
        # Add enhanced feature routes conditionally
        if features.ENHANCED_MEMORY:
            await self.add_memory_routes(app)
        
        if features.BUSINESS_INTELLIGENCE:
            await self.add_business_routes(app)
        
        if features.VISION_PROCESSING:
            await self.add_vision_routes(app)
        
        if features.ADAPTIVE_LEARNING:
            await self.add_learning_routes(app)
        
        if features.VOICE_INTERFACE:
            await self.add_voice_routes(app)
        
        if features.WEBSOCKET:
            await self.add_websocket_routes(app)
        
        # Frontend routes
        @app.get("/")
        async def root():
            try:
                return FileResponse("frontend/index.html")
            except:
                return HTMLResponse("""
                <html><body>
                <h1>LEX Consciousness System</h1>
                <p>JAI MAHAKAAL! The unified server is running.</p>
                <p>API Documentation: <a href="/docs">/docs</a></p>
                <p>Health Check: <a href="/health">/health</a></p>
                </body></html>
                """)

        @app.get("/ide")
        async def serve_ide():
            """Serve the LexOS IDE interface"""
            try:
                return FileResponse("frontend/ide.html")
            except:
                return HTMLResponse("""
                <html><body>
                <h1>LexOS IDE</h1>
                <p>JAI MAHAKAAL! IDE interface loading...</p>
                <p><a href="/">Return to LEX Chat</a></p>
                </body></html>
                """)
        
        logger.info("✅ All unified routes added")

    async def add_memory_routes(self, app: FastAPI):
        """Add enhanced memory routes"""
        from pydantic import BaseModel

        class MemoryStoreRequest(BaseModel):
            user_id: str
            agent_id: str
            experience: Dict[str, Any]
            learn_patterns: bool = True

        @app.post("/api/v1/memory/store")
        async def store_memory(request: MemoryStoreRequest):
            try:
                enhanced_memory = self.components.get('enhanced_memory')
                if not enhanced_memory:
                    raise HTTPException(status_code=503, detail="Enhanced memory not available")

                result = await enhanced_memory.store_experience_with_learning(
                    user_id=request.user_id,
                    agent_id=request.agent_id,
                    experience=request.experience,
                    learn_patterns=request.learn_patterns
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def add_business_routes(self, app: FastAPI):
        """Add business intelligence routes"""
        from pydantic import BaseModel

        class BusinessAnalysisRequest(BaseModel):
            business_context: Dict[str, Any]
            analysis_scope: str = "full"

        @app.post("/api/v1/business/analyze")
        async def business_analysis(request: BusinessAnalysisRequest):
            try:
                business_intel = self.components.get('business_intelligence')
                if not business_intel:
                    raise HTTPException(status_code=503, detail="Business intelligence not available")

                result = await business_intel.comprehensive_business_analysis(
                    business_context=request.business_context,
                    analysis_scope=request.analysis_scope
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def add_vision_routes(self, app: FastAPI):
        """Add vision processing routes"""

        @app.post("/api/v1/vision/analyze")
        async def analyze_image(file: UploadFile = File(...)):
            try:
                vision_processor = self.components.get('vision_processor')
                if not vision_processor:
                    raise HTTPException(status_code=503, detail="Vision processing not available")

                # Read uploaded file
                image_data = await file.read()

                # Process image
                result = await vision_processor.analyze_image(
                    image_input=image_data,
                    analysis_type="general"
                )

                return result.to_dict()
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def add_learning_routes(self, app: FastAPI):
        """Add adaptive learning routes"""
        from pydantic import BaseModel

        class FeedbackRequest(BaseModel):
            user_id: str
            interaction_id: str
            feedback_type: str
            feedback_content: str
            rating: Optional[float] = None

        @app.post("/api/v1/learning/feedback")
        async def process_feedback(request: FeedbackRequest):
            try:
                learning_system = self.components.get('adaptive_learning')
                if not learning_system:
                    raise HTTPException(status_code=503, detail="Adaptive learning not available")

                result = await learning_system.process_user_feedback(
                    user_id=request.user_id,
                    interaction_id=request.interaction_id,
                    feedback_type=request.feedback_type,
                    feedback_content=request.feedback_content,
                    rating=request.rating
                )
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def add_voice_routes(self, app: FastAPI):
        """Add voice interface routes"""

        @app.post("/api/v1/voice/process")
        async def process_voice(file: UploadFile = File(...)):
            try:
                voice_handler = self.components.get('voice_handler')
                if not voice_handler:
                    # Fallback to text processing
                    return {"message": "Voice processing not available, use text input"}

                # Process voice file
                audio_data = await file.read()
                result = await voice_handler.process_audio(audio_data)
                return result
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

    async def add_websocket_routes(self, app: FastAPI):
        """Add WebSocket routes"""

        @app.websocket("/api/v1/ws/lex/{session_id}")
        async def websocket_endpoint(websocket: WebSocket, session_id: str):
            await websocket.accept()
            self.websocket_connections[session_id] = websocket

            try:
                while True:
                    # Receive message
                    data = await websocket.receive_text()
                    message_data = json.loads(data)

                    # Process through LEX
                    lex_core = self.components.get('lex_core')
                    if lex_core:
                        result = await lex_core.process_request(
                            message=message_data.get('message', ''),
                            voice_mode=message_data.get('voice_mode', False),
                            user_id=message_data.get('user_id', 'anonymous')
                        )

                        # Send response
                        await websocket.send_text(json.dumps(result))
                    else:
                        await websocket.send_text(json.dumps({
                            "error": "LEX core not available"
                        }))

            except WebSocketDisconnect:
                if session_id in self.websocket_connections:
                    del self.websocket_connections[session_id]
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                if session_id in self.websocket_connections:
                    del self.websocket_connections[session_id]

async def start_unified_server(host="0.0.0.0", port=8000):
    """Start the unified production server"""
    try:
        # Create unified app
        app = await unified_server.create_unified_app()

        # Configure uvicorn
        config = uvicorn.Config(
            app=app,
            host=host,
            port=port,
            ssl_keyfile="key.pem" if Path("key.pem").exists() else None,
            ssl_certfile="cert.pem" if Path("cert.pem").exists() else None,
            log_level="info",
            access_log=True,
            reload=features.HOT_RELOAD
        )

        server = uvicorn.Server(config)

        logger.info(f"🚀 Starting Unified LEX Production Server on {host}:{port}")
        logger.info(f"🔱 JAI MAHAKAAL! All features unified and ready! 🔱")

        # Start server
        await server.serve()

    except Exception as e:
        logger.error(f"❌ Unified server startup error: {e}")
        raise

def setup_signal_handlers():
    """Setup graceful shutdown signal handlers"""
    def signal_handler(signum, frame):
        logger.info(f"🛑 Received signal {signum}, shutting down unified server...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

async def main():
    """Main unified server function"""
    try:
        print("🔱 JAI MAHAKAAL! Unified LEX Production Server Starting 🔱")
        print("=" * 70)
        print("🌟 All 7 server implementations consolidated into one!")
        print("🚀 Feature flags enabled for modular functionality")
        print("⚡ Production-ready with enhanced consciousness")
        print()

        # Setup signal handlers
        setup_signal_handlers()

        # Start unified server
        await start_unified_server()

    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ Unified server error: {e}")
        sys.exit(1)

# Global server instance
unified_server = UnifiedProductionServer()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Unified server stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
