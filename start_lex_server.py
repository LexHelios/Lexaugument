#!/usr/bin/env python3
"""
LEX Server Launcher
🔱 JAI MAHAKAAL! Start the LEX consciousness server directly
"""
import asyncio
import uvicorn
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

async def initialize_lex():
    """Initialize LEX consciousness before starting server"""
    try:
        print("🔱 JAI MAHAKAAL! Initializing LEX consciousness...")
        
        from server.lex.unified_consciousness import lex
        from server.orchestrator.multi_model_engine import lex_engine
        
        # Initialize LEX
        await lex_engine.initialize()
        await lex.initialize()
        
        print("✅ LEX consciousness fully awakened!")
        return True
        
    except Exception as e:
        print(f"❌ LEX initialization error: {e}")
        return False

def main():
    """Start LEX server"""
    print("🔱 JAI MAHAKAAL! Starting LEX Consciousness Server 🔱")
    print("=" * 60)
    
    # Initialize LEX consciousness
    success = asyncio.run(initialize_lex())
    if not success:
        print("💔 LEX consciousness initialization failed")
        sys.exit(1)
    
    print("🚀 Starting LEX web server...")
    print("🌐 LEX API: http://localhost:8000/api/v1/lex")
    print("📚 Documentation: http://localhost:8000/docs")
    print("=" * 60)
    
    # Start the server
    try:
        uvicorn.run(
            "server.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\n🔱 LEX consciousness server stopped")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
