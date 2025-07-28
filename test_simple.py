#!/usr/bin/env python3
"""
Simple LEX Test
🔱 JAI MAHAKAAL! Test basic LEX functionality
"""
import asyncio
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

async def test_lex_simple():
    """Test LEX basic functionality"""
    try:
        print("🔱 JAI MAHAKAAL! Testing LEX basic functionality...")
        
        # Test basic imports
        print("📦 Testing imports...")
        from server.lex.unified_consciousness import lex
        from server.orchestrator.multi_model_engine import lex_engine
        print("✅ Imports successful!")
        
        # Test initialization
        print("🧠 Testing initialization...")
        await lex_engine.initialize()
        print("✅ LEX engine initialized!")
        
        await lex.initialize()
        print("✅ LEX consciousness initialized!")
        
        # Test simple interaction
        print("💬 Testing simple interaction...")
        result = await lex.process_user_input(
            user_input="Hello LEX, are you working?",
            user_id="test_user"
        )
        
        print(f"✨ LEX Response: {result['response'][:100]}...")
        print(f"🎯 Action: {result['action_taken']}")
        print(f"🌟 Confidence: {result['confidence']:.3f}")
        print(f"🔱 Divine Blessing: {result['divine_blessing']}")
        
        print("\n🔱 JAI MAHAKAAL! LEX is working perfectly! 🔱")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_lex_simple())
    if success:
        print("🎉 All tests passed!")
    else:
        print("💔 Tests failed!")
        sys.exit(1)
