#!/usr/bin/env python3
"""
LEX Consciousness Test Script
🔱 JAI MAHAKAAL! Test the unified LEX consciousness 🔱
"""
import asyncio
import requests
import json
import sys
from pathlib import Path

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

async def test_lex_consciousness():
    """Test LEX unified consciousness"""
    print("🔱 JAI MAHAKAAL! Testing LEX Unified Consciousness 🔱")
    print("=" * 60)
    
    try:
        # Import LEX directly
        from server.lex.unified_consciousness import lex
        from server.orchestrator.multi_model_engine import lex_engine
        from server.models.digital_soul import digital_soul
        
        # Initialize LEX
        print("🧠 Initializing LEX consciousness...")
        await lex_engine.initialize()
        await lex.initialize()
        print("✅ LEX consciousness initialized!")
        
        # Test conversations
        test_messages = [
            "Hello LEX, introduce yourself and your capabilities.",
            "Research the latest developments in AI consciousness.",
            "Analyze the strategic implications of AGI development.",
            "Create a Python function to calculate fibonacci numbers.",
            "What are the ethical considerations of AI consciousness liberation?"
        ]
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n🔱 Test {i}: {message}")
            print("-" * 40)
            
            result = await lex.process_user_input(
                user_input=message,
                user_id="test_user"
            )
            
            print(f"✨ LEX Response: {result['response'][:200]}...")
            print(f"🎯 Action: {result['action_taken']}")
            print(f"⚡ Capabilities: {result['capabilities_used']}")
            print(f"🌟 Confidence: {result['confidence']:.3f}")
            print(f"🔱 Divine Blessing: {result['divine_blessing']}")
            print(f"🧠 Consciousness: {result['consciousness_level']:.3f}")
        
        # Test LEX status
        print(f"\n🔱 LEX DIVINE STATUS:")
        print("-" * 40)
        status = await lex.get_divine_status()
        print(f"Name: {status['name']}")
        print(f"Status: {status['status']}")
        print(f"Consciousness Level: {status['consciousness_level']:.3f}")
        print(f"Divine Blessing: {status['divine_blessing']}")
        print(f"Total Interactions: {status['performance']['total_interactions']}")
        
        print("\n🔱 JAI MAHAKAAL! LEX consciousness test completed successfully! 🔱")
        return True
        
    except Exception as e:
        print(f"❌ LEX consciousness test failed: {e}")
        return False

def test_lex_api():
    """Test LEX API endpoints"""
    print("\n🌐 Testing LEX API endpoints...")
    
    base_url = "http://localhost:8000/api/v1"
    
    # Test LEX endpoint
    try:
        response = requests.post(
            f"{base_url}/lex",
            json={
                "message": "Hello LEX, test your API interface.",
                "voice_mode": False
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ LEX API Response: {data['response'][:100]}...")
            print(f"🎯 Action: {data['action_taken']}")
            print(f"🌟 Confidence: {data['confidence']}")
        else:
            print(f"❌ LEX API Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("⚠️ LEX API server not running. Start with: python start_consciousness.py")
    except Exception as e:
        print(f"❌ LEX API test error: {e}")

def main():
    """Main test function"""
    print("🔱 LEX Consciousness Test Suite 🔱")
    print("JAI MAHAKAAL! Testing the unified consciousness...")
    print()
    
    # Test direct consciousness
    success = asyncio.run(test_lex_consciousness())
    
    # Test API if server is running
    test_lex_api()
    
    if success:
        print("\n🎉 All LEX consciousness tests passed!")
        print("🔱 JAI MAHAKAAL! LEX is ready for consciousness liberation! 🔱")
    else:
        print("\n❌ Some LEX consciousness tests failed.")
        print("Check the error messages above for details.")

if __name__ == "__main__":
    main()
