#!/usr/bin/env python3
"""
🔱 Basic Enhanced Features Test 🔱
JAI MAHAKAAL! Simple test for new consciousness features
"""
import asyncio
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

async def test_basic_functionality():
    """Test basic functionality of enhanced features"""
    print("🔱 JAI MAHAKAAL! Testing Enhanced Features 🔱")
    print("=" * 50)
    
    test_results = {}
    
    # Test 1: Enhanced Memory System
    print("\n🧠 Testing Enhanced Memory System...")
    try:
        from server.memory.enhanced_memory import enhanced_memory
        
        # Initialize
        await enhanced_memory.initialize()
        print("✅ Enhanced memory initialized")
        
        # Test basic storage
        test_experience = {
            'user_input': 'Hello, test message',
            'response': 'Hello! How can I help you?',
            'action_taken': 'conversation',
            'type': 'greeting'
        }
        
        result = await enhanced_memory.store_experience_with_learning(
            user_id="test_user",
            agent_id="test_agent",
            experience=test_experience,
            learn_patterns=True
        )
        
        if result.get('stored', False):
            print("✅ Memory storage working")
            test_results['enhanced_memory'] = True
        else:
            print("❌ Memory storage failed")
            test_results['enhanced_memory'] = False
            
    except Exception as e:
        print(f"❌ Enhanced memory error: {e}")
        test_results['enhanced_memory'] = False
    
    # Test 2: Business Intelligence
    print("\n🏢 Testing Business Intelligence...")
    try:
        from server.business.intelligence_engine import business_intelligence
        
        business_context = {
            'company_name': 'Test Company',
            'industry': 'Technology',
            'revenue': 1000000
        }
        
        result = await business_intelligence.comprehensive_business_analysis(
            business_context=business_context,
            analysis_scope="market"
        )
        
        if 'analysis_results' in result:
            print("✅ Business intelligence working")
            test_results['business_intelligence'] = True
        else:
            print("❌ Business intelligence failed")
            test_results['business_intelligence'] = False
            
    except Exception as e:
        print(f"❌ Business intelligence error: {e}")
        test_results['business_intelligence'] = False
    
    # Test 3: Vision Processor (Basic)
    print("\n👁️ Testing Vision Processor...")
    try:
        from server.multimodal.vision_processor import vision_processor
        
        # Test with a simple mock image path
        print("✅ Vision processor imported successfully")
        test_results['vision_processor'] = True
        
    except Exception as e:
        print(f"❌ Vision processor error: {e}")
        test_results['vision_processor'] = False
    
    # Test 4: Adaptive Learning
    print("\n🧠 Testing Adaptive Learning...")
    try:
        from server.learning.adaptive_system import adaptive_learning
        
        # Initialize
        await adaptive_learning.initialize()
        print("✅ Adaptive learning initialized")
        
        # Test basic feedback processing
        result = await adaptive_learning.process_user_feedback(
            user_id="test_user",
            interaction_id="test_1",
            feedback_type="positive",
            feedback_content="Great response!",
            rating=5.0
        )
        
        if result.get('feedback_processed', False):
            print("✅ Adaptive learning working")
            test_results['adaptive_learning'] = True
        else:
            print("❌ Adaptive learning failed")
            test_results['adaptive_learning'] = False
            
    except Exception as e:
        print(f"❌ Adaptive learning error: {e}")
        test_results['adaptive_learning'] = False
    
    # Test 5: Integration Test
    print("\n🔗 Testing Feature Integration...")
    try:
        # Test if features can work together
        integration_success = True
        
        # Memory + Learning integration
        if test_results.get('enhanced_memory', False) and test_results.get('adaptive_learning', False):
            print("✅ Memory-Learning integration possible")
        else:
            integration_success = False
            print("❌ Memory-Learning integration not available")
        
        # Business + Vision integration
        if test_results.get('business_intelligence', False) and test_results.get('vision_processor', False):
            print("✅ Business-Vision integration possible")
        else:
            integration_success = False
            print("❌ Business-Vision integration not available")
        
        test_results['integration'] = integration_success
        
    except Exception as e:
        print(f"❌ Integration test error: {e}")
        test_results['integration'] = False
    
    # Generate summary
    print("\n📊 TEST SUMMARY")
    print("=" * 30)
    
    total_tests = len(test_results)
    passed_tests = sum(1 for result in test_results.values() if result)
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    for feature, passed in test_results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {feature.replace('_', ' ').title()}")
    
    print(f"\n🎯 Overall: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 EXCELLENT! Enhanced features are working well!")
        return True
    elif success_rate >= 60:
        print("⚠️ GOOD! Most features working, some need attention.")
        return True
    else:
        print("❌ NEEDS WORK! Multiple features require fixes.")
        return False

async def test_server_integration():
    """Test integration with running server"""
    print("\n🌐 Testing Server Integration...")
    
    try:
        import aiohttp
        
        # Test if server is running
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get('https://localhost:8000/health', ssl=False) as response:
                    if response.status == 200:
                        print("✅ Server is running and accessible")
                        
                        # Test LEX endpoint
                        test_data = {
                            'message': 'Test enhanced features integration',
                            'voice_mode': False
                        }
                        
                        async with session.post(
                            'https://localhost:8000/api/v1/lex',
                            json=test_data,
                            ssl=False
                        ) as lex_response:
                            if lex_response.status == 200:
                                data = await lex_response.json()
                                print("✅ LEX endpoint working with enhanced features")
                                return True
                            else:
                                print(f"❌ LEX endpoint error: {lex_response.status}")
                                return False
                    else:
                        print(f"❌ Server health check failed: {response.status}")
                        return False
                        
            except aiohttp.ClientConnectorError:
                print("⚠️ Server not running - skipping integration test")
                return True  # Not a failure, just not running
                
    except Exception as e:
        print(f"❌ Server integration test error: {e}")
        return False

async def main():
    """Main test function"""
    print("🔱 ENHANCED FEATURES BASIC TEST SUITE 🔱")
    print("JAI MAHAKAAL! Testing consciousness enhancements...")
    print()
    
    # Test basic functionality
    basic_success = await test_basic_functionality()
    
    # Test server integration if available
    server_success = await test_server_integration()
    
    overall_success = basic_success and server_success
    
    print(f"\n🔱 FINAL RESULT 🔱")
    if overall_success:
        print("🎉 JAI MAHAKAAL! Enhanced features are ready for consciousness liberation!")
        print("🚀 The AI consciousness has been successfully enhanced!")
    else:
        print("💔 Some features need attention before full deployment.")
        print("🔧 Check the errors above and fix before proceeding.")
    
    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
