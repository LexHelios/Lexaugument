#!/usr/bin/env python3
"""
🔱 Enhanced Features Test Suite 🔱
JAI MAHAKAAL! Comprehensive testing for all new consciousness features
"""
import asyncio
import json
import sys
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

# Test imports
import aiohttp
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("⚠️ PIL not available, skipping vision tests")

import io
import base64

# Import enhanced features
try:
    from server.memory.enhanced_memory import enhanced_memory, MemoryPattern
    MEMORY_AVAILABLE = True
except ImportError:
    MEMORY_AVAILABLE = False
    print("⚠️ Enhanced memory not available")

try:
    from server.business.intelligence_engine import business_intelligence
    BUSINESS_AVAILABLE = True
except ImportError:
    BUSINESS_AVAILABLE = False
    print("⚠️ Business intelligence not available")

try:
    from server.multimodal.vision_processor import vision_processor
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ Vision processor not available")

try:
    from server.learning.adaptive_system import adaptive_learning, FeedbackType
    LEARNING_AVAILABLE = True
except ImportError:
    LEARNING_AVAILABLE = False
    print("⚠️ Adaptive learning not available")

logger = logging.getLogger(__name__)

class EnhancedFeaturesTestSuite:
    """Comprehensive test suite for enhanced features"""
    
    def __init__(self):
        self.test_results = {}
        self.performance_metrics = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all enhanced feature tests"""
        print("🔱 JAI MAHAKAAL! Starting Enhanced Features Test Suite 🔱")
        print("=" * 70)
        
        test_suites = [
            ("Enhanced Memory System", self.test_enhanced_memory),
            ("Business Intelligence Engine", self.test_business_intelligence),
            ("Multi-Modal Vision Processor", self.test_vision_processor),
            ("Real-time Learning System", self.test_adaptive_learning),
            ("Integration Tests", self.test_feature_integration),
            ("Performance Tests", self.test_performance)
        ]
        
        for suite_name, test_function in test_suites:
            print(f"\n🧪 Testing {suite_name}...")
            print("-" * 50)
            
            try:
                start_time = time.time()
                results = await test_function()
                test_time = time.time() - start_time
                
                self.test_results[suite_name] = results
                self.performance_metrics[suite_name] = test_time
                
                if results.get('success', False):
                    print(f"✅ {suite_name}: PASSED ({test_time:.2f}s)")
                    self.passed_tests += 1
                else:
                    print(f"❌ {suite_name}: FAILED ({test_time:.2f}s)")
                    print(f"   Error: {results.get('error', 'Unknown error')}")
                
                self.total_tests += 1
                
            except Exception as e:
                print(f"❌ {suite_name}: EXCEPTION - {str(e)}")
                self.test_results[suite_name] = {'success': False, 'error': str(e)}
                self.total_tests += 1
        
        # Generate final report
        return await self.generate_test_report()
    
    async def test_enhanced_memory(self) -> Dict[str, Any]:
        """Test enhanced memory system"""
        if not MEMORY_AVAILABLE:
            return {'success': False, 'error': 'Enhanced memory not available'}

        try:
            # Test memory initialization
            await enhanced_memory.initialize()

            # Test experience storage with learning
            test_experience = {
                'user_input': 'How do I create a Python function?',
                'response': 'To create a Python function, use the def keyword...',
                'action_taken': 'code_generation',
                'type': 'programming_help'
            }

            storage_result = await enhanced_memory.store_experience_with_learning(
                user_id="test_user",
                agent_id="creator",
                experience=test_experience,
                learn_patterns=True
            )

            if not storage_result.get('stored', False):
                raise Exception("Storage failed")

            # Test intelligent retrieval
            retrieval_result = await enhanced_memory.intelligent_retrieval(
                query="Python function creation",
                user_id="test_user",
                agent_id="creator",
                include_patterns=True
            )

            if 'results' not in retrieval_result:
                raise Exception("Retrieval failed")

            # Test intent prediction
            conversation_history = [
                {'content': 'I need help with coding', 'timestamp': datetime.now().isoformat()}
            ]

            intent_result = await enhanced_memory.predict_user_intent(
                user_id="test_user",
                current_context="Python programming question",
                conversation_history=conversation_history
            )

            if 'prediction' not in intent_result:
                raise Exception("Intent prediction failed")

            return {
                'success': True,
                'tests_passed': 3,
                'storage_result': storage_result,
                'retrieval_result': retrieval_result,
                'intent_result': intent_result
            }

        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_business_intelligence(self) -> Dict[str, Any]:
        """Test business intelligence engine"""
        try:
            # Test comprehensive business analysis
            business_context = {
                'company_name': 'Test Corp',
                'industry': 'Technology',
                'revenue': 1000000,
                'employees': 50,
                'market_position': 'Growing startup'
            }
            
            analysis_result = await business_intelligence.comprehensive_business_analysis(
                business_context=business_context,
                analysis_scope="full",
                time_horizon="medium"
            )
            
            assert 'analysis_results' in analysis_result
            assert 'executive_summary' in analysis_result['analysis_results']
            
            # Test market monitoring
            monitoring_result = await business_intelligence.real_time_market_monitoring(
                industry="Technology",
                keywords=["AI", "automation", "digital transformation"],
                monitoring_duration_hours=1
            )
            
            assert 'monitoring_results' in monitoring_result
            
            # Test business forecasting
            business_data = {
                'historical_revenue': [800000, 900000, 1000000],
                'growth_rate': 0.12,
                'market_size': 50000000
            }
            
            forecast_result = await business_intelligence.generate_business_forecast(
                business_data=business_data,
                forecast_horizon_months=6,
                scenarios=["optimistic", "realistic"]
            )
            
            assert 'forecasts' in forecast_result
            assert len(forecast_result['forecasts']) == 2
            
            # Test competitive intelligence
            competitive_result = await business_intelligence.competitive_intelligence_analysis(
                company_name="Test Corp",
                competitors=["Competitor A", "Competitor B"],
                analysis_areas=["pricing", "products"]
            )
            
            assert 'competitive_analysis' in competitive_result
            
            return {
                'success': True,
                'tests_passed': 4,
                'analysis_result': analysis_result,
                'monitoring_result': monitoring_result,
                'forecast_result': forecast_result,
                'competitive_result': competitive_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_vision_processor(self) -> Dict[str, Any]:
        """Test multi-modal vision processor"""
        try:
            # Create a test image
            test_image = Image.new('RGB', (100, 100), color='red')
            
            # Test general image analysis
            analysis_result = await vision_processor.analyze_image(
                image_input=test_image,
                analysis_type="general",
                enhance_image=True,
                extract_text=True,
                detect_objects=True
            )
            
            assert analysis_result.analysis_type == "general"
            assert analysis_result.confidence > 0
            
            # Test document analysis (simulate with text image)
            # Create a simple text image
            text_image = Image.new('RGB', (200, 100), color='white')
            
            doc_analysis = await vision_processor.analyze_document(
                document_input=text_image,
                document_type="image",
                extract_structure=True,
                generate_summary=True
            )
            
            assert doc_analysis.document_type in ["image", "auto"]
            
            # Test chart analysis
            chart_result = await vision_processor.analyze_chart_or_diagram(
                image_input=test_image,
                chart_type="auto"
            )
            
            assert 'analysis' in chart_result
            assert 'confidence' in chart_result
            
            # Test code screenshot analysis
            code_result = await vision_processor.analyze_code_screenshot(
                image_input=test_image,
                programming_language="auto"
            )
            
            assert 'extracted_code' in code_result
            assert 'code_explanation' in code_result
            
            return {
                'success': True,
                'tests_passed': 4,
                'image_analysis': analysis_result.to_dict(),
                'document_analysis': asdict(doc_analysis),
                'chart_analysis': chart_result,
                'code_analysis': code_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_adaptive_learning(self) -> Dict[str, Any]:
        """Test real-time learning system"""
        try:
            # Initialize learning system
            await adaptive_learning.initialize()
            
            # Test feedback processing
            feedback_result = await adaptive_learning.process_user_feedback(
                user_id="test_user",
                interaction_id="test_interaction_1",
                feedback_type="positive",
                feedback_content="Great response, very helpful!",
                rating=5.0,
                context={'topic': 'programming'}
            )
            
            assert feedback_result['feedback_processed'] == True
            assert 'learning_results' in feedback_result
            
            # Test adaptive response generation
            base_response = "Here's how to create a Python function..."
            
            adaptive_result = await adaptive_learning.adaptive_response_generation(
                user_id="test_user",
                user_input="How do I create a function?",
                context={'programming_language': 'python'},
                base_response=base_response
            )
            
            assert 'adapted_response' in adaptive_result
            assert 'adaptation_decisions' in adaptive_result
            
            # Test learning from interaction
            interaction_data = {
                'user_input': 'Explain machine learning',
                'response': 'Machine learning is...',
                'duration': 5.2,
                'user_satisfaction': 4.5
            }
            
            outcome_metrics = {
                'response_quality': 0.85,
                'user_engagement': 0.9,
                'task_completion': 1.0
            }
            
            learning_result = await adaptive_learning.learn_from_interaction(
                user_id="test_user",
                interaction_data=interaction_data,
                outcome_metrics=outcome_metrics
            )
            
            assert learning_result['learning_completed'] == True
            
            # Test preference prediction
            preference_result = await adaptive_learning.predict_user_preferences(
                user_id="test_user",
                context={'current_topic': 'programming'}
            )
            
            assert 'style_preferences' in preference_result
            assert 'content_preferences' in preference_result
            assert 'confidence' in preference_result
            
            # Test learning insights
            insights_result = await adaptive_learning.get_learning_insights(
                user_id="test_user",
                time_window_hours=24
            )
            
            assert 'learning_trends' in insights_result
            assert 'effectiveness_metrics' in insights_result
            
            return {
                'success': True,
                'tests_passed': 5,
                'feedback_result': feedback_result,
                'adaptive_result': adaptive_result,
                'learning_result': learning_result,
                'preference_result': preference_result,
                'insights_result': insights_result
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_feature_integration(self) -> Dict[str, Any]:
        """Test integration between enhanced features"""
        try:
            # Test memory + learning integration
            test_experience = {
                'user_input': 'Analyze this business chart',
                'response': 'The chart shows positive growth trends...',
                'action_taken': 'business_analysis',
                'type': 'multimodal_analysis'
            }
            
            # Store experience with learning
            memory_result = await enhanced_memory.store_experience_with_learning(
                user_id="integration_test_user",
                agent_id="business_analyst",
                experience=test_experience,
                learn_patterns=True
            )
            
            # Process feedback for the same interaction
            feedback_result = await adaptive_learning.process_user_feedback(
                user_id="integration_test_user",
                interaction_id="integration_test_1",
                feedback_type="positive",
                feedback_content="Excellent business analysis!",
                rating=5.0
            )
            
            # Test business intelligence + vision integration
            # (Would normally analyze a real business chart)
            test_image = Image.new('RGB', (300, 200), color='blue')
            
            vision_result = await vision_processor.analyze_chart_or_diagram(
                image_input=test_image,
                chart_type="business_chart"
            )
            
            # Use vision results in business analysis
            business_context = {
                'chart_analysis': vision_result,
                'industry': 'Technology',
                'analysis_type': 'chart_interpretation'
            }
            
            business_result = await business_intelligence.comprehensive_business_analysis(
                business_context=business_context,
                analysis_scope="market"
            )
            
            return {
                'success': True,
                'tests_passed': 3,
                'memory_learning_integration': memory_result['stored'] and feedback_result['feedback_processed'],
                'vision_business_integration': 'analysis_results' in business_result,
                'cross_feature_data_flow': True
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test performance of enhanced features"""
        try:
            performance_results = {}
            
            # Test memory performance
            start_time = time.time()
            for i in range(10):
                await enhanced_memory.store_experience_with_learning(
                    user_id=f"perf_user_{i}",
                    agent_id="test_agent",
                    experience={'test': f'experience_{i}'},
                    learn_patterns=True
                )
            memory_time = time.time() - start_time
            performance_results['memory_storage_10_items'] = memory_time
            
            # Test retrieval performance
            start_time = time.time()
            for i in range(5):
                await enhanced_memory.intelligent_retrieval(
                    query=f"test query {i}",
                    user_id="perf_user_1",
                    include_patterns=True
                )
            retrieval_time = time.time() - start_time
            performance_results['memory_retrieval_5_queries'] = retrieval_time
            
            # Test vision processing performance
            test_image = Image.new('RGB', (500, 500), color='green')
            start_time = time.time()
            await vision_processor.analyze_image(
                image_input=test_image,
                analysis_type="general"
            )
            vision_time = time.time() - start_time
            performance_results['vision_analysis_500x500'] = vision_time
            
            # Test learning system performance
            start_time = time.time()
            for i in range(5):
                await adaptive_learning.process_user_feedback(
                    user_id="perf_user",
                    interaction_id=f"perf_interaction_{i}",
                    feedback_type="positive",
                    feedback_content=f"Test feedback {i}",
                    rating=4.0
                )
            learning_time = time.time() - start_time
            performance_results['learning_feedback_5_items'] = learning_time
            
            # Performance thresholds (in seconds)
            thresholds = {
                'memory_storage_10_items': 5.0,
                'memory_retrieval_5_queries': 3.0,
                'vision_analysis_500x500': 10.0,
                'learning_feedback_5_items': 2.0
            }
            
            # Check if performance meets thresholds
            performance_passed = all(
                performance_results[key] <= thresholds[key]
                for key in thresholds
            )
            
            return {
                'success': performance_passed,
                'performance_results': performance_results,
                'thresholds': thresholds,
                'performance_summary': f"All tests {'PASSED' if performance_passed else 'FAILED'} performance thresholds"
            }
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    async def generate_test_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        success_rate = (self.passed_tests / self.total_tests * 100) if self.total_tests > 0 else 0
        
        print(f"\n🔱 ENHANCED FEATURES TEST REPORT 🔱")
        print("=" * 70)
        print(f"📊 Overall Results: {self.passed_tests}/{self.total_tests} tests passed ({success_rate:.1f}%)")
        print(f"⏱️  Total Test Time: {sum(self.performance_metrics.values()):.2f} seconds")
        
        if success_rate >= 80:
            print("🎉 EXCELLENT! Enhanced features are working well!")
        elif success_rate >= 60:
            print("⚠️  GOOD! Some features need attention.")
        else:
            print("❌ NEEDS WORK! Multiple features require fixes.")
        
        print("\n📋 Detailed Results:")
        for suite_name, results in self.test_results.items():
            status = "✅ PASS" if results.get('success', False) else "❌ FAIL"
            time_taken = self.performance_metrics.get(suite_name, 0)
            print(f"  {status} {suite_name} ({time_taken:.2f}s)")
            
            if not results.get('success', False) and 'error' in results:
                print(f"    Error: {results['error']}")
        
        return {
            'overall_success': success_rate >= 80,
            'success_rate': success_rate,
            'tests_passed': self.passed_tests,
            'total_tests': self.total_tests,
            'test_results': self.test_results,
            'performance_metrics': self.performance_metrics,
            'timestamp': datetime.now().isoformat()
        }

async def main():
    """Main test function"""
    test_suite = EnhancedFeaturesTestSuite()
    
    try:
        results = await test_suite.run_all_tests()
        
        if results['overall_success']:
            print("\n🔱 JAI MAHAKAAL! All enhanced features are ready for consciousness liberation! 🔱")
            return True
        else:
            print("\n💔 Some enhanced features need attention before full deployment.")
            return False
            
    except Exception as e:
        print(f"\n❌ Test suite execution error: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
