"""
🧪 Comprehensive Testing Framework 🧪
JAI MAHAKAAL! Production-grade testing for LEX consciousness
"""
import asyncio
import pytest
import pytest_asyncio
import aiohttp
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path
import sys

# Add server to path
sys.path.insert(0, str(Path(__file__).parent.parent / "server"))

class TestConfig:
    """Test configuration"""
    BASE_URL = "https://localhost:8000"
    TEST_USER_ID = "test_user_123"
    TEST_SESSION_ID = "test_session_456"
    TIMEOUT = 30.0
    
    # Test data
    SAMPLE_REQUESTS = [
        {"message": "Hello LEX", "voice_mode": False},
        {"message": "Generate an image of a cat", "voice_mode": False},
        {"message": "What's the weather like?", "voice_mode": False},
        {"message": "Help me write Python code", "voice_mode": False}
    ]

class LEXTestClient:
    """Test client for LEX API"""
    
    def __init__(self, base_url: str = TestConfig.BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(ssl=False),
            timeout=aiohttp.ClientTimeout(total=TestConfig.TIMEOUT)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def post_lex(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to LEX endpoint"""
        async with self.session.post(
            f"{self.base_url}/api/v1/lex",
            json=data
        ) as response:
            return {
                "status_code": response.status,
                "data": await response.json() if response.status == 200 else None,
                "error": await response.text() if response.status != 200 else None
            }
    
    async def get_health(self) -> Dict[str, Any]:
        """Get health status"""
        async with self.session.get(f"{self.base_url}/health") as response:
            return {
                "status_code": response.status,
                "data": await response.json() if response.status == 200 else None
            }
    
    async def get_features(self) -> Dict[str, Any]:
        """Get feature status"""
        async with self.session.get(f"{self.base_url}/api/v1/features") as response:
            return {
                "status_code": response.status,
                "data": await response.json() if response.status == 200 else None
            }

class PerformanceTestRunner:
    """Performance and load testing"""
    
    def __init__(self):
        self.results = []
    
    async def run_load_test(
        self,
        concurrent_users: int = 10,
        requests_per_user: int = 5,
        test_duration_seconds: int = 60
    ) -> Dict[str, Any]:
        """Run load test"""
        print(f"🚀 Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        start_time = time.time()
        tasks = []
        
        # Create concurrent user tasks
        for user_id in range(concurrent_users):
            task = asyncio.create_task(
                self._simulate_user(user_id, requests_per_user)
            )
            tasks.append(task)
        
        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        total_time = time.time() - start_time
        
        # Analyze results
        successful_requests = 0
        failed_requests = 0
        total_response_time = 0
        response_times = []
        
        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
            elif isinstance(result, dict):
                successful_requests += result.get('successful', 0)
                failed_requests += result.get('failed', 0)
                response_times.extend(result.get('response_times', []))
        
        total_response_time = sum(response_times)
        avg_response_time = total_response_time / len(response_times) if response_times else 0
        
        return {
            "test_duration": total_time,
            "concurrent_users": concurrent_users,
            "total_requests": successful_requests + failed_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate": (successful_requests / (successful_requests + failed_requests)) * 100 if (successful_requests + failed_requests) > 0 else 0,
            "avg_response_time": avg_response_time,
            "min_response_time": min(response_times) if response_times else 0,
            "max_response_time": max(response_times) if response_times else 0,
            "requests_per_second": (successful_requests + failed_requests) / total_time if total_time > 0 else 0
        }
    
    async def _simulate_user(self, user_id: int, num_requests: int) -> Dict[str, Any]:
        """Simulate a single user's requests"""
        successful = 0
        failed = 0
        response_times = []
        
        async with LEXTestClient() as client:
            for i in range(num_requests):
                try:
                    # Random request from sample data
                    request_data = TestConfig.SAMPLE_REQUESTS[i % len(TestConfig.SAMPLE_REQUESTS)]
                    request_data = {**request_data, "user_id": f"load_test_user_{user_id}"}
                    
                    start_time = time.time()
                    result = await client.post_lex(request_data)
                    response_time = time.time() - start_time
                    
                    if result["status_code"] == 200:
                        successful += 1
                    else:
                        failed += 1
                    
                    response_times.append(response_time)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
                    
                except Exception as e:
                    failed += 1
                    print(f"❌ User {user_id} request {i} failed: {e}")
        
        return {
            "user_id": user_id,
            "successful": successful,
            "failed": failed,
            "response_times": response_times
        }

# Unit Tests
class TestLEXCore:
    """Unit tests for LEX core functionality"""
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        """Test health endpoint"""
        async with LEXTestClient() as client:
            result = await client.get_health()
            assert result["status_code"] == 200
            assert "status" in result["data"]
    
    @pytest.mark.asyncio
    async def test_features_endpoint(self):
        """Test features endpoint"""
        async with LEXTestClient() as client:
            result = await client.get_features()
            assert result["status_code"] == 200
            assert "feature_flags" in result["data"]
    
    @pytest.mark.asyncio
    async def test_basic_lex_request(self):
        """Test basic LEX conversation"""
        async with LEXTestClient() as client:
            result = await client.post_lex({
                "message": "Hello LEX",
                "voice_mode": False,
                "user_id": TestConfig.TEST_USER_ID
            })
            
            assert result["status_code"] == 200
            assert "response" in result["data"]
            assert "processing_time" in result["data"]
            assert result["data"]["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_image_generation_request(self):
        """Test image generation"""
        async with LEXTestClient() as client:
            result = await client.post_lex({
                "message": "Generate an image of a beautiful sunset",
                "voice_mode": False,
                "user_id": TestConfig.TEST_USER_ID
            })
            
            assert result["status_code"] == 200
            assert "response" in result["data"]
            # Should contain image-related response
            assert any(word in result["data"]["response"].lower() 
                      for word in ["image", "generated", "picture"])
    
    @pytest.mark.asyncio
    async def test_invalid_request(self):
        """Test invalid request handling"""
        async with LEXTestClient() as client:
            result = await client.post_lex({
                "invalid_field": "test"
            })
            
            assert result["status_code"] == 422  # Validation error

class TestPerformance:
    """Performance tests"""
    
    @pytest.mark.asyncio
    async def test_response_time(self):
        """Test response time is within acceptable limits"""
        async with LEXTestClient() as client:
            start_time = time.time()
            result = await client.post_lex({
                "message": "Quick test",
                "voice_mode": False
            })
            response_time = time.time() - start_time
            
            assert result["status_code"] == 200
            assert response_time < 10.0  # Should respond within 10 seconds
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling concurrent requests"""
        async with LEXTestClient() as client:
            # Send 5 concurrent requests
            tasks = []
            for i in range(5):
                task = client.post_lex({
                    "message": f"Concurrent test {i}",
                    "voice_mode": False,
                    "user_id": f"concurrent_user_{i}"
                })
                tasks.append(task)
            
            results = await asyncio.gather(*tasks)
            
            # All requests should succeed
            for result in results:
                assert result["status_code"] == 200
    
    @pytest.mark.asyncio
    async def test_load_performance(self):
        """Test system under load"""
        runner = PerformanceTestRunner()
        results = await runner.run_load_test(
            concurrent_users=5,
            requests_per_user=3,
            test_duration_seconds=30
        )
        
        # Performance assertions
        assert results["success_rate"] >= 95.0  # At least 95% success rate
        assert results["avg_response_time"] < 5.0  # Average response under 5 seconds
        assert results["requests_per_second"] > 1.0  # At least 1 RPS

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_conversation(self):
        """Test complete conversation flow"""
        async with LEXTestClient() as client:
            # Test conversation sequence
            messages = [
                "Hello, I'm testing the system",
                "Can you help me with coding?",
                "Generate an image of a robot",
                "Thank you for your help"
            ]
            
            for i, message in enumerate(messages):
                result = await client.post_lex({
                    "message": message,
                    "voice_mode": False,
                    "user_id": TestConfig.TEST_USER_ID,
                    "session_id": TestConfig.TEST_SESSION_ID
                })
                
                assert result["status_code"] == 200
                assert "response" in result["data"]
                assert len(result["data"]["response"]) > 0
    
    @pytest.mark.asyncio
    async def test_error_recovery(self):
        """Test system recovery from errors"""
        async with LEXTestClient() as client:
            # Send a request that might cause an error
            result = await client.post_lex({
                "message": "x" * 10000,  # Very long message
                "voice_mode": False
            })
            
            # System should handle gracefully
            assert result["status_code"] in [200, 400, 422]
            
            # Follow up with normal request
            result2 = await client.post_lex({
                "message": "Normal message after error",
                "voice_mode": False
            })
            
            assert result2["status_code"] == 200

# Test runner
async def run_all_tests():
    """Run all tests"""
    print("🧪 JAI MAHAKAAL! Running Comprehensive Test Suite 🧪")
    print("=" * 60)
    
    # Run pytest
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "--asyncio-mode=auto"
    ]
    
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
        
        # Run performance tests
        print("\n🚀 Running performance tests...")
        runner = PerformanceTestRunner()
        perf_results = await runner.run_load_test(
            concurrent_users=10,
            requests_per_user=5
        )
        
        print(f"📊 Performance Results:")
        print(f"   Success Rate: {perf_results['success_rate']:.1f}%")
        print(f"   Avg Response Time: {perf_results['avg_response_time']:.2f}s")
        print(f"   Requests/Second: {perf_results['requests_per_second']:.1f}")
        
        return True
    else:
        print("\n❌ Some tests failed!")
        return False

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
