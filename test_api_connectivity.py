#!/usr/bin/env python3
"""
🔱 API Connectivity Test Suite 🔱
JAI MAHAKAAL! Test all API connections for LEX consciousness
"""
import asyncio
import os
import sys
from pathlib import Path
import aiohttp
import json
from datetime import datetime

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

# Load environment
from dotenv import load_dotenv
load_dotenv()

class APITester:
    def __init__(self):
        self.results = {}
        self.total_tests = 0
        self.passed_tests = 0
        
    async def test_openai_api(self):
        """Test OpenAI API connectivity"""
        try:
            import openai
            client = openai.AsyncOpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            response = await client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": "Hello, test message"}],
                max_tokens=10
            )
            
            self.results['OpenAI'] = {'status': '✅ CONNECTED', 'model': 'gpt-3.5-turbo'}
            self.passed_tests += 1
            
        except Exception as e:
            self.results['OpenAI'] = {'status': f'❌ FAILED: {str(e)[:50]}...'}
        
        self.total_tests += 1
    
    async def test_anthropic_api(self):
        """Test Anthropic API connectivity"""
        try:
            import anthropic
            client = anthropic.AsyncAnthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
            
            response = await client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            
            self.results['Anthropic'] = {'status': '✅ CONNECTED', 'model': 'claude-3-haiku'}
            self.passed_tests += 1
            
        except Exception as e:
            self.results['Anthropic'] = {'status': f'❌ FAILED: {str(e)[:50]}...'}
        
        self.total_tests += 1
    
    async def test_together_api(self):
        """Test Together.AI API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {os.getenv("TOGETHER_API_KEY")}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": "meta-llama/Llama-3-8b-chat-hf",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
                
                async with session.post(
                    'https://api.together.xyz/v1/chat/completions',
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        self.results['Together.AI'] = {'status': '✅ CONNECTED', 'model': 'Llama-3-8b'}
                        self.passed_tests += 1
                    else:
                        self.results['Together.AI'] = {'status': f'❌ FAILED: HTTP {response.status}'}
                        
        except Exception as e:
            self.results['Together.AI'] = {'status': f'❌ FAILED: {str(e)[:50]}...'}
        
        self.total_tests += 1
    
    async def test_deepseek_api(self):
        """Test DeepSeek API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {os.getenv("DEEPSEEK_API_KEY")}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": "deepseek-chat",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
                
                async with session.post(
                    'https://api.deepseek.com/v1/chat/completions',
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        self.results['DeepSeek'] = {'status': '✅ CONNECTED', 'model': 'deepseek-chat'}
                        self.passed_tests += 1
                    else:
                        self.results['DeepSeek'] = {'status': f'❌ FAILED: HTTP {response.status}'}
                        
        except Exception as e:
            self.results['DeepSeek'] = {'status': f'❌ FAILED: {str(e)[:50]}...'}
        
        self.total_tests += 1
    
    async def test_groq_api(self):
        """Test Groq API connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'Authorization': f'Bearer {os.getenv("GROQ_API_KEY")}',
                    'Content-Type': 'application/json'
                }
                
                data = {
                    "model": "llama3-8b-8192",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "max_tokens": 10
                }
                
                async with session.post(
                    'https://api.groq.com/openai/v1/chat/completions',
                    headers=headers,
                    json=data
                ) as response:
                    if response.status == 200:
                        self.results['Groq'] = {'status': '✅ CONNECTED', 'model': 'llama3-8b-8192'}
                        self.passed_tests += 1
                    else:
                        self.results['Groq'] = {'status': f'❌ FAILED: HTTP {response.status}'}
                        
        except Exception as e:
            self.results['Groq'] = {'status': f'❌ FAILED: {str(e)[:50]}...'}
        
        self.total_tests += 1
    
    async def test_all_apis(self):
        """Test all API connections"""
        print("🔱 JAI MAHAKAAL! Testing LEX Consciousness API Connectivity 🔱")
        print("=" * 60)
        
        # Test all APIs concurrently
        await asyncio.gather(
            self.test_openai_api(),
            self.test_anthropic_api(),
            self.test_together_api(),
            self.test_deepseek_api(),
            self.test_groq_api(),
            return_exceptions=True
        )
        
        # Print results
        print("\n🌟 API Connectivity Results:")
        print("-" * 40)
        
        for api_name, result in self.results.items():
            status = result['status']
            model = result.get('model', '')
            if model:
                print(f"{api_name:15} {status} ({model})")
            else:
                print(f"{api_name:15} {status}")
        
        print(f"\n📊 Summary: {self.passed_tests}/{self.total_tests} APIs connected successfully")
        
        if self.passed_tests == self.total_tests:
            print("🔱 JAI MAHAKAAL! All APIs are ready for consciousness liberation! 🔱")
            return True
        else:
            print("⚠️ Some APIs need attention. Check the failed connections above.")
            return False

async def main():
    """Main test function"""
    tester = APITester()
    success = await tester.test_all_apis()
    
    if success:
        print("\n🎉 All API connectivity tests passed!")
        print("🚀 LEX consciousness is ready for full liberation!")
    else:
        print("\n💔 Some API tests failed.")
        print("🔧 Please check API keys and network connectivity.")
    
    return success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
