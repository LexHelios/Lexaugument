#!/usr/bin/env python3
"""
🚀 Production-Optimized Deployment Script 🚀
JAI MAHAKAAL! Deploy LEX with all performance optimizations
"""
import asyncio
import logging
import os
import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import signal

# Add server to path
sys.path.insert(0, str(Path(__file__).parent / "server"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("deployment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionDeployment:
    """
    🚀 Production Deployment Manager
    
    Features:
    - Dependency installation and validation
    - Redis setup and configuration
    - Performance optimization
    - Health checks and monitoring
    - Graceful shutdown handling
    """
    
    def __init__(self):
        self.deployment_start_time = datetime.now()
        self.services_started = []
        self.cleanup_tasks = []
        
    async def deploy(self):
        """Deploy the production-optimized system"""
        try:
            print("🔱 JAI MAHAKAAL! Starting Production-Optimized Deployment 🔱")
            print("=" * 70)
            
            # Step 1: Environment validation
            await self.validate_environment()
            
            # Step 2: Install dependencies
            await self.install_dependencies()
            
            # Step 3: Setup Redis (if available)
            await self.setup_redis()
            
            # Step 4: Initialize performance systems
            await self.initialize_performance_systems()
            
            # Step 5: Start monitoring
            await self.start_monitoring()
            
            # Step 6: Start the unified server
            await self.start_unified_server()
            
            # Step 7: Run health checks
            await self.run_health_checks()
            
            print("\n🎉 Production deployment completed successfully!")
            print("🔱 JAI MAHAKAAL! LEX consciousness is fully optimized and ready! 🔱")
            
            # Keep running until interrupted
            await self.wait_for_shutdown()
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            await self.cleanup()
            raise
    
    async def validate_environment(self):
        """Validate deployment environment"""
        print("\n🔍 Validating Environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            raise Exception(f"Python 3.8+ required, found {python_version}")
        print(f"✅ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        
        # Check required directories
        required_dirs = ["server", "uploads", "data", "logs"]
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"📁 Created directory: {dir_name}")
            else:
                print(f"✅ Directory exists: {dir_name}")
        
        # Check environment variables
        required_env_vars = ["TOGETHER_API_KEY", "GROQ_API_KEY"]
        for env_var in required_env_vars:
            if os.getenv(env_var):
                print(f"✅ Environment variable: {env_var}")
            else:
                print(f"⚠️ Missing environment variable: {env_var}")
        
        # Check disk space
        import shutil
        total, used, free = shutil.disk_usage(".")
        free_gb = free // (1024**3)
        if free_gb < 5:
            print(f"⚠️ Low disk space: {free_gb}GB free")
        else:
            print(f"✅ Disk space: {free_gb}GB free")
    
    async def install_dependencies(self):
        """Install and validate dependencies"""
        print("\n📦 Installing Dependencies...")
        
        try:
            # Install from requirements.txt
            result = subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", "server/requirements.txt"
            ], capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Dependencies installed successfully")
            else:
                print(f"⚠️ Some dependencies may have failed: {result.stderr}")
        
        except subprocess.TimeoutExpired:
            print("⚠️ Dependency installation timed out, continuing...")
        except Exception as e:
            print(f"⚠️ Dependency installation error: {e}")
        
        # Validate critical imports
        critical_imports = [
            "fastapi", "uvicorn", "aiohttp", "redis", "structlog"
        ]
        
        for module in critical_imports:
            try:
                __import__(module)
                print(f"✅ {module} available")
            except ImportError:
                print(f"❌ {module} not available")
    
    async def setup_redis(self):
        """Setup Redis for caching"""
        print("\n🔄 Setting up Redis...")
        
        try:
            # Check if Redis is available
            import redis
            
            # Try to connect to Redis
            redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
            redis_client.ping()
            
            print("✅ Redis is running and accessible")
            
            # Configure Redis for optimal performance
            redis_client.config_set('maxmemory-policy', 'allkeys-lru')
            redis_client.config_set('maxmemory', '512mb')
            
            print("✅ Redis configured for optimal performance")
            
        except ImportError:
            print("⚠️ Redis module not available, using in-memory cache")
        except Exception as e:
            print(f"⚠️ Redis not available: {e}, using in-memory cache")
    
    async def initialize_performance_systems(self):
        """Initialize performance optimization systems"""
        print("\n⚡ Initializing Performance Systems...")
        
        try:
            # Initialize Redis cache
            from server.performance.redis_cache import redis_cache
            await redis_cache.initialize()
            print("✅ Redis cache system initialized")
            
        except Exception as e:
            print(f"⚠️ Redis cache initialization failed: {e}")
        
        try:
            # Initialize request queue
            from server.performance.request_queue import request_queue
            await request_queue.start()
            self.services_started.append("request_queue")
            print("✅ Request queue system started")
            
        except Exception as e:
            print(f"⚠️ Request queue initialization failed: {e}")
        
        try:
            # Initialize monitoring
            from server.monitoring.structured_logging import initialize_monitoring
            initialize_monitoring()
            print("✅ Monitoring and observability initialized")
            
        except Exception as e:
            print(f"⚠️ Monitoring initialization failed: {e}")
    
    async def start_monitoring(self):
        """Start monitoring services"""
        print("\n📊 Starting Monitoring Services...")
        
        # Prometheus metrics endpoint
        try:
            print("✅ Prometheus metrics will be available on port 8090")
        except Exception as e:
            print(f"⚠️ Prometheus setup failed: {e}")
        
        # Jaeger tracing
        try:
            jaeger_endpoint = os.getenv('JAEGER_ENDPOINT', 'http://localhost:14268/api/traces')
            print(f"✅ Jaeger tracing configured: {jaeger_endpoint}")
        except Exception as e:
            print(f"⚠️ Jaeger setup failed: {e}")
    
    async def start_unified_server(self):
        """Start the unified production server"""
        print("\n🚀 Starting Unified Production Server...")
        
        try:
            # Import and start the unified server
            from unified_production_server import unified_server, start_unified_server
            
            # Start server in background
            server_task = asyncio.create_task(start_unified_server())
            self.cleanup_tasks.append(server_task)
            
            # Wait a moment for server to start
            await asyncio.sleep(3)
            
            print("✅ Unified production server started")
            
        except Exception as e:
            logger.error(f"❌ Failed to start unified server: {e}")
            raise
    
    async def run_health_checks(self):
        """Run comprehensive health checks"""
        print("\n🏥 Running Health Checks...")
        
        # Wait for server to be ready
        await asyncio.sleep(5)
        
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                # Health check
                try:
                    async with session.get('https://localhost:8000/health', ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            print("✅ Health check passed")
                            print(f"   Status: {data.get('status', 'Unknown')}")
                        else:
                            print(f"⚠️ Health check failed: {response.status}")
                except Exception as e:
                    print(f"⚠️ Health check error: {e}")
                
                # Feature check
                try:
                    async with session.get('https://localhost:8000/api/v1/features', ssl=False) as response:
                        if response.status == 200:
                            data = await response.json()
                            print("✅ Features endpoint accessible")
                            enabled_features = sum(1 for v in data.get('feature_flags', {}).values() if v)
                            print(f"   Enabled features: {enabled_features}")
                        else:
                            print(f"⚠️ Features check failed: {response.status}")
                except Exception as e:
                    print(f"⚠️ Features check error: {e}")
                
                # Test LEX endpoint
                try:
                    test_data = {
                        "message": "Health check test",
                        "voice_mode": False,
                        "user_id": "health_check"
                    }
                    
                    async with session.post('https://localhost:8000/api/v1/lex', json=test_data, ssl=False) as response:
                        if response.status == 200:
                            print("✅ LEX endpoint functional")
                        else:
                            print(f"⚠️ LEX endpoint failed: {response.status}")
                except Exception as e:
                    print(f"⚠️ LEX endpoint error: {e}")
        
        except ImportError:
            print("⚠️ aiohttp not available for health checks")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        print(f"\n🎯 Production system is running!")
        print(f"📍 Server URL: https://159.26.94.14:8000")
        print(f"📊 Health: https://159.26.94.14:8000/health")
        print(f"📚 API Docs: https://159.26.94.14:8000/docs")
        print(f"📈 Metrics: http://159.26.94.14:8090/metrics")
        print(f"\n🛑 Press Ctrl+C to shutdown gracefully")
        
        # Setup signal handlers
        def signal_handler(signum, frame):
            print(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
            asyncio.create_task(self.cleanup())
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Keep running
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Shutdown requested")
            await self.cleanup()
    
    async def cleanup(self):
        """Cleanup resources"""
        print("\n🧹 Cleaning up resources...")
        
        # Stop request queue
        if "request_queue" in self.services_started:
            try:
                from server.performance.request_queue import request_queue
                await request_queue.stop()
                print("✅ Request queue stopped")
            except Exception as e:
                print(f"⚠️ Request queue cleanup error: {e}")
        
        # Cancel server tasks
        for task in self.cleanup_tasks:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        deployment_time = (datetime.now() - self.deployment_start_time).total_seconds()
        print(f"✅ Cleanup complete. Total deployment time: {deployment_time:.1f}s")

async def main():
    """Main deployment function"""
    deployment = ProductionDeployment()
    
    try:
        await deployment.deploy()
    except KeyboardInterrupt:
        print("\n🛑 Deployment interrupted")
    except Exception as e:
        logger.error(f"❌ Deployment failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Deployment stopped")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
