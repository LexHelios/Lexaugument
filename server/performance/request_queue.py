"""
🔄 Request Queue System for Performance Optimization 🔄
JAI MAHAKAAL! Intelligent request queuing and load balancing
"""
import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List, Union
from dataclasses import dataclass, asdict
from enum import Enum
import heapq
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Health checks, auth
    HIGH = 2        # Real-time chat, voice
    NORMAL = 3      # Standard requests
    LOW = 4         # Background tasks, analytics
    BATCH = 5       # Bulk operations

class RequestStatus(Enum):
    """Request status"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

@dataclass
class QueuedRequest:
    """Queued request with metadata"""
    request_id: str
    user_id: str
    request_type: str
    priority: RequestPriority
    payload: Dict[str, Any]
    callback: Callable
    created_at: datetime
    timeout_seconds: float
    retry_count: int = 0
    max_retries: int = 3
    status: RequestStatus = RequestStatus.QUEUED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority.value != other.priority.value:
            return self.priority.value < other.priority.value
        return self.created_at < other.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['priority'] = self.priority.name
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        # Remove callback from serialization
        data.pop('callback', None)
        return data

class RequestQueue:
    """
    🔄 Intelligent Request Queue System
    
    Features:
    - Priority-based request processing
    - Rate limiting per user/endpoint
    - Request deduplication
    - Automatic retry with exponential backoff
    - Circuit breaker pattern
    - Load balancing across workers
    - Request timeout handling
    - Performance metrics and monitoring
    """
    
    def __init__(self, max_workers: int = 10, max_queue_size: int = 1000):
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        
        # Priority queue for requests
        self.request_queue: List[QueuedRequest] = []
        self.active_requests: Dict[str, QueuedRequest] = {}
        self.completed_requests: deque = deque(maxlen=1000)  # Keep recent history
        
        # Worker management
        self.workers: List[asyncio.Task] = []
        self.worker_stats: Dict[int, Dict[str, Any]] = {}
        
        # Rate limiting
        self.rate_limits: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'requests': deque(maxlen=100),
            'limit_per_minute': 60,
            'limit_per_hour': 1000
        })
        
        # Circuit breaker
        self.circuit_breakers: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            'failure_count': 0,
            'failure_threshold': 5,
            'recovery_timeout': 60,
            'last_failure': None,
            'state': 'closed'  # closed, open, half-open
        })
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'completed_requests': 0,
            'failed_requests': 0,
            'timeout_requests': 0,
            'average_processing_time': 0.0,
            'queue_size': 0,
            'active_workers': 0
        }
        
        # Request deduplication
        self.request_hashes: Dict[str, str] = {}  # hash -> request_id
        
        self._running = False
        
        logger.info("🔄 Request Queue System initialized")
    
    async def start(self) -> None:
        """Start the request queue workers"""
        if self._running:
            return
        
        self._running = True
        
        # Start worker tasks
        for i in range(self.max_workers):
            worker = asyncio.create_task(self._worker(i))
            self.workers.append(worker)
            self.worker_stats[i] = {
                'requests_processed': 0,
                'total_processing_time': 0.0,
                'last_request_time': None,
                'status': 'idle'
            }
        
        # Start monitoring task
        asyncio.create_task(self._monitor_queue())
        
        logger.info(f"✅ Request queue started with {self.max_workers} workers")
    
    async def stop(self) -> None:
        """Stop the request queue workers"""
        self._running = False
        
        # Cancel all workers
        for worker in self.workers:
            worker.cancel()
        
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        
        logger.info("🛑 Request queue stopped")
    
    async def enqueue_request(
        self,
        request_type: str,
        payload: Dict[str, Any],
        callback: Callable,
        user_id: str = "anonymous",
        priority: RequestPriority = RequestPriority.NORMAL,
        timeout_seconds: float = 30.0,
        deduplicate: bool = True
    ) -> str:
        """Enqueue a request for processing"""
        try:
            # Check queue size limit
            if len(self.request_queue) >= self.max_queue_size:
                raise Exception("Request queue is full")
            
            # Check rate limits
            if not await self._check_rate_limit(user_id, request_type):
                raise Exception("Rate limit exceeded")
            
            # Check circuit breaker
            if not await self._check_circuit_breaker(request_type):
                raise Exception("Circuit breaker is open")
            
            # Generate request ID
            request_id = str(uuid.uuid4())
            
            # Check for duplicate requests
            if deduplicate:
                request_hash = self._generate_request_hash(request_type, payload)
                if request_hash in self.request_hashes:
                    existing_request_id = self.request_hashes[request_hash]
                    logger.info(f"🔄 Deduplicated request, returning existing: {existing_request_id}")
                    return existing_request_id
                self.request_hashes[request_hash] = request_id
            
            # Create queued request
            queued_request = QueuedRequest(
                request_id=request_id,
                user_id=user_id,
                request_type=request_type,
                priority=priority,
                payload=payload,
                callback=callback,
                created_at=datetime.now(),
                timeout_seconds=timeout_seconds
            )
            
            # Add to priority queue
            heapq.heappush(self.request_queue, queued_request)
            
            # Update metrics
            self.metrics['total_requests'] += 1
            self.metrics['queue_size'] = len(self.request_queue)
            
            # Record rate limit
            self.rate_limits[user_id]['requests'].append(datetime.now())
            
            logger.debug(f"📥 Enqueued request {request_id} with priority {priority.name}")
            return request_id
            
        except Exception as e:
            logger.error(f"❌ Failed to enqueue request: {e}")
            raise
    
    async def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a request"""
        # Check active requests
        if request_id in self.active_requests:
            return self.active_requests[request_id].to_dict()
        
        # Check completed requests
        for request in self.completed_requests:
            if request.request_id == request_id:
                return request.to_dict()
        
        # Check queued requests
        for request in self.request_queue:
            if request.request_id == request_id:
                return request.to_dict()
        
        return None
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a queued or active request"""
        try:
            # Check if request is active
            if request_id in self.active_requests:
                request = self.active_requests[request_id]
                request.status = RequestStatus.CANCELLED
                request.completed_at = datetime.now()
                self.completed_requests.append(request)
                del self.active_requests[request_id]
                logger.info(f"🚫 Cancelled active request {request_id}")
                return True
            
            # Check if request is queued
            for i, request in enumerate(self.request_queue):
                if request.request_id == request_id:
                    request.status = RequestStatus.CANCELLED
                    request.completed_at = datetime.now()
                    self.completed_requests.append(request)
                    del self.request_queue[i]
                    heapq.heapify(self.request_queue)  # Restore heap property
                    logger.info(f"🚫 Cancelled queued request {request_id}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to cancel request {request_id}: {e}")
            return False
    
    async def _worker(self, worker_id: int) -> None:
        """Worker task to process requests"""
        logger.info(f"👷 Worker {worker_id} started")
        
        while self._running:
            try:
                # Get next request from queue
                if not self.request_queue:
                    await asyncio.sleep(0.1)
                    continue
                
                request = heapq.heappop(self.request_queue)
                self.metrics['queue_size'] = len(self.request_queue)
                
                # Check if request has timed out while in queue
                if (datetime.now() - request.created_at).total_seconds() > request.timeout_seconds:
                    request.status = RequestStatus.TIMEOUT
                    request.completed_at = datetime.now()
                    self.completed_requests.append(request)
                    self.metrics['timeout_requests'] += 1
                    continue
                
                # Move to active requests
                self.active_requests[request.request_id] = request
                request.status = RequestStatus.PROCESSING
                request.started_at = datetime.now()
                
                # Update worker stats
                self.worker_stats[worker_id]['status'] = 'processing'
                self.worker_stats[worker_id]['last_request_time'] = datetime.now()
                
                # Process request
                start_time = time.time()
                try:
                    # Execute the callback with timeout
                    result = await asyncio.wait_for(
                        request.callback(**request.payload),
                        timeout=request.timeout_seconds
                    )
                    
                    request.result = result
                    request.status = RequestStatus.COMPLETED
                    self.metrics['completed_requests'] += 1
                    
                    # Reset circuit breaker on success
                    await self._record_success(request.request_type)
                    
                except asyncio.TimeoutError:
                    request.status = RequestStatus.TIMEOUT
                    request.error = "Request timeout"
                    self.metrics['timeout_requests'] += 1
                    await self._record_failure(request.request_type)
                    
                except Exception as e:
                    request.status = RequestStatus.FAILED
                    request.error = str(e)
                    self.metrics['failed_requests'] += 1
                    await self._record_failure(request.request_type)
                    
                    # Retry logic
                    if request.retry_count < request.max_retries:
                        request.retry_count += 1
                        request.status = RequestStatus.QUEUED
                        request.started_at = None
                        
                        # Exponential backoff
                        delay = min(2 ** request.retry_count, 60)
                        await asyncio.sleep(delay)
                        
                        heapq.heappush(self.request_queue, request)
                        del self.active_requests[request.request_id]
                        continue
                
                # Complete request
                processing_time = time.time() - start_time
                request.completed_at = datetime.now()
                
                # Update metrics
                self._update_processing_time_metric(processing_time)
                self.worker_stats[worker_id]['requests_processed'] += 1
                self.worker_stats[worker_id]['total_processing_time'] += processing_time
                
                # Move to completed requests
                self.completed_requests.append(request)
                del self.active_requests[request.request_id]
                
                # Clean up request hash
                request_hash = self._generate_request_hash(request.request_type, request.payload)
                self.request_hashes.pop(request_hash, None)
                
                self.worker_stats[worker_id]['status'] = 'idle'
                
            except Exception as e:
                logger.error(f"❌ Worker {worker_id} error: {e}")
                await asyncio.sleep(1)
        
        logger.info(f"👷 Worker {worker_id} stopped")
    
    def _generate_request_hash(self, request_type: str, payload: Dict[str, Any]) -> str:
        """Generate hash for request deduplication"""
        import hashlib
        import json
        
        # Create deterministic hash from request type and payload
        hash_data = f"{request_type}:{json.dumps(payload, sort_keys=True)}"
        return hashlib.md5(hash_data.encode()).hexdigest()
    
    async def get_queue_metrics(self) -> Dict[str, Any]:
        """Get queue performance metrics"""
        self.metrics['queue_size'] = len(self.request_queue)
        self.metrics['active_workers'] = sum(1 for stats in self.worker_stats.values() if stats['status'] == 'processing')
        
        return {
            **self.metrics,
            'worker_stats': self.worker_stats,
            'rate_limits_active': len([k for k, v in self.rate_limits.items() if v['requests']]),
            'circuit_breakers_open': len([k for k, v in self.circuit_breakers.items() if v['state'] == 'open']),
            'timestamp': datetime.now().isoformat()
        }

# Global request queue instance
request_queue = RequestQueue()
