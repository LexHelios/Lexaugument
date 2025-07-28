"""
📊 Structured Logging and OpenTelemetry Integration 📊
JAI MAHAKAAL! Production-grade observability and monitoring
"""
import logging
import json
import time
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional, List
from contextlib import contextmanager
import structlog
from opentelemetry import trace, metrics
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.exporter.prometheus import PrometheusMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.aiohttp_client import AioHttpClientInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
import uuid

class StructuredLogger:
    """
    📊 Structured Logging System
    
    Features:
    - JSON structured logging
    - Request correlation IDs
    - Performance metrics
    - Error tracking
    - User activity logging
    - Security event logging
    """
    
    def __init__(self):
        self.service_name = "lex-consciousness"
        self.service_version = "3.0.0"
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Configure structlog
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        # Setup standard logging
        logging.basicConfig(
            format="%(message)s",
            stream=sys.stdout,
            level=logging.INFO,
        )
        
        self.logger = structlog.get_logger()
        
    def get_logger(self, name: str = None) -> structlog.BoundLogger:
        """Get a structured logger instance"""
        if name:
            return structlog.get_logger(name)
        return self.logger
    
    def log_request(
        self,
        request_id: str,
        user_id: str,
        endpoint: str,
        method: str,
        status_code: int,
        processing_time: float,
        **kwargs
    ):
        """Log HTTP request"""
        self.logger.info(
            "http_request",
            request_id=request_id,
            user_id=user_id,
            endpoint=endpoint,
            method=method,
            status_code=status_code,
            processing_time_ms=round(processing_time * 1000, 2),
            **kwargs
        )
    
    def log_model_request(
        self,
        request_id: str,
        model_name: str,
        prompt_length: int,
        response_length: int,
        processing_time: float,
        cache_hit: bool = False,
        **kwargs
    ):
        """Log AI model request"""
        self.logger.info(
            "model_request",
            request_id=request_id,
            model_name=model_name,
            prompt_length=prompt_length,
            response_length=response_length,
            processing_time_ms=round(processing_time * 1000, 2),
            cache_hit=cache_hit,
            **kwargs
        )
    
    def log_error(
        self,
        error: Exception,
        request_id: str = None,
        user_id: str = None,
        context: Dict[str, Any] = None,
        **kwargs
    ):
        """Log error with context"""
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            request_id=request_id,
            user_id=user_id,
            context=context or {},
            **kwargs
        )
    
    def log_security_event(
        self,
        event_type: str,
        user_id: str,
        ip_address: str,
        details: Dict[str, Any] = None,
        **kwargs
    ):
        """Log security-related events"""
        self.logger.warning(
            "security_event",
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            details=details or {},
            **kwargs
        )
    
    def log_performance_metric(
        self,
        metric_name: str,
        value: float,
        unit: str,
        tags: Dict[str, str] = None,
        **kwargs
    ):
        """Log performance metrics"""
        self.logger.info(
            "performance_metric",
            metric_name=metric_name,
            value=value,
            unit=unit,
            tags=tags or {},
            **kwargs
        )

class OpenTelemetrySetup:
    """
    🔍 OpenTelemetry Tracing and Metrics Setup
    
    Features:
    - Distributed tracing
    - Custom metrics
    - Jaeger integration
    - Prometheus metrics
    - Automatic instrumentation
    """
    
    def __init__(self):
        self.service_name = "lex-consciousness"
        self.service_version = "3.0.0"
        self.environment = os.getenv('ENVIRONMENT', 'development')
        
        # Jaeger configuration
        self.jaeger_endpoint = os.getenv('JAEGER_ENDPOINT', 'http://localhost:14268/api/traces')
        
        # Prometheus configuration
        self.prometheus_port = int(os.getenv('PROMETHEUS_PORT', '8090'))
        
        self.tracer = None
        self.meter = None
        
    def setup_tracing(self):
        """Setup distributed tracing"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment
            })
            
            # Setup tracer provider
            tracer_provider = TracerProvider(resource=resource)
            trace.set_tracer_provider(tracer_provider)
            
            # Setup Jaeger exporter
            jaeger_exporter = JaegerExporter(
                endpoint=self.jaeger_endpoint,
            )
            
            # Add span processor
            span_processor = BatchSpanProcessor(jaeger_exporter)
            tracer_provider.add_span_processor(span_processor)
            
            # Get tracer
            self.tracer = trace.get_tracer(__name__)
            
            # Auto-instrument FastAPI
            FastAPIInstrumentor.instrument()
            AioHttpClientInstrumentor.instrument()
            RedisInstrumentor.instrument()
            
            print("✅ OpenTelemetry tracing configured")
            
        except Exception as e:
            print(f"⚠️ OpenTelemetry tracing setup failed: {e}")
    
    def setup_metrics(self):
        """Setup metrics collection"""
        try:
            # Create resource
            resource = Resource.create({
                "service.name": self.service_name,
                "service.version": self.service_version,
                "deployment.environment": self.environment
            })
            
            # Setup Prometheus metric reader
            prometheus_reader = PrometheusMetricReader(port=self.prometheus_port)
            
            # Setup meter provider
            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[prometheus_reader]
            )
            metrics.set_meter_provider(meter_provider)
            
            # Get meter
            self.meter = metrics.get_meter(__name__)
            
            print(f"✅ Prometheus metrics available on port {self.prometheus_port}")
            
        except Exception as e:
            print(f"⚠️ Metrics setup failed: {e}")
    
    def get_tracer(self):
        """Get tracer instance"""
        return self.tracer
    
    def get_meter(self):
        """Get meter instance"""
        return self.meter

class MetricsCollector:
    """
    📈 Custom Metrics Collector
    
    Collects business and technical metrics
    """
    
    def __init__(self, meter):
        self.meter = meter
        
        # Create metrics instruments
        self.request_counter = meter.create_counter(
            name="lex_requests_total",
            description="Total number of requests",
            unit="1"
        )
        
        self.request_duration = meter.create_histogram(
            name="lex_request_duration_seconds",
            description="Request duration in seconds",
            unit="s"
        )
        
        self.model_requests = meter.create_counter(
            name="lex_model_requests_total",
            description="Total number of model requests",
            unit="1"
        )
        
        self.cache_hits = meter.create_counter(
            name="lex_cache_hits_total",
            description="Total number of cache hits",
            unit="1"
        )
        
        self.queue_size = meter.create_up_down_counter(
            name="lex_queue_size",
            description="Current queue size",
            unit="1"
        )
        
        self.active_users = meter.create_up_down_counter(
            name="lex_active_users",
            description="Number of active users",
            unit="1"
        )
        
        self.error_counter = meter.create_counter(
            name="lex_errors_total",
            description="Total number of errors",
            unit="1"
        )
    
    def record_request(self, endpoint: str, method: str, status_code: int, duration: float):
        """Record HTTP request metrics"""
        self.request_counter.add(1, {
            "endpoint": endpoint,
            "method": method,
            "status_code": str(status_code)
        })
        
        self.request_duration.record(duration, {
            "endpoint": endpoint,
            "method": method
        })
    
    def record_model_request(self, model_name: str, success: bool, cache_hit: bool = False):
        """Record model request metrics"""
        self.model_requests.add(1, {
            "model": model_name,
            "success": str(success)
        })
        
        if cache_hit:
            self.cache_hits.add(1, {"model": model_name})
    
    def record_queue_size(self, size: int):
        """Record current queue size"""
        self.queue_size.add(size)
    
    def record_active_users(self, count: int):
        """Record active user count"""
        self.active_users.add(count)
    
    def record_error(self, error_type: str, endpoint: str = None):
        """Record error occurrence"""
        attributes = {"error_type": error_type}
        if endpoint:
            attributes["endpoint"] = endpoint
        
        self.error_counter.add(1, attributes)

class RequestTracker:
    """
    🔍 Request Tracking and Correlation
    
    Tracks requests across services with correlation IDs
    """
    
    def __init__(self, logger: StructuredLogger, tracer):
        self.logger = logger
        self.tracer = tracer
        self.active_requests: Dict[str, Dict[str, Any]] = {}
    
    @contextmanager
    def track_request(self, request_id: str = None, operation_name: str = "request"):
        """Context manager for tracking requests"""
        if not request_id:
            request_id = str(uuid.uuid4())
        
        start_time = time.time()
        
        # Start span
        with self.tracer.start_as_current_span(operation_name) as span:
            span.set_attribute("request.id", request_id)
            
            # Track request
            self.active_requests[request_id] = {
                "start_time": start_time,
                "operation": operation_name,
                "span": span
            }
            
            try:
                yield request_id
            except Exception as e:
                # Record error in span
                span.record_exception(e)
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                raise
            finally:
                # Clean up
                processing_time = time.time() - start_time
                span.set_attribute("request.duration", processing_time)
                
                if request_id in self.active_requests:
                    del self.active_requests[request_id]
    
    def get_active_requests(self) -> Dict[str, Dict[str, Any]]:
        """Get currently active requests"""
        return {
            req_id: {
                "operation": data["operation"],
                "duration": time.time() - data["start_time"]
            }
            for req_id, data in self.active_requests.items()
        }

# Global instances
structured_logger = StructuredLogger()
otel_setup = OpenTelemetrySetup()
metrics_collector = None
request_tracker = None

def initialize_monitoring():
    """Initialize all monitoring components"""
    global metrics_collector, request_tracker
    
    # Setup OpenTelemetry
    otel_setup.setup_tracing()
    otel_setup.setup_metrics()
    
    # Initialize metrics collector
    if otel_setup.get_meter():
        metrics_collector = MetricsCollector(otel_setup.get_meter())
    
    # Initialize request tracker
    if otel_setup.get_tracer():
        request_tracker = RequestTracker(structured_logger, otel_setup.get_tracer())
    
    print("✅ Monitoring system initialized")

# Auto-initialize if imported
if os.getenv('ENABLE_MONITORING', 'true').lower() == 'true':
    try:
        initialize_monitoring()
    except Exception as e:
        print(f"⚠️ Monitoring initialization failed: {e}")
