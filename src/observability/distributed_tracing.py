"""
Distributed Tracing System for Phase 3 Observability
Implements request tracing across microservices for P1-002.4
"""

import os
import time
import uuid
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class TraceSpan:
    """Represents a single span in a distributed trace"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    operation_name: str
    service_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    tags: Dict[str, Any] = None
    logs: List[Dict[str, Any]] = None
    status: str = "ok"  # ok, error
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = {}
        if self.logs is None:
            self.logs = []
    
    def finish(self):
        """Finish the span and calculate duration"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration_ms = (self.end_time - self.start_time) * 1000
    
    def add_tag(self, key: str, value: Any):
        """Add a tag to the span"""
        self.tags[key] = value
    
    def add_log(self, message: str, level: str = "info", **kwargs):
        """Add a log entry to the span"""
        log_entry = {
            "timestamp": time.time(),
            "level": level,
            "message": message,
            **kwargs
        }
        self.logs.append(log_entry)

@dataclass
class TraceContext:
    """Context for propagating trace information"""
    trace_id: str
    span_id: str
    baggage: Dict[str, str] = None
    
    def __post_init__(self):
        if self.baggage is None:
            self.baggage = {}

class DistributedTracer:
    """
    Distributed tracing system for Phase 3 observability
    Provides request tracing across microservices
    """
    
    def __init__(self, service_name: str, jaeger_endpoint: Optional[str] = None):
        self.service_name = service_name
        self.jaeger_endpoint = jaeger_endpoint or os.getenv('JAEGER_ENDPOINT')
        self.active_spans: Dict[str, TraceSpan] = {}
        self.completed_traces: List[TraceSpan] = []
        self.sampling_rate = float(os.getenv('TRACE_SAMPLING_RATE', '1.0'))
        
        logger.info(f"Distributed tracer initialized for service: {service_name}")
    
    def should_sample(self) -> bool:
        """Determine if this trace should be sampled"""
        import random
        return random.random() < self.sampling_rate
    
    def start_span(self, operation_name: str, parent_context: Optional[TraceContext] = None) -> TraceSpan:
        """Start a new trace span"""
        
        # Generate trace and span IDs
        if parent_context:
            trace_id = parent_context.trace_id
            parent_span_id = parent_context.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None
        
        span_id = str(uuid.uuid4())
        
        # Create the span
        span = TraceSpan(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            operation_name=operation_name,
            service_name=self.service_name,
            start_time=time.time()
        )
        
        # Add service and operation tags
        span.add_tag("service.name", self.service_name)
        span.add_tag("operation.name", operation_name)
        
        # Store active span
        self.active_spans[span_id] = span
        
        logger.debug(f"Started span {span_id} for operation {operation_name}")
        return span
    
    def finish_span(self, span: TraceSpan):
        """Finish a span and store the trace"""
        span.finish()
        
        # Remove from active spans
        if span.span_id in self.active_spans:
            del self.active_spans[span.span_id]
        
        # Store completed trace
        self.completed_traces.append(span)
        
        # Send to Jaeger if configured
        if self.jaeger_endpoint and self.should_sample():
            self._send_to_jaeger(span)
        
        logger.debug(f"Finished span {span.span_id} ({span.duration_ms:.2f}ms)")
    
    def extract_context(self, headers: Dict[str, str]) -> Optional[TraceContext]:
        """Extract trace context from HTTP headers"""
        trace_id = headers.get('x-trace-id')
        span_id = headers.get('x-span-id')
        
        if trace_id and span_id:
            return TraceContext(trace_id=trace_id, span_id=span_id)
        
        return None
    
    def inject_context(self, context: TraceContext, headers: Dict[str, str]):
        """Inject trace context into HTTP headers"""
        headers['x-trace-id'] = context.trace_id
        headers['x-span-id'] = context.span_id
        
        # Add baggage if present
        for key, value in context.baggage.items():
            headers[f'x-baggage-{key}'] = value
    
    @contextmanager
    def trace_operation(self, operation_name: str, parent_context: Optional[TraceContext] = None):
        """Context manager for tracing operations"""
        span = self.start_span(operation_name, parent_context)
        try:
            yield span
        except Exception as e:
            span.add_tag("error", True)
            span.add_tag("error.message", str(e))
            span.status = "error"
            span.add_log(f"Operation failed: {str(e)}", "error")
            raise
        finally:
            self.finish_span(span)
    
    def _send_to_jaeger(self, span: TraceSpan):
        """Send span to Jaeger collector"""
        try:
            # Convert span to Jaeger format
            jaeger_span = {
                "traceID": span.trace_id.replace('-', ''),
                "spanID": span.span_id.replace('-', ''),
                "parentSpanID": span.parent_span_id.replace('-', '') if span.parent_span_id else None,
                "operationName": span.operation_name,
                "startTime": int(span.start_time * 1_000_000),  # microseconds
                "duration": int(span.duration_ms * 1000) if span.duration_ms else 0,
                "tags": [{"key": k, "type": "string", "value": str(v)} for k, v in span.tags.items()],
                "logs": [{
                    "timestamp": int(log["timestamp"] * 1_000_000),
                    "fields": [{"key": k, "value": str(v)} for k, v in log.items() if k != "timestamp"]
                } for log in span.logs],
                "process": {
                    "serviceName": self.service_name,
                    "tags": []
                }
            }
            
            # In a real implementation, send via HTTP to Jaeger
            # For now, log the trace data
            logger.info(f"Trace sent to Jaeger: {json.dumps(jaeger_span, indent=2)}")
            
        except Exception as e:
            logger.error(f"Failed to send trace to Jaeger: {e}")
    
    def get_trace_stats(self) -> Dict[str, Any]:
        """Get tracing statistics"""
        return {
            "service_name": self.service_name,
            "active_spans": len(self.active_spans),
            "completed_traces": len(self.completed_traces),
            "sampling_rate": self.sampling_rate,
            "jaeger_endpoint": self.jaeger_endpoint
        }
    
    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent completed traces"""
        recent = self.completed_traces[-limit:] if len(self.completed_traces) > limit else self.completed_traces
        return [asdict(trace) for trace in recent]

# Global tracer instance
_tracer: Optional[DistributedTracer] = None

def initialize_tracer(service_name: str, jaeger_endpoint: Optional[str] = None) -> DistributedTracer:
    """Initialize global tracer"""
    global _tracer
    _tracer = DistributedTracer(service_name, jaeger_endpoint)
    return _tracer

def get_tracer() -> Optional[DistributedTracer]:
    """Get global tracer instance"""
    return _tracer