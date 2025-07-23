"""
Load Balancer and Service Registry
Handles service discovery, load balancing, and auto-scaling
"""

import os
import time
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from contextlib import asynccontextmanager
import random

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import aiohttp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LB_CONFIG = {
    'name': 'load-balancer',
    'version': '1.0.0',
    'port': int(os.getenv('LB_PORT', 8000)),
    'health_check_interval_seconds': int(os.getenv('HEALTH_CHECK_INTERVAL', 30)),
    'service_timeout_seconds': int(os.getenv('SERVICE_TIMEOUT', 120)),
    'max_retry_attempts': int(os.getenv('MAX_RETRY_ATTEMPTS', 3)),
    'load_balancing_strategy': os.getenv('LB_STRATEGY', 'round_robin'),  # round_robin, least_connections, weighted
    'auto_scaling_enabled': os.getenv('AUTO_SCALING_ENABLED', 'true').lower() == 'true',
    'scale_up_threshold': float(os.getenv('SCALE_UP_THRESHOLD', 0.8)),  # 80% utilization
    'scale_down_threshold': float(os.getenv('SCALE_DOWN_THRESHOLD', 0.3)),  # 30% utilization
    'min_instances': int(os.getenv('MIN_INSTANCES', 1)),
    'max_instances': int(os.getenv('MAX_INSTANCES', 10))
}

# Data models
@dataclass
class ServiceInstance:
    """Represents a service instance"""
    id: str
    name: str
    host: str
    port: int
    health_endpoint: str
    capabilities: List[str]
    status: str = "healthy"  # healthy, unhealthy, scaling
    last_health_check: Optional[datetime] = None
    response_time_ms: float = 0.0
    active_connections: int = 0
    total_requests: int = 0
    error_count: int = 0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    load_score: float = 0.0

@dataclass
class LoadBalancingMetrics:
    """Load balancing metrics"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time_ms: float = 0.0
    current_load: float = 0.0
    active_instances: int = 0

# Pydantic models for API
class ServiceRegistration(BaseModel):
    name: str
    host: str = Field(default="localhost")
    port: int
    health_endpoint: str
    capabilities: List[str] = Field(default_factory=list)
    max_concurrent_sessions: int = Field(default=100)

class ProxyRequest(BaseModel):
    method: str
    path: str
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Dict[str, Any]] = None
    target_service: Optional[str] = None

class HealthStatus(BaseModel):
    status: str
    timestamp: datetime
    active_services: int
    total_requests: int
    avg_response_time_ms: float
    load_balancing_metrics: Dict[str, Any]

# Global state
service_registry: Dict[str, List[ServiceInstance]] = {}
lb_metrics = LoadBalancingMetrics()
request_counter = 0

class LoadBalancer:
    """Intelligent load balancer with auto-scaling"""
    
    def __init__(self):
        self.current_request_index = {}  # For round-robin
        
    def select_service_instance(self, service_name: str) -> Optional[ServiceInstance]:
        """Select the best service instance based on configured strategy"""
        if service_name not in service_registry:
            return None
        
        healthy_instances = [
            instance for instance in service_registry[service_name]
            if instance.status == "healthy"
        ]
        
        if not healthy_instances:
            return None
        
        strategy = LB_CONFIG['load_balancing_strategy']
        
        if strategy == 'round_robin':
            return self._round_robin_selection(service_name, healthy_instances)
        elif strategy == 'least_connections':
            return self._least_connections_selection(healthy_instances)
        elif strategy == 'weighted':
            return self._weighted_selection(healthy_instances)
        else:
            # Default to round robin
            return self._round_robin_selection(service_name, healthy_instances)
    
    def _round_robin_selection(self, service_name: str, instances: List[ServiceInstance]) -> ServiceInstance:
        """Round-robin load balancing"""
        if service_name not in self.current_request_index:
            self.current_request_index[service_name] = 0
        
        index = self.current_request_index[service_name] % len(instances)
        self.current_request_index[service_name] += 1
        
        return instances[index]
    
    def _least_connections_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Select instance with least active connections"""
        return min(instances, key=lambda x: x.active_connections)
    
    def _weighted_selection(self, instances: List[ServiceInstance]) -> ServiceInstance:
        """Weighted selection based on performance metrics"""
        # Calculate weights based on inverse load score
        weights = []
        for instance in instances:
            # Lower load score = higher weight
            weight = max(0.1, 1.0 - instance.load_score)
            weights.append(weight)
        
        # Weighted random selection
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(instances)
        
        normalized_weights = [w / total_weight for w in weights]
        return random.choices(instances, weights=normalized_weights)[0]
    
    async def proxy_request(self, target_instance: ServiceInstance, method: str, 
                          path: str, headers: Dict[str, str], body: Optional[Dict] = None) -> Dict[str, Any]:
        """Proxy request to target service instance"""
        url = f"http://{target_instance.host}:{target_instance.port}{path}"
        
        # Update connection count
        target_instance.active_connections += 1
        target_instance.total_requests += 1
        
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == 'GET':
                    async with session.get(url, headers=headers) as response:
                        result = await response.json()
                        status_code = response.status
                elif method.upper() == 'POST':
                    async with session.post(url, headers=headers, json=body) as response:
                        result = await response.json()
                        status_code = response.status
                else:
                    raise HTTPException(status_code=405, detail=f"Method {method} not supported")
                
                # Update metrics
                response_time = (time.time() - start_time) * 1000
                target_instance.response_time_ms = (
                    target_instance.response_time_ms * 0.9 + response_time * 0.1
                )  # Exponential moving average
                
                if status_code >= 400:
                    target_instance.error_count += 1
                
                return {
                    'status_code': status_code,
                    'data': result,
                    'response_time_ms': response_time,
                    'instance_id': target_instance.id
                }
                
        except Exception as e:
            target_instance.error_count += 1
            logger.error(f"Proxy request failed: {e}")
            raise HTTPException(status_code=503, detail=f"Service unavailable: {str(e)}")
        finally:
            target_instance.active_connections -= 1

# Initialize load balancer
load_balancer = LoadBalancer()

# Service discovery and health checking
async def health_check_service(instance: ServiceInstance) -> bool:
    """Check health of a service instance"""
    try:
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.get(
                instance.health_endpoint,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_time = (time.time() - start_time) * 1000
                
                if response.status == 200:
                    health_data = await response.json()
                    
                    # Update instance metrics from health check
                    instance.response_time_ms = response_time
                    instance.last_health_check = datetime.now()
                    
                    # Extract metrics if available
                    if 'memory_usage_mb' in health_data:
                        instance.memory_usage = health_data['memory_usage_mb']
                    if 'active_sessions' in health_data:
                        instance.active_connections = health_data['active_sessions']
                    
                    # Calculate load score (0.0 = no load, 1.0 = full load)
                    instance.load_score = min(1.0, (
                        instance.active_connections / 100 * 0.4 +  # Connection load
                        instance.memory_usage / 512 * 0.3 +        # Memory load
                        (instance.error_count / max(instance.total_requests, 1)) * 0.3  # Error rate
                    ))
                    
                    instance.status = "healthy"
                    return True
                else:
                    instance.status = "unhealthy"
                    return False
                    
    except Exception as e:
        logger.warning(f"Health check failed for {instance.id}: {e}")
        instance.status = "unhealthy"
        return False

async def periodic_health_checks():
    """Periodically check health of all registered services"""
    while True:
        try:
            for service_name, instances in service_registry.items():
                for instance in instances:
                    await health_check_service(instance)
            
            # Check for auto-scaling opportunities
            if LB_CONFIG['auto_scaling_enabled']:
                await check_auto_scaling()
            
            await asyncio.sleep(LB_CONFIG['health_check_interval_seconds'])
            
        except Exception as e:
            logger.error(f"Health check task error: {e}")
            await asyncio.sleep(30)

async def check_auto_scaling():
    """Check if auto-scaling is needed"""
    for service_name, instances in service_registry.items():
        healthy_instances = [i for i in instances if i.status == "healthy"]
        
        if not healthy_instances:
            continue
        
        # Calculate average load across instances
        avg_load = sum(i.load_score for i in healthy_instances) / len(healthy_instances)
        
        current_count = len(healthy_instances)
        
        # Scale up if load is high and we're below max instances
        if (avg_load > LB_CONFIG['scale_up_threshold'] and 
            current_count < LB_CONFIG['max_instances']):
            logger.info(f"Auto-scaling up {service_name}: load={avg_load:.2f}, instances={current_count}")
            # In a real implementation, this would trigger container/VM creation
            await simulate_scale_up(service_name)
        
        # Scale down if load is low and we're above min instances
        elif (avg_load < LB_CONFIG['scale_down_threshold'] and 
              current_count > LB_CONFIG['min_instances']):
            logger.info(f"Auto-scaling down {service_name}: load={avg_load:.2f}, instances={current_count}")
            # In a real implementation, this would terminate instances
            await simulate_scale_down(service_name)

async def simulate_scale_up(service_name: str):
    """Simulate scaling up a service (placeholder for actual scaling logic)"""
    # In a real implementation, this would:
    # 1. Create new container/VM instance
    # 2. Wait for it to be ready
    # 3. Add it to the service registry
    logger.info(f"Simulating scale-up for {service_name}")

async def simulate_scale_down(service_name: str):
    """Simulate scaling down a service (placeholder for actual scaling logic)"""
    # In a real implementation, this would:
    # 1. Select instance to terminate
    # 2. Drain connections gracefully
    # 3. Remove from registry
    # 4. Terminate container/VM
    logger.info(f"Simulating scale-down for {service_name}")

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    # Startup
    logger.info(f"Starting {LB_CONFIG['name']} v{LB_CONFIG['version']}")
    
    # Start health checking task
    health_task = asyncio.create_task(periodic_health_checks())
    
    yield
    
    # Shutdown
    health_task.cancel()
    logger.info("Shutting down load balancer")

app = FastAPI(
    title="Deep Tree Echo Load Balancer",
    description="Intelligent load balancer with service discovery and auto-scaling",
    version=LB_CONFIG['version'],
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Endpoints

@app.post("/api/services/register")
async def register_service(registration: ServiceRegistration):
    """Register a new service instance"""
    global request_counter
    
    instance_id = f"{registration.name}-{int(time.time())}-{request_counter}"
    request_counter += 1
    
    instance = ServiceInstance(
        id=instance_id,
        name=registration.name,
        host=registration.host,
        port=registration.port,
        health_endpoint=registration.health_endpoint,
        capabilities=registration.capabilities
    )
    
    # Add to registry
    if registration.name not in service_registry:
        service_registry[registration.name] = []
    
    service_registry[registration.name].append(instance)
    
    logger.info(f"Registered service instance: {instance_id}")
    
    return {
        'instance_id': instance_id,
        'status': 'registered',
        'timestamp': datetime.now()
    }

@app.post("/api/proxy/{service_name}")
async def proxy_to_service(service_name: str, proxy_request: ProxyRequest):
    """Proxy request to appropriate service instance"""
    global lb_metrics
    
    # Select target instance
    target_instance = load_balancer.select_service_instance(service_name)
    if not target_instance:
        raise HTTPException(
            status_code=503, 
            detail=f"No healthy instances available for service: {service_name}"
        )
    
    # Proxy the request
    try:
        result = await load_balancer.proxy_request(
            target_instance,
            proxy_request.method,
            proxy_request.path,
            proxy_request.headers,
            proxy_request.body
        )
        
        # Update metrics
        lb_metrics.total_requests += 1
        lb_metrics.successful_requests += 1
        lb_metrics.avg_response_time_ms = (
            lb_metrics.avg_response_time_ms * 0.9 + 
            result['response_time_ms'] * 0.1
        )
        
        return result
        
    except HTTPException as e:
        lb_metrics.total_requests += 1
        lb_metrics.failed_requests += 1
        raise e

@app.get("/api/services")
async def list_services():
    """List all registered services and their instances"""
    services = {}
    
    for service_name, instances in service_registry.items():
        services[service_name] = {
            'instances': [asdict(instance) for instance in instances],
            'healthy_count': len([i for i in instances if i.status == "healthy"]),
            'total_count': len(instances)
        }
    
    return {
        'services': services,
        'load_balancing_strategy': LB_CONFIG['load_balancing_strategy'],
        'auto_scaling_enabled': LB_CONFIG['auto_scaling_enabled']
    }

@app.get("/api/services/{service_name}/instances")
async def get_service_instances(service_name: str):
    """Get instances for a specific service"""
    if service_name not in service_registry:
        raise HTTPException(status_code=404, detail="Service not found")
    
    instances = service_registry[service_name]
    
    return {
        'service_name': service_name,
        'instances': [asdict(instance) for instance in instances],
        'healthy_count': len([i for i in instances if i.status == "healthy"]),
        'total_count': len(instances),
        'avg_load_score': sum(i.load_score for i in instances) / len(instances) if instances else 0
    }

@app.delete("/api/services/{service_name}/instances/{instance_id}")
async def unregister_service_instance(service_name: str, instance_id: str):
    """Unregister a service instance"""
    if service_name not in service_registry:
        raise HTTPException(status_code=404, detail="Service not found")
    
    instances = service_registry[service_name]
    instance_to_remove = None
    
    for instance in instances:
        if instance.id == instance_id:
            instance_to_remove = instance
            break
    
    if not instance_to_remove:
        raise HTTPException(status_code=404, detail="Instance not found")
    
    instances.remove(instance_to_remove)
    logger.info(f"Unregistered service instance: {instance_id}")
    
    return {
        'instance_id': instance_id,
        'status': 'unregistered',
        'timestamp': datetime.now()
    }

@app.get("/health", response_model=HealthStatus)
async def health_check():
    """Load balancer health check"""
    active_services = sum(
        len([i for i in instances if i.status == "healthy"])
        for instances in service_registry.values()
    )
    
    return HealthStatus(
        status="healthy",
        timestamp=datetime.now(),
        active_services=active_services,
        total_requests=lb_metrics.total_requests,
        avg_response_time_ms=lb_metrics.avg_response_time_ms,
        load_balancing_metrics={
            'successful_requests': lb_metrics.successful_requests,
            'failed_requests': lb_metrics.failed_requests,
            'success_rate': (
                lb_metrics.successful_requests / lb_metrics.total_requests
                if lb_metrics.total_requests > 0 else 1.0
            ),
            'strategy': LB_CONFIG['load_balancing_strategy'],
            'auto_scaling_enabled': LB_CONFIG['auto_scaling_enabled']
        }
    )

@app.get("/metrics")
async def get_metrics():
    """Prometheus-compatible metrics"""
    metrics = []
    
    # Service registry metrics
    total_instances = sum(len(instances) for instances in service_registry.values())
    healthy_instances = sum(
        len([i for i in instances if i.status == "healthy"])
        for instances in service_registry.values()
    )
    
    metrics.extend([
        f"load_balancer_total_services {len(service_registry)}",
        f"load_balancer_total_instances {total_instances}",
        f"load_balancer_healthy_instances {healthy_instances}",
        f"load_balancer_total_requests {lb_metrics.total_requests}",
        f"load_balancer_successful_requests {lb_metrics.successful_requests}",
        f"load_balancer_failed_requests {lb_metrics.failed_requests}",
        f"load_balancer_avg_response_time_ms {lb_metrics.avg_response_time_ms}"
    ])
    
    # Per-service metrics
    for service_name, instances in service_registry.items():
        service_healthy = len([i for i in instances if i.status == "healthy"])
        service_total = len(instances)
        avg_load = sum(i.load_score for i in instances) / len(instances) if instances else 0
        
        metrics.extend([
            f"service_instances_total{{service=\"{service_name}\"}} {service_total}",
            f"service_instances_healthy{{service=\"{service_name}\"}} {service_healthy}",
            f"service_avg_load_score{{service=\"{service_name}\"}} {avg_load}"
        ])
    
    return {"metrics": "\n".join(metrics)}

@app.post("/api/scaling/{service_name}/scale-up")
async def manual_scale_up(service_name: str):
    """Manually trigger scale-up for a service"""
    if service_name not in service_registry:
        raise HTTPException(status_code=404, detail="Service not found")
    
    await simulate_scale_up(service_name)
    
    return {
        'service_name': service_name,
        'action': 'scale_up',
        'timestamp': datetime.now(),
        'status': 'initiated'
    }

@app.post("/api/scaling/{service_name}/scale-down")
async def manual_scale_down(service_name: str):
    """Manually trigger scale-down for a service"""
    if service_name not in service_registry:
        raise HTTPException(status_code=404, detail="Service not found")
    
    instances = service_registry[service_name]
    if len([i for i in instances if i.status == "healthy"]) <= LB_CONFIG['min_instances']:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot scale down below minimum instances ({LB_CONFIG['min_instances']})"
        )
    
    await simulate_scale_down(service_name)
    
    return {
        'service_name': service_name,
        'action': 'scale_down',
        'timestamp': datetime.now(),
        'status': 'initiated'
    }

if __name__ == "__main__":
    uvicorn.run(
        "load_balancer:app",
        host="0.0.0.0",
        port=LB_CONFIG['port'],
        reload=False,
        log_level="info"
    )