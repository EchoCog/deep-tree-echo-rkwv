"""
Third-Party Integration Framework
Base classes and utilities for creating integrations with external services
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json
import asyncio
from enum import Enum

logger = logging.getLogger(__name__)

class IntegrationType(Enum):
    """Types of integrations supported"""
    WEBHOOK = "webhook"
    DATABASE = "database"  
    CLOUD_SERVICE = "cloud_service"
    MESSAGING = "messaging"
    AUTHENTICATION = "authentication"
    STORAGE = "storage"
    ANALYTICS = "analytics"
    NOTIFICATION = "notification"

class IntegrationStatus(Enum):
    """Integration status states"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    PENDING = "pending"
    DISABLED = "disabled"

@dataclass
class IntegrationConfig:
    """Configuration for an integration"""
    name: str
    type: IntegrationType
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    credentials: Dict[str, str] = field(default_factory=dict)
    rate_limit: Optional[int] = None
    timeout: int = 30
    retry_attempts: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass 
class IntegrationEvent:
    """Event data for integration operations"""
    event_type: str
    data: Dict[str, Any]
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseIntegration(ABC):
    """Base class for all third-party integrations"""
    
    def __init__(self, config: IntegrationConfig):
        self.config = config
        self.status = IntegrationStatus.PENDING
        self.last_error: Optional[str] = None
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'last_activity': None
        }
    
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the integration"""
        pass
    
    @abstractmethod
    async def send_event(self, event: IntegrationEvent) -> bool:
        """Send an event to the external service"""
        pass
    
    @abstractmethod
    async def receive_event(self) -> Optional[IntegrationEvent]:
        """Receive an event from the external service"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the integration is healthy"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        pass
    
    def update_stats(self, success: bool):
        """Update integration statistics"""
        self.stats['total_requests'] += 1
        self.stats['last_activity'] = datetime.now()
        
        if success:
            self.stats['successful_requests'] += 1
            self.status = IntegrationStatus.ACTIVE
            self.last_error = None
        else:
            self.stats['failed_requests'] += 1
            self.status = IntegrationStatus.ERROR
    
    def get_info(self) -> Dict[str, Any]:
        """Get integration information"""
        return {
            'name': self.config.name,
            'type': self.config.type.value,
            'status': self.status.value,
            'enabled': self.config.enabled,
            'last_error': self.last_error,
            'stats': self.stats
        }

class WebhookIntegration(BaseIntegration):
    """Webhook-based integration"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.webhook_url = config.settings.get('webhook_url')
        self.secret = config.credentials.get('webhook_secret')
        self.headers = config.settings.get('headers', {})
    
    async def initialize(self) -> bool:
        """Initialize webhook integration"""
        if not self.webhook_url:
            self.last_error = "Webhook URL is required"
            return False
        
        try:
            # Test webhook endpoint
            health_check = await self.health_check()
            if health_check:
                self.status = IntegrationStatus.ACTIVE
                return True
            else:
                self.status = IntegrationStatus.ERROR
                return False
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def send_event(self, event: IntegrationEvent) -> bool:
        """Send event via webhook"""
        try:
            import aiohttp
            
            payload = {
                'event_type': event.event_type,
                'data': event.data,
                'source': event.source,
                'timestamp': event.timestamp.isoformat(),
                'correlation_id': event.correlation_id,
                'metadata': event.metadata
            }
            
            headers = {'Content-Type': 'application/json'}
            headers.update(self.headers)
            
            if self.secret:
                # Add webhook signature if secret is provided
                import hmac
                import hashlib
                signature = hmac.new(
                    self.secret.encode(),
                    json.dumps(payload).encode(),
                    hashlib.sha256
                ).hexdigest()
                headers['X-Webhook-Signature'] = f'sha256={signature}'
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    success = 200 <= response.status < 300
                    self.update_stats(success)
                    
                    if not success:
                        self.last_error = f"HTTP {response.status}: {await response.text()}"
                    
                    return success
                    
        except Exception as e:
            self.last_error = str(e)
            self.update_stats(False)
            logger.error(f"Webhook integration error: {e}")
            return False
    
    async def receive_event(self) -> Optional[IntegrationEvent]:
        """Webhooks are push-based, so this returns None"""
        return None
    
    async def health_check(self) -> bool:
        """Check webhook endpoint health"""
        try:
            import aiohttp
            
            # Send a simple health check request
            async with aiohttp.ClientSession() as session:
                async with session.head(
                    self.webhook_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    return 200 <= response.status < 500  # Allow 4xx responses for health check
                    
        except Exception as e:
            logger.warning(f"Webhook health check failed: {e}")
            return False

class DatabaseIntegration(BaseIntegration):
    """Database integration base"""
    
    def __init__(self, config: IntegrationConfig):
        super().__init__(config)
        self.connection_string = config.credentials.get('connection_string')
        self.database_type = config.settings.get('database_type', 'postgresql')
        self.table_name = config.settings.get('table_name', 'echo_events')
    
    async def initialize(self) -> bool:
        """Initialize database connection"""
        if not self.connection_string:
            self.last_error = "Database connection string is required"
            return False
        
        try:
            # Database-specific initialization would go here
            # For now, we'll just mark as active
            self.status = IntegrationStatus.ACTIVE
            return True
        except Exception as e:
            self.last_error = str(e)
            self.status = IntegrationStatus.ERROR
            return False
    
    async def send_event(self, event: IntegrationEvent) -> bool:
        """Store event in database"""
        try:
            # Mock database operation
            # In real implementation, this would use appropriate database driver
            success = True  # Simulate successful database write
            self.update_stats(success)
            return success
            
        except Exception as e:
            self.last_error = str(e)
            self.update_stats(False)
            logger.error(f"Database integration error: {e}")
            return False
    
    async def receive_event(self) -> Optional[IntegrationEvent]:
        """Retrieve event from database"""
        # Mock implementation
        return None
    
    async def health_check(self) -> bool:
        """Check database connection health"""
        try:
            # Mock health check
            return True
        except Exception as e:
            logger.warning(f"Database health check failed: {e}")
            return False

class IntegrationManager:
    """Manages multiple integrations"""
    
    def __init__(self):
        self.integrations: Dict[str, BaseIntegration] = {}
        self.event_queue = asyncio.Queue()
        self.is_running = False
    
    def register_integration(self, integration: BaseIntegration) -> None:
        """Register a new integration"""
        self.integrations[integration.config.name] = integration
        logger.info(f"Registered integration: {integration.config.name}")
    
    def unregister_integration(self, name: str) -> None:
        """Remove an integration"""
        if name in self.integrations:
            del self.integrations[name]
            logger.info(f"Unregistered integration: {name}")
    
    async def initialize_all(self) -> Dict[str, bool]:
        """Initialize all registered integrations"""
        results = {}
        
        for name, integration in self.integrations.items():
            try:
                if integration.config.enabled:
                    success = await integration.initialize()
                    results[name] = success
                    
                    if success:
                        logger.info(f"Integration '{name}' initialized successfully")
                    else:
                        logger.error(f"Integration '{name}' failed to initialize: {integration.last_error}")
                else:
                    results[name] = False
                    logger.info(f"Integration '{name}' is disabled")
                    
            except Exception as e:
                logger.error(f"Error initializing integration '{name}': {e}")
                results[name] = False
        
        return results
    
    async def send_event_to_all(self, event: IntegrationEvent) -> Dict[str, bool]:
        """Send an event to all active integrations"""
        results = {}
        
        tasks = []
        for name, integration in self.integrations.items():
            if (integration.config.enabled and 
                integration.status == IntegrationStatus.ACTIVE):
                tasks.append(self._send_event_with_name(name, integration, event))
        
        if tasks:
            task_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, (name, result) in enumerate(zip(
                [name for name, integration in self.integrations.items() 
                 if integration.config.enabled and integration.status == IntegrationStatus.ACTIVE],
                task_results
            )):
                if isinstance(result, Exception):
                    logger.error(f"Integration '{name}' error: {result}")
                    results[name] = False
                else:
                    results[name] = result
        
        return results
    
    async def _send_event_with_name(self, name: str, integration: BaseIntegration, event: IntegrationEvent) -> bool:
        """Helper to send event and return result with name"""
        try:
            return await integration.send_event(event)
        except Exception as e:
            logger.error(f"Error sending event to integration '{name}': {e}")
            return False
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all integrations"""
        results = {}
        
        for name, integration in self.integrations.items():
            try:
                if integration.config.enabled:
                    health = await integration.health_check()
                    results[name] = health
                    
                    if not health:
                        integration.status = IntegrationStatus.ERROR
                else:
                    results[name] = False
                    
            except Exception as e:
                logger.error(f"Health check error for integration '{name}': {e}")
                results[name] = False
                integration.status = IntegrationStatus.ERROR
        
        return results
    
    def get_integration_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all integrations"""
        return {
            name: integration.get_info() 
            for name, integration in self.integrations.items()
        }
    
    async def cleanup_all(self) -> None:
        """Cleanup all integrations"""
        for integration in self.integrations.values():
            try:
                await integration.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up integration '{integration.config.name}': {e}")

# Convenience function to create common integrations
def create_webhook_integration(name: str, webhook_url: str, secret: Optional[str] = None) -> WebhookIntegration:
    """Create a webhook integration with common settings"""
    config = IntegrationConfig(
        name=name,
        type=IntegrationType.WEBHOOK,
        settings={'webhook_url': webhook_url},
        credentials={'webhook_secret': secret} if secret else {}
    )
    return WebhookIntegration(config)

def create_database_integration(name: str, connection_string: str, table_name: str = 'echo_events') -> DatabaseIntegration:
    """Create a database integration with common settings"""
    config = IntegrationConfig(
        name=name,
        type=IntegrationType.DATABASE,
        credentials={'connection_string': connection_string},
        settings={'table_name': table_name}
    )
    return DatabaseIntegration(config)