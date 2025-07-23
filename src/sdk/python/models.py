"""
Deep Tree Echo SDK Data Models
Data classes for API responses and requests
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class MembraneOutput:
    """Output from a cognitive membrane"""
    membrane_type: str
    response: str
    confidence: float
    processing_time: float

@dataclass 
class CognitiveState:
    """Cognitive state information"""
    declarative_memory_items: int = 0
    procedural_memory_items: int = 0
    episodic_memory_items: int = 0
    temporal_context_length: int = 0
    current_goals: int = 0
    last_updated: Optional[datetime] = None

@dataclass
class CognitiveResult:
    """Result of cognitive processing"""
    input_text: str
    integrated_response: str
    processing_time: float
    session_id: str
    membrane_outputs: List[MembraneOutput] = field(default_factory=list)
    cognitive_state: Optional[CognitiveState] = None
    timestamp: Optional[datetime] = None
    confidence: Optional[float] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CognitiveResult':
        """Create CognitiveResult from API response"""
        membrane_outputs = []
        if 'membrane_outputs' in data:
            membrane_outputs = [
                MembraneOutput(**output) for output in data['membrane_outputs']
            ]
        
        cognitive_state = None
        if 'cognitive_state' in data and data['cognitive_state']:
            cognitive_state = CognitiveState(**data['cognitive_state'])
        
        timestamp = None
        if 'timestamp' in data and data['timestamp']:
            timestamp = datetime.fromisoformat(data['timestamp'].replace('Z', '+00:00'))
        
        return cls(
            input_text=data.get('input_text', data.get('input', '')),
            integrated_response=data.get('integrated_response', data.get('output', '')),
            processing_time=data.get('processing_time', 0.0),
            session_id=data.get('session_id', ''),
            membrane_outputs=membrane_outputs,
            cognitive_state=cognitive_state,
            timestamp=timestamp,
            confidence=data.get('confidence')
        )

@dataclass
class SessionConfiguration:
    """Session configuration parameters"""
    temperature: float = 0.8
    max_context_length: int = 2048
    memory_persistence: bool = True

@dataclass
class SessionInfo:
    """Cognitive session information"""
    session_id: str
    status: str
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    total_tokens_processed: int = 0
    configuration: Optional[SessionConfiguration] = None
    cognitive_state: Optional[CognitiveState] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SessionInfo':
        """Create SessionInfo from API response"""
        created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        last_activity = datetime.fromisoformat(data['last_activity'].replace('Z', '+00:00'))
        
        configuration = None
        if 'configuration' in data and data['configuration']:
            configuration = SessionConfiguration(**data['configuration'])
        
        cognitive_state = None
        if 'cognitive_state' in data and data['cognitive_state']:
            cognitive_state = CognitiveState(**data['cognitive_state'])
        
        return cls(
            session_id=data['session_id'],
            status=data['status'],
            created_at=created_at,
            last_activity=last_activity,
            message_count=data.get('message_count', 0),
            total_tokens_processed=data.get('total_tokens_processed', 0),
            configuration=configuration,
            cognitive_state=cognitive_state,
            metadata=data.get('metadata', {})
        )

@dataclass
class MemoryItem:
    """Memory item from the cognitive system"""
    id: str
    content: str
    memory_type: str
    relevance_score: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryItem':
        """Create MemoryItem from API response"""
        created_at = datetime.fromisoformat(data['created_at'].replace('Z', '+00:00'))
        last_accessed = datetime.fromisoformat(data['last_accessed'].replace('Z', '+00:00'))
        
        return cls(
            id=data['id'],
            content=data['content'],
            memory_type=data['memory_type'],
            relevance_score=data['relevance_score'],
            created_at=created_at,
            last_accessed=last_accessed,
            access_count=data.get('access_count', 0),
            metadata=data.get('metadata', {})
        )

@dataclass
class SystemServices:
    """System services status"""
    cognitive_processing: bool = False
    memory_system: bool = False
    api_server: bool = False

@dataclass
class SystemPerformance:
    """System performance metrics"""
    response_time_ms: float = 0.0
    throughput_rpm: float = 0.0
    cache_hit_rate: float = 0.0

@dataclass
class SystemInfo:
    """System information"""
    status: str
    version: str
    uptime: float
    echo_system_initialized: bool

@dataclass
class SystemStatus:
    """System status information"""
    system: SystemInfo
    services: SystemServices
    performance: SystemPerformance
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'SystemStatus':
        """Create SystemStatus from API response"""
        system = SystemInfo(**data['system'])
        services = SystemServices(**data['services'])
        performance = SystemPerformance(**data['performance'])
        
        return cls(
            system=system,
            services=services,
            performance=performance
        )

@dataclass
class UsageAnalytics:
    """Usage analytics information"""
    total_requests: int
    successful_requests: int
    error_requests: int
    average_response_time: float
    api_tier: str
    quota_remaining: int
    period: str = "last_30_days"
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UsageAnalytics':
        """Create UsageAnalytics from API response"""
        return cls(**data)

@dataclass
class QuotaInfo:
    """API quota information"""
    tier: str
    hour_usage: int
    hour_limit: int  
    day_usage: int
    day_limit: int
    allowed: bool
    
    @property
    def hour_remaining(self) -> int:
        """Calculate remaining hourly quota"""
        return max(0, self.hour_limit - self.hour_usage)
    
    @property
    def day_remaining(self) -> int:
        """Calculate remaining daily quota"""
        return max(0, self.day_limit - self.day_usage)
    
    @property
    def usage_percentage(self) -> float:
        """Calculate usage percentage for the day"""
        return (self.day_usage / self.day_limit * 100) if self.day_limit > 0 else 0.0