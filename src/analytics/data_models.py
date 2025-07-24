"""
Data Models for Analytics System
Defines data structures for events, sessions, metrics, and queries.
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum


class EventType(Enum):
    """Types of analytics events"""
    USER_ACTION = "user_action"
    SYSTEM_METRIC = "system_metric"
    COGNITIVE_PROCESS = "cognitive_process"
    API_REQUEST = "api_request"
    ERROR = "error"
    PERFORMANCE = "performance"


class SessionStatus(Enum):
    """User session status"""
    ACTIVE = "active"
    EXPIRED = "expired"
    TERMINATED = "terminated"


@dataclass
class AnalyticsEvent:
    """Base analytics event data structure"""
    event_id: str
    event_type: EventType
    timestamp: datetime
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid.uuid4())
        if self.data is None:
            self.data = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/transmission"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        result['event_type'] = self.event_type.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AnalyticsEvent':
        """Create from dictionary"""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        data['event_type'] = EventType(data['event_type'])
        return cls(**data)


@dataclass
class UserSession:
    """User session tracking"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    last_activity: datetime
    status: SessionStatus
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    events_count: int = 0
    cognitive_processes: int = 0
    api_calls: int = 0
    
    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
    
    @property
    def duration(self) -> timedelta:
        """Session duration"""
        return self.last_activity - self.start_time
    
    @property
    def is_active(self) -> bool:
        """Check if session is still active (within 30 minutes)"""
        return (datetime.now() - self.last_activity) < timedelta(minutes=30)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now()
        if not self.is_active and self.status == SessionStatus.ACTIVE:
            self.status = SessionStatus.EXPIRED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['last_activity'] = self.last_activity.isoformat()
        result['status'] = self.status.value
        return result


@dataclass
class SystemMetrics:
    """System performance metrics"""
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: float
    throughput: float
    error_rate: float
    active_sessions: int
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class CognitiveProcessingMetrics:
    """Cognitive processing performance metrics"""
    timestamp: datetime
    session_id: str
    membrane_type: str  # memory, reasoning, grammar
    processing_time: float
    input_tokens: int
    output_tokens: int
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


@dataclass
class AnalyticsQuery:
    """Analytics query structure"""
    query_id: str
    query_type: str  # aggregation, trend, comparison, prediction
    filters: Dict[str, Any]
    time_range: Dict[str, datetime]
    grouping: List[str]
    metrics: List[str]
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.query_id is None:
            self.query_id = str(uuid.uuid4())
        if self.created_at is None:
            self.created_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = asdict(self)
        result['created_at'] = self.created_at.isoformat()
        result['time_range'] = {
            k: v.isoformat() for k, v in self.time_range.items()
        }
        return result


class DataWarehouse:
    """Simple in-memory data warehouse for analytics data"""
    
    def __init__(self):
        self.events: List[AnalyticsEvent] = []
        self.sessions: Dict[str, UserSession] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.cognitive_metrics: List[CognitiveProcessingMetrics] = []
        self._index_by_timestamp = {}
        self._index_by_user = {}
        self._index_by_session = {}
    
    def add_event(self, event: AnalyticsEvent):
        """Add an analytics event"""
        self.events.append(event)
        self._update_indexes(event)
    
    def add_session(self, session: UserSession):
        """Add or update a user session"""
        self.sessions[session.session_id] = session
    
    def add_system_metrics(self, metrics: SystemMetrics):
        """Add system metrics"""
        self.system_metrics.append(metrics)
    
    def add_cognitive_metrics(self, metrics: CognitiveProcessingMetrics):
        """Add cognitive processing metrics"""
        self.cognitive_metrics.append(metrics)
    
    def _update_indexes(self, event: AnalyticsEvent):
        """Update internal indexes for faster querying"""
        # Index by timestamp (hour buckets)
        hour_bucket = event.timestamp.replace(minute=0, second=0, microsecond=0)
        if hour_bucket not in self._index_by_timestamp:
            self._index_by_timestamp[hour_bucket] = []
        self._index_by_timestamp[hour_bucket].append(event)
        
        # Index by user
        if event.user_id:
            if event.user_id not in self._index_by_user:
                self._index_by_user[event.user_id] = []
            self._index_by_user[event.user_id].append(event)
        
        # Index by session
        if event.session_id:
            if event.session_id not in self._index_by_session:
                self._index_by_session[event.session_id] = []
            self._index_by_session[event.session_id].append(event)
    
    def query_events(self, 
                    start_time: Optional[datetime] = None,
                    end_time: Optional[datetime] = None,
                    event_type: Optional[EventType] = None,
                    user_id: Optional[str] = None,
                    session_id: Optional[str] = None) -> List[AnalyticsEvent]:
        """Query events with filters"""
        events = self.events
        
        if start_time:
            events = [e for e in events if e.timestamp >= start_time]
        if end_time:
            events = [e for e in events if e.timestamp <= end_time]
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if user_id:
            events = [e for e in events if e.user_id == user_id]
        if session_id:
            events = [e for e in events if e.session_id == session_id]
        
        return events
    
    def get_active_sessions(self) -> List[UserSession]:
        """Get currently active sessions"""
        return [s for s in self.sessions.values() if s.is_active]
    
    def cleanup_old_data(self, retention_days: int = 365):
        """Clean up old data based on retention policy"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        
        # Clean events
        self.events = [e for e in self.events if e.timestamp > cutoff_date]
        
        # Clean system metrics
        self.system_metrics = [m for m in self.system_metrics if m.timestamp > cutoff_date]
        
        # Clean cognitive metrics
        self.cognitive_metrics = [m for m in self.cognitive_metrics if m.timestamp > cutoff_date]
        
        # Rebuild indexes
        self._index_by_timestamp.clear()
        self._index_by_user.clear()
        self._index_by_session.clear()
        
        for event in self.events:
            self._update_indexes(event)