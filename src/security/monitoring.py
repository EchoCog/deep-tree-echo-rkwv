"""
Security Monitor
Handles security event monitoring, intrusion detection, and audit logging
"""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import re
import hashlib

logger = logging.getLogger(__name__)

class SecurityEventType(Enum):
    """Types of security events"""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILED = "login_failed"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    PERMISSION_DENIED = "permission_denied"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    API_ACCESS = "api_access"
    DATA_ACCESS = "data_access"
    ADMIN_ACTION = "admin_action"
    SECURITY_VIOLATION = "security_violation"
    ACCOUNT_LOCKED = "account_locked"
    INTRUSION_ATTEMPT = "intrusion_attempt"

class AlertLevel(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class SecurityEvent:
    """Security event data model"""
    event_id: str
    event_type: SecurityEventType
    user_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    timestamp: datetime
    details: Dict[str, Any]
    alert_level: AlertLevel = AlertLevel.LOW
    resource_id: Optional[str] = None
    session_id: Optional[str] = None

@dataclass
class SecurityAlert:
    """Security alert data model"""
    alert_id: str
    alert_type: str
    alert_level: AlertLevel
    message: str
    timestamp: datetime
    related_events: List[str] = field(default_factory=list)
    acknowledged: bool = False
    resolved: bool = False

class SecurityMonitor:
    """Comprehensive security monitoring and intrusion detection system"""
    
    def __init__(self, alert_retention_days: int = 90):
        self.events: deque = deque(maxlen=10000)  # Keep last 10k events
        self.alerts: Dict[str, SecurityAlert] = {}
        self.alert_retention_days = alert_retention_days
        
        # Security metrics
        self.login_attempts: defaultdict = defaultdict(int)
        self.ip_access_counts: defaultdict = defaultdict(int)
        self.failed_logins_by_ip: defaultdict = defaultdict(list)
        self.failed_logins_by_user: defaultdict = defaultdict(list)
        self.suspicious_ips: Set[str] = set()
        self.blocked_ips: Set[str] = set()
        
        # Detection thresholds
        self.thresholds = {
            'max_failed_logins_per_hour': 5,
            'max_failed_logins_per_ip_per_hour': 10,
            'max_api_requests_per_minute': 100,
            'suspicious_user_agent_patterns': [
                r'bot', r'crawler', r'spider', r'scraper', r'automated'
            ],
            'suspicious_ip_patterns': [
                # Add known malicious IP patterns
            ]
        }
        
        # Rate limiting windows
        self.rate_windows = {
            'api_requests': defaultdict(lambda: deque(maxlen=1000)),
            'login_attempts': defaultdict(lambda: deque(maxlen=100))
        }
    
    def log_security_event(self, event_type: SecurityEventType, user_id: Optional[str] = None,
                          ip_address: Optional[str] = None, user_agent: Optional[str] = None,
                          details: Dict[str, Any] = None, resource_id: Optional[str] = None,
                          session_id: Optional[str] = None) -> str:
        """Log a security event and check for alerts"""
        
        event_id = self._generate_event_id()
        
        event = SecurityEvent(
            event_id=event_id,
            event_type=event_type,
            user_id=user_id,
            ip_address=ip_address,
            user_agent=user_agent,
            timestamp=datetime.now(),
            details=details or {},
            resource_id=resource_id,
            session_id=session_id
        )
        
        # Determine alert level
        event.alert_level = self._calculate_alert_level(event)
        
        # Store event
        self.events.append(event)
        
        # Update metrics
        self._update_security_metrics(event)
        
        # Check for security alerts
        self._check_security_alerts(event)
        
        # Log to standard logger
        logger.info(f"Security event: {event_type.value} - User: {user_id}, IP: {ip_address}")
        
        return event_id
    
    def _generate_event_id(self) -> str:
        """Generate unique event ID"""
        timestamp = datetime.now().isoformat()
        return hashlib.md5(timestamp.encode()).hexdigest()[:16]
    
    def _calculate_alert_level(self, event: SecurityEvent) -> AlertLevel:
        """Calculate alert level based on event type and context"""
        high_risk_events = {
            SecurityEventType.INTRUSION_ATTEMPT,
            SecurityEventType.SECURITY_VIOLATION,
            SecurityEventType.ACCOUNT_LOCKED
        }
        
        medium_risk_events = {
            SecurityEventType.LOGIN_FAILED,
            SecurityEventType.PERMISSION_DENIED,
            SecurityEventType.SUSPICIOUS_ACTIVITY
        }
        
        if event.event_type in high_risk_events:
            return AlertLevel.CRITICAL
        elif event.event_type in medium_risk_events:
            return AlertLevel.MEDIUM
        else:
            return AlertLevel.LOW
    
    def _update_security_metrics(self, event: SecurityEvent):
        """Update security metrics based on event"""
        
        # Update login attempt metrics
        if event.event_type in [SecurityEventType.LOGIN_SUCCESS, SecurityEventType.LOGIN_FAILED]:
            if event.user_id:
                self.login_attempts[event.user_id] += 1
            if event.ip_address:
                self.ip_access_counts[event.ip_address] += 1
        
        # Track failed login attempts
        if event.event_type == SecurityEventType.LOGIN_FAILED:
            if event.ip_address:
                self.failed_logins_by_ip[event.ip_address].append(event.timestamp)
            if event.user_id:
                self.failed_logins_by_user[event.user_id].append(event.timestamp)
        
        # Update rate limiting windows
        if event.event_type == SecurityEventType.API_ACCESS:
            if event.ip_address:
                self.rate_windows['api_requests'][event.ip_address].append(event.timestamp)
    
    def _check_security_alerts(self, event: SecurityEvent):
        """Check if event triggers any security alerts"""
        
        # Check for brute force attacks
        self._check_brute_force_attacks(event)
        
        # Check for suspicious user agents
        self._check_suspicious_user_agent(event)
        
        # Check for rate limit violations
        self._check_rate_limits(event)
        
        # Check for suspicious patterns
        self._check_suspicious_patterns(event)
    
    def _check_brute_force_attacks(self, event: SecurityEvent):
        """Check for brute force login attempts"""
        if event.event_type != SecurityEventType.LOGIN_FAILED:
            return
        
        now = datetime.now()
        hour_ago = now - timedelta(hours=1)
        
        # Check failed logins by IP
        if event.ip_address:
            recent_failures = [
                timestamp for timestamp in self.failed_logins_by_ip[event.ip_address]
                if timestamp > hour_ago
            ]
            
            if len(recent_failures) >= self.thresholds['max_failed_logins_per_ip_per_hour']:
                self._create_alert(
                    alert_type="brute_force_ip",
                    level=AlertLevel.HIGH,
                    message=f"Brute force attack detected from IP {event.ip_address}",
                    related_events=[event.event_id]
                )
                self.suspicious_ips.add(event.ip_address)
        
        # Check failed logins by user
        if event.user_id:
            recent_failures = [
                timestamp for timestamp in self.failed_logins_by_user[event.user_id]
                if timestamp > hour_ago
            ]
            
            if len(recent_failures) >= self.thresholds['max_failed_logins_per_hour']:
                self._create_alert(
                    alert_type="brute_force_user",
                    level=AlertLevel.MEDIUM,
                    message=f"Multiple failed login attempts for user {event.user_id}",
                    related_events=[event.event_id]
                )
    
    def _check_suspicious_user_agent(self, event: SecurityEvent):
        """Check for suspicious user agent patterns"""
        if not event.user_agent:
            return
        
        user_agent_lower = event.user_agent.lower()
        
        for pattern in self.thresholds['suspicious_user_agent_patterns']:
            if re.search(pattern, user_agent_lower):
                self._create_alert(
                    alert_type="suspicious_user_agent",
                    level=AlertLevel.MEDIUM,
                    message=f"Suspicious user agent detected: {event.user_agent}",
                    related_events=[event.event_id]
                )
                break
    
    def _check_rate_limits(self, event: SecurityEvent):
        """Check for rate limit violations"""
        if event.event_type != SecurityEventType.API_ACCESS or not event.ip_address:
            return
        
        now = datetime.now()
        minute_ago = now - timedelta(minutes=1)
        
        recent_requests = [
            timestamp for timestamp in self.rate_windows['api_requests'][event.ip_address]
            if timestamp > minute_ago
        ]
        
        if len(recent_requests) >= self.thresholds['max_api_requests_per_minute']:
            self._create_alert(
                alert_type="rate_limit_violation",
                level=AlertLevel.MEDIUM,
                message=f"Rate limit violation from IP {event.ip_address}",
                related_events=[event.event_id]
            )
    
    def _check_suspicious_patterns(self, event: SecurityEvent):
        """Check for other suspicious patterns"""
        # Check for access from blocked IPs
        if event.ip_address in self.blocked_ips:
            self._create_alert(
                alert_type="blocked_ip_access",
                level=AlertLevel.HIGH,
                message=f"Access attempt from blocked IP {event.ip_address}",
                related_events=[event.event_id]
            )
        
        # Check for privilege escalation attempts
        if event.event_type == SecurityEventType.PERMISSION_DENIED:
            # Look for patterns of permission denied events
            recent_denials = [
                e for e in self.events
                if (e.event_type == SecurityEventType.PERMISSION_DENIED and
                    e.user_id == event.user_id and
                    datetime.now() - e.timestamp < timedelta(minutes=5))
            ]
            
            if len(recent_denials) >= 3:
                self._create_alert(
                    alert_type="privilege_escalation_attempt",
                    level=AlertLevel.HIGH,
                    message=f"Possible privilege escalation attempt by user {event.user_id}",
                    related_events=[e.event_id for e in recent_denials]
                )
    
    def _create_alert(self, alert_type: str, level: AlertLevel, message: str, related_events: List[str]):
        """Create a security alert"""
        alert_id = self._generate_event_id()
        
        alert = SecurityAlert(
            alert_id=alert_id,
            alert_type=alert_type,
            alert_level=level,
            message=message,
            timestamp=datetime.now(),
            related_events=related_events
        )
        
        self.alerts[alert_id] = alert
        
        # Log alert
        logger.warning(f"Security Alert [{level.value}]: {message}")
        
        # Trigger automated response for critical alerts
        if level == AlertLevel.CRITICAL:
            self._trigger_automated_response(alert)
    
    def _trigger_automated_response(self, alert: SecurityAlert):
        """Trigger automated response for critical alerts"""
        logger.critical(f"Triggering automated response for alert: {alert.alert_id}")
        
        # Example automated responses
        if alert.alert_type == "brute_force_ip":
            # Could automatically block IP
            for event_id in alert.related_events:
                event = self._get_event_by_id(event_id)
                if event and event.ip_address:
                    self.block_ip(event.ip_address, "Automated: Brute force attack")
    
    def block_ip(self, ip_address: str, reason: str):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        self.log_security_event(
            SecurityEventType.ADMIN_ACTION,
            details={'action': 'block_ip', 'ip': ip_address, 'reason': reason}
        )
        logger.warning(f"Blocked IP address {ip_address}: {reason}")
    
    def unblock_ip(self, ip_address: str, admin_user_id: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            self.log_security_event(
                SecurityEventType.ADMIN_ACTION,
                user_id=admin_user_id,
                details={'action': 'unblock_ip', 'ip': ip_address}
            )
            logger.info(f"Unblocked IP address {ip_address} by admin {admin_user_id}")
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked"""
        return ip_address in self.blocked_ips
    
    def is_ip_suspicious(self, ip_address: str) -> bool:
        """Check if IP address is suspicious"""
        return ip_address in self.suspicious_ips
    
    def acknowledge_alert(self, alert_id: str, admin_user_id: str) -> bool:
        """Acknowledge a security alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].acknowledged = True
            self.log_security_event(
                SecurityEventType.ADMIN_ACTION,
                user_id=admin_user_id,
                details={'action': 'acknowledge_alert', 'alert_id': alert_id}
            )
            return True
        return False
    
    def resolve_alert(self, alert_id: str, admin_user_id: str) -> bool:
        """Resolve a security alert"""
        if alert_id in self.alerts:
            self.alerts[alert_id].resolved = True
            self.log_security_event(
                SecurityEventType.ADMIN_ACTION,
                user_id=admin_user_id,
                details={'action': 'resolve_alert', 'alert_id': alert_id}
            )
            return True
        return False
    
    def get_security_events(self, user_id: Optional[str] = None, event_type: Optional[SecurityEventType] = None,
                           hours: int = 24) -> List[SecurityEvent]:
        """Get security events with filters"""
        cutoff = datetime.now() - timedelta(hours=hours)
        
        filtered_events = []
        for event in self.events:
            if event.timestamp < cutoff:
                continue
            
            if user_id and event.user_id != user_id:
                continue
            
            if event_type and event.event_type != event_type:
                continue
            
            filtered_events.append(event)
        
        return sorted(filtered_events, key=lambda e: e.timestamp, reverse=True)
    
    def get_active_alerts(self, level: Optional[AlertLevel] = None) -> List[SecurityAlert]:
        """Get active security alerts"""
        alerts = []
        for alert in self.alerts.values():
            if alert.resolved:
                continue
            
            if level and alert.alert_level != level:
                continue
            
            alerts.append(alert)
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security summary for the specified time period"""
        cutoff = datetime.now() - timedelta(hours=hours)
        recent_events = [e for e in self.events if e.timestamp >= cutoff]
        
        # Count events by type
        event_counts = defaultdict(int)
        for event in recent_events:
            event_counts[event.event_type.value] += 1
        
        # Count alerts by level
        alert_counts = defaultdict(int)
        for alert in self.alerts.values():
            if alert.timestamp >= cutoff:
                alert_counts[alert.alert_level.value] += 1
        
        return {
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_counts': dict(event_counts),
            'total_alerts': len([a for a in self.alerts.values() if a.timestamp >= cutoff]),
            'alert_counts': dict(alert_counts),
            'blocked_ips': len(self.blocked_ips),
            'suspicious_ips': len(self.suspicious_ips),
            'unique_users': len(set(e.user_id for e in recent_events if e.user_id)),
            'unique_ips': len(set(e.ip_address for e in recent_events if e.ip_address))
        }
    
    def generate_audit_report(self, user_id: Optional[str] = None, days: int = 30) -> Dict[str, Any]:
        """Generate comprehensive audit report"""
        cutoff = datetime.now() - timedelta(days=days)
        
        # Filter events for the period
        audit_events = [e for e in self.events if e.timestamp >= cutoff]
        if user_id:
            audit_events = [e for e in audit_events if e.user_id == user_id]
        
        # Organize events by type
        events_by_type = defaultdict(list)
        for event in audit_events:
            events_by_type[event.event_type.value].append({
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'ip_address': event.ip_address,
                'details': event.details
            })
        
        # Get alerts for the period
        audit_alerts = [
            {
                'alert_id': alert.alert_id,
                'alert_type': alert.alert_type,
                'level': alert.alert_level.value,
                'message': alert.message,
                'timestamp': alert.timestamp.isoformat(),
                'acknowledged': alert.acknowledged,
                'resolved': alert.resolved
            }
            for alert in self.alerts.values()
            if alert.timestamp >= cutoff
        ]
        
        return {
            'report_generated': datetime.now().isoformat(),
            'audit_period_days': days,
            'user_id': user_id,
            'total_events': len(audit_events),
            'events_by_type': dict(events_by_type),
            'total_alerts': len(audit_alerts),
            'alerts': audit_alerts,
            'security_summary': self.get_security_summary(days * 24)
        }
    
    def _get_event_by_id(self, event_id: str) -> Optional[SecurityEvent]:
        """Get event by ID"""
        for event in self.events:
            if event.event_id == event_id:
                return event
        return None
    
    def cleanup_old_data(self):
        """Clean up old alerts and data"""
        cutoff = datetime.now() - timedelta(days=self.alert_retention_days)
        
        # Remove old resolved alerts
        old_alerts = [
            alert_id for alert_id, alert in self.alerts.items()
            if alert.resolved and alert.timestamp < cutoff
        ]
        
        for alert_id in old_alerts:
            del self.alerts[alert_id]
        
        logger.debug(f"Cleaned up {len(old_alerts)} old alerts")