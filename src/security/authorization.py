"""
Authorization System
Handles role-based access control, permissions, and resource-level access controls
"""

from typing import Dict, Any, Optional, List, Set
from dataclasses import dataclass, field
from datetime import datetime
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class ResourceType(Enum):
    """Resource types for access control"""
    SESSION = "session"
    MEMORY = "memory"
    API = "api"
    ADMIN = "admin"
    USER_PROFILE = "user_profile"
    SYSTEM_METRICS = "system_metrics"

class Action(Enum):
    """Actions that can be performed on resources"""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    EXECUTE = "execute"
    ADMIN = "admin"

@dataclass
class Permission:
    """Permission data model"""
    name: str
    resource_type: ResourceType
    actions: Set[Action]
    description: str = ""
    conditions: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Role:
    """Role data model"""
    name: str
    permissions: Set[str]  # Permission names
    description: str = ""
    is_system_role: bool = False
    created_at: datetime = field(default_factory=datetime.now)

class AuthorizationSystem:
    """Comprehensive role-based access control system"""
    
    def __init__(self):
        self.permissions: Dict[str, Permission] = {}
        self.roles: Dict[str, Role] = {}
        self.user_roles: Dict[str, Set[str]] = {}  # user_id -> role names
        self.resource_ownership: Dict[str, str] = {}  # resource_id -> user_id
        self._initialize_default_permissions()
        self._initialize_default_roles()
    
    def _initialize_default_permissions(self):
        """Initialize default system permissions"""
        # Session permissions
        self.add_permission(Permission(
            name="session.create",
            resource_type=ResourceType.SESSION,
            actions={Action.CREATE},
            description="Create new cognitive sessions"
        ))
        
        self.add_permission(Permission(
            name="session.read",
            resource_type=ResourceType.SESSION,
            actions={Action.READ},
            description="Read session information"
        ))
        
        self.add_permission(Permission(
            name="session.update",
            resource_type=ResourceType.SESSION,
            actions={Action.UPDATE},
            description="Update session settings"
        ))
        
        self.add_permission(Permission(
            name="session.delete",
            resource_type=ResourceType.SESSION,
            actions={Action.DELETE},
            description="Delete sessions"
        ))
        
        # Memory permissions
        self.add_permission(Permission(
            name="memory.create",
            resource_type=ResourceType.MEMORY,
            actions={Action.CREATE},
            description="Store new memories"
        ))
        
        self.add_permission(Permission(
            name="memory.read",
            resource_type=ResourceType.MEMORY,
            actions={Action.READ},
            description="Read and search memories"
        ))
        
        self.add_permission(Permission(
            name="memory.update",
            resource_type=ResourceType.MEMORY,
            actions={Action.UPDATE},
            description="Update existing memories"
        ))
        
        self.add_permission(Permission(
            name="memory.delete",
            resource_type=ResourceType.MEMORY,
            actions={Action.DELETE},
            description="Delete memories"
        ))
        
        self.add_permission(Permission(
            name="memory.consolidate",
            resource_type=ResourceType.MEMORY,
            actions={Action.EXECUTE},
            description="Execute memory consolidation"
        ))
        
        # API permissions
        self.add_permission(Permission(
            name="api.cognitive_process",
            resource_type=ResourceType.API,
            actions={Action.EXECUTE},
            description="Process cognitive inputs"
        ))
        
        self.add_permission(Permission(
            name="api.status",
            resource_type=ResourceType.API,
            actions={Action.READ},
            description="Read system status"
        ))
        
        self.add_permission(Permission(
            name="api.metrics",
            resource_type=ResourceType.API,
            actions={Action.READ},
            description="Read system metrics"
        ))
        
        # User profile permissions
        self.add_permission(Permission(
            name="profile.read",
            resource_type=ResourceType.USER_PROFILE,
            actions={Action.READ},
            description="Read user profiles"
        ))
        
        self.add_permission(Permission(
            name="profile.update",
            resource_type=ResourceType.USER_PROFILE,
            actions={Action.UPDATE},
            description="Update user profiles"
        ))
        
        # Admin permissions
        self.add_permission(Permission(
            name="admin.users",
            resource_type=ResourceType.ADMIN,
            actions={Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE},
            description="Manage users"
        ))
        
        self.add_permission(Permission(
            name="admin.roles",
            resource_type=ResourceType.ADMIN,
            actions={Action.CREATE, Action.READ, Action.UPDATE, Action.DELETE},
            description="Manage roles and permissions"
        ))
        
        self.add_permission(Permission(
            name="admin.system",
            resource_type=ResourceType.ADMIN,
            actions={Action.READ, Action.UPDATE, Action.EXECUTE},
            description="System administration"
        ))
        
        self.add_permission(Permission(
            name="admin.audit",
            resource_type=ResourceType.ADMIN,
            actions={Action.READ},
            description="Access audit logs"
        ))
    
    def _initialize_default_roles(self):
        """Initialize default system roles"""
        # Guest role - minimal access
        self.add_role(Role(
            name="guest",
            permissions={
                "api.status"
            },
            description="Guest access with minimal permissions",
            is_system_role=True
        ))
        
        # User role - standard user access
        self.add_role(Role(
            name="user",
            permissions={
                "session.create",
                "session.read", 
                "session.update",
                "memory.create",
                "memory.read",
                "api.cognitive_process",
                "api.status",
                "profile.read",
                "profile.update"
            },
            description="Standard user with cognitive processing access",
            is_system_role=True
        ))
        
        # Premium user role - enhanced access
        self.add_role(Role(
            name="premium_user",
            permissions={
                "session.create",
                "session.read",
                "session.update", 
                "session.delete",
                "memory.create",
                "memory.read",
                "memory.update",
                "memory.consolidate",
                "api.cognitive_process",
                "api.status",
                "api.metrics",
                "profile.read",
                "profile.update"
            },
            description="Premium user with enhanced cognitive features",
            is_system_role=True
        ))
        
        # Moderator role - content management
        self.add_role(Role(
            name="moderator",
            permissions={
                "session.create",
                "session.read",
                "session.update",
                "session.delete",
                "memory.create",
                "memory.read",
                "memory.update",
                "memory.delete",
                "memory.consolidate",
                "api.cognitive_process",
                "api.status",
                "api.metrics",
                "profile.read",
                "profile.update"
            },
            description="Moderator with content management capabilities",
            is_system_role=True
        ))
        
        # Admin role - full access
        self.add_role(Role(
            name="admin",
            permissions={
                "session.create",
                "session.read",
                "session.update",
                "session.delete",
                "memory.create",
                "memory.read",
                "memory.update",
                "memory.delete",
                "memory.consolidate",
                "api.cognitive_process",
                "api.status",
                "api.metrics",
                "profile.read",
                "profile.update",
                "admin.users",
                "admin.roles",
                "admin.system",
                "admin.audit"
            },
            description="Administrator with full system access",
            is_system_role=True
        ))
    
    def add_permission(self, permission: Permission):
        """Add a new permission"""
        self.permissions[permission.name] = permission
        logger.debug(f"Added permission: {permission.name}")
    
    def add_role(self, role: Role):
        """Add a new role"""
        self.roles[role.name] = role
        logger.debug(f"Added role: {role.name}")
    
    def assign_role_to_user(self, user_id: str, role_name: str) -> bool:
        """Assign role to user"""
        if role_name not in self.roles:
            logger.warning(f"Attempted to assign unknown role: {role_name}")
            return False
        
        if user_id not in self.user_roles:
            self.user_roles[user_id] = set()
        
        self.user_roles[user_id].add(role_name)
        logger.info(f"Assigned role {role_name} to user {user_id}")
        return True
    
    def remove_role_from_user(self, user_id: str, role_name: str) -> bool:
        """Remove role from user"""
        if user_id not in self.user_roles:
            return False
        
        if role_name in self.user_roles[user_id]:
            self.user_roles[user_id].remove(role_name)
            logger.info(f"Removed role {role_name} from user {user_id}")
            return True
        
        return False
    
    def get_user_roles(self, user_id: str) -> Set[str]:
        """Get all roles for a user"""
        return self.user_roles.get(user_id, set())
    
    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user through their roles"""
        user_roles = self.get_user_roles(user_id)
        permissions = set()
        
        for role_name in user_roles:
            if role_name in self.roles:
                permissions.update(self.roles[role_name].permissions)
        
        return permissions
    
    def has_permission(self, user_id: str, permission_name: str, resource_id: Optional[str] = None) -> bool:
        """Check if user has specific permission"""
        # Get user permissions
        user_permissions = self.get_user_permissions(user_id)
        
        if permission_name not in user_permissions:
            return False
        
        # Check resource ownership for resource-specific permissions
        if resource_id and self._requires_ownership_check(permission_name):
            return self._check_resource_ownership(user_id, resource_id)
        
        return True
    
    def has_role(self, user_id: str, role_name: str) -> bool:
        """Check if user has specific role"""
        return role_name in self.get_user_roles(user_id)
    
    def can_access_resource(self, user_id: str, resource_type: ResourceType, action: Action, resource_id: Optional[str] = None) -> bool:
        """Check if user can perform action on resource type"""
        # Find permissions that match resource type and action
        matching_permissions = []
        for perm_name, permission in self.permissions.items():
            if permission.resource_type == resource_type and action in permission.actions:
                matching_permissions.append(perm_name)
        
        # Check if user has any matching permission
        user_permissions = self.get_user_permissions(user_id)
        has_permission = any(perm in user_permissions for perm in matching_permissions)
        
        if not has_permission:
            return False
        
        # Check resource ownership if needed
        if resource_id:
            return self._check_resource_ownership(user_id, resource_id)
        
        return True
    
    def set_resource_owner(self, resource_id: str, user_id: str):
        """Set ownership of a resource"""
        self.resource_ownership[resource_id] = user_id
        logger.debug(f"Set owner of resource {resource_id} to user {user_id}")
    
    def get_resource_owner(self, resource_id: str) -> Optional[str]:
        """Get owner of a resource"""
        return self.resource_ownership.get(resource_id)
    
    def _requires_ownership_check(self, permission_name: str) -> bool:
        """Check if permission requires ownership verification"""
        ownership_permissions = {
            "session.read",
            "session.update", 
            "session.delete",
            "memory.read",
            "memory.update",
            "memory.delete",
            "profile.read",
            "profile.update"
        }
        return permission_name in ownership_permissions
    
    def _check_resource_ownership(self, user_id: str, resource_id: str) -> bool:
        """Check if user owns the resource or has admin privileges"""
        # Check direct ownership
        if self.get_resource_owner(resource_id) == user_id:
            return True
        
        # Check admin privileges
        if self.has_role(user_id, "admin") or self.has_role(user_id, "moderator"):
            return True
        
        return False
    
    def get_accessible_resources(self, user_id: str, resource_type: ResourceType) -> List[str]:
        """Get list of resources user can access"""
        accessible = []
        
        # If user has admin/moderator role, they can access all resources
        if self.has_role(user_id, "admin") or self.has_role(user_id, "moderator"):
            return list(self.resource_ownership.keys())
        
        # Otherwise, only return owned resources
        for resource_id, owner_id in self.resource_ownership.items():
            if owner_id == user_id:
                accessible.append(resource_id)
        
        return accessible
    
    def create_custom_role(self, name: str, permissions: Set[str], description: str = "") -> bool:
        """Create a custom role"""
        if name in self.roles:
            return False
        
        # Validate permissions exist
        invalid_permissions = permissions - set(self.permissions.keys())
        if invalid_permissions:
            logger.warning(f"Invalid permissions in role creation: {invalid_permissions}")
            return False
        
        role = Role(
            name=name,
            permissions=permissions,
            description=description,
            is_system_role=False
        )
        
        self.add_role(role)
        logger.info(f"Created custom role: {name}")
        return True
    
    def update_role_permissions(self, role_name: str, permissions: Set[str]) -> bool:
        """Update permissions for a role"""
        if role_name not in self.roles:
            return False
        
        role = self.roles[role_name]
        if role.is_system_role:
            logger.warning(f"Attempted to modify system role: {role_name}")
            return False
        
        # Validate permissions exist
        invalid_permissions = permissions - set(self.permissions.keys())
        if invalid_permissions:
            logger.warning(f"Invalid permissions in role update: {invalid_permissions}")
            return False
        
        role.permissions = permissions
        logger.info(f"Updated permissions for role: {role_name}")
        return True
    
    def delete_role(self, role_name: str) -> bool:
        """Delete a custom role"""
        if role_name not in self.roles:
            return False
        
        role = self.roles[role_name]
        if role.is_system_role:
            logger.warning(f"Attempted to delete system role: {role_name}")
            return False
        
        # Remove role from all users
        for user_id in self.user_roles:
            self.user_roles[user_id].discard(role_name)
        
        del self.roles[role_name]
        logger.info(f"Deleted role: {role_name}")
        return True
    
    def get_role_info(self, role_name: str) -> Optional[Dict[str, Any]]:
        """Get role information"""
        if role_name not in self.roles:
            return None
        
        role = self.roles[role_name]
        return {
            'name': role.name,
            'permissions': list(role.permissions),
            'description': role.description,
            'is_system_role': role.is_system_role,
            'created_at': role.created_at.isoformat()
        }
    
    def get_permission_info(self, permission_name: str) -> Optional[Dict[str, Any]]:
        """Get permission information"""
        if permission_name not in self.permissions:
            return None
        
        permission = self.permissions[permission_name]
        return {
            'name': permission.name,
            'resource_type': permission.resource_type.value,
            'actions': [action.value for action in permission.actions],
            'description': permission.description,
            'conditions': permission.conditions
        }
    
    def audit_user_access(self, user_id: str) -> Dict[str, Any]:
        """Generate audit report for user access"""
        return {
            'user_id': user_id,
            'roles': list(self.get_user_roles(user_id)),
            'permissions': list(self.get_user_permissions(user_id)),
            'owned_resources': [
                resource_id for resource_id, owner_id in self.resource_ownership.items()
                if owner_id == user_id
            ],
            'audit_timestamp': datetime.now().isoformat()
        }