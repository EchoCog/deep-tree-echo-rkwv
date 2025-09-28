"""
Namespace Management System
Manages nested namespaces for agents and arenas with hierarchical permissions
"""

import logging
import time
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum

from .core import PermissionLevel

logger = logging.getLogger(__name__)

class NamespaceType(Enum):
    """Types of namespaces"""
    AGENT = "agent"
    ARENA = "arena"
    MEMORY = "memory"
    SYSTEM = "system"

@dataclass
class Namespace:
    """Represents a namespace with hierarchical structure"""
    name: str
    namespace_type: NamespaceType
    parent: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    permissions: Set[PermissionLevel] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    children: Set[str] = field(default_factory=set)
    active: bool = True
    
    def get_full_path(self) -> str:
        """Get the full namespace path"""
        if self.parent:
            return f"{self.parent}.{self.name}"
        return self.name
    
    def is_ancestor_of(self, other_namespace: str) -> bool:
        """Check if this namespace is an ancestor of another"""
        full_path = self.get_full_path()
        return other_namespace.startswith(f"{full_path}.")
    
    def get_depth(self) -> int:
        """Get the depth level of this namespace"""
        if not self.parent:
            return 0
        return self.parent.count('.') + 1

class NamespaceManager:
    """
    Manages the nested namespace system for EchoSpace architecture
    """
    
    def __init__(self):
        self.namespaces: Dict[str, Namespace] = {}
        self._initialize_default_namespaces()
    
    def _initialize_default_namespaces(self):
        """Initialize the default namespace structure"""
        
        # Create root namespaces
        self.create_namespace("ActualMarduk", NamespaceType.AGENT, permissions={
            PermissionLevel.FULL
        })
        
        self.create_namespace("Marduk-Space", NamespaceType.ARENA, permissions={
            PermissionLevel.FULL
        })
        
        self.create_namespace("Marduk-Sandbox", NamespaceType.ARENA, permissions={
            PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE
        })
        
        self.create_namespace("Marduk-Memory", NamespaceType.MEMORY, permissions={
            PermissionLevel.READ, PermissionLevel.WRITE
        })
        
        self.create_namespace("EchoCog", NamespaceType.AGENT, permissions={
            PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE
        })
        
        self.create_namespace("EchoSpace", NamespaceType.ARENA, permissions={
            PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE
        })
        
        logger.info("Default namespaces initialized")
    
    def create_namespace(
        self,
        name: str,
        namespace_type: NamespaceType,
        parent: Optional[str] = None,
        permissions: Optional[Set[PermissionLevel]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Namespace:
        """Create a new namespace"""
        
        full_name = f"{parent}.{name}" if parent else name
        
        if full_name in self.namespaces:
            raise ValueError(f"Namespace already exists: {full_name}")
        
        if parent and parent not in self.namespaces:
            raise ValueError(f"Parent namespace does not exist: {parent}")
        
        namespace = Namespace(
            name=name,
            namespace_type=namespace_type,
            parent=parent,
            permissions=permissions or set(),
            metadata=metadata or {}
        )
        
        self.namespaces[full_name] = namespace
        
        # Update parent's children
        if parent:
            parent_ns = self.namespaces[parent]
            parent_ns.children.add(full_name)
        
        logger.info(f"Namespace created: {full_name} ({namespace_type.value})")
        return namespace
    
    def get_namespace(self, name: str) -> Optional[Namespace]:
        """Get a namespace by name"""
        return self.namespaces.get(name)
    
    def create_virtual_marduk_namespace(self, marduk_id: str) -> Namespace:
        """Create a namespace for a Virtual Marduk"""
        namespace_name = f"VirtualMarduk-{marduk_id}"
        
        return self.create_namespace(
            name=namespace_name,
            namespace_type=NamespaceType.AGENT,
            permissions={PermissionLevel.READ, PermissionLevel.WRITE, PermissionLevel.EXECUTE},
            metadata={
                'marduk_type': 'virtual',
                'marduk_id': marduk_id,
                'sandbox_access': True
            }
        )
    
    def get_virtual_marduk_namespaces(self) -> List[Namespace]:
        """Get all Virtual Marduk namespaces"""
        return [
            ns for ns in self.namespaces.values()
            if (ns.namespace_type == NamespaceType.AGENT and 
                ns.metadata.get('marduk_type') == 'virtual')
        ]
    
    def check_namespace_permission(
        self,
        namespace_name: str,
        permission: PermissionLevel
    ) -> bool:
        """Check if a namespace has a specific permission"""
        namespace = self.get_namespace(namespace_name)
        if not namespace:
            return False
        
        return (permission in namespace.permissions or 
                PermissionLevel.FULL in namespace.permissions)
    
    def get_namespace_hierarchy(self, root: Optional[str] = None) -> Dict[str, Any]:
        """Get the namespace hierarchy tree"""
        
        def build_tree(namespace_name: str) -> Dict[str, Any]:
            namespace = self.namespaces.get(namespace_name)
            if not namespace:
                return {}
            
            tree = {
                'name': namespace.name,
                'type': namespace.namespace_type.value,
                'full_path': namespace.get_full_path(),
                'permissions': [p.value for p in namespace.permissions],
                'created_at': namespace.created_at,
                'active': namespace.active,
                'children': {}
            }
            
            for child_name in namespace.children:
                tree['children'][child_name] = build_tree(child_name)
            
            return tree
        
        if root:
            return build_tree(root)
        
        # Build trees for all root namespaces (those without parents)
        trees = {}
        for name, namespace in self.namespaces.items():
            if not namespace.parent:
                trees[name] = build_tree(name)
        
        return trees
    
    def list_namespaces(
        self,
        namespace_type: Optional[NamespaceType] = None,
        parent: Optional[str] = None,
        active_only: bool = True
    ) -> List[Namespace]:
        """List namespaces with optional filtering"""
        
        namespaces = []
        for namespace in self.namespaces.values():
            # Filter by type
            if namespace_type and namespace.namespace_type != namespace_type:
                continue
            
            # Filter by parent
            if parent and namespace.parent != parent:
                continue
            
            # Filter by active status
            if active_only and not namespace.active:
                continue
            
            namespaces.append(namespace)
        
        return sorted(namespaces, key=lambda ns: ns.get_full_path())
    
    def deactivate_namespace(self, name: str) -> bool:
        """Deactivate a namespace (soft delete)"""
        namespace = self.get_namespace(name)
        if namespace:
            namespace.active = False
            logger.info(f"Namespace deactivated: {name}")
            return True
        return False
    
    def activate_namespace(self, name: str) -> bool:
        """Reactivate a namespace"""
        namespace = self.get_namespace(name)
        if namespace:
            namespace.active = True
            logger.info(f"Namespace activated: {name}")
            return True
        return False
    
    def delete_namespace(self, name: str, force: bool = False) -> bool:
        """
        Delete a namespace permanently.
        Requires force=True for namespaces with children.
        """
        namespace = self.get_namespace(name)
        if not namespace:
            return False
        
        # Check for children
        if namespace.children and not force:
            raise ValueError(f"Cannot delete namespace with children: {name}. Use force=True.")
        
        # Remove from parent's children
        if namespace.parent:
            parent_ns = self.namespaces.get(namespace.parent)
            if parent_ns:
                parent_ns.children.discard(name)
        
        # Delete children if force is True
        if force:
            for child_name in list(namespace.children):
                self.delete_namespace(child_name, force=True)
        
        # Delete the namespace
        del self.namespaces[name]
        logger.info(f"Namespace deleted: {name}")
        return True
    
    def get_stats(self) -> Dict[str, Any]:
        """Get namespace system statistics"""
        
        type_counts = {}
        permission_counts = {}
        depth_counts = {}
        
        for namespace in self.namespaces.values():
            # Count by type
            ns_type = namespace.namespace_type.value
            type_counts[ns_type] = type_counts.get(ns_type, 0) + 1
            
            # Count by permissions
            for perm in namespace.permissions:
                perm_name = perm.value
                permission_counts[perm_name] = permission_counts.get(perm_name, 0) + 1
            
            # Count by depth
            depth = namespace.get_depth()
            depth_counts[depth] = depth_counts.get(depth, 0) + 1
        
        return {
            'total_namespaces': len(self.namespaces),
            'active_namespaces': sum(1 for ns in self.namespaces.values() if ns.active),
            'type_distribution': type_counts,
            'permission_distribution': permission_counts,
            'depth_distribution': depth_counts,
            'virtual_marduks': len(self.get_virtual_marduk_namespaces()),
            'timestamp': time.time()
        }