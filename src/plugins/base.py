"""
Plugin Architecture System
Extensible plugin system for Deep Tree Echo cognitive architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime
import logging
import inspect
import importlib
import pkgutil
from pathlib import Path
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PluginType(Enum):
    """Types of plugins supported"""
    COGNITIVE_PROCESSOR = "cognitive_processor"
    MEMORY_PROVIDER = "memory_provider"
    INTEGRATION = "integration"
    MIDDLEWARE = "middleware"
    ANALYZER = "analyzer"
    TRANSFORMER = "transformer"
    VALIDATOR = "validator"
    EXTENSION = "extension"

class PluginStatus(Enum):
    """Plugin status states"""
    LOADED = "loaded"
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    DISABLED = "disabled"

@dataclass
class PluginMetadata:
    """Plugin metadata information"""
    name: str
    version: str
    description: str
    author: str
    plugin_type: PluginType
    dependencies: List[str] = field(default_factory=list)
    api_version: str = "1.0.0"
    license: str = "MIT"
    homepage: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    min_echo_version: str = "1.0.0"
    max_echo_version: Optional[str] = None

@dataclass
class PluginConfig:
    """Plugin configuration"""
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    priority: int = 50  # Lower numbers = higher priority
    auto_load: bool = True
    permissions: List[str] = field(default_factory=list)

class PluginHook:
    """Decorator for plugin hook points"""
    
    def __init__(self, hook_name: str, priority: int = 50):
        self.hook_name = hook_name
        self.priority = priority
    
    def __call__(self, func: Callable) -> Callable:
        func._plugin_hook = self.hook_name
        func._hook_priority = self.priority
        return func

class BasePlugin(ABC):
    """Base class for all plugins"""
    
    def __init__(self, config: PluginConfig):
        self.config = config
        self.status = PluginStatus.LOADED
        self.metadata: Optional[PluginMetadata] = None
        self.last_error: Optional[str] = None
        self.created_at = datetime.now()
        self.stats = {
            'activations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'last_activity': None
        }
    
    @abstractmethod
    def get_metadata(self) -> PluginMetadata:
        """Return plugin metadata"""
        pass
    
    @abstractmethod
    async def initialize(self, echo_system: Any) -> bool:
        """Initialize the plugin with the Echo system"""
        pass
    
    @abstractmethod
    async def activate(self) -> bool:
        """Activate the plugin"""
        pass
    
    @abstractmethod
    async def deactivate(self) -> bool:
        """Deactivate the plugin"""
        pass
    
    async def cleanup(self) -> None:
        """Cleanup plugin resources"""
        pass
    
    def update_stats(self, operation_success: bool = True):
        """Update plugin statistics"""
        self.stats['last_activity'] = datetime.now()
        
        if operation_success:
            self.stats['successful_operations'] += 1
        else:
            self.stats['failed_operations'] += 1
    
    def get_info(self) -> Dict[str, Any]:
        """Get plugin information"""
        metadata = self.get_metadata()
        return {
            'name': metadata.name,
            'version': metadata.version,
            'description': metadata.description,
            'author': metadata.author,
            'type': metadata.plugin_type.value,
            'status': self.status.value,
            'enabled': self.config.enabled,
            'priority': self.config.priority,
            'created_at': self.created_at.isoformat(),
            'last_error': self.last_error,
            'stats': self.stats,
            'dependencies': metadata.dependencies,
            'tags': metadata.tags
        }

class CognitiveProcessorPlugin(BasePlugin):
    """Plugin for extending cognitive processing capabilities"""
    
    @abstractmethod
    async def process_input(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process cognitive input"""
        pass
    
    @abstractmethod
    def get_supported_operations(self) -> List[str]:
        """Get list of supported operations"""
        pass

class MemoryProviderPlugin(BasePlugin):
    """Plugin for extending memory capabilities"""
    
    @abstractmethod
    async def store_memory(self, content: str, memory_type: str, metadata: Dict[str, Any]) -> str:
        """Store memory item"""
        pass
    
    @abstractmethod
    async def retrieve_memory(self, memory_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve memory item"""
        pass
    
    @abstractmethod
    async def search_memory(self, query: str, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search memory items"""
        pass

class MiddlewarePlugin(BasePlugin):
    """Plugin for request/response middleware"""
    
    @abstractmethod
    async def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process incoming request"""
        pass
    
    @abstractmethod
    async def process_response(self, response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process outgoing response"""
        pass

class PluginRegistry:
    """Registry for managing plugins"""
    
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self.plugin_types: Dict[PluginType, List[str]] = {
            plugin_type: [] for plugin_type in PluginType
        }
        self.hooks: Dict[str, List[Callable]] = {}
        self.plugin_paths: List[Path] = []
    
    def add_plugin_path(self, path: Union[str, Path]) -> None:
        """Add a path to search for plugins"""
        plugin_path = Path(path)
        if plugin_path.exists() and plugin_path.is_dir():
            self.plugin_paths.append(plugin_path)
            logger.info(f"Added plugin path: {plugin_path}")
        else:
            logger.warning(f"Plugin path does not exist: {plugin_path}")
    
    def register_plugin(self, plugin: BasePlugin) -> bool:
        """Register a plugin instance"""
        try:
            metadata = plugin.get_metadata()
            
            if metadata.name in self.plugins:
                logger.warning(f"Plugin '{metadata.name}' is already registered")
                return False
            
            plugin.metadata = metadata
            self.plugins[metadata.name] = plugin
            self.plugin_types[metadata.plugin_type].append(metadata.name)
            
            # Register plugin hooks
            self._register_plugin_hooks(plugin)
            
            logger.info(f"Registered plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error registering plugin: {e}")
            return False
    
    def unregister_plugin(self, name: str) -> bool:
        """Unregister a plugin"""
        if name not in self.plugins:
            logger.warning(f"Plugin '{name}' is not registered")
            return False
        
        try:
            plugin = self.plugins[name]
            
            # Deactivate if active
            if plugin.status == PluginStatus.ACTIVE:
                asyncio.create_task(plugin.deactivate())
            
            # Cleanup
            asyncio.create_task(plugin.cleanup())
            
            # Remove from registry
            plugin_type = plugin.metadata.plugin_type
            if name in self.plugin_types[plugin_type]:
                self.plugin_types[plugin_type].remove(name)
            
            # Unregister hooks
            self._unregister_plugin_hooks(plugin)
            
            del self.plugins[name]
            
            logger.info(f"Unregistered plugin: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unregistering plugin '{name}': {e}")
            return False
    
    def _register_plugin_hooks(self, plugin: BasePlugin) -> None:
        """Register hooks from a plugin"""
        for method_name in dir(plugin):
            method = getattr(plugin, method_name)
            if hasattr(method, '_plugin_hook'):
                hook_name = method._plugin_hook
                priority = getattr(method, '_hook_priority', 50)
                
                if hook_name not in self.hooks:
                    self.hooks[hook_name] = []
                
                # Insert maintaining priority order (lower numbers first)
                inserted = False
                for i, (existing_priority, existing_method) in enumerate(self.hooks[hook_name]):
                    if priority < existing_priority:
                        self.hooks[hook_name].insert(i, (priority, method))
                        inserted = True
                        break
                
                if not inserted:
                    self.hooks[hook_name].append((priority, method))
                
                logger.debug(f"Registered hook '{hook_name}' from plugin '{plugin.metadata.name}'")
    
    def _unregister_plugin_hooks(self, plugin: BasePlugin) -> None:
        """Unregister hooks from a plugin"""
        for hook_name, hook_methods in self.hooks.items():
            self.hooks[hook_name] = [
                (priority, method) for priority, method in hook_methods
                if not (hasattr(method, '__self__') and method.__self__ == plugin)
            ]
    
    async def load_plugins_from_path(self, path: Path) -> Dict[str, bool]:
        """Load plugins from a directory"""
        results = {}
        
        if not path.exists() or not path.is_dir():
            logger.warning(f"Plugin path does not exist: {path}")
            return results
        
        # Look for Python plugin files
        for plugin_file in path.glob("*.py"):
            if plugin_file.name.startswith("__"):
                continue
            
            try:
                # Import the plugin module
                spec = importlib.util.spec_from_file_location(
                    plugin_file.stem, plugin_file
                )
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for plugin classes
                for name, obj in inspect.getmembers(module, inspect.isclass):
                    if (issubclass(obj, BasePlugin) and 
                        obj != BasePlugin and 
                        not inspect.isabstract(obj)):
                        
                        try:
                            # Create plugin instance with default config
                            plugin_config = PluginConfig()
                            plugin_instance = obj(plugin_config)
                            
                            success = self.register_plugin(plugin_instance)
                            results[f"{plugin_file.stem}.{name}"] = success
                            
                        except Exception as e:
                            logger.error(f"Error creating plugin instance '{name}': {e}")
                            results[f"{plugin_file.stem}.{name}"] = False
                            
            except Exception as e:
                logger.error(f"Error loading plugin file '{plugin_file}': {e}")
                results[str(plugin_file)] = False
        
        return results
    
    async def initialize_all_plugins(self, echo_system: Any) -> Dict[str, bool]:
        """Initialize all registered plugins"""
        results = {}
        
        # Sort plugins by priority for initialization
        sorted_plugins = sorted(
            self.plugins.items(),
            key=lambda x: x[1].config.priority
        )
        
        for name, plugin in sorted_plugins:
            if plugin.config.enabled and plugin.config.auto_load:
                try:
                    success = await plugin.initialize(echo_system)
                    results[name] = success
                    
                    if success:
                        plugin.status = PluginStatus.INACTIVE
                        logger.info(f"Initialized plugin: {name}")
                    else:
                        plugin.status = PluginStatus.ERROR
                        logger.error(f"Failed to initialize plugin: {name}")
                        
                except Exception as e:
                    logger.error(f"Error initializing plugin '{name}': {e}")
                    plugin.last_error = str(e)
                    plugin.status = PluginStatus.ERROR
                    results[name] = False
            else:
                results[name] = False
        
        return results
    
    async def activate_plugin(self, name: str) -> bool:
        """Activate a specific plugin"""
        if name not in self.plugins:
            logger.error(f"Plugin '{name}' not found")
            return False
        
        plugin = self.plugins[name]
        
        if plugin.status != PluginStatus.INACTIVE:
            logger.warning(f"Plugin '{name}' is not in inactive state")
            return False
        
        try:
            success = await plugin.activate()
            
            if success:
                plugin.status = PluginStatus.ACTIVE
                plugin.stats['activations'] += 1
                logger.info(f"Activated plugin: {name}")
            else:
                plugin.status = PluginStatus.ERROR
                logger.error(f"Failed to activate plugin: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error activating plugin '{name}': {e}")
            plugin.last_error = str(e)
            plugin.status = PluginStatus.ERROR
            return False
    
    async def deactivate_plugin(self, name: str) -> bool:
        """Deactivate a specific plugin"""
        if name not in self.plugins:
            logger.error(f"Plugin '{name}' not found")
            return False
        
        plugin = self.plugins[name]
        
        if plugin.status != PluginStatus.ACTIVE:
            logger.warning(f"Plugin '{name}' is not active")
            return False
        
        try:
            success = await plugin.deactivate()
            
            if success:
                plugin.status = PluginStatus.INACTIVE
                logger.info(f"Deactivated plugin: {name}")
            else:
                plugin.status = PluginStatus.ERROR
                logger.error(f"Failed to deactivate plugin: {name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error deactivating plugin '{name}': {e}")
            plugin.last_error = str(e)
            plugin.status = PluginStatus.ERROR
            return False
    
    async def call_hook(self, hook_name: str, *args, **kwargs) -> List[Any]:
        """Call all registered hooks for a given hook name"""
        results = []
        
        if hook_name in self.hooks:
            for priority, method in self.hooks[hook_name]:
                try:
                    if inspect.iscoroutinefunction(method):
                        result = await method(*args, **kwargs)
                    else:
                        result = method(*args, **kwargs)
                    
                    results.append(result)
                    
                except Exception as e:
                    logger.error(f"Error calling hook '{hook_name}': {e}")
                    results.append(None)
        
        return results
    
    def get_plugins_by_type(self, plugin_type: PluginType) -> List[BasePlugin]:
        """Get all plugins of a specific type"""
        plugin_names = self.plugin_types.get(plugin_type, [])
        return [self.plugins[name] for name in plugin_names if name in self.plugins]
    
    def get_active_plugins(self) -> List[BasePlugin]:
        """Get all active plugins"""
        return [
            plugin for plugin in self.plugins.values()
            if plugin.status == PluginStatus.ACTIVE
        ]
    
    def get_plugin_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all plugins"""
        return {
            name: plugin.get_info()
            for name, plugin in self.plugins.items()
        }
    
    async def cleanup_all_plugins(self) -> None:
        """Cleanup all plugins"""
        for plugin in self.plugins.values():
            try:
                if plugin.status == PluginStatus.ACTIVE:
                    await plugin.deactivate()
                await plugin.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up plugin '{plugin.metadata.name}': {e}")

# Global plugin registry instance
plugin_registry = PluginRegistry()

# Decorator for creating simple plugins
def echo_plugin(name: str, version: str, description: str, plugin_type: PluginType):
    """Decorator to create a simple plugin from a class"""
    def decorator(cls):
        class WrappedPlugin(BasePlugin):
            def get_metadata(self) -> PluginMetadata:
                return PluginMetadata(
                    name=name,
                    version=version,
                    description=description,
                    author="Unknown",
                    plugin_type=plugin_type
                )
            
            async def initialize(self, echo_system: Any) -> bool:
                if hasattr(cls, 'initialize'):
                    return await cls.initialize(self, echo_system)
                return True
            
            async def activate(self) -> bool:
                if hasattr(cls, 'activate'):
                    return await cls.activate(self)
                return True
            
            async def deactivate(self) -> bool:
                if hasattr(cls, 'deactivate'):
                    return await cls.deactivate(self)
                return True
        
        # Copy methods from original class
        for attr_name in dir(cls):
            if not attr_name.startswith('_') and attr_name not in ['initialize', 'activate', 'deactivate']:
                attr = getattr(cls, attr_name)
                if callable(attr):
                    setattr(WrappedPlugin, attr_name, attr)
        
        return WrappedPlugin
    
    return decorator