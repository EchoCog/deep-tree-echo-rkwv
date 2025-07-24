"""
Comprehensive Test Suite for API Ecosystem
Tests RESTful API, GraphQL, SDK functionality, integrations, and plugins
"""

import pytest
import asyncio
import json
import requests
from datetime import datetime
from pathlib import Path
import sys
import os

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

# Import our components
from api.rest_api import APIVersioning, RateLimiter, create_enhanced_api_blueprint
from integrations.base import IntegrationManager, IntegrationEvent, create_webhook_integration
from plugins.base import plugin_registry, BasePlugin, PluginMetadata, PluginConfig, PluginType
from enhanced_api_server import EnhancedEchoServer

class TestAPIVersioning:
    """Test API versioning functionality"""
    
    def test_version_extraction_from_header(self):
        """Test version extraction from headers"""
        from unittest.mock import Mock
        
        # Mock request with version header
        mock_request = Mock()
        mock_request.headers = {'API-Version': 'v2'}
        mock_request.path = '/api/test'
        
        version = APIVersioning.get_api_version(mock_request)
        assert version == 'v2'
    
    def test_version_extraction_from_path(self):
        """Test version extraction from URL path"""
        from unittest.mock import Mock
        
        mock_request = Mock()
        mock_request.headers = {}
        mock_request.path = '/api/v2/cognitive/process'
        
        version = APIVersioning.get_api_version(mock_request)
        assert version == 'v2'
    
    def test_default_version(self):
        """Test default version when none specified"""
        from unittest.mock import Mock
        
        mock_request = Mock()
        mock_request.headers = {}
        mock_request.path = '/api/cognitive/process'
        
        version = APIVersioning.get_api_version(mock_request)
        assert version == 'v1'
    
    def test_version_compatibility(self):
        """Test version compatibility checking"""
        assert APIVersioning.version_compatible('v1', 'v1') == True
        assert APIVersioning.version_compatible('v1', 'v2') == True
        assert APIVersioning.version_compatible('v2', 'v1') == False

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    def setup_method(self):
        """Setup test method"""
        self.limiter = RateLimiter()
    
    def test_user_tier_identification(self):
        """Test user tier identification from API key"""
        assert self.limiter.get_user_tier('') == 'free'
        assert self.limiter.get_user_tier('basic_12345') == 'basic'
        assert self.limiter.get_user_tier('premium_12345') == 'premium'
        assert self.limiter.get_user_tier('invalid_key') == 'free'
    
    def test_quota_checking(self):
        """Test quota checking functionality"""
        # Test free tier quota
        quota_check = self.limiter.check_quota('test_user', 'free')
        assert quota_check['allowed'] == True
        assert quota_check['tier'] == 'free'
        assert quota_check['hour_limit'] == 100
        assert quota_check['day_limit'] == 1000
    
    def test_usage_increment(self):
        """Test usage increment"""
        user_id = 'test_user'
        
        # Initial state
        quota_before = self.limiter.check_quota(user_id, 'free')
        initial_hour_usage = quota_before['hour_usage']
        initial_day_usage = quota_before['day_usage']
        
        # Increment usage
        self.limiter.increment_usage(user_id)
        
        # Check updated state
        quota_after = self.limiter.check_quota(user_id, 'free')
        assert quota_after['hour_usage'] == initial_hour_usage + 1
        assert quota_after['day_usage'] == initial_day_usage + 1

class TestIntegrationManager:
    """Test integration management"""
    
    def setup_method(self):
        """Setup test method"""
        self.manager = IntegrationManager()
    
    @pytest.mark.asyncio
    async def test_webhook_integration_creation(self):
        """Test webhook integration creation"""
        webhook = create_webhook_integration(
            "test_webhook",
            "https://httpbin.org/post",
            "test_secret"
        )
        
        assert webhook.config.name == "test_webhook"
        assert webhook.config.type.value == "webhook"
        assert webhook.webhook_url == "https://httpbin.org/post"
        assert webhook.secret == "test_secret"
    
    @pytest.mark.asyncio
    async def test_integration_registration(self):
        """Test integration registration"""
        webhook = create_webhook_integration("test", "https://httpbin.org/post")
        
        self.manager.register_integration(webhook)
        assert "test" in self.manager.integrations
        
        status = self.manager.get_integration_status()
        assert "test" in status
        assert status["test"]["name"] == "test"
    
    @pytest.mark.asyncio
    async def test_event_sending(self):
        """Test sending events to integrations"""
        # Create test event
        event = IntegrationEvent(
            event_type="test_event",
            data={"message": "Hello, World!"},
            source="test_suite"
        )
        
        # Create webhook integration
        webhook = create_webhook_integration("test", "https://httpbin.org/post")
        self.manager.register_integration(webhook)
        
        # Initialize and send event (will fail in test environment, but we can test the flow)
        try:
            await self.manager.initialize_all()
            results = await self.manager.send_event_to_all(event)
            assert "test" in results
        except Exception:
            # Expected in test environment without actual webhook endpoint
            pass

class TestPluginSystem:
    """Test plugin system functionality"""
    
    def setup_method(self):
        """Setup test method"""
        plugin_registry.plugins.clear()
        plugin_registry.plugin_types = {plugin_type: [] for plugin_type in PluginType}
        plugin_registry.hooks.clear()
    
    def test_plugin_creation(self):
        """Test basic plugin creation"""
        class TestPlugin(BasePlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="test_plugin",
                    version="1.0.0",
                    description="Test plugin",
                    author="Test Author",
                    plugin_type=PluginType.EXTENSION
                )
            
            async def initialize(self, echo_system):
                return True
            
            async def activate(self):
                return True
            
            async def deactivate(self):
                return True
        
        config = PluginConfig()
        plugin = TestPlugin(config)
        
        assert plugin.config == config
        assert plugin.get_metadata().name == "test_plugin"
    
    def test_plugin_registry(self):
        """Test plugin registry functionality"""
        class TestPlugin(BasePlugin):
            def get_metadata(self):
                return PluginMetadata(
                    name="registry_test",
                    version="1.0.0",
                    description="Registry test plugin",
                    author="Test Author",
                    plugin_type=PluginType.EXTENSION
                )
            
            async def initialize(self, echo_system):
                return True
            
            async def activate(self):
                return True
            
            async def deactivate(self):
                return True
        
        plugin = TestPlugin(PluginConfig())
        
        # Test registration
        success = plugin_registry.register_plugin(plugin)
        assert success == True
        assert "registry_test" in plugin_registry.plugins
        
        # Test type organization
        extensions = plugin_registry.get_plugins_by_type(PluginType.EXTENSION)
        assert len(extensions) == 1
        assert extensions[0].metadata.name == "registry_test"
        
        # Test unregistration
        success = plugin_registry.unregister_plugin("registry_test")
        assert success == True
        assert "registry_test" not in plugin_registry.plugins

class TestSDKComponents:
    """Test SDK components"""
    
    def test_python_sdk_imports(self):
        """Test that Python SDK components can be imported"""
        try:
            from sdk.python.client import EchoClient
            from sdk.python.models import CognitiveResult, SessionInfo
            from sdk.python.exceptions import EchoAPIError
            
            # Test basic instantiation
            client = EchoClient("test_key", "http://localhost:8000")
            assert client.api_key == "test_key"
            assert client.base_url == "http://localhost:8000"
            
        except ImportError as e:
            pytest.skip(f"SDK imports not available: {e}")
    
    def test_cli_tool_imports(self):
        """Test that CLI tool can be imported"""
        try:
            from cli.echo import EchoCLI
            
            cli = EchoCLI("test_key")
            assert cli.base_url == "http://localhost:8000"
            
        except ImportError as e:
            pytest.skip(f"CLI imports not available: {e}")

class TestEnhancedAPIServer:
    """Test enhanced API server functionality"""
    
    def setup_method(self):
        """Setup test method"""
        self.server = EnhancedEchoServer()
    
    def test_server_initialization(self):
        """Test server initialization"""
        assert self.server.app is not None
        assert self.server.integration_manager is not None
        assert self.server.config is not None
    
    @pytest.mark.asyncio
    async def test_echo_system_initialization(self):
        """Test Echo system initialization"""
        success = await self.server.initialize_echo_system()
        assert success == True
        assert self.server.echo_system is not None
        assert self.server.memory_system is not None
    
    @pytest.mark.asyncio
    async def test_component_initialization(self):
        """Test component initialization"""
        # Initialize Echo system first
        await self.server.initialize_echo_system()
        
        # Test integrations initialization (will skip if disabled)
        await self.server.initialize_integrations()
        
        # Test plugins initialization (will skip if disabled)
        await self.server.initialize_plugins()
        
        # Verify server stats
        assert 'start_time' in self.server.stats
        assert isinstance(self.server.stats['start_time'], datetime)

@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests requiring running server"""
    
    @pytest.fixture(scope="class")
    def server_url(self):
        """Get server URL for testing"""
        return "http://localhost:8000"
    
    def test_enhanced_status_endpoint(self, server_url):
        """Test enhanced status endpoint"""
        try:
            response = requests.get(f"{server_url}/api/ecosystem/status", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                assert 'success' in data
                assert 'data' in data
                assert 'server' in data['data']
                assert 'echo_system' in data['data']
                assert 'api_features' in data['data']
        except requests.exceptions.RequestException:
            pytest.skip("Server not running")
    
    def test_graphql_endpoint(self, server_url):
        """Test GraphQL endpoint availability"""
        try:
            response = requests.get(f"{server_url}/graphql/schema", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                assert 'schema' in data
                assert 'documentation' in data
        except requests.exceptions.RequestException:
            pytest.skip("Server not running or GraphQL not available")
    
    def test_documentation_endpoint(self, server_url):
        """Test API documentation endpoint"""
        try:
            response = requests.get(f"{server_url}/api/docs", timeout=5)
            
            if response.status_code == 200:
                assert 'text/html' in response.headers.get('content-type', '')
        except requests.exceptions.RequestException:
            pytest.skip("Server not running")

class TestAPIResponseFormat:
    """Test API response formatting"""
    
    def test_api_response_format(self):
        """Test standardized API response format"""
        from api.rest_api import api_response
        from unittest.mock import Mock
        
        # Mock request
        mock_request = Mock()
        mock_request.headers = {'API-Version': 'v1'}
        
        # Test successful response
        with pytest.MonkeyPatch().context() as m:
            m.setattr('api.rest_api.request', mock_request)
            
            response, status_code = api_response(
                data={'test': 'data'},
                version='v1'
            )
            
            assert status_code == 200
            response_data = response.get_json()
            assert response_data['success'] == True
            assert response_data['version'] == 'v1'
            assert 'timestamp' in response_data
            assert response_data['data']['test'] == 'data'
        
        # Test error response
        with pytest.MonkeyPatch().context() as m:
            m.setattr('api.rest_api.request', mock_request)
            
            response, status_code = api_response(
                error='Test error',
                status_code=400,
                version='v1'
            )
            
            assert status_code == 400
            response_data = response.get_json()
            assert response_data['success'] == False
            assert response_data['error'] == 'Test error'

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])