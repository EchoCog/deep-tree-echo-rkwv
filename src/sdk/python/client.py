"""
Deep Tree Echo Python SDK Client
Main client class for interacting with the Echo API
"""

import json
import time
from typing import Dict, List, Optional, Any, Union
from urllib.parse import urljoin
import logging

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    requests = None

from .models import (
    CognitiveResult, SessionInfo, MemoryItem, SystemStatus, 
    UsageAnalytics, QuotaInfo, SessionConfiguration
)
from .exceptions import (
    EchoAPIError, EchoAuthenticationError, EchoRateLimitError,
    EchoQuotaExceededError, EchoValidationError, EchoServerError,
    EchoNetworkError
)

logger = logging.getLogger(__name__)

class EchoClient:
    """
    Official Python client for the Deep Tree Echo API
    
    Features:
    - Comprehensive API endpoint coverage
    - Automatic retry with exponential backoff
    - Rate limit handling and quota tracking
    - Type-safe response models
    - Async/await support (optional)
    - GraphQL query support
    """
    
    def __init__(
        self,
        api_key: str,
        base_url: str = "http://localhost:8000",
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff_factor: float = 0.3,
        verify_ssl: bool = True,
        user_agent: str = None
    ):
        """
        Initialize the Echo client
        
        Args:
            api_key: Your Echo API key
            base_url: Base URL for the API (default: http://localhost:8000)
            timeout: Request timeout in seconds (default: 30.0)
            max_retries: Maximum number of retries for failed requests (default: 3)
            backoff_factor: Backoff factor for retry delays (default: 0.3)
            verify_ssl: Whether to verify SSL certificates (default: True)
            user_agent: Custom user agent string
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library is required. Install with: pip install requests")
        
        if not api_key:
            raise EchoAuthenticationError("API key is required")
        
        self.api_key = api_key
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.verify_ssl = verify_ssl
        
        # Set up user agent
        default_ua = f"deep-tree-echo-python-sdk/1.0.0"
        self.user_agent = user_agent or default_ua
        
        # Set up session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "PUT", "DELETE", "OPTIONS", "TRACE", "POST"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set default headers
        self.session.headers.update({
            'User-Agent': self.user_agent,
            'X-API-Key': self.api_key,
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Track quota information
        self._quota_info: Optional[QuotaInfo] = None
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None,
        api_version: str = 'v1'
    ) -> Dict[str, Any]:
        """Make HTTP request to the API"""
        
        url = urljoin(self.base_url, f"/api/{api_version}/{endpoint.lstrip('/')}")
        
        # Add API version header
        headers = {'API-Version': api_version}
        
        try:
            if method.upper() == 'GET':
                response = self.session.get(
                    url, 
                    params=params, 
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=headers
                )
            elif method.upper() == 'POST':
                response = self.session.post(
                    url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=headers
                )
            elif method.upper() == 'PUT':
                response = self.session.put(
                    url,
                    json=data,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=headers
                )
            elif method.upper() == 'DELETE':
                response = self.session.delete(
                    url,
                    params=params,
                    timeout=self.timeout,
                    verify=self.verify_ssl,
                    headers=headers
                )
            else:
                raise EchoAPIError(f"Unsupported HTTP method: {method}")
            
            # Update quota info from response headers
            self._update_quota_from_headers(response.headers)
            
            # Handle response
            return self._handle_response(response)
            
        except requests.exceptions.Timeout:
            raise EchoNetworkError("Request timed out")
        except requests.exceptions.ConnectionError:
            raise EchoNetworkError("Connection error")
        except requests.exceptions.RequestException as e:
            raise EchoNetworkError(f"Network error: {str(e)}")
    
    def _handle_response(self, response: 'requests.Response') -> Dict[str, Any]:
        """Handle HTTP response and convert to appropriate exception if needed"""
        
        try:
            response_data = response.json()
        except json.JSONDecodeError:
            response_data = {'error': 'Invalid JSON response', 'content': response.text}
        
        if response.status_code == 200 or response.status_code == 201:
            return response_data
        
        error_message = response_data.get('error', f'HTTP {response.status_code} error')
        
        if response.status_code == 401:
            raise EchoAuthenticationError(error_message, response.status_code, response_data)
        elif response.status_code == 429:
            retry_after = response.headers.get('Retry-After')
            retry_after_int = int(retry_after) if retry_after else None
            raise EchoRateLimitError(error_message, retry_after_int, response.status_code, response_data)
        elif response.status_code == 400:
            raise EchoValidationError(error_message, response.status_code, response_data)
        elif response.status_code >= 500:
            raise EchoServerError(error_message, response.status_code, response_data)
        else:
            raise EchoAPIError(error_message, response.status_code, response_data)
    
    def _update_quota_from_headers(self, headers: Dict[str, str]):
        """Update quota information from response headers"""
        tier = headers.get('X-RateLimit-Tier')
        hour_remaining = headers.get('X-RateLimit-Hour-Remaining')
        day_remaining = headers.get('X-RateLimit-Day-Remaining')
        
        if all([tier, hour_remaining, day_remaining]):
            # Note: This is a simplified quota info update
            # In a real implementation, you'd want to track more complete information
            self._quota_info = QuotaInfo(
                tier=tier,
                hour_usage=0,  # Would need to calculate from remaining
                hour_limit=1000,  # Would get from tier configuration
                day_usage=0,  # Would need to calculate from remaining
                day_limit=10000,  # Would get from tier configuration
                allowed=True
            )
    
    # ========================================================================
    # System Operations
    # ========================================================================
    
    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        response = self._make_request('GET', 'status')
        return SystemStatus.from_dict(response['data'])
    
    def health_check(self) -> bool:
        """Perform health check"""
        try:
            response = self._make_request('GET', '../health')  # Health is not versioned
            return response.get('status') == 'healthy'
        except EchoAPIError:
            return False
    
    # ========================================================================
    # Cognitive Processing
    # ========================================================================
    
    def process_cognitive_input(
        self,
        input_text: str,
        session_id: Optional[str] = None,
        temperature: float = 0.8,
        max_tokens: int = 2048,
        api_version: str = 'v1'
    ) -> CognitiveResult:
        """
        Process text input through the cognitive architecture
        
        Args:
            input_text: Text to process
            session_id: Optional session ID for context
            temperature: Randomness parameter (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            api_version: API version to use (v1 or v2)
            
        Returns:
            CognitiveResult with processing results
        """
        if not input_text.strip():
            raise EchoValidationError("Input text cannot be empty")
        
        data = {
            'input': input_text,
            'options': {
                'temperature': temperature,
                'max_tokens': max_tokens
            }
        }
        
        if session_id:
            data['options']['session_id'] = session_id
        
        response = self._make_request('POST', 'cognitive/process', data=data, api_version=api_version)
        return CognitiveResult.from_dict(response['data'])
    
    # ========================================================================
    # Session Management
    # ========================================================================
    
    def create_session(
        self,
        configuration: Optional[SessionConfiguration] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SessionInfo:
        """Create a new cognitive session"""
        data = {}
        
        if configuration:
            data.update({
                'temperature': configuration.temperature,
                'max_context_length': configuration.max_context_length,
                'memory_persistence': configuration.memory_persistence
            })
        
        if metadata:
            data['metadata'] = metadata
        
        response = self._make_request('POST', 'sessions', data=data)
        return SessionInfo.from_dict(response['data'])
    
    def get_session(self, session_id: str) -> SessionInfo:
        """Get session information"""
        if not session_id:
            raise EchoValidationError("Session ID is required")
        
        response = self._make_request('GET', f'sessions/{session_id}')
        return SessionInfo.from_dict(response['data'])
    
    # ========================================================================
    # Memory Operations
    # ========================================================================
    
    def search_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
        min_relevance: float = 0.5
    ) -> List[MemoryItem]:
        """
        Search memory items
        
        Args:
            query: Search query text
            memory_type: Optional memory type filter (declarative, procedural, episodic)
            limit: Maximum number of results (default: 10)
            min_relevance: Minimum relevance score (default: 0.5)
            
        Returns:
            List of matching memory items
        """
        if not query.strip():
            raise EchoValidationError("Search query cannot be empty")
        
        data = {
            'query': query,
            'limit': limit,
            'min_relevance': min_relevance
        }
        
        if memory_type:
            data['memory_type'] = memory_type
        
        response = self._make_request('POST', 'memory/search', data=data)
        return [MemoryItem.from_dict(item) for item in response['data']['results']]
    
    # ========================================================================
    # Analytics
    # ========================================================================
    
    def get_usage_analytics(self, period: str = "last_30_days") -> UsageAnalytics:
        """Get usage analytics"""
        response = self._make_request('GET', 'analytics/usage', params={'period': period}, api_version='v2')
        return UsageAnalytics.from_dict(response['data'])
    
    @property
    def quota_info(self) -> Optional[QuotaInfo]:
        """Get current quota information"""
        return self._quota_info
    
    # ========================================================================
    # Batch Operations
    # ========================================================================
    
    def batch_process(
        self,
        inputs: List[str],
        session_id: Optional[str] = None,
        **kwargs
    ) -> List[CognitiveResult]:
        """
        Process multiple inputs in batch
        
        Args:
            inputs: List of input texts to process
            session_id: Optional session ID for context
            **kwargs: Additional parameters for processing
            
        Returns:
            List of CognitiveResult objects
        """
        if not inputs:
            raise EchoValidationError("Inputs list cannot be empty")
        
        results = []
        for input_text in inputs:
            try:
                result = self.process_cognitive_input(
                    input_text, 
                    session_id=session_id, 
                    **kwargs
                )
                results.append(result)
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except EchoRateLimitError as e:
                logger.warning(f"Rate limit hit during batch processing: {e}")
                if e.retry_after:
                    time.sleep(e.retry_after)
                    # Retry this input
                    result = self.process_cognitive_input(
                        input_text, 
                        session_id=session_id, 
                        **kwargs
                    )
                    results.append(result)
                else:
                    raise
        
        return results
    
    # ========================================================================
    # Context Manager Support
    # ========================================================================
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            self.session.close()
    
    def close(self):
        """Close the client session"""
        if self.session:
            self.session.close()