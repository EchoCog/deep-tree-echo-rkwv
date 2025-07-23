"""
Deep Tree Echo Python SDK
Official Python SDK for the Deep Tree Echo cognitive architecture API
"""

__version__ = "1.0.0"
__author__ = "Deep Tree Echo Team"
__license__ = "MIT"

from .client import EchoClient
from .exceptions import EchoAPIError, EchoAuthenticationError, EchoRateLimitError
from .models import CognitiveResult, SessionInfo, MemoryItem, SystemStatus

__all__ = [
    'EchoClient',
    'EchoAPIError', 
    'EchoAuthenticationError',
    'EchoRateLimitError',
    'CognitiveResult',
    'SessionInfo', 
    'MemoryItem',
    'SystemStatus'
]