"""
Deep Tree Echo SDK Exceptions
Custom exception classes for the SDK
"""

class EchoAPIError(Exception):
    """Base exception for all Echo API errors"""
    
    def __init__(self, message: str, status_code: int = None, response: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response = response or {}
    
    def __str__(self):
        if self.status_code:
            return f"HTTP {self.status_code}: {self.message}"
        return self.message

class EchoAuthenticationError(EchoAPIError):
    """Authentication related errors"""
    pass

class EchoRateLimitError(EchoAPIError):
    """Rate limiting errors"""
    
    def __init__(self, message: str, retry_after: int = None, **kwargs):
        super().__init__(message, **kwargs)
        self.retry_after = retry_after

class EchoQuotaExceededError(EchoRateLimitError):
    """Quota exceeded errors"""
    pass

class EchoValidationError(EchoAPIError):
    """Input validation errors"""
    pass

class EchoServerError(EchoAPIError):
    """Server-side errors"""
    pass

class EchoNetworkError(EchoAPIError):
    """Network connectivity errors"""
    pass