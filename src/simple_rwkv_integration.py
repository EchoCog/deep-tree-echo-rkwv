"""
Simple RWKV Integration

Simplified RWKV integration using pip package for easy setup and usage
within the Deep Tree Echo framework.
"""

import logging
import os
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Try to import the RWKV package
try:
    from rwkv.model import RWKV
    from rwkv.utils import PIPELINE, PIPELINE_ARGS
    RWKV_AVAILABLE = True
    logger.info("RWKV package available via pip install")
except ImportError:
    RWKV_AVAILABLE = False
    logger.warning("RWKV package not available. Install with: pip install rwkv")
    # Create mock classes for fallback
    class RWKV:
        def __init__(self, *args, **kwargs):
            pass
    class PIPELINE:
        def __init__(self, *args, **kwargs):
            pass
    class PIPELINE_ARGS:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

@dataclass
class SimpleRWKVConfig:
    """Configuration for simple RWKV integration"""
    model_path: str = ""
    strategy: str = "cpu fp32"
    chunk_len: int = 256
    max_tokens: int = 200
    temperature: float = 1.0
    top_p: float = 0.9
    alpha_frequency: float = 0.1
    alpha_presence: float = 0.1
    token_ban: List[int] = None
    token_stop: List[int] = None

class SimpleRWKVIntegration:
    """
    Simplified RWKV integration using the pip-installable RWKV package.
    Provides easy setup and usage for basic RWKV functionality.
    """
    
    def __init__(self, config: Optional[SimpleRWKVConfig] = None):
        """Initialize the simple RWKV integration"""
        self.config = config or SimpleRWKVConfig()
        self.model = None
        self.pipeline = None
        self._is_initialized = False
        
    def initialize(self, model_path: Optional[str] = None) -> bool:
        """Initialize the RWKV model and pipeline"""
        try:
            # Use provided model path or config model path
            model_path = model_path or self.config.model_path
            
            if not RWKV_AVAILABLE:
                logger.warning("RWKV package not available. Using mock mode.")
                self._is_initialized = True
                return True
            
            if not model_path:
                logger.warning("No model path provided. Using mock initialization.")
                self._is_initialized = True
                return True
            
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                # Still initialize in mock mode
                self._is_initialized = True
                return True
                
            # Initialize RWKV model
            self.model = RWKV(
                model=model_path,
                strategy=self.config.strategy,
                chunk_len=self.config.chunk_len
            )
            
            # Initialize pipeline
            self.pipeline = PIPELINE(self.model, "20B_tokenizer.json")
            
            self._is_initialized = True
            logger.info(f"Successfully initialized RWKV model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RWKV model: {e}")
            # Fall back to mock mode
            self._is_initialized = True
            return True
    
    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """Generate text using the RWKV model"""
        if not self.is_available():
            logger.warning("RWKV integration not available")
            return self._mock_generate(prompt)
        
        try:
            # Merge kwargs with config defaults
            generation_args = PIPELINE_ARGS(
                temperature=kwargs.get('temperature', self.config.temperature),
                top_p=kwargs.get('top_p', self.config.top_p),
                alpha_frequency=kwargs.get('alpha_frequency', self.config.alpha_frequency),
                alpha_presence=kwargs.get('alpha_presence', self.config.alpha_presence),
                token_ban=kwargs.get('token_ban', self.config.token_ban or []),
                token_stop=kwargs.get('token_stop', self.config.token_stop or []),
                chunk_len=self.config.chunk_len
            )
            
            # Generate text
            result = self.pipeline.generate(
                prompt,
                token_count=kwargs.get('max_tokens', self.config.max_tokens),
                args=generation_args
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return self._mock_generate(prompt)
    
    def _mock_generate(self, prompt: str) -> str:
        """Mock text generation for fallback"""
        return f"[Mock RWKV Response] Echo: {prompt[:50]}... (RWKV not fully initialized)"
    
    def is_available(self) -> bool:
        """Check if the RWKV integration is available"""
        if not self._is_initialized:
            return False
        # In mock mode (when RWKV package not available), we're still available for mock responses
        if not RWKV_AVAILABLE:
            return True
        # When RWKV package is available, check if model is loaded
        return self.model is not None
    
    def get_info(self) -> Dict[str, Any]:
        """Get information about the RWKV integration"""
        return {
            'backend': 'simple_rwkv',
            'available': self.is_available(),
            'rwkv_package_available': RWKV_AVAILABLE,
            'model_loaded': self.model is not None,
            'model_path': self.config.model_path,
            'strategy': self.config.strategy,
            'chunk_len': self.config.chunk_len,
            'version': self.get_version()
        }
    
    def get_version(self) -> str:
        """Get version information"""
        version = "Simple RWKV Integration 1.0.0"
        if RWKV_AVAILABLE:
            try:
                # Try to get RWKV package version
                import rwkv
                if hasattr(rwkv, '__version__'):
                    version += f" (RWKV {rwkv.__version__})"
                else:
                    version += " (RWKV package installed)"
            except Exception:
                version += " (RWKV version unknown)"
        else:
            version += " (RWKV package not available)"
        return version
    
    def chat(self, message: str, context: Optional[str] = None) -> str:
        """Simple chat interface"""
        if context:
            prompt = f"{context}\n\nUser: {message}\n\nAssistant:"
        else:
            prompt = f"User: {message}\n\nAssistant:"
            
        response = self.generate(prompt)
        
        # Clean up the response to extract just the assistant's reply
        if response and "Assistant:" in response:
            response = response.split("Assistant:")[-1].strip()
        
        return response or "I couldn't generate a response."
    
    def reset(self):
        """Reset the model state"""
        if self.model and hasattr(self.model, 'clear'):
            try:
                self.model.clear()
                logger.info("Model state reset")
            except Exception as e:
                logger.warning(f"Failed to reset model state: {e}")

# Factory function for easy instantiation
def create_simple_rwkv(model_path: Optional[str] = None, **kwargs) -> SimpleRWKVIntegration:
    """Create and initialize a simple RWKV integration"""
    config = SimpleRWKVConfig(model_path=model_path or "", **kwargs)
    integration = SimpleRWKVIntegration(config)
    integration.initialize()
    return integration

# Default instance for quick usage
default_rwkv = None

def get_default_rwkv() -> SimpleRWKVIntegration:
    """Get the default RWKV integration instance"""
    global default_rwkv
    if default_rwkv is None:
        default_rwkv = create_simple_rwkv()
    return default_rwkv