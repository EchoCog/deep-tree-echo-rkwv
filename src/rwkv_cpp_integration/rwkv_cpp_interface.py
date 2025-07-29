"""
RWKV C++ Interface

High-performance C++ interface for RWKV model inference within the Deep Tree Echo framework.
"""

import os
import sys
import ctypes
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class RWKVModelConfig:
    """Configuration for RWKV model loading"""
    model_path: str
    threads: int = 4
    gpu_layers: int = 0
    context_length: int = 2048
    vocab_size: int = 65536
    
@dataclass 
class RWKVGenerationConfig:
    """Configuration for RWKV text generation"""
    max_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 40
    alpha_frequency: float = 0.0
    alpha_presence: float = 0.0
    penalty_decay: float = 0.996

class RWKVCppInterface:
    """
    High-performance C++ interface for RWKV model operations.
    Provides optimized inference capabilities for the Deep Tree Echo framework.
    """
    
    def __init__(self, config: Optional[RWKVModelConfig] = None):
        """Initialize the RWKV C++ interface"""
        self.config = config or RWKVModelConfig(model_path="")
        self._lib = None
        self._model_handle = None
        self._context_lock = threading.Lock()
        self._is_initialized = False
        
    def initialize(self) -> bool:
        """Initialize the C++ backend"""
        try:
            if not self._load_library():
                logger.warning("C++ library not available, falling back to Python implementation")
                return False
                
            if self.config.model_path and os.path.exists(self.config.model_path):
                return self._load_model(self.config.model_path)
            else:
                logger.info("No model path specified, C++ interface ready but no model loaded")
                self._is_initialized = True
                return True
                
        except Exception as e:
            logger.error(f"Failed to initialize RWKV C++ interface: {e}")
            return False
    
    def _load_library(self) -> bool:
        """Load the RWKV C++ shared library"""
        try:
            library_path = self._auto_detect_library_path()
            if not library_path or not os.path.exists(library_path):
                logger.warning(f"RWKV C++ library not found at {library_path}")
                return False
                
            self._lib = ctypes.CDLL(library_path)
            self._setup_function_signatures()
            logger.info(f"Successfully loaded RWKV C++ library from {library_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load RWKV C++ library: {e}")
            return False
    
    def _auto_detect_library_path(self) -> Optional[str]:
        """Auto-detect the RWKV C++ library path"""
        # Common library names and locations
        possible_names = [
            'libecho_rwkv_cpp_bridge.so',
            'libecho_rwkv_cpp.so', 
            'librwkv_cpp.so',
            'echo_rwkv_cpp_bridge.dll',
            'echo_rwkv_cpp.dll',
            'rwkv_cpp.dll'
        ]
        
        # Search locations
        search_paths = [
            os.path.dirname(__file__),  # Current directory
            os.path.join(os.path.dirname(__file__), '..', '..', 'build'),  # Build directory
            os.path.join(os.path.dirname(__file__), '..', '..', 'lib'),   # Lib directory
            '/usr/local/lib',
            '/usr/lib',
            os.path.expanduser('~/.local/lib')
        ]
        
        for search_path in search_paths:
            for lib_name in possible_names:
                full_path = os.path.join(search_path, lib_name)
                if os.path.exists(full_path):
                    return full_path
                    
        return None
    
    def _setup_function_signatures(self):
        """Setup C function signatures"""
        if not self._lib:
            return
            
        # Model loading functions
        self._lib.rwkv_load_model.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.rwkv_load_model.restype = ctypes.c_void_p
        
        # Inference functions
        self._lib.rwkv_inference.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
        self._lib.rwkv_inference.restype = ctypes.c_char_p
        
        # Cleanup functions
        self._lib.rwkv_free_model.argtypes = [ctypes.c_void_p]
        self._lib.rwkv_free_model.restype = None
        
    def _load_model(self, model_path: str) -> bool:
        """Load RWKV model"""
        try:
            if not self._lib:
                return False
                
            model_path_bytes = model_path.encode('utf-8')
            self._model_handle = self._lib.rwkv_load_model(model_path_bytes, self.config.threads)
            
            if self._model_handle:
                self._is_initialized = True
                logger.info(f"Successfully loaded RWKV model from {model_path}")
                return True
            else:
                logger.error(f"Failed to load RWKV model from {model_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error loading RWKV model: {e}")
            return False
    
    def generate(self, 
                prompt: str, 
                config: Optional[RWKVGenerationConfig] = None) -> Optional[str]:
        """Generate text using the RWKV model"""
        if not self.is_available():
            logger.warning("RWKV C++ interface not available")
            return None
            
        generation_config = config or RWKVGenerationConfig()
        
        try:
            with self._context_lock:
                if not self._model_handle:
                    logger.error("No model loaded")
                    return None
                    
                prompt_bytes = prompt.encode('utf-8')
                result_ptr = self._lib.rwkv_inference(
                    self._model_handle, 
                    prompt_bytes, 
                    generation_config.max_tokens
                )
                
                if result_ptr:
                    result = ctypes.string_at(result_ptr).decode('utf-8')
                    return result
                else:
                    logger.error("Inference failed")
                    return None
                    
        except Exception as e:
            logger.error(f"Error during text generation: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the C++ interface is available and ready"""
        return self._is_initialized and self._lib is not None
    
    def get_version(self) -> str:
        """Get the version of the RWKV C++ backend"""
        if not self._lib:
            return "C++ backend not available"
            
        try:
            if hasattr(self._lib, 'rwkv_get_version'):
                version_ptr = self._lib.rwkv_get_version()
                if version_ptr:
                    return ctypes.string_at(version_ptr).decode('utf-8')
            return "1.0.0-cpp"
        except Exception:
            return "Unknown version"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model"""
        return {
            'model_path': self.config.model_path,
            'threads': self.config.threads,
            'gpu_layers': self.config.gpu_layers,
            'context_length': self.config.context_length,
            'vocab_size': self.config.vocab_size,
            'backend': 'rwkv.cpp',
            'version': self.get_version(),
            'available': self.is_available(),
            'loaded': self._model_handle is not None
        }
    
    def __del__(self):
        """Cleanup resources"""
        try:
            if self._model_handle and self._lib:
                self._lib.rwkv_free_model(self._model_handle)
                self._model_handle = None
        except Exception:
            pass

# Factory function for easier instantiation
def create_rwkv_cpp_interface(model_path: Optional[str] = None, **kwargs) -> RWKVCppInterface:
    """Create and initialize an RWKV C++ interface"""
    if model_path:
        config = RWKVModelConfig(model_path=model_path, **kwargs)
    else:
        config = RWKVModelConfig(model_path="", **kwargs)
        
    interface = RWKVCppInterface(config)
    interface.initialize()
    return interface