"""
Deep Tree Echo RWKV.cpp Python Bridge

This module provides a Python interface to the high-performance rwkv.cpp library
for distributed agentic cognitive micro-kernel processing within the Deep Tree Echo framework.
"""

import ctypes
import os
import sys
import threading
from ctypes import c_int, c_uint32, c_float, c_char_p, c_size_t, c_void_p, POINTER
from typing import Optional, Tuple, List, Dict, Any
import logging

# Try to import numpy, but make it optional
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

logger = logging.getLogger(__name__)

class EchoRWKVCppError(Exception):
    """Exception raised by RWKV.cpp bridge operations"""
    pass

class EchoRWKVCppBridge:
    """
    Python bridge to rwkv.cpp for Deep Tree Echo cognitive processing
    """
    
    # Error codes
    SUCCESS = 0
    ERROR_INVALID_ARGS = 1
    ERROR_MODEL_LOAD = 2
    ERROR_INFERENCE = 3
    ERROR_MEMORY = 4
    ERROR_THREAD = 5
    
    def __init__(self):
        self._lib = None
        self._contexts = {}
        self._context_lock = threading.Lock()
        self._load_library()
        
    def _load_library(self):
        """Load the shared library with platform-specific naming"""
        library_names = []
        
        if sys.platform.startswith('win'):
            library_names = ['echo_rwkv_cpp_bridge.dll', 'libecho_rwkv_cpp_bridge.dll']
        elif sys.platform.startswith('darwin'):
            library_names = ['libecho_rwkv_cpp_bridge.dylib', 'libecho_rwkv_cpp_bridge.so']
        else:
            library_names = ['libecho_rwkv_cpp_bridge.so', 'echo_rwkv_cpp_bridge.so']
        
        # Search paths
        search_paths = [
            os.path.dirname(os.path.abspath(__file__)),  # Same directory as this file
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'src'),
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'build', 'bin'),
            '/usr/local/lib',
            '/usr/lib'
        ]
        
        for path in search_paths:
            for lib_name in library_names:
                lib_path = os.path.join(path, lib_name)
                if os.path.exists(lib_path):
                    try:
                        self._lib = ctypes.CDLL(lib_path)
                        self._setup_function_signatures()
                        logger.info(f"Loaded RWKV.cpp bridge from: {lib_path}")
                        return
                    except OSError as e:
                        logger.warning(f"Failed to load {lib_path}: {e}")
                        continue
        
        # If we reach here, library wasn't found
        logger.warning("RWKV.cpp bridge library not found. Using fallback mode.")
        self._lib = None
    
    def _setup_function_signatures(self):
        """Setup function signatures for the C library"""
        if not self._lib:
            return
        
        # echo_rwkv_init_model
        self._lib.echo_rwkv_init_model.argtypes = [c_char_p, c_uint32, c_uint32]
        self._lib.echo_rwkv_init_model.restype = c_int
        
        # echo_rwkv_eval
        self._lib.echo_rwkv_eval.argtypes = [c_int, c_uint32, POINTER(c_float), POINTER(c_float), POINTER(c_float)]
        self._lib.echo_rwkv_eval.restype = c_int
        
        # echo_rwkv_generate_text
        self._lib.echo_rwkv_generate_text.argtypes = [c_int, c_char_p, c_uint32, c_float, c_float, c_char_p, c_size_t]
        self._lib.echo_rwkv_generate_text.restype = c_int
        
        # echo_rwkv_get_n_vocab
        self._lib.echo_rwkv_get_n_vocab.argtypes = [c_int]
        self._lib.echo_rwkv_get_n_vocab.restype = c_uint32
        
        # echo_rwkv_get_n_embed
        self._lib.echo_rwkv_get_n_embed.argtypes = [c_int]
        self._lib.echo_rwkv_get_n_embed.restype = c_uint32
        
        # echo_rwkv_get_state_size
        self._lib.echo_rwkv_get_state_size.argtypes = [c_int]
        self._lib.echo_rwkv_get_state_size.restype = c_size_t
        
        # echo_rwkv_get_logits_size
        self._lib.echo_rwkv_get_logits_size.argtypes = [c_int]
        self._lib.echo_rwkv_get_logits_size.restype = c_size_t
        
        # echo_rwkv_free_model
        self._lib.echo_rwkv_free_model.argtypes = [c_int]
        self._lib.echo_rwkv_free_model.restype = c_int
        
        # echo_rwkv_get_version
        self._lib.echo_rwkv_get_version.argtypes = []
        self._lib.echo_rwkv_get_version.restype = c_char_p
        
        # echo_rwkv_init_library
        self._lib.echo_rwkv_init_library.argtypes = []
        self._lib.echo_rwkv_init_library.restype = c_int
        
        # echo_rwkv_cleanup_library
        self._lib.echo_rwkv_cleanup_library.argtypes = []
        self._lib.echo_rwkv_cleanup_library.restype = None
        
        # Initialize library
        result = self._lib.echo_rwkv_init_library()
        if result != self.SUCCESS:
            raise EchoRWKVCppError(f"Failed to initialize RWKV.cpp bridge library: {result}")
    
    def is_available(self) -> bool:
        """Check if RWKV.cpp bridge is available"""
        return self._lib is not None
    
    def get_version(self) -> str:
        """Get bridge version information"""
        if not self._lib:
            return "RWKV.cpp bridge not available"
        
        try:
            version_bytes = self._lib.echo_rwkv_get_version()
            return version_bytes.decode('utf-8')
        except Exception as e:
            logger.error(f"Error getting version: {e}")
            return "Unknown version"
    
    def load_model(self, model_path: str, thread_count: int = 4, gpu_layers: int = 0) -> int:
        """
        Load RWKV model for cognitive processing
        
        Args:
            model_path: Path to RWKV model file in ggml format
            thread_count: Number of threads for parallel processing
            gpu_layers: Number of layers to offload to GPU
            
        Returns:
            Model context ID for use in subsequent operations
            
        Raises:
            EchoRWKVCppError: If model loading fails
        """
        if not self._lib:
            raise EchoRWKVCppError("RWKV.cpp bridge not available")
        
        if not os.path.exists(model_path):
            raise EchoRWKVCppError(f"Model file not found: {model_path}")
        
        model_path_bytes = model_path.encode('utf-8')
        context_id = self._lib.echo_rwkv_init_model(model_path_bytes, thread_count, gpu_layers)
        
        if context_id < 0:
            error_code = -context_id
            error_msg = {
                self.ERROR_INVALID_ARGS: "Invalid arguments",
                self.ERROR_MODEL_LOAD: "Model loading failed",
                self.ERROR_MEMORY: "Memory allocation failed"
            }.get(error_code, f"Unknown error: {error_code}")
            raise EchoRWKVCppError(f"Failed to load model: {error_msg}")
        
        # Store context info
        with self._context_lock:
            self._contexts[context_id] = {
                'model_path': model_path,
                'thread_count': thread_count,
                'gpu_layers': gpu_layers,
                'n_vocab': self._lib.echo_rwkv_get_n_vocab(context_id),
                'n_embed': self._lib.echo_rwkv_get_n_embed(context_id),
                'state_size': self._lib.echo_rwkv_get_state_size(context_id),
                'logits_size': self._lib.echo_rwkv_get_logits_size(context_id)
            }
        
        logger.info(f"Loaded RWKV model {model_path} with context ID {context_id}")
        return context_id
    
    def generate_text(self, context_id: int, prompt: str, max_tokens: int = 100, 
                     temperature: float = 0.8, top_p: float = 0.9) -> str:
        """
        Generate text using RWKV model for cognitive processing
        
        Args:
            context_id: Model context ID from load_model()
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            
        Returns:
            Generated text response
            
        Raises:
            EchoRWKVCppError: If generation fails
        """
        if not self._lib:
            raise EchoRWKVCppError("RWKV.cpp bridge not available")
        
        if context_id not in self._contexts:
            raise EchoRWKVCppError(f"Invalid context ID: {context_id}")
        
        # Prepare buffers
        prompt_bytes = prompt.encode('utf-8')
        output_buffer = ctypes.create_string_buffer(4096)  # 4KB buffer
        
        result = self._lib.echo_rwkv_generate_text(
            context_id, prompt_bytes, max_tokens, 
            c_float(temperature), c_float(top_p),
            output_buffer, len(output_buffer)
        )
        
        if result < 0:
            error_code = -result
            error_msg = {
                self.ERROR_INVALID_ARGS: "Invalid arguments",
                self.ERROR_INFERENCE: "Inference failed"
            }.get(error_code, f"Unknown error: {error_code}")
            raise EchoRWKVCppError(f"Text generation failed: {error_msg}")
        
        return output_buffer.value.decode('utf-8')
    
    def eval_token(self, context_id: int, token: int, 
                  state_in: Optional[Any] = None) -> Tuple[Any, Any]:
        """
        Evaluate a single token through the RWKV model
        
        Args:
            context_id: Model context ID
            token: Input token ID
            state_in: Input state (optional, numpy array if available)
            
        Returns:
            Tuple of (state_out, logits_out) as arrays
            
        Raises:
            EchoRWKVCppError: If evaluation fails or numpy not available
        """
        if not self._lib:
            raise EchoRWKVCppError("RWKV.cpp bridge not available")
        
        if not NUMPY_AVAILABLE:
            raise EchoRWKVCppError("numpy is required for eval_token but not available")
        
        if context_id not in self._contexts:
            raise EchoRWKVCppError(f"Invalid context ID: {context_id}")
        
        ctx_info = self._contexts[context_id]
        
        # Prepare state buffers
        state_out = np.zeros(ctx_info['state_size'], dtype=np.float32)
        logits_out = np.zeros(ctx_info['logits_size'], dtype=np.float32)
        
        # Convert numpy arrays to ctypes pointers
        state_in_ptr = None
        if state_in is not None:
            if state_in.size != ctx_info['state_size']:
                raise EchoRWKVCppError(f"State size mismatch: expected {ctx_info['state_size']}, got {state_in.size}")
            state_in_ptr = state_in.ctypes.data_as(POINTER(c_float))
        
        state_out_ptr = state_out.ctypes.data_as(POINTER(c_float))
        logits_out_ptr = logits_out.ctypes.data_as(POINTER(c_float))
        
        result = self._lib.echo_rwkv_eval(context_id, token, state_in_ptr, state_out_ptr, logits_out_ptr)
        
        if result != self.SUCCESS:
            raise EchoRWKVCppError(f"Token evaluation failed: error code {result}")
        
        return state_out, logits_out
    
    def get_model_info(self, context_id: int) -> Dict[str, Any]:
        """Get information about a loaded model"""
        if context_id not in self._contexts:
            raise EchoRWKVCppError(f"Invalid context ID: {context_id}")
        
        return self._contexts[context_id].copy()
    
    def free_model(self, context_id: int) -> None:
        """
        Free model resources
        
        Args:
            context_id: Model context ID to free
            
        Raises:
            EchoRWKVCppError: If freeing fails
        """
        if not self._lib:
            return  # Nothing to free if library not loaded
        
        if context_id not in self._contexts:
            logger.warning(f"Attempted to free unknown context ID: {context_id}")
            return
        
        result = self._lib.echo_rwkv_free_model(context_id)
        
        if result != self.SUCCESS:
            logger.error(f"Failed to free model context {context_id}: error code {result}")
        
        with self._context_lock:
            if context_id in self._contexts:
                del self._contexts[context_id]
        
        logger.info(f"Freed RWKV model context {context_id}")
    
    def cleanup(self) -> None:
        """Cleanup all resources"""
        if not self._lib:
            return
        
        # Free all contexts
        with self._context_lock:
            for context_id in list(self._contexts.keys()):
                self.free_model(context_id)
        
        # Cleanup library
        self._lib.echo_rwkv_cleanup_library()
        logger.info("RWKV.cpp bridge cleanup completed")
    
    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error during RWKV.cpp bridge cleanup: {e}")

# Global bridge instance
_bridge_instance = None

def get_rwkv_cpp_bridge() -> EchoRWKVCppBridge:
    """Get the global RWKV.cpp bridge instance"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = EchoRWKVCppBridge()
    return _bridge_instance

def test_rwkv_cpp_integration():
    """Test the RWKV.cpp integration"""
    bridge = get_rwkv_cpp_bridge()
    
    print(f"RWKV.cpp Bridge Available: {bridge.is_available()}")
    print(f"Version: {bridge.get_version()}")
    print(f"Numpy Available: {NUMPY_AVAILABLE}")
    
    # If bridge is available, we could test with a model
    # This would require an actual RWKV model file
    if bridge.is_available():
        print("RWKV.cpp bridge is ready for Deep Tree Echo cognitive processing!")
        if not NUMPY_AVAILABLE:
            print("Note: Install numpy for full functionality: pip install numpy")
    else:
        print("RWKV.cpp bridge not available - using fallback mode")

if __name__ == "__main__":
    test_rwkv_cpp_integration()