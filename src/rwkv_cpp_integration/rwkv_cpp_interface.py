"""
RWKV C++ Interface

High-performance C++ interface for RWKV model inference within the Deep Tree Echo framework.
"""

import os
import sys
import ctypes
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from datetime import datetime

# Add the rwkv.cpp python path to sys.path
RWKV_CPP_PATH = os.path.join(os.path.dirname(__file__), '../../external/rwkv-cpp/python')
if RWKV_CPP_PATH not in sys.path:
    sys.path.insert(0, RWKV_CPP_PATH)

try:
    from rwkv_cpp import rwkv_cpp_shared_library, rwkv_cpp_model
    from tokenizer_util import get_tokenizer
    import sampling
    RWKV_CPP_AVAILABLE = True
except ImportError as e:
    RWKV_CPP_AVAILABLE = False
    print(f"Warning: rwkv.cpp not available: {e}")

# Import the base interface from the main framework
try:
    from echo_rwkv_bridge import RWKVModelInterface, CognitiveContext
except ImportError:
    # Define minimal interface if not available
    from abc import ABC, abstractmethod
    
    class RWKVModelInterface(ABC):
        @abstractmethod
        async def initialize(self, model_config: Dict[str, Any]) -> bool:
            pass
        
        @abstractmethod 
        async def generate_response(self, prompt: str, context: Any) -> str:
            pass
            
        @abstractmethod
        async def encode_memory(self, memory_item: Dict[str, Any]) -> Union[List[float], Any]:
            pass
            
        @abstractmethod
        async def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
            pass
            
        @abstractmethod
        def get_model_state(self) -> Dict[str, Any]:
            pass

    # Mock CognitiveContext if not available
    @dataclass
    class CognitiveContext:
        session_id: str
        user_input: str
        conversation_history: List[Dict[str, Any]]
        memory_state: Dict[str, Any]
        processing_goals: List[str]
        temporal_context: List[str]
        metadata: Dict[str, Any]
import threading
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class RWKVCppConfig:
    """Configuration for RWKV.cpp integration"""
    model_path: str
    library_path: str = None  # Will auto-detect if not provided
    thread_count: int = 4
    gpu_layer_count: int = 0
    context_length: int = 2048
    temperature: float = 0.8
    top_p: float = 0.9
    top_k: int = 40
    max_tokens: int = 200
    tokenizer_type: str = "world"  # "world", "20B", or "pile"
    enable_memory_optimization: bool = True
    memory_limit_mb: int = 600  # WebVM constraint
    batch_size: int = 1
    cache_tokens: bool = True

class RWKVCppInterface(RWKVModelInterface):
    """
    High-performance RWKV.cpp implementation of RWKVModelInterface
    Provides C++ optimized RWKV inference for the Deep Tree Echo cognitive architecture
    """
    
    def __init__(self):
        self.config: Optional[RWKVCppConfig] = None
        self.library = None
        self.model = None
        self.tokenizer_encode = None
        self.tokenizer_decode = None
        self.state = None
        self.logits = None
        self.initialized = False
        self.memory_store = []
        self.token_cache = {}
        self.generation_stats = {
            'total_tokens': 0,
            'total_time': 0.0,
            'avg_tokens_per_second': 0.0
        }
        
    async def initialize(self, model_config: Dict[str, Any]) -> bool:
        """Initialize RWKV.cpp model with cognitive architecture integration"""
        
        if not RWKV_CPP_AVAILABLE:
            logger.error("RWKV.cpp not available - falling back to mock implementation")
            return await self._initialize_fallback(model_config)
        
        try:
            # Parse configuration
            self.config = RWKVCppConfig(**model_config)
            
            # Auto-detect library path if not provided
            if not self.config.library_path:
                self.config.library_path = self._auto_detect_library_path()
            
            if not self.config.library_path or not os.path.exists(self.config.library_path):
                logger.error(f"RWKV library not found at {self.config.library_path}")
                return await self._initialize_fallback(model_config)
            
            # Load the shared library
            self.library = rwkv_cpp_shared_library.load_rwkv_shared_library()
            
            if not self.library:
                logger.error("Failed to load RWKV.cpp shared library")
                return await self._initialize_fallback(model_config)
            
            # Check if model file exists
            if not os.path.exists(self.config.model_path):
                logger.error(f"Model file not found: {self.config.model_path}")
                return await self._initialize_fallback(model_config)
            
            # Initialize the model
            self.model = rwkv_cpp_model.RWKVModel(
                self.library,
                self.config.model_path,
                thread_count=self.config.thread_count,
                gpu_layer_count=self.config.gpu_layer_count
            )
            
            # Set up tokenizer
            self.tokenizer_decode, self.tokenizer_encode = get_tokenizer(
                self.config.tokenizer_type, 
                self.model.n_vocab
            )
            
            # Initialize state and logits buffers
            self.state = None  # Will be initialized on first use
            self.logits = None
            
            self.initialized = True
            
            logger.info(f"RWKV.cpp initialized successfully")
            logger.info(f"Model vocab size: {self.model.n_vocab}")
            logger.info(f"Model embedding size: {self.model.n_embed}")
            logger.info(f"Model layers: {self.model.n_layer}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RWKV.cpp: {e}")
            return await self._initialize_fallback(model_config)
    
    def _auto_detect_library_path(self) -> Optional[str]:
        """Auto-detect the path to the RWKV.cpp shared library"""
        
        # Check relative path from the integration module
        base_dir = os.path.dirname(__file__)
        potential_paths = [
            os.path.join(base_dir, '../../external/rwkv-cpp/librwkv.so'),
            os.path.join(base_dir, '../../external/rwkv-cpp/librwkv.dylib'),
            os.path.join(base_dir, '../../external/rwkv-cpp/rwkv.dll'),
            '/usr/local/lib/librwkv.so',
            '/usr/lib/librwkv.so',
            './librwkv.so',
            './rwkv.dll'
        ]
        
        for path in potential_paths:
            full_path = Path(path).resolve()
            if os.path.exists(full_path):
                logger.info(f"Auto-detected RWKV library at: {full_path}")
                return full_path
        
        logger.warning("Could not auto-detect RWKV.cpp library path")
        return None
    
    async def _initialize_fallback(self, model_config: Dict[str, Any]) -> bool:
        """Fallback initialization when RWKV.cpp is not available"""
        logger.warning("Initializing fallback mock implementation")
        self.initialized = True
        self.config = RWKVCppConfig(**model_config)
        return True
    
    async def generate_response(self, prompt: str, context: CognitiveContext) -> str:
        """Generate response using RWKV.cpp optimized inference"""
        
        if not self.initialized:
            return "Error: RWKV.cpp interface not initialized"
        
        if not self.model:
            return self._generate_fallback_response(prompt, context)
        
        try:
            start_time = datetime.now()
            
            # Tokenize the prompt
            prompt_tokens = self.tokenizer_encode(prompt)
            
            # If state is None, process the prompt to initialize state
            if self.state is None:
                self.logits, self.state = self.model.eval_sequence_in_chunks(
                    prompt_tokens, None, None, None, use_numpy=True
                )
            else:
                # For conversation continuation, process only new tokens
                # This is a simplified version - in practice you'd want more sophisticated state management
                for token in prompt_tokens[-10:]:  # Process last 10 tokens to maintain context
                    self.logits, self.state = self.model.eval(
                        token, self.state, self.state, self.logits, use_numpy=True
                    )
            
            # Generate response tokens
            response_tokens = []
            current_state = np.copy(self.state)
            current_logits = np.copy(self.logits)
            
            for _ in range(self.config.max_tokens):
                # Sample next token
                token = sampling.sample_logits(
                    current_logits,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k
                )
                
                response_tokens.append(token)
                
                # Check for end of sequence or natural stopping points
                if token == 0:  # End token
                    break
                
                # Decode current tokens to check for natural stops
                current_text = self.tokenizer_decode(response_tokens)
                if self._is_natural_stop(current_text):
                    break
                
                # Get next logits
                current_logits, current_state = self.model.eval(
                    token, current_state, current_state, current_logits, use_numpy=True
                )
            
            # Decode the response
            response_text = self.tokenizer_decode(response_tokens)
            
            # Clean up the response
            response_text = self._clean_response(response_text)
            
            # Update generation statistics
            end_time = datetime.now()
            generation_time = (end_time - start_time).total_seconds()
            tokens_generated = len(response_tokens)
            
            self.generation_stats['total_tokens'] += tokens_generated
            self.generation_stats['total_time'] += generation_time
            if self.generation_stats['total_time'] > 0:
                self.generation_stats['avg_tokens_per_second'] = (
                    self.generation_stats['total_tokens'] / self.generation_stats['total_time']
                )
            
            logger.debug(f"Generated {tokens_generated} tokens in {generation_time:.3f}s")
            
            return response_text
            
        except Exception as e:
            logger.error(f"Error in RWKV.cpp generation: {e}")
            return self._generate_fallback_response(prompt, context)
    
    def _generate_fallback_response(self, prompt: str, context: CognitiveContext) -> str:
        """Generate fallback response when RWKV.cpp is not available"""
        return f"Mock RWKV.cpp response for: {prompt[:50]}... (C++ backend not available)"
    
    def _is_natural_stop(self, text: str) -> bool:
        """Check if the generated text has reached a natural stopping point"""
        # Simple heuristics for natural stopping points
        if len(text) < 10:
            return False
        
        # Stop at sentence endings followed by capital letters or end of meaningful content
        natural_stops = ['. ', '! ', '? ', '\n\n', '.\n']
        for stop in natural_stops:
            if text.endswith(stop):
                return True
        
        # Stop if we're repeating content
        if len(text) > 50:
            last_30 = text[-30:]
            if last_30 in text[:-30]:
                return True
        
        return False
    
    def _clean_response(self, text: str) -> str:
        """Clean up the generated response"""
        if not text:
            return text
        
        # Remove trailing incomplete sentences
        text = text.strip()
        
        # Remove common artifacts
        artifacts = ['<|endoftext|>', '<|end|>', '<|im_end|>']
        for artifact in artifacts:
            text = text.replace(artifact, '')
        
        # Ensure reasonable length
        if len(text) > 500:
            # Find last complete sentence within reasonable length
            sentences = text.split('. ')
            truncated = []
            total_length = 0
            
            for sentence in sentences:
                if total_length + len(sentence) < 400:
                    truncated.append(sentence)
                    total_length += len(sentence)
                else:
                    break
            
            if truncated:
                text = '. '.join(truncated)
                if not text.endswith('.'):
                    text += '.'
        
        return text.strip()
    
    async def encode_memory(self, memory_item: Dict[str, Any]) -> Union[List[float], Any]:
        """Encode memory item using RWKV.cpp model embeddings"""
        
        if not self.model:
            # Fallback encoding
            content = str(memory_item.get('content', ''))
            return [hash(content[i:i+10]) % 1000 / 1000.0 for i in range(min(512, len(content)))]
        
        try:
            # Use the model's embedding layer to encode memory
            content = str(memory_item.get('content', ''))
            tokens = self.tokenizer_encode(content[:100])  # Limit content length
            
            # Get embeddings by running through model (simplified approach)
            if tokens:
                logits, _ = self.model.eval_sequence_in_chunks(tokens, None, None, None, use_numpy=True)
                # Use logits as a form of embedding (simplified)
                embedding = logits[:min(self.EMBEDDING_SIZE, len(logits))].tolist()
                return embedding
            else:
                return [0.0] * self.EMBEDDING_SIZE  # Default embedding
                
        except Exception as e:
            logger.error(f"Error encoding memory with RWKV.cpp: {e}")
            # Fallback to simple encoding
            content = str(memory_item.get('content', ''))
            return [hash(content[i:i+10]) % 1000 / 1000.0 for i in range(min(self.EMBEDDING_SIZE, len(content)))]
    
    async def retrieve_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories using RWKV.cpp enhanced similarity search"""
        
        if not self.memory_store:
            return []
        
        try:
            # Encode the query
            query_encoding = await self.encode_memory({'content': query})
            
            # Simple cosine similarity search (in production, use vector database)
            similarities = []
            for i, memory in enumerate(self.memory_store):
                memory_encoding = memory.get('encoding', [])
                if memory_encoding and len(memory_encoding) == len(query_encoding):
                    # Compute cosine similarity
                    dot_product = sum(a * b for a, b in zip(query_encoding, memory_encoding))
                    norm_a = sum(a * a for a in query_encoding) ** 0.5
                    norm_b = sum(b * b for b in memory_encoding) ** 0.5
                    
                    if norm_a > 0 and norm_b > 0:
                        similarity = dot_product / (norm_a * norm_b)
                        similarities.append((i, similarity))
            
            # Sort by similarity and return top_k
            similarities.sort(key=lambda x: x[1], reverse=True)
            
            results = []
            for i, similarity in similarities[:top_k]:
                memory = self.memory_store[i].copy()
                memory['similarity'] = similarity
                results.append(memory)
            
            return results
            
        except Exception as e:
            logger.error(f"Error retrieving memories: {e}")
            return self.memory_store[-top_k:] if self.memory_store else []
    
    def get_model_state(self) -> Dict[str, Any]:
        """Get comprehensive model state information"""
        
        base_state = {
            'initialized': self.initialized,
            'model_type': 'rwkv_cpp',
            'config': asdict(self.config) if self.config else {},
            'generation_stats': self.generation_stats,
            'memory_items': len(self.memory_store),
            'rwkv_cpp_available': RWKV_CPP_AVAILABLE
        }
        
        if self.model:
            base_state.update({
                'model_loaded': True,
                'vocab_size': self.model.n_vocab,
                'embedding_size': self.model.n_embed,
                'layer_count': self.model.n_layer,
                'state_initialized': self.state is not None,
                'token_cache_size': len(self.token_cache)
            })
        else:
            base_state.update({
                'model_loaded': False,
                'fallback_mode': True
            })
        
        return base_state
    
    def reset_state(self):
        """Reset the model state for new conversation"""
        self.state = None
        self.logits = None
        logger.debug("RWKV.cpp model state reset")
    
    async def save_memory(self, memory_item: Dict[str, Any]):
        """Save memory item with encoding for future retrieval"""
        try:
            # Add encoding to memory item for similarity search
            encoding = await self.encode_memory(memory_item)
            
            memory_with_encoding = memory_item.copy()
            memory_with_encoding['encoding'] = encoding
            memory_with_encoding['timestamp'] = datetime.now().isoformat()
            
            self.memory_store.append(memory_with_encoding)
            
            # Limit memory store size for WebVM constraints
            if len(self.memory_store) > 1000:
                self.memory_store = self.memory_store[-800:]  # Keep most recent 800
                
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
    
    def __del__(self):
        """Clean up resources"""
        if self.model:
            try:
                self.model.free()
            except:
                pass
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
