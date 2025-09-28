"""
RWKV.cpp Integration for Deep Tree Echo Framework
Provides high-performance C++ RWKV inference integration with the cognitive architecture
"""

try:
    from .rwkv_cpp_interface import RWKVCppInterface, RWKVCppConfig
    from .rwkv_cpp_cognitive_bridge import RWKVCppCognitiveBridge
    RWKV_CPP_AVAILABLE = True
except ImportError:
    # Fallback when RWKV.cpp is not available
    RWKV_CPP_AVAILABLE = False
    class RWKVCppInterface:
        def __init__(self, *args, **kwargs):
            raise ImportError("RWKV.cpp not available")
    
    class RWKVCppConfig:
        def __init__(self, *args, **kwargs):
            raise ImportError("RWKV.cpp not available")
    
    class RWKVCppCognitiveBridge:
        def __init__(self, *args, **kwargs):
            raise ImportError("RWKV.cpp not available")

# For backwards compatibility
try:
    from .rwkv_cpp_bridge import RWKVCppBridge
except ImportError:
    RWKVCppBridge = RWKVCppInterface

# Additional interfaces
RWKVCppMembraneProcessor = RWKVCppCognitiveBridge

def create_rwkv_processor(*args, **kwargs):
    """Create RWKV processor instance"""
    if RWKV_CPP_AVAILABLE:
        return RWKVCppCognitiveBridge(*args, **kwargs)
    else:
        raise ImportError("RWKV.cpp not available")

__all__ = [
    'RWKVCppInterface',
    'RWKVCppConfig', 
    'RWKVCppCognitiveBridge',
    'RWKVCppBridge',
    'RWKVCppMembraneProcessor',
    'create_rwkv_processor',
    'RWKV_CPP_AVAILABLE'
]
