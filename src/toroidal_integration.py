"""
Toroidal Integration Layer
Integrates the Toroidal Cognitive System with existing Echo-RWKV infrastructure
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import asdict

from toroidal_cognitive_system import (
    ToroidalCognitiveSystem, 
    ToroidalResponse,
    create_toroidal_cognitive_system
)

try:
    from echo_rwkv_bridge import (
        CognitiveContext,
        IntegratedCognitiveResponse,
        EchoRWKVIntegrationEngine
    )
    ECHO_RWKV_AVAILABLE = True
except ImportError:
    ECHO_RWKV_AVAILABLE = False
    # Create fallback classes
    class CognitiveContext:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class IntegratedCognitiveResponse:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

logger = logging.getLogger(__name__)

class ToroidalEchoRWKVBridge:
    """Bridge between Toroidal Cognitive System and Echo-RWKV infrastructure"""
    
    def __init__(self, buffer_size: int = 1000, use_real_rwkv: bool = False):
        self.toroidal_system = create_toroidal_cognitive_system(buffer_size)
        self.use_real_rwkv = use_real_rwkv
        
        # Initialize Echo-RWKV integration if available
        if ECHO_RWKV_AVAILABLE and use_real_rwkv:
            self.echo_rwkv_engine = EchoRWKVIntegrationEngine(use_real_rwkv=use_real_rwkv)
        else:
            self.echo_rwkv_engine = None
            
        logger.info(f"Toroidal-Echo-RWKV Bridge initialized (RWKV: {self.use_real_rwkv}, Available: {ECHO_RWKV_AVAILABLE})")
    
    async def initialize(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Initialize the integration bridge"""
        try:
            if self.echo_rwkv_engine:
                await self.echo_rwkv_engine.initialize(config or {})
            
            logger.info("Toroidal-Echo-RWKV Bridge initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize bridge: {e}")
            return False
    
    async def process_cognitive_input(self, 
                                    user_input: str,
                                    session_id: str = "default",
                                    conversation_history: Optional[list] = None,
                                    memory_state: Optional[Dict[str, Any]] = None,
                                    processing_goals: Optional[list] = None) -> ToroidalResponse:
        """Process input through the Toroidal Cognitive System with optional RWKV enhancement"""
        
        # Prepare context
        context = {
            "session_id": session_id,
            "conversation_history": conversation_history or [],
            "memory_state": memory_state or {},
            "processing_goals": processing_goals or []
        }
        
        # Process through Toroidal system
        toroidal_response = await self.toroidal_system.process_input(user_input, context)
        
        # Enhance with RWKV if available
        if self.echo_rwkv_engine:
            enhanced_response = await self._enhance_with_rwkv(toroidal_response, context)
            return enhanced_response
        
        return toroidal_response
    
    async def _enhance_with_rwkv(self, 
                                toroidal_response: ToroidalResponse, 
                                context: Dict[str, Any]) -> ToroidalResponse:
        """Enhance Toroidal response with RWKV processing"""
        try:
            # Create cognitive context for RWKV
            cognitive_context = CognitiveContext(
                session_id=context.get("session_id", "default"),
                user_input=toroidal_response.user_input,
                conversation_history=context.get("conversation_history", []),
                memory_state=context.get("memory_state", {}),
                processing_goals=context.get("processing_goals", []),
                temporal_context=[],
                metadata={"toroidal_processing": True}
            )
            
            # Process through RWKV
            rwkv_response = await self.echo_rwkv_engine.process_cognitive_input(cognitive_context)
            
            # Enhance the toroidal response with RWKV insights
            enhanced_output = self._merge_responses(toroidal_response, rwkv_response)
            
            # Update the toroidal response
            toroidal_response.synchronized_output = enhanced_output
            toroidal_response.reflection += "\n\n**RWKV Enhancement**: Integrated with advanced language model processing for enhanced coherence and factual accuracy."
            
            return toroidal_response
            
        except Exception as e:
            logger.warning(f"RWKV enhancement failed: {e}")
            return toroidal_response
    
    def _merge_responses(self, toroidal_response: ToroidalResponse, rwkv_response) -> str:
        """Merge Toroidal and RWKV responses intelligently"""
        # Start with the toroidal synchronized output
        merged_output = toroidal_response.synchronized_output
        
        # Add RWKV insights if available
        if hasattr(rwkv_response, 'integrated_output') and rwkv_response.integrated_output:
            merged_output += (
                f"\n\n---\n\n"
                f"## **RWKV Language Model Enhancement**\n\n"
                f"{rwkv_response.integrated_output}\n\n"
                f"**Integration Confidence**: {getattr(rwkv_response, 'confidence_score', 0.0):.3f}"
            )
        
        return merged_output
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "toroidal_system": self.toroidal_system.get_system_metrics(),
            "rwkv_integration": {
                "enabled": self.use_real_rwkv,
                "available": ECHO_RWKV_AVAILABLE,
                "engine_initialized": self.echo_rwkv_engine is not None
            },
            "bridge_status": "operational"
        }
        
        return status

class ToroidalRESTAPI:
    """REST API wrapper for Toroidal Cognitive System"""
    
    def __init__(self, bridge: ToroidalEchoRWKVBridge):
        self.bridge = bridge
    
    async def process_query(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query through the Toroidal system"""
        try:
            user_input = request_data.get("input", "")
            session_id = request_data.get("session_id", "default")
            conversation_history = request_data.get("conversation_history", [])
            memory_state = request_data.get("memory_state", {})
            processing_goals = request_data.get("processing_goals", [])
            
            if not user_input.strip():
                return {"error": "Empty input provided"}
            
            # Process through bridge
            response = await self.bridge.process_cognitive_input(
                user_input=user_input,
                session_id=session_id,
                conversation_history=conversation_history,
                memory_state=memory_state,
                processing_goals=processing_goals
            )
            
            # Convert to dictionary for JSON serialization
            return {
                "success": True,
                "response": {
                    "user_input": response.user_input,
                    "echo_response": {
                        "hemisphere": response.echo_response.hemisphere,
                        "response_text": response.echo_response.response_text,
                        "processing_time": response.echo_response.processing_time,
                        "confidence": response.echo_response.confidence,
                        "cognitive_markers": response.echo_response.cognitive_markers
                    },
                    "marduk_response": {
                        "hemisphere": response.marduk_response.hemisphere,
                        "response_text": response.marduk_response.response_text,
                        "processing_time": response.marduk_response.processing_time,
                        "confidence": response.marduk_response.confidence,
                        "cognitive_markers": response.marduk_response.cognitive_markers
                    },
                    "synchronized_output": response.synchronized_output,
                    "reflection": response.reflection,
                    "total_processing_time": response.total_processing_time,
                    "convergence_metrics": response.convergence_metrics
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {"error": str(e), "success": False}
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        try:
            return {
                "success": True,
                "status": self.bridge.get_system_status()
            }
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {"error": str(e), "success": False}

# Factory functions for easy instantiation
def create_toroidal_bridge(buffer_size: int = 1000, use_real_rwkv: bool = False) -> ToroidalEchoRWKVBridge:
    """Create a new Toroidal-Echo-RWKV Bridge"""
    return ToroidalEchoRWKVBridge(buffer_size, use_real_rwkv)

def create_toroidal_api(bridge: ToroidalEchoRWKVBridge) -> ToroidalRESTAPI:
    """Create a new Toroidal REST API wrapper"""
    return ToroidalRESTAPI(bridge)