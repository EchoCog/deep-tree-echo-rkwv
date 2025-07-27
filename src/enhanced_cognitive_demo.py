#!/usr/bin/env python3
"""
Deep Tree Echo Enhanced Application with Foundation Components
Demonstrates integration of P0 foundation components with existing cognitive architecture.
"""

import os
import json
import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

# Foundation components
from rwkv_model_foundation import get_model_manager, RWKVModelConfig
from persistent_memory_foundation import get_persistent_memory_system, PersistentMemorySystem
from security_framework_foundation import get_security_framework, SecurityFramework, UserRole

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class EnhancedCognitiveResponse:
    """Enhanced response with foundation component integration"""
    user_input: str
    ai_response: str
    session_id: str
    user_id: str
    model_used: str
    memory_context: List[Dict[str, Any]]
    processing_time: float
    confidence: float
    metadata: Dict[str, Any]

class EnhancedCognitiveProcessor:
    """Enhanced cognitive processor using foundation components"""
    
    def __init__(self):
        # Initialize foundation components
        self.model_manager = get_model_manager(memory_limit_mb=600)
        self.memory_system = get_persistent_memory_system("/tmp/echo_enhanced.db")
        self.security = get_security_framework()
        
        # Load optimal model
        self.current_model = None
        self._load_optimal_model()
        
        logger.info("Enhanced cognitive processor initialized")
    
    def _load_optimal_model(self):
        """Load the optimal RWKV model for current constraints"""
        optimal_config = self.model_manager.get_optimal_model()
        if optimal_config:
            success = self.model_manager.load_model(optimal_config)
            if success:
                self.current_model = optimal_config
                logger.info(f"Loaded model: {optimal_config.model_name}")
            else:
                logger.warning("Failed to load optimal model, using fallback")
        else:
            logger.warning("No suitable model found for memory constraints")
    
    def authenticate_user(self, username: str, password: str, ip_address: str = "127.0.0.1") -> Optional[str]:
        """Authenticate user and return session ID"""
        session_id = self.security.authenticate_user(
            username=username,
            password=password,
            ip_address=ip_address,
            user_agent="DeepTreeEcho/1.0"
        )
        
        if session_id:
            logger.info(f"User {username} authenticated successfully")
        else:
            logger.warning(f"Authentication failed for user {username}")
        
        return session_id
    
    def process_cognitive_request(self, session_id: str, user_input: str, ip_address: str = "127.0.0.1") -> Optional[EnhancedCognitiveResponse]:
        """Process a cognitive request with full security and memory integration"""
        start_time = time.time()
        
        # Validate session
        user = self.security.validate_session(session_id, ip_address)
        if not user:
            logger.warning(f"Invalid session: {session_id}")
            return None
        
        logger.info(f"Processing request from user {user.username}")
        
        try:
            # Get conversation context from memory
            conversation_history = self.memory_system.get_conversation_history(
                session_id, max_turns=5
            )
            
            # Search for relevant knowledge
            relevant_memories = self.memory_system.search_knowledge(
                user_input, context_session=session_id
            )
            
            # Process through cognitive membranes (simplified for demo)
            ai_response = self._generate_response(user_input, conversation_history, relevant_memories)
            
            # Store conversation turn
            user_mem_id, ai_mem_id = self.memory_system.store_conversation_turn(
                session_id=session_id,
                user_input=user_input,
                ai_response=ai_response,
                metadata={
                    "user_id": user.user_id,
                    "model": self.current_model.model_name if self.current_model else "mock",
                    "processing_time": time.time() - start_time
                }
            )
            
            processing_time = time.time() - start_time
            
            # Create enhanced response
            response = EnhancedCognitiveResponse(
                user_input=user_input,
                ai_response=ai_response,
                session_id=session_id,
                user_id=user.user_id,
                model_used=self.current_model.model_name if self.current_model else "mock",
                memory_context=[{
                    "type": "conversation",
                    "turns": len(conversation_history)
                }, {
                    "type": "knowledge",
                    "relevant_items": len(relevant_memories)
                }],
                processing_time=processing_time,
                confidence=0.85,  # Mock confidence
                metadata={
                    "user_memory_id": user_mem_id,
                    "ai_memory_id": ai_mem_id,
                    "model_memory_usage": self.model_manager.get_memory_usage()
                }
            )
            
            logger.info(f"Cognitive request processed in {processing_time:.3f}s")
            return response
            
        except Exception as e:
            logger.error(f"Error processing cognitive request: {e}")
            return None
    
    def _generate_response(self, user_input: str, history: List[Dict], memories: List) -> str:
        """Generate AI response (simplified implementation)"""
        # This would integrate with real RWKV model when available
        # For now, provide a context-aware mock response
        
        context_info = ""
        if history:
            context_info += f" (Building on {len(history)} previous exchanges)"
        
        if memories:
            context_info += f" (Drawing from {len(memories)} relevant memories)"
        
        response = f"I understand your question: '{user_input}'{context_info}. "
        
        # Simple response generation based on input
        if "hello" in user_input.lower():
            response += "Hello! I'm ready to help with your cognitive processing needs."
        elif "memory" in user_input.lower():
            response += f"I have access to {len(memories)} relevant memories and our conversation history."
        elif "model" in user_input.lower():
            model_name = self.current_model.model_name if self.current_model else "mock model"
            response += f"I'm currently using {model_name} for cognitive processing."
        else:
            response += "I'm processing your request through my cognitive membranes (Memory, Reasoning, Grammar) and will provide a thoughtful response based on available context."
        
        return response
    
    def get_system_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive system status for authenticated user"""
        user = self.security.validate_session(session_id)
        if not user:
            return None
        
        model_status = self.model_manager.get_memory_usage()
        memory_status = self.memory_system.get_system_status()
        security_status = self.security.get_security_status()
        
        return {
            "user": {
                "username": user.username,
                "role": user.role.value,
                "session_active": True
            },
            "models": {
                "current_model": self.current_model.model_name if self.current_model else None,
                "memory_usage": model_status,
                "available_models": len(self.model_manager.get_available_models())
            },
            "memory": {
                "total_memories": memory_status.get("total_memories", 0),
                "active_sessions": memory_status.get("active_sessions", 0),
                "storage_type": memory_status.get("storage_type", "unknown")
            },
            "security": {
                "storage_type": security_status.get("storage_type", "unknown"),
                "session_timeout": security_status.get("session_timeout", 0)
            },
            "timestamp": time.time()
        }
    
    def logout_user(self, session_id: str):
        """Logout user and cleanup session"""
        self.security.logout_user(session_id)
        logger.info(f"User logged out: {session_id}")

def demo_enhanced_system():
    """Demonstrate the enhanced system capabilities"""
    print("=" * 60)
    print("Deep Tree Echo Enhanced System Demo")
    print("=" * 60)
    
    # Initialize enhanced processor
    processor = EnhancedCognitiveProcessor()
    
    # Authenticate user (using default admin)
    session_id = processor.authenticate_user("admin", "admin123")
    if not session_id:
        print("Authentication failed!")
        return
    
    print(f"✓ Authenticated successfully (Session: {session_id[:8]}...)")
    
    # Test cognitive processing
    test_inputs = [
        "Hello! Can you tell me about artificial intelligence?",
        "What can you tell me about your memory capabilities?",
        "What model are you using for processing?",
        "How do you handle complex reasoning tasks?"
    ]
    
    for i, user_input in enumerate(test_inputs, 1):
        print(f"\n--- Interaction {i} ---")
        print(f"User: {user_input}")
        
        response = processor.process_cognitive_request(session_id, user_input)
        if response:
            print(f"AI: {response.ai_response}")
            print(f"Model: {response.model_used}")
            print(f"Processing time: {response.processing_time:.3f}s")
            print(f"Memory context: {response.memory_context}")
        else:
            print("Failed to process request")
    
    # Get system status
    print(f"\n--- System Status ---")
    status = processor.get_system_status(session_id)
    if status:
        print(json.dumps(status, indent=2, default=str))
    
    # Logout
    processor.logout_user(session_id)
    print(f"\n✓ User logged out successfully")

if __name__ == "__main__":
    demo_enhanced_system()