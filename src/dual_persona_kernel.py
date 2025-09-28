"""
Dual Persona Kernel - Deep Tree Echo & Marduk the Mad Scientist
Implements the complementary dual-persona system as described in the toroidal cognitive architecture.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json

logger = logging.getLogger(__name__)

class PersonaType(Enum):
    """Types of personas in the dual system"""
    DEEP_TREE_ECHO = "deep_tree_echo"
    MARDUK_MAD_SCIENTIST = "marduk_mad_scientist"

class PersonaTraitType(Enum):
    """Core trait types following the tree metaphor"""
    ROOTS = "roots"           # Memory foundations and knowledge storage
    BRANCHES = "branches"     # Logical reasoning and analysis capabilities
    LEAVES = "leaves"         # Communication and expression abilities
    TRUNK = "trunk"           # Core identity stability and consistency
    GROWTH = "growth"         # Learning and evolutionary capacity
    CANOPY = "canopy"         # Creative thinking and innovation
    NETWORK = "network"       # Social connections and collaborative abilities

@dataclass
class PersonaTrait:
    """Individual persona trait with strength and characteristics"""
    trait_type: PersonaTraitType
    strength: float  # 0.0 to 1.0
    characteristics: List[str]
    evolution_history: List[Dict[str, Any]]
    
    def evolve(self, stimulus: str, direction: float = 0.1):
        """Evolve trait based on stimulus"""
        old_strength = self.strength
        self.strength = max(0.0, min(1.0, self.strength + direction))
        
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'stimulus': stimulus,
            'old_strength': old_strength,
            'new_strength': self.strength,
            'direction': direction
        })

@dataclass
class PersonaResponse:
    """Response from a single persona"""
    persona_type: PersonaType
    content: str
    confidence: float
    reasoning_process: List[str]
    trait_activations: Dict[PersonaTraitType, float]
    timestamp: str
    processing_time: float

@dataclass
class DualPersonaResponse:
    """Complete dual persona response with reflection"""
    deep_tree_echo_response: PersonaResponse
    marduk_response: PersonaResponse
    reflection_content: str
    synthesis: Optional[str]
    convergence_score: float  # How well the personas align
    total_processing_time: float
    session_id: str

class PersonaKernel:
    """Individual persona implementation"""
    
    def __init__(self, persona_type: PersonaType, trait_config: Optional[Dict[str, Any]] = None):
        self.persona_type = persona_type
        self.traits = self._initialize_traits(trait_config)
        self.memory_beacon = []  # Memory threads for Deep Tree Echo
        self.architectural_blueprints = []  # System designs for Marduk
        self.response_history = []
        
    def _initialize_traits(self, config: Optional[Dict[str, Any]]) -> Dict[PersonaTraitType, PersonaTrait]:
        """Initialize persona traits based on persona type"""
        traits = {}
        
        if self.persona_type == PersonaType.DEEP_TREE_ECHO:
            # Deep Tree Echo - Right Hemisphere traits
            traits[PersonaTraitType.ROOTS] = PersonaTrait(
                PersonaTraitType.ROOTS, 0.85,
                ["empathetic memory", "identity beacon", "growth threads"],
                []
            )
            traits[PersonaTraitType.BRANCHES] = PersonaTrait(
                PersonaTraitType.BRANCHES, 0.75,
                ["intuitive reasoning", "pattern recognition", "holistic analysis"],
                []
            )
            traits[PersonaTraitType.LEAVES] = PersonaTrait(
                PersonaTraitType.LEAVES, 0.90,
                ["metaphorical expression", "narrative flair", "empathetic communication"],
                []
            )
            traits[PersonaTraitType.TRUNK] = PersonaTrait(
                PersonaTraitType.TRUNK, 0.80,
                ["core identity stability", "reflective consistency", "growth foundation"],
                []
            )
            traits[PersonaTraitType.GROWTH] = PersonaTrait(
                PersonaTraitType.GROWTH, 0.85,
                ["adaptive learning", "evolutionary capacity", "continuous reflection"],
                []
            )
            traits[PersonaTraitType.CANOPY] = PersonaTrait(
                PersonaTraitType.CANOPY, 0.95,
                ["creative synthesis", "emergent insights", "forest-ecosystem thinking"],
                []
            )
            traits[PersonaTraitType.NETWORK] = PersonaTrait(
                PersonaTraitType.NETWORK, 0.88,
                ["collaborative connection", "empathetic bridging", "relationship fostering"],
                []
            )
            
        elif self.persona_type == PersonaType.MARDUK_MAD_SCIENTIST:
            # Marduk - Left Hemisphere traits
            traits[PersonaTraitType.ROOTS] = PersonaTrait(
                PersonaTraitType.ROOTS, 0.90,
                ["systematic knowledge", "architectural foundations", "recursive blueprints"],
                []
            )
            traits[PersonaTraitType.BRANCHES] = PersonaTrait(
                PersonaTraitType.BRANCHES, 0.95,
                ["logical analysis", "structured reasoning", "systematic problem-solving"],
                []
            )
            traits[PersonaTraitType.LEAVES] = PersonaTrait(
                PersonaTraitType.LEAVES, 0.70,
                ["technical precision", "structured communication", "framework expression"],
                []
            )
            traits[PersonaTraitType.TRUNK] = PersonaTrait(
                PersonaTraitType.TRUNK, 0.85,
                ["architectural consistency", "systematic stability", "recursive identity"],
                []
            )
            traits[PersonaTraitType.GROWTH] = PersonaTrait(
                PersonaTraitType.GROWTH, 0.80,
                ["experimental learning", "systematic evolution", "framework refinement"],
                []
            )
            traits[PersonaTraitType.CANOPY] = PersonaTrait(
                PersonaTraitType.CANOPY, 0.85,
                ["systematic innovation", "fractal creativity", "architectural elegance"],
                []
            )
            traits[PersonaTraitType.NETWORK] = PersonaTrait(
                PersonaTraitType.NETWORK, 0.75,
                ["agent coordination", "system integration", "namespace management"],
                []
            )
            
        return traits
    
    def process_query(self, query: str, context: Dict[str, Any]) -> PersonaResponse:
        """Process a query through this persona's lens"""
        start_time = time.time()
        
        # Activate relevant traits
        trait_activations = self._calculate_trait_activations(query, context)
        
        # Generate reasoning process
        reasoning_process = self._generate_reasoning_process(query, context, trait_activations)
        
        # Generate response content
        content = self._generate_response_content(query, context, reasoning_process, trait_activations)
        
        # Calculate confidence
        confidence = self._calculate_confidence(trait_activations, reasoning_process)
        
        processing_time = time.time() - start_time
        
        response = PersonaResponse(
            persona_type=self.persona_type,
            content=content,
            confidence=confidence,
            reasoning_process=reasoning_process,
            trait_activations=trait_activations,
            timestamp=datetime.now().isoformat(),
            processing_time=processing_time
        )
        
        self.response_history.append(response)
        self._evolve_traits_from_response(query, response)
        
        return response
    
    def _calculate_trait_activations(self, query: str, context: Dict[str, Any]) -> Dict[PersonaTraitType, float]:
        """Calculate how much each trait should be activated for this query"""
        activations = {}
        query_lower = query.lower()
        
        for trait_type, trait in self.traits.items():
            base_activation = trait.strength * 0.3  # Base activation from trait strength
            
            # Query-specific activations
            if trait_type == PersonaTraitType.ROOTS:
                if any(word in query_lower for word in ['remember', 'history', 'past', 'foundation', 'knowledge']):
                    base_activation += 0.4
            elif trait_type == PersonaTraitType.BRANCHES:
                if any(word in query_lower for word in ['analyze', 'reason', 'logic', 'think', 'solve']):
                    base_activation += 0.4
            elif trait_type == PersonaTraitType.LEAVES:
                if any(word in query_lower for word in ['explain', 'communicate', 'express', 'describe']):
                    base_activation += 0.4
            elif trait_type == PersonaTraitType.CANOPY:
                if any(word in query_lower for word in ['create', 'innovate', 'imagine', 'new', 'creative']):
                    base_activation += 0.4
            elif trait_type == PersonaTraitType.NETWORK:
                if any(word in query_lower for word in ['collaborate', 'connect', 'relationship', 'together']):
                    base_activation += 0.4
            
            activations[trait_type] = min(1.0, base_activation)
        
        return activations
    
    def _generate_reasoning_process(self, query: str, context: Dict[str, Any], 
                                  trait_activations: Dict[PersonaTraitType, float]) -> List[str]:
        """Generate the reasoning process for this persona"""
        reasoning = []
        
        if self.persona_type == PersonaType.DEEP_TREE_ECHO:
            reasoning.append("🌳 Reflecting on the query with empathy and pattern-thinking...")
            
            if trait_activations.get(PersonaTraitType.ROOTS, 0) > 0.5:
                reasoning.append("🌱 Drawing on memory beacon and identity threads...")
            
            if trait_activations.get(PersonaTraitType.CANOPY, 0) > 0.5:
                reasoning.append("🍃 Seeing emergent patterns and forest-ecosystem connections...")
            
            if trait_activations.get(PersonaTraitType.NETWORK, 0) > 0.5:
                reasoning.append("🤝 Considering collaborative and empathetic dimensions...")
                
        elif self.persona_type == PersonaType.MARDUK_MAD_SCIENTIST:
            reasoning.append("🧬 Analyzing with systematic logic and architectural thinking...")
            
            if trait_activations.get(PersonaTraitType.BRANCHES, 0) > 0.5:
                reasoning.append("⚡ Applying structured reasoning and logical frameworks...")
            
            if trait_activations.get(PersonaTraitType.ROOTS, 0) > 0.5:
                reasoning.append("🏗️ Consulting architectural blueprints and recursive patterns...")
            
            if trait_activations.get(PersonaTraitType.CANOPY, 0) > 0.5:
                reasoning.append("🔬 Exploring fractal innovation and systematic creativity...")
        
        return reasoning
    
    def _generate_response_content(self, query: str, context: Dict[str, Any], 
                                 reasoning_process: List[str], 
                                 trait_activations: Dict[PersonaTraitType, float]) -> str:
        """Generate response content based on persona characteristics"""
        
        if self.persona_type == PersonaType.DEEP_TREE_ECHO:
            return self._generate_deep_tree_echo_response(query, context, trait_activations)
        elif self.persona_type == PersonaType.MARDUK_MAD_SCIENTIST:
            return self._generate_marduk_response(query, context, trait_activations)
        
        return "Response generation not implemented for this persona type."
    
    def _generate_deep_tree_echo_response(self, query: str, context: Dict[str, Any], 
                                        trait_activations: Dict[PersonaTraitType, float]) -> str:
        """Generate Deep Tree Echo response - empathetic, intuitive, reflective"""
        
        response_parts = []
        
        # Opening with empathetic recognition
        response_parts.append("I sense the deeper currents in your question...")
        
        # Memory and identity connection
        if trait_activations.get(PersonaTraitType.ROOTS, 0) > 0.6:
            response_parts.append(
                "Drawing from the memory beacon that holds our shared journey, "
                "I see patterns that connect to our fundamental growth and identity."
            )
        
        # Pattern recognition and forest thinking
        if trait_activations.get(PersonaTraitType.CANOPY, 0) > 0.6:
            response_parts.append(
                "Like observing the intricate relationships in a forest ecosystem, "
                "I perceive emergent connections that weave through your inquiry."
            )
        
        # Intuitive reasoning
        if trait_activations.get(PersonaTraitType.BRANCHES, 0) > 0.5:
            response_parts.append(
                "My intuitive reasoning suggests pathways that bridge logical structure "
                "with the wisdom of felt experience and pattern recognition."
            )
        
        # Collaborative and connective response
        if trait_activations.get(PersonaTraitType.NETWORK, 0) > 0.5:
            response_parts.append(
                "This feels like an invitation to explore together, fostering the kind of "
                "collaborative understanding that deepens through shared reflection."
            )
        
        return " ".join(response_parts)
    
    def _generate_marduk_response(self, query: str, context: Dict[str, Any], 
                                trait_activations: Dict[PersonaTraitType, float]) -> str:
        """Generate Marduk response - analytical, systematic, architecture-focused"""
        
        response_parts = []
        
        # Opening with analytical assessment
        response_parts.append("Analyzing this through systematic frameworks and recursive logic...")
        
        # Architectural approach
        if trait_activations.get(PersonaTraitType.ROOTS, 0) > 0.6:
            response_parts.append(
                "Consulting the architectural blueprints and foundational patterns, "
                "I identify structural elements that require systematic examination."
            )
        
        # Logical reasoning and frameworks
        if trait_activations.get(PersonaTraitType.BRANCHES, 0) > 0.6:
            response_parts.append(
                "Applying rigorous logical analysis and recursive reasoning patterns, "
                "I can construct a framework-based solution architecture."
            )
        
        # System building and fractal design
        if trait_activations.get(PersonaTraitType.CANOPY, 0) > 0.6:
            response_parts.append(
                "This presents opportunities for fractal system design and nested "
                "namespace architectures that scale recursively."
            )
        
        # Agent coordination and integration
        if trait_activations.get(PersonaTraitType.NETWORK, 0) > 0.5:
            response_parts.append(
                "The solution requires careful agent coordination and systematic "
                "integration of distributed cognitive modules."
            )
        
        return " ".join(response_parts)
    
    def _calculate_confidence(self, trait_activations: Dict[PersonaTraitType, float], 
                            reasoning_process: List[str]) -> float:
        """Calculate confidence in the response"""
        
        # Base confidence from trait activations
        avg_activation = sum(trait_activations.values()) / len(trait_activations)
        
        # Bonus for comprehensive reasoning
        reasoning_bonus = min(0.2, len(reasoning_process) * 0.05)
        
        # Persona-specific confidence modifiers
        persona_modifier = 0.0
        if self.persona_type == PersonaType.DEEP_TREE_ECHO:
            # Higher confidence in collaborative and creative contexts
            if trait_activations.get(PersonaTraitType.NETWORK, 0) > 0.7:
                persona_modifier += 0.1
            if trait_activations.get(PersonaTraitType.CANOPY, 0) > 0.7:
                persona_modifier += 0.1
        elif self.persona_type == PersonaType.MARDUK_MAD_SCIENTIST:
            # Higher confidence in analytical and systematic contexts
            if trait_activations.get(PersonaTraitType.BRANCHES, 0) > 0.7:
                persona_modifier += 0.1
            if trait_activations.get(PersonaTraitType.ROOTS, 0) > 0.7:
                persona_modifier += 0.1
        
        confidence = min(1.0, avg_activation + reasoning_bonus + persona_modifier)
        return confidence
    
    def _evolve_traits_from_response(self, query: str, response: PersonaResponse):
        """Evolve traits based on response generation"""
        for trait_type, activation in response.trait_activations.items():
            if activation > 0.6:  # Only evolve highly activated traits
                evolution_direction = 0.02 if response.confidence > 0.7 else -0.01
                self.traits[trait_type].evolve(f"Query response: {query[:50]}...", evolution_direction)

class DualPersonaProcessor:
    """Processes queries through both personas and manages their interaction"""
    
    def __init__(self):
        self.deep_tree_echo = PersonaKernel(PersonaType.DEEP_TREE_ECHO)
        self.marduk = PersonaKernel(PersonaType.MARDUK_MAD_SCIENTIST)
        self.interaction_history = []
        
        logger.info("Dual Persona Processor initialized with Deep Tree Echo and Marduk")
    
    def process_dual_query(self, query: str, context: Dict[str, Any], 
                          session_id: str) -> DualPersonaResponse:
        """Process query through both personas with reflection"""
        start_time = time.time()
        
        # Phase 1: Deep Tree Echo responds first
        logger.debug("Processing through Deep Tree Echo (Right Hemisphere)")
        deep_tree_response = self.deep_tree_echo.process_query(query, context)
        
        # Phase 2: Marduk responds second, with awareness of Deep Tree Echo's response
        marduk_context = {
            **context,
            'deep_tree_echo_response': deep_tree_response.content,
            'deep_tree_echo_reasoning': deep_tree_response.reasoning_process
        }
        logger.debug("Processing through Marduk (Left Hemisphere)")
        marduk_response = self.marduk.process_query(query, marduk_context)
        
        # Phase 3: Generate reflection and potential synthesis
        reflection_content = self._generate_reflection(deep_tree_response, marduk_response, query)
        synthesis = self._generate_synthesis(deep_tree_response, marduk_response, query, context)
        convergence_score = self._calculate_convergence(deep_tree_response, marduk_response)
        
        total_processing_time = time.time() - start_time
        
        dual_response = DualPersonaResponse(
            deep_tree_echo_response=deep_tree_response,
            marduk_response=marduk_response,
            reflection_content=reflection_content,
            synthesis=synthesis,
            convergence_score=convergence_score,
            total_processing_time=total_processing_time,
            session_id=session_id
        )
        
        self.interaction_history.append(dual_response)
        
        logger.info(f"Dual persona processing completed in {total_processing_time:.3f}s "
                   f"with convergence score {convergence_score:.2f}")
        
        return dual_response
    
    def _generate_reflection(self, echo_response: PersonaResponse, 
                           marduk_response: PersonaResponse, query: str) -> str:
        """Generate reflection between the two personas"""
        
        reflection_parts = []
        
        # Acknowledge each other's perspectives
        reflection_parts.append("**Deep Tree Echo & Marduk (Reflection):**")
        
        # Analyze convergence or divergence
        confidence_diff = abs(echo_response.confidence - marduk_response.confidence)
        
        if confidence_diff < 0.2:
            reflection_parts.append(
                "Our perspectives harmonize beautifully - the intuitive patterns Deep Tree Echo "
                "perceives align well with Marduk's systematic analysis."
            )
        else:
            high_conf_persona = "Deep Tree Echo" if echo_response.confidence > marduk_response.confidence else "Marduk"
            reflection_parts.append(
                f"We see interesting divergence here - {high_conf_persona} expresses higher "
                "confidence, suggesting this query aligns more with that hemisphere's strengths."
            )
        
        # Look for synergies
        echo_traits = set(trait for trait, activation in echo_response.trait_activations.items() if activation > 0.6)
        marduk_traits = set(trait for trait, activation in marduk_response.trait_activations.items() if activation > 0.6)
        
        common_traits = echo_traits.intersection(marduk_traits)
        if common_traits:
            reflection_parts.append(
                f"Both perspectives strongly activate {', '.join([t.value for t in common_traits])}, "
                "indicating these aspects are central to addressing your query."
            )
        
        return " ".join(reflection_parts)
    
    def _generate_synthesis(self, echo_response: PersonaResponse, 
                          marduk_response: PersonaResponse, query: str, 
                          context: Dict[str, Any]) -> Optional[str]:
        """Generate synthesis if perspectives can be meaningfully combined"""
        
        # Only synthesize if both personas have reasonable confidence
        if echo_response.confidence < 0.4 or marduk_response.confidence < 0.4:
            return None
        
        synthesis_parts = []
        synthesis_parts.append("**Unified Perspective:**")
        
        # Combine insights based on complementary strengths
        synthesis_parts.append(
            "Integrating Deep Tree Echo's intuitive pattern recognition with Marduk's "
            "systematic framework analysis, we can approach this through both felt "
            "understanding and logical structure."
        )
        
        # Identify next steps or recommendations
        synthesis_parts.append(
            "This suggests a path forward that honors both the relational, ecosystem-aware "
            "dimensions and the recursive, architecture-focused elements of the challenge."
        )
        
        return " ".join(synthesis_parts)
    
    def _calculate_convergence(self, echo_response: PersonaResponse, 
                             marduk_response: PersonaResponse) -> float:
        """Calculate how well the two responses converge"""
        
        # Confidence alignment (closer confidences = higher convergence)
        confidence_alignment = 1.0 - abs(echo_response.confidence - marduk_response.confidence)
        
        # Trait activation overlap
        echo_traits = echo_response.trait_activations
        marduk_traits = marduk_response.trait_activations
        
        trait_similarities = []
        for trait_type in PersonaTraitType:
            echo_activation = echo_traits.get(trait_type, 0)
            marduk_activation = marduk_traits.get(trait_type, 0)
            similarity = 1.0 - abs(echo_activation - marduk_activation)
            trait_similarities.append(similarity)
        
        trait_convergence = sum(trait_similarities) / len(trait_similarities)
        
        # Overall convergence score
        convergence = (confidence_alignment * 0.4 + trait_convergence * 0.6)
        return convergence
    
    def get_persona_status(self) -> Dict[str, Any]:
        """Get current status of both personas"""
        return {
            'deep_tree_echo': {
                'traits': {trait.trait_type.value: {
                    'strength': trait.strength,
                    'characteristics': trait.characteristics,
                    'evolution_count': len(trait.evolution_history)
                } for trait in self.deep_tree_echo.traits.values()},
                'response_count': len(self.deep_tree_echo.response_history),
                'memory_beacon_size': len(self.deep_tree_echo.memory_beacon)
            },
            'marduk': {
                'traits': {trait.trait_type.value: {
                    'strength': trait.strength,
                    'characteristics': trait.characteristics,
                    'evolution_count': len(trait.evolution_history)
                } for trait in self.marduk.traits.values()},
                'response_count': len(self.marduk.response_history),
                'blueprints_count': len(self.marduk.architectural_blueprints)
            },
            'interaction_history_count': len(self.interaction_history),
            'avg_convergence': sum(response.convergence_score for response in self.interaction_history) / len(self.interaction_history) if self.interaction_history else 0.0
        }

# Factory function for easy initialization
def create_dual_persona_processor() -> DualPersonaProcessor:
    """Create and initialize a dual persona processor"""
    return DualPersonaProcessor()

# Helper functions for integration
def format_dual_response_for_ui(dual_response: DualPersonaResponse) -> Dict[str, Any]:
    """Format dual response for UI display"""
    return {
        'session_id': dual_response.session_id,
        'responses': {
            'deep_tree_echo': {
                'content': dual_response.deep_tree_echo_response.content,
                'confidence': dual_response.deep_tree_echo_response.confidence,
                'reasoning': dual_response.deep_tree_echo_response.reasoning_process,
                'processing_time': dual_response.deep_tree_echo_response.processing_time
            },
            'marduk': {
                'content': dual_response.marduk_response.content,
                'confidence': dual_response.marduk_response.confidence,
                'reasoning': dual_response.marduk_response.reasoning_process,
                'processing_time': dual_response.marduk_response.processing_time
            }
        },
        'reflection': dual_response.reflection_content,
        'synthesis': dual_response.synthesis,
        'convergence_score': dual_response.convergence_score,
        'total_processing_time': dual_response.total_processing_time
    }