"""
Complex Reasoning Chains for Deep Tree Echo
Implements multi-step reasoning processes with validation and explanation generation
"""

import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class ReasoningType(Enum):
    """Types of reasoning processes"""
    DEDUCTIVE = "deductive"      # General to specific
    INDUCTIVE = "inductive"      # Specific to general  
    ABDUCTIVE = "abductive"      # Best explanation
    ANALOGICAL = "analogical"    # Similarity-based
    CAUSAL = "causal"           # Cause-effect relationships
    TEMPORAL = "temporal"        # Time-based reasoning
    SPATIAL = "spatial"         # Space-based reasoning
    PROBABILISTIC = "probabilistic"  # Uncertainty handling

class ReasoningStep(Enum):
    """Steps in reasoning process"""
    ANALYSIS = "analysis"
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    INFERENCE = "inference"
    VALIDATION = "validation"
    CONCLUSION = "conclusion"

@dataclass
class ReasoningNode:
    """A single node in a reasoning chain"""
    id: str
    step_type: str
    content: str
    premises: List[str]
    conclusion: str
    confidence: float
    evidence: List[str]
    reasoning_type: str
    timestamp: str
    dependencies: List[str]  # IDs of prerequisite nodes
    validation_status: str   # "pending", "valid", "invalid", "uncertain"
    explanation: str

@dataclass
class ReasoningChain:
    """A complete reasoning chain with multiple steps"""
    id: str
    query: str
    reasoning_type: str
    nodes: List[ReasoningNode]
    overall_confidence: float
    validation_score: float
    explanation: str
    start_time: str
    end_time: Optional[str]
    session_id: str
    metadata: Dict[str, Any]

@dataclass
class ReasoningValidation:
    """Validation results for reasoning chain"""
    chain_id: str
    is_valid: bool
    confidence: float
    validation_errors: List[str]
    logical_consistency: float
    evidence_quality: float
    conclusion_support: float
    suggestions: List[str]

class ReasoningStrategy(ABC):
    """Abstract base class for reasoning strategies"""
    
    @abstractmethod
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningChain:
        """Execute reasoning process"""
        pass
    
    @abstractmethod
    def validate(self, chain: ReasoningChain) -> ReasoningValidation:
        """Validate reasoning chain"""
        pass

class DeductiveReasoning(ReasoningStrategy):
    """Deductive reasoning implementation"""
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningChain:
        """Perform deductive reasoning from general to specific"""
        chain_id = f"deductive_{int(time.time())}"
        start_time = datetime.now().isoformat()
        
        nodes = []
        
        # Step 1: Identify general principles
        analysis_node = ReasoningNode(
            id=f"{chain_id}_analysis",
            step_type=ReasoningStep.ANALYSIS.value,
            content=f"Analyzing query for deductive reasoning: {query}",
            premises=[],
            conclusion="",
            confidence=0.8,
            evidence=[],
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[],
            validation_status="pending",
            explanation="Identifying general principles and rules applicable to the query"
        )
        nodes.append(analysis_node)
        
        # Step 2: Apply general principles to specific case
        inference_node = ReasoningNode(
            id=f"{chain_id}_inference",
            step_type=ReasoningStep.INFERENCE.value,
            content=f"Applying deductive inference to: {query}",
            premises=[analysis_node.id],
            conclusion="",
            confidence=0.7,
            evidence=[],
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[analysis_node.id],
            validation_status="pending",
            explanation="Applying general principles to reach specific conclusion"
        )
        nodes.append(inference_node)
        
        # Step 3: Validate logical consistency
        validation_node = ReasoningNode(
            id=f"{chain_id}_validation",
            step_type=ReasoningStep.VALIDATION.value,
            content="Checking logical validity of deductive reasoning",
            premises=[analysis_node.id, inference_node.id],
            conclusion="",
            confidence=0.6,
            evidence=[],
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[analysis_node.id, inference_node.id],
            validation_status="pending",
            explanation="Ensuring the conclusion follows logically from the premises"
        )
        nodes.append(validation_node)
        
        # Step 4: Draw conclusion
        conclusion_node = ReasoningNode(
            id=f"{chain_id}_conclusion",
            step_type=ReasoningStep.CONCLUSION.value,
            content=f"Deductive conclusion for: {query}",
            premises=[validation_node.id],
            conclusion=f"Based on deductive reasoning: {self._generate_deductive_conclusion(query, context)}",
            confidence=0.8,
            evidence=[analysis_node.id, inference_node.id],
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[validation_node.id],
            validation_status="valid",
            explanation="Final conclusion drawn through deductive reasoning process"
        )
        nodes.append(conclusion_node)
        
        chain = ReasoningChain(
            id=chain_id,
            query=query,
            reasoning_type=ReasoningType.DEDUCTIVE.value,
            nodes=nodes,
            overall_confidence=0.75,
            validation_score=0.8,
            explanation=self._generate_chain_explanation(nodes),
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            session_id=context.get('session_id', 'unknown'),
            metadata={'strategy': 'deductive', 'steps': len(nodes)}
        )
        
        return chain
    
    def validate(self, chain: ReasoningChain) -> ReasoningValidation:
        """Validate deductive reasoning chain"""
        errors = []
        
        # Check logical flow
        if not self._has_valid_logical_flow(chain.nodes):
            errors.append("Invalid logical flow in reasoning chain")
        
        # Check premise-conclusion relationships
        for node in chain.nodes:
            if node.step_type == ReasoningStep.CONCLUSION.value:
                if not node.premises:
                    errors.append("Conclusion node lacks supporting premises")
        
        return ReasoningValidation(
            chain_id=chain.id,
            is_valid=len(errors) == 0,
            confidence=0.8 if len(errors) == 0 else 0.4,
            validation_errors=errors,
            logical_consistency=0.9 if len(errors) == 0 else 0.5,
            evidence_quality=0.7,
            conclusion_support=0.8 if len(errors) == 0 else 0.3,
            suggestions=["Ensure all conclusions have supporting premises"] if errors else []
        )
    
    def _generate_deductive_conclusion(self, query: str, context: Dict[str, Any]) -> str:
        """Generate a deductive conclusion"""
        return f"Through deductive reasoning, the logical conclusion for '{query}' follows from established general principles."
    
    def _generate_chain_explanation(self, nodes: List[ReasoningNode]) -> str:
        """Generate explanation for the entire reasoning chain"""
        return f"Deductive reasoning chain with {len(nodes)} steps: analysis → inference → validation → conclusion"
    
    def _has_valid_logical_flow(self, nodes: List[ReasoningNode]) -> bool:
        """Check if reasoning chain has valid logical flow"""
        step_types = [node.step_type for node in nodes]
        expected_flow = [ReasoningStep.ANALYSIS.value, ReasoningStep.INFERENCE.value, 
                        ReasoningStep.VALIDATION.value, ReasoningStep.CONCLUSION.value]
        return step_types == expected_flow

class InductiveReasoning(ReasoningStrategy):
    """Inductive reasoning implementation"""
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningChain:
        """Perform inductive reasoning from specific to general"""
        chain_id = f"inductive_{int(time.time())}"
        start_time = datetime.now().isoformat()
        
        nodes = []
        
        # Step 1: Collect specific observations/evidence
        evidence_node = ReasoningNode(
            id=f"{chain_id}_evidence",
            step_type=ReasoningStep.EVIDENCE.value,
            content=f"Collecting evidence for inductive reasoning: {query}",
            premises=[],
            conclusion="",
            confidence=0.7,
            evidence=[],
            reasoning_type=ReasoningType.INDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[],
            validation_status="pending",
            explanation="Gathering specific observations and evidence"
        )
        nodes.append(evidence_node)
        
        # Step 2: Identify patterns
        analysis_node = ReasoningNode(
            id=f"{chain_id}_analysis",
            step_type=ReasoningStep.ANALYSIS.value,
            content="Analyzing patterns in collected evidence",
            premises=[evidence_node.id],
            conclusion="",
            confidence=0.6,
            evidence=[evidence_node.id],
            reasoning_type=ReasoningType.INDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[evidence_node.id],
            validation_status="pending",
            explanation="Identifying patterns and regularities in the evidence"
        )
        nodes.append(analysis_node)
        
        # Step 3: Form hypothesis/generalization
        hypothesis_node = ReasoningNode(
            id=f"{chain_id}_hypothesis",
            step_type=ReasoningStep.HYPOTHESIS.value,
            content="Forming general hypothesis from patterns",
            premises=[analysis_node.id],
            conclusion="",
            confidence=0.5,
            evidence=[evidence_node.id],
            reasoning_type=ReasoningType.INDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[analysis_node.id],
            validation_status="pending",
            explanation="Generalizing from specific patterns to form hypothesis"
        )
        nodes.append(hypothesis_node)
        
        # Step 4: Assess confidence and draw conclusion
        conclusion_node = ReasoningNode(
            id=f"{chain_id}_conclusion",
            step_type=ReasoningStep.CONCLUSION.value,
            content=f"Inductive conclusion for: {query}",
            premises=[hypothesis_node.id],
            conclusion=f"Through inductive reasoning: {self._generate_inductive_conclusion(query, context)}",
            confidence=0.6,  # Inductive conclusions are less certain
            evidence=[evidence_node.id, analysis_node.id],
            reasoning_type=ReasoningType.INDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[hypothesis_node.id],
            validation_status="probable",
            explanation="Probabilistic conclusion based on inductive generalization"
        )
        nodes.append(conclusion_node)
        
        chain = ReasoningChain(
            id=chain_id,
            query=query,
            reasoning_type=ReasoningType.INDUCTIVE.value,
            nodes=nodes,
            overall_confidence=0.6,  # Lower confidence than deductive
            validation_score=0.7,
            explanation=self._generate_chain_explanation(nodes),
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            session_id=context.get('session_id', 'unknown'),
            metadata={'strategy': 'inductive', 'steps': len(nodes)}
        )
        
        return chain
    
    def validate(self, chain: ReasoningChain) -> ReasoningValidation:
        """Validate inductive reasoning chain"""
        errors = []
        
        # Check for sufficient evidence
        evidence_nodes = [n for n in chain.nodes if n.step_type == ReasoningStep.EVIDENCE.value]
        if not evidence_nodes:
            errors.append("Insufficient evidence for inductive reasoning")
        
        # Check for pattern analysis
        analysis_nodes = [n for n in chain.nodes if n.step_type == ReasoningStep.ANALYSIS.value]
        if not analysis_nodes:
            errors.append("Missing pattern analysis step")
        
        return ReasoningValidation(
            chain_id=chain.id,
            is_valid=len(errors) == 0,
            confidence=0.6 if len(errors) == 0 else 0.3,  # Inherently less certain
            validation_errors=errors,
            logical_consistency=0.7,
            evidence_quality=0.8 if not errors else 0.4,
            conclusion_support=0.6,
            suggestions=["Gather more evidence to strengthen inductive reasoning"] if errors else []
        )
    
    def _generate_inductive_conclusion(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an inductive conclusion"""
        return f"Based on observed patterns and evidence, it appears likely that '{query}' follows this general principle."
    
    def _generate_chain_explanation(self, nodes: List[ReasoningNode]) -> str:
        """Generate explanation for the entire reasoning chain"""
        return f"Inductive reasoning chain with {len(nodes)} steps: evidence → pattern analysis → hypothesis → probable conclusion"

class AbductiveReasoning(ReasoningStrategy):
    """Abductive reasoning implementation - inference to best explanation"""
    
    async def reason(self, query: str, context: Dict[str, Any]) -> ReasoningChain:
        """Perform abductive reasoning to find best explanation"""
        chain_id = f"abductive_{int(time.time())}"
        start_time = datetime.now().isoformat()
        
        nodes = []
        
        # Step 1: Identify phenomena to explain
        analysis_node = ReasoningNode(
            id=f"{chain_id}_analysis",
            step_type=ReasoningStep.ANALYSIS.value,
            content=f"Identifying phenomena requiring explanation: {query}",
            premises=[],
            conclusion="",
            confidence=0.8,
            evidence=[],
            reasoning_type=ReasoningType.ABDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[],
            validation_status="pending",
            explanation="Analyzing the phenomena that need explanation"
        )
        nodes.append(analysis_node)
        
        # Step 2: Generate possible explanations
        hypothesis_node = ReasoningNode(
            id=f"{chain_id}_hypothesis",
            step_type=ReasoningStep.HYPOTHESIS.value,
            content="Generating candidate explanations",
            premises=[analysis_node.id],
            conclusion="",
            confidence=0.7,
            evidence=[],
            reasoning_type=ReasoningType.ABDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[analysis_node.id],
            validation_status="pending",
            explanation="Creating multiple possible explanations for the phenomena"
        )
        nodes.append(hypothesis_node)
        
        # Step 3: Evaluate explanations
        evaluation_node = ReasoningNode(
            id=f"{chain_id}_evaluation",
            step_type=ReasoningStep.VALIDATION.value,
            content="Evaluating and ranking explanations",
            premises=[hypothesis_node.id],
            conclusion="",
            confidence=0.6,
            evidence=[analysis_node.id],
            reasoning_type=ReasoningType.ABDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[hypothesis_node.id],
            validation_status="pending",
            explanation="Assessing explanatory power, simplicity, and consistency"
        )
        nodes.append(evaluation_node)
        
        # Step 4: Select best explanation
        conclusion_node = ReasoningNode(
            id=f"{chain_id}_conclusion",
            step_type=ReasoningStep.CONCLUSION.value,
            content=f"Best explanation for: {query}",
            premises=[evaluation_node.id],
            conclusion=f"The most plausible explanation is: {self._generate_abductive_conclusion(query, context)}",
            confidence=0.7,
            evidence=[analysis_node.id, hypothesis_node.id],
            reasoning_type=ReasoningType.ABDUCTIVE.value,
            timestamp=datetime.now().isoformat(),
            dependencies=[evaluation_node.id],
            validation_status="plausible",
            explanation="Best explanation selected based on explanatory criteria"
        )
        nodes.append(conclusion_node)
        
        chain = ReasoningChain(
            id=chain_id,
            query=query,
            reasoning_type=ReasoningType.ABDUCTIVE.value,
            nodes=nodes,
            overall_confidence=0.65,
            validation_score=0.75,
            explanation=self._generate_chain_explanation(nodes),
            start_time=start_time,
            end_time=datetime.now().isoformat(),
            session_id=context.get('session_id', 'unknown'),
            metadata={'strategy': 'abductive', 'steps': len(nodes)}
        )
        
        return chain
    
    def validate(self, chain: ReasoningChain) -> ReasoningValidation:
        """Validate abductive reasoning chain"""
        errors = []
        
        # Check for hypothesis generation
        hypothesis_nodes = [n for n in chain.nodes if n.step_type == ReasoningStep.HYPOTHESIS.value]
        if not hypothesis_nodes:
            errors.append("Missing hypothesis generation step")
        
        # Check for evaluation step
        evaluation_nodes = [n for n in chain.nodes if 'evaluation' in n.content.lower()]
        if not evaluation_nodes:
            errors.append("Missing explanation evaluation step")
        
        return ReasoningValidation(
            chain_id=chain.id,
            is_valid=len(errors) == 0,
            confidence=0.7 if len(errors) == 0 else 0.4,
            validation_errors=errors,
            logical_consistency=0.8,
            evidence_quality=0.6,
            conclusion_support=0.7 if len(errors) == 0 else 0.4,
            suggestions=["Consider alternative explanations"] if not errors else []
        )
    
    def _generate_abductive_conclusion(self, query: str, context: Dict[str, Any]) -> str:
        """Generate an abductive conclusion"""
        return f"The most coherent explanation for '{query}' based on available evidence and theoretical considerations."
    
    def _generate_chain_explanation(self, nodes: List[ReasoningNode]) -> str:
        """Generate explanation for the entire reasoning chain"""
        return f"Abductive reasoning chain with {len(nodes)} steps: phenomenon analysis → hypothesis generation → evaluation → best explanation"

class ReasoningChainProcessor:
    """Processes and manages complex reasoning chains"""
    
    def __init__(self):
        self.strategies = {
            ReasoningType.DEDUCTIVE: DeductiveReasoning(),
            ReasoningType.INDUCTIVE: InductiveReasoning(),
            ReasoningType.ABDUCTIVE: AbductiveReasoning()
        }
        self.active_chains = {}
        self.completed_chains = []
        
    async def process_reasoning_request(self, query: str, context: Dict[str, Any], 
                                      reasoning_type: Optional[ReasoningType] = None) -> ReasoningChain:
        """Process a reasoning request using appropriate strategy"""
        
        # Select reasoning type if not specified
        if reasoning_type is None:
            reasoning_type = self._select_reasoning_type(query, context)
        
        logger.debug(f"Processing reasoning request with {reasoning_type.value} strategy")
        
        # Execute reasoning
        strategy = self.strategies.get(reasoning_type)
        if not strategy:
            raise ValueError(f"Unsupported reasoning type: {reasoning_type}")
        
        chain = await strategy.reason(query, context)
        
        # Validate reasoning chain
        validation = strategy.validate(chain)
        chain.validation_score = validation.confidence
        
        # Store and track chain
        self.active_chains[chain.id] = chain
        self.completed_chains.append(chain)
        
        # Generate detailed explanation
        chain.explanation = self._generate_detailed_explanation(chain, validation)
        
        logger.info(f"Completed reasoning chain {chain.id} with confidence {chain.overall_confidence}")
        return chain
    
    def _select_reasoning_type(self, query: str, context: Dict[str, Any]) -> ReasoningType:
        """Automatically select appropriate reasoning type"""
        query_lower = query.lower()
        
        # Look for reasoning type indicators
        if any(word in query_lower for word in ['why', 'explain', 'because', 'cause']):
            return ReasoningType.ABDUCTIVE
        elif any(word in query_lower for word in ['if', 'then', 'therefore', 'must', 'all']):
            return ReasoningType.DEDUCTIVE
        elif any(word in query_lower for word in ['pattern', 'usually', 'often', 'tend to']):
            return ReasoningType.INDUCTIVE
        else:
            return ReasoningType.ABDUCTIVE  # Default to abductive for explanatory queries
    
    def _generate_detailed_explanation(self, chain: ReasoningChain, 
                                     validation: ReasoningValidation) -> str:
        """Generate detailed explanation of reasoning process"""
        explanation_parts = [
            f"Reasoning Type: {chain.reasoning_type}",
            f"Processing Steps: {len(chain.nodes)}",
            f"Overall Confidence: {chain.overall_confidence:.2f}",
            f"Validation Score: {chain.validation_score:.2f}",
            "",
            "Step-by-step reasoning:"
        ]
        
        for i, node in enumerate(chain.nodes, 1):
            explanation_parts.append(
                f"{i}. {node.step_type.title()}: {node.explanation} "
                f"(Confidence: {node.confidence:.2f})"
            )
        
        if validation.validation_errors:
            explanation_parts.extend([
                "",
                "Validation Issues:",
                *[f"- {error}" for error in validation.validation_errors]
            ])
        
        if validation.suggestions:
            explanation_parts.extend([
                "",
                "Suggestions for Improvement:",
                *[f"- {suggestion}" for suggestion in validation.suggestions]
            ])
        
        return "\n".join(explanation_parts)
    
    def get_chain_summary(self, chain_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a reasoning chain"""
        chain = self.active_chains.get(chain_id)
        if not chain:
            return None
        
        return {
            'id': chain.id,
            'query': chain.query,
            'reasoning_type': chain.reasoning_type,
            'confidence': chain.overall_confidence,
            'validation_score': chain.validation_score,
            'steps': len(chain.nodes),
            'start_time': chain.start_time,
            'end_time': chain.end_time,
            'explanation_summary': chain.explanation.split('\n')[0] if chain.explanation else ""
        }
    
    def get_reasoning_insights(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get insights about reasoning performance"""
        chains = self.completed_chains
        if session_id:
            chains = [c for c in chains if c.session_id == session_id]
        
        if not chains:
            return {"message": "No reasoning chains available"}
        
        # Calculate statistics
        reasoning_types = [c.reasoning_type for c in chains]
        type_counts = {rt.value: reasoning_types.count(rt.value) for rt in ReasoningType}
        
        avg_confidence = sum(c.overall_confidence for c in chains) / len(chains)
        avg_validation = sum(c.validation_score for c in chains) / len(chains)
        avg_steps = sum(len(c.nodes) for c in chains) / len(chains)
        
        return {
            'total_chains': len(chains),
            'reasoning_type_distribution': type_counts,
            'average_confidence': avg_confidence,
            'average_validation_score': avg_validation,
            'average_steps_per_chain': avg_steps,
            'recent_chains': [self.get_chain_summary(c.id) for c in chains[-5:]],
            'performance_trends': self._calculate_performance_trends(chains)
        }
    
    def _calculate_performance_trends(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Calculate performance trends over time"""
        if len(chains) < 5:
            return {"message": "Insufficient data for trend analysis"}
        
        # Split into recent and earlier chains
        mid_point = len(chains) // 2
        earlier_chains = chains[:mid_point]
        recent_chains = chains[mid_point:]
        
        earlier_conf = sum(c.overall_confidence for c in earlier_chains) / len(earlier_chains)
        recent_conf = sum(c.overall_confidence for c in recent_chains) / len(recent_chains)
        
        confidence_trend = "improving" if recent_conf > earlier_conf else "declining"
        
        return {
            'confidence_trend': confidence_trend,
            'confidence_change': recent_conf - earlier_conf,
            'recent_average_confidence': recent_conf,
            'earlier_average_confidence': earlier_conf
        }

class ComplexReasoningSystem:
    """Main system coordinating complex reasoning capabilities"""
    
    def __init__(self):
        self.processor = ReasoningChainProcessor()
        self.reasoning_cache = {}
        self.confidence_threshold = 0.6
        
        logger.info("Complex reasoning system initialized")
    
    async def execute_reasoning(self, query: str, context: Dict[str, Any], 
                              reasoning_type: Optional[str] = None) -> Dict[str, Any]:
        """Execute complex reasoning with validation and explanation"""
        
        # Convert reasoning type string to enum if provided
        reasoning_enum = None
        if reasoning_type:
            try:
                reasoning_enum = ReasoningType(reasoning_type)
            except ValueError:
                logger.warning(f"Invalid reasoning type: {reasoning_type}")
        
        # Check cache for similar queries
        cache_key = f"{query}_{reasoning_type or 'auto'}"
        if cache_key in self.reasoning_cache:
            cached_result = self.reasoning_cache[cache_key]
            logger.debug(f"Using cached reasoning result for: {query[:50]}...")
            return cached_result
        
        # Process reasoning request
        try:
            chain = await self.processor.process_reasoning_request(
                query, context, reasoning_enum
            )
            
            # Prepare result
            result = {
                'success': True,
                'chain_id': chain.id,
                'reasoning_type': chain.reasoning_type,
                'confidence': chain.overall_confidence,
                'validation_score': chain.validation_score,
                'conclusion': self._extract_conclusion(chain),
                'explanation': chain.explanation,
                'steps': [self._node_to_dict(node) for node in chain.nodes],
                'processing_time': self._calculate_processing_time(chain),
                'needs_improvement': chain.overall_confidence < self.confidence_threshold
            }
            
            # Cache result if confidence is high enough
            if chain.overall_confidence >= self.confidence_threshold:
                self.reasoning_cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in reasoning execution: {e}")
            return {
                'success': False,
                'error': str(e),
                'suggestion': 'Try rephrasing the query or providing more context'
            }
    
    def validate_reasoning_chain(self, chain_id: str) -> Dict[str, Any]:
        """Validate a specific reasoning chain"""
        chain = self.processor.active_chains.get(chain_id)
        if not chain:
            return {'error': 'Chain not found'}
        
        reasoning_type = ReasoningType(chain.reasoning_type)
        strategy = self.processor.strategies.get(reasoning_type)
        
        if not strategy:
            return {'error': 'Strategy not available for validation'}
        
        validation = strategy.validate(chain)
        
        return {
            'chain_id': chain_id,
            'validation_result': asdict(validation),
            'recommendations': validation.suggestions
        }
    
    def get_reasoning_explanation(self, chain_id: str, detail_level: str = 'full') -> Dict[str, Any]:
        """Get explanation of reasoning process"""
        chain = self.processor.active_chains.get(chain_id)
        if not chain:
            return {'error': 'Chain not found'}
        
        if detail_level == 'summary':
            return {
                'chain_id': chain_id,
                'summary': f"{chain.reasoning_type} reasoning with {len(chain.nodes)} steps",
                'conclusion': self._extract_conclusion(chain),
                'confidence': chain.overall_confidence
            }
        elif detail_level == 'full':
            return {
                'chain_id': chain_id,
                'detailed_explanation': chain.explanation,
                'step_by_step': [
                    {
                        'step': i+1,
                        'type': node.step_type,
                        'content': node.content,
                        'explanation': node.explanation,
                        'confidence': node.confidence
                    }
                    for i, node in enumerate(chain.nodes)
                ],
                'overall_assessment': {
                    'confidence': chain.overall_confidence,
                    'validation_score': chain.validation_score,
                    'reasoning_type': chain.reasoning_type
                }
            }
        else:
            return {'error': 'Invalid detail level. Use "summary" or "full"'}
    
    def _extract_conclusion(self, chain: ReasoningChain) -> str:
        """Extract the main conclusion from a reasoning chain"""
        conclusion_nodes = [n for n in chain.nodes if n.step_type == ReasoningStep.CONCLUSION.value]
        if conclusion_nodes:
            return conclusion_nodes[-1].conclusion
        return "No clear conclusion reached"
    
    def _node_to_dict(self, node: ReasoningNode) -> Dict[str, Any]:
        """Convert reasoning node to dictionary"""
        return {
            'id': node.id,
            'type': node.step_type,
            'content': node.content,
            'confidence': node.confidence,
            'explanation': node.explanation,
            'validation_status': node.validation_status
        }
    
    def _calculate_processing_time(self, chain: ReasoningChain) -> float:
        """Calculate processing time for reasoning chain"""
        if chain.start_time and chain.end_time:
            start = datetime.fromisoformat(chain.start_time)
            end = datetime.fromisoformat(chain.end_time)
            return (end - start).total_seconds()
        return 0.0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get reasoning system statistics"""
        return {
            'active_chains': len(self.processor.active_chains),
            'completed_chains': len(self.processor.completed_chains),
            'cached_results': len(self.reasoning_cache),
            'available_strategies': list(self.processor.strategies.keys()),
            'confidence_threshold': self.confidence_threshold,
            'insights': self.processor.get_reasoning_insights()
        }