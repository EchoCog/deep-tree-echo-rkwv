"""
Explanation Generation System for Deep Tree Echo
Implements Task 2.6: Generate human-readable explanations of reasoning processes and conclusions
"""

import time
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import json

logger = logging.getLogger(__name__)

class ExplanationStyle(Enum):
    """Different explanation styles for different user preferences"""
    TECHNICAL = "technical"      # Detailed technical explanation
    CONVERSATIONAL = "conversational"  # Natural, conversational tone
    BULLET_POINTS = "bullet_points"    # Structured bullet format
    NARRATIVE = "narrative"      # Story-like explanation
    MINIMAL = "minimal"         # Brief, concise explanations

class ExplanationLevel(Enum):
    """Detail levels for explanations"""
    OVERVIEW = "overview"        # High-level summary
    DETAILED = "detailed"        # Comprehensive explanation
    STEP_BY_STEP = "step_by_step"  # Each reasoning step explained
    INTERACTIVE = "interactive"   # Questions and clarifications

@dataclass
class ExplanationTemplate:
    """Template for generating explanations"""
    template_id: str
    name: str
    style: ExplanationStyle
    level: ExplanationLevel
    introduction_template: str
    step_template: str
    conclusion_template: str
    validation_template: str
    user_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExplanationRequest:
    """Request for explanation generation"""
    content_type: str  # 'reasoning_chain', 'memory_retrieval', 'cognitive_process'
    content_data: Dict[str, Any]
    target_audience: str = "general"  # 'general', 'technical', 'beginner', 'expert'
    style_preference: ExplanationStyle = ExplanationStyle.CONVERSATIONAL
    detail_level: ExplanationLevel = ExplanationLevel.DETAILED
    max_length: Optional[int] = None
    include_confidence: bool = True
    include_alternatives: bool = False
    personalization_context: Optional[Dict[str, Any]] = None

@dataclass
class GeneratedExplanation:
    """Generated explanation with metadata"""
    explanation_id: str
    request: ExplanationRequest
    generated_text: str
    confidence_score: float
    generation_time: float
    word_count: int
    reading_time_minutes: float
    clarity_score: float
    sections: Dict[str, str] = field(default_factory=dict)
    interactive_elements: List[Dict[str, Any]] = field(default_factory=list)

class ExplanationTemplateLibrary:
    """Library of explanation templates"""
    
    def __init__(self):
        self.templates = {}
        self._initialize_default_templates()
    
    def _initialize_default_templates(self):
        """Initialize default explanation templates"""
        
        # Technical style templates
        self.templates['technical_detailed'] = ExplanationTemplate(
            template_id='technical_detailed',
            name='Technical Detailed',
            style=ExplanationStyle.TECHNICAL,
            level=ExplanationLevel.DETAILED,
            introduction_template="Technical Analysis: {reasoning_type} processing initiated for query: '{query}'",
            step_template="Step {step_number} ({step_type}): {explanation} [Confidence: {confidence:.2f}]",
            conclusion_template="Analysis Result: {conclusion} (Overall confidence: {overall_confidence:.2f})",
            validation_template="Validation: {validation_score:.2f} | Issues: {issues} | Recommendations: {recommendations}"
        )
        
        # Conversational style templates  
        self.templates['conversational_detailed'] = ExplanationTemplate(
            template_id='conversational_detailed',
            name='Conversational Detailed',
            style=ExplanationStyle.CONVERSATIONAL,
            level=ExplanationLevel.DETAILED,
            introduction_template="Let me walk you through how I analyzed your question: '{query}'",
            step_template="First, I {explanation_friendly}. I'm {confidence_text} about this step.",
            conclusion_template="Based on this analysis, {conclusion_friendly}",
            validation_template="To check my work, I found {validation_friendly}"
        )
        
        # Bullet point style templates
        self.templates['bullet_detailed'] = ExplanationTemplate(
            template_id='bullet_detailed',
            name='Bullet Point Detailed',
            style=ExplanationStyle.BULLET_POINTS,
            level=ExplanationLevel.DETAILED,
            introduction_template="## Analysis of: {query}\n\n**Reasoning approach:** {reasoning_type}",
            step_template="• **{step_type}:** {explanation} *(confidence: {confidence:.1f})*",
            conclusion_template="## Conclusion\n{conclusion}",
            validation_template="## Quality Assessment\n• Validation score: {validation_score:.2f}\n• {validation_details}"
        )
        
        # Narrative style templates
        self.templates['narrative_detailed'] = ExplanationTemplate(
            template_id='narrative_detailed',
            name='Narrative Detailed',
            style=ExplanationStyle.NARRATIVE,
            level=ExplanationLevel.DETAILED,
            introduction_template="When you asked '{query}', I embarked on a {reasoning_type} journey to find the answer.",
            step_template="Along the way, I {explanation_narrative}, which gave me {confidence_narrative} in this direction.",
            conclusion_template="At the end of this journey, I arrived at: {conclusion}",
            validation_template="Looking back on this path, {validation_narrative}"
        )
    
    def get_template(self, style: ExplanationStyle, level: ExplanationLevel) -> Optional[ExplanationTemplate]:
        """Get best matching template"""
        template_key = f"{style.value}_{level.value}"
        if template_key in self.templates:
            return self.templates[template_key]
        
        # Fallback to conversational detailed
        return self.templates.get('conversational_detailed')
    
    def add_custom_template(self, template: ExplanationTemplate):
        """Add a custom template"""
        self.templates[template.template_id] = template

class ClarityOptimizer:
    """Optimizes explanation clarity and readability"""
    
    def __init__(self):
        self.jargon_replacements = {
            'instantiate': 'create',
            'utilize': 'use',
            'facilitate': 'help',
            'methodology': 'method',
            'paradigm': 'approach',
            'heuristic': 'rule of thumb',
            'optimal': 'best',
            'empirical': 'based on experience'
        }
    
    def optimize_clarity(self, text: str, target_audience: str = "general") -> str:
        """Optimize text for clarity"""
        optimized = text
        
        # Replace technical jargon for general audiences
        if target_audience in ['general', 'beginner']:
            for jargon, replacement in self.jargon_replacements.items():
                optimized = optimized.replace(jargon, replacement)
        
        # Break up long sentences
        optimized = self._break_long_sentences(optimized)
        
        # Add transition words for better flow
        optimized = self._improve_flow(optimized)
        
        return optimized
    
    def _break_long_sentences(self, text: str) -> str:
        """Break up sentences that are too long"""
        sentences = text.split('. ')
        improved_sentences = []
        
        for sentence in sentences:
            if len(sentence) > 150:  # Long sentence threshold
                # Try to split on conjunctions
                for conjunction in [', and ', ', but ', ', however ', ', therefore ']:
                    if conjunction in sentence:
                        parts = sentence.split(conjunction, 1)
                        improved_sentences.append(parts[0] + '.')
                        improved_sentences.append(parts[1].strip().capitalize())
                        break
                else:
                    improved_sentences.append(sentence)
            else:
                improved_sentences.append(sentence)
        
        return '. '.join(improved_sentences)
    
    def _improve_flow(self, text: str) -> str:
        """Add transition words to improve flow"""
        sentences = text.split('. ')
        if len(sentences) < 2:
            return text
        
        transitions = ['Additionally', 'Furthermore', 'As a result', 'Consequently', 'Moreover']
        improved = [sentences[0]]
        
        for i, sentence in enumerate(sentences[1:], 1):
            if i % 3 == 0 and len(sentences) > 3:  # Add transitions periodically
                transition = transitions[min(i // 3 - 1, len(transitions) - 1)]
                improved.append(f"{transition}, {sentence.lower()}")
            else:
                improved.append(sentence)
        
        return '. '.join(improved)
    
    def calculate_clarity_score(self, text: str) -> float:
        """Calculate readability/clarity score"""
        if not text:
            return 0.0
        
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.0
        
        # Simple readability metrics
        avg_words_per_sentence = len(words) / len(sentences)
        avg_chars_per_word = sum(len(word) for word in words) / len(words)
        
        # Penalize very long sentences and words
        sentence_penalty = max(0, (avg_words_per_sentence - 20) * 0.05)
        word_penalty = max(0, (avg_chars_per_word - 6) * 0.1)
        
        # Base score starts high and gets penalized
        score = 1.0 - sentence_penalty - word_penalty
        
        # Bonus for good structure (headings, bullet points)
        if '##' in text or '•' in text:
            score += 0.1
        
        return max(0.0, min(1.0, score))

class ExplanationGenerator:
    """Main explanation generation system"""
    
    def __init__(self):
        self.template_library = ExplanationTemplateLibrary()
        self.clarity_optimizer = ClarityOptimizer()
        self.generated_explanations = []
        
        logger.info("Explanation generation system initialized")
    
    def generate_explanation(self, request: ExplanationRequest) -> GeneratedExplanation:
        """Generate explanation based on request"""
        start_time = time.time()
        
        try:
            # Select appropriate template
            template = self.template_library.get_template(
                request.style_preference, 
                request.detail_level
            )
            
            if not template:
                raise ValueError(f"No template found for {request.style_preference}, {request.detail_level}")
            
            # Generate explanation content
            explanation_text = self._generate_content(request, template)
            
            # Optimize for clarity
            if request.target_audience != 'technical':
                explanation_text = self.clarity_optimizer.optimize_clarity(
                    explanation_text, request.target_audience
                )
            
            # Apply length constraints
            if request.max_length and len(explanation_text) > request.max_length:
                explanation_text = self._truncate_explanation(explanation_text, request.max_length)
            
            # Calculate metrics
            generation_time = time.time() - start_time
            word_count = len(explanation_text.split())
            reading_time = word_count / 200  # Average reading speed: 200 WPM
            clarity_score = self.clarity_optimizer.calculate_clarity_score(explanation_text)
            
            # Create generated explanation
            explanation = GeneratedExplanation(
                explanation_id=f"exp_{int(time.time())}_{len(self.generated_explanations)}",
                request=request,
                generated_text=explanation_text,
                confidence_score=self._calculate_explanation_confidence(request),
                generation_time=generation_time,
                word_count=word_count,
                reading_time_minutes=reading_time,
                clarity_score=clarity_score,
                sections=self._extract_sections(explanation_text),
                interactive_elements=self._generate_interactive_elements(request)
            )
            
            self.generated_explanations.append(explanation)
            logger.info(f"Generated explanation {explanation.explanation_id} in {generation_time:.2f}s")
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            # Return fallback explanation
            return self._generate_fallback_explanation(request, str(e))
    
    def _generate_content(self, request: ExplanationRequest, template: ExplanationTemplate) -> str:
        """Generate the actual explanation content"""
        content_data = request.content_data
        
        if request.content_type == 'reasoning_chain':
            return self._explain_reasoning_chain(content_data, template, request)
        elif request.content_type == 'memory_retrieval':
            return self._explain_memory_retrieval(content_data, template)
        elif request.content_type == 'cognitive_process':
            return self._explain_cognitive_process(content_data, template)
        else:
            return self._explain_generic_content(content_data, template)
    
    def _explain_reasoning_chain(self, chain_data: Dict[str, Any], template: ExplanationTemplate, request: ExplanationRequest) -> str:
        """Generate explanation for reasoning chain"""
        parts = []
        
        # Introduction
        introduction = template.introduction_template.format(
            reasoning_type=chain_data.get('reasoning_type', 'analytical'),
            query=chain_data.get('query', 'your question')
        )
        parts.append(introduction)
        
        # Steps
        steps = chain_data.get('steps', [])
        for i, step in enumerate(steps, 1):
            confidence = step.get('confidence', 0.8)
            confidence_text = self._confidence_to_text(confidence)
            
            if template.style == ExplanationStyle.CONVERSATIONAL:
                step_text = template.step_template.format(
                    explanation_friendly=self._make_friendly(step.get('explanation', '')),
                    confidence_text=confidence_text
                )
            else:
                step_text = template.step_template.format(
                    step_number=i,
                    step_type=step.get('step_type', 'analysis'),
                    explanation=step.get('explanation', ''),
                    confidence=confidence
                )
            parts.append(step_text)
        
        # Conclusion
        conclusion = template.conclusion_template.format(
            conclusion=chain_data.get('conclusion', 'analysis complete'),
            conclusion_friendly=self._make_friendly(chain_data.get('conclusion', 'analysis complete')),
            overall_confidence=chain_data.get('overall_confidence', 0.8)
        )
        parts.append(conclusion)
        
        # Validation if requested
        if request.include_confidence:
            validation_data = chain_data.get('validation', {})
            validation_text = template.validation_template.format(
                validation_score=validation_data.get('score', 0.8),
                issues=len(validation_data.get('issues', [])),
                recommendations=len(validation_data.get('recommendations', [])),
                validation_friendly=self._make_friendly(f"the validation score was {validation_data.get('score', 0.8):.1f}"),
                validation_details=self._format_validation_details(validation_data)
            )
            parts.append(validation_text)
        
        return '\n\n'.join(parts)
    
    def _explain_memory_retrieval(self, memory_data: Dict[str, Any], template: ExplanationTemplate) -> str:
        """Generate explanation for memory retrieval process"""
        return f"Memory retrieval explanation for: {memory_data.get('query', 'unknown')}"
    
    def _explain_cognitive_process(self, process_data: Dict[str, Any], template: ExplanationTemplate) -> str:
        """Generate explanation for cognitive process"""
        return f"Cognitive process explanation: {process_data.get('process_type', 'unknown')}"
    
    def _explain_generic_content(self, content_data: Dict[str, Any], template: ExplanationTemplate) -> str:
        """Generate explanation for generic content"""
        return f"Generic explanation for: {json.dumps(content_data, indent=2)}"
    
    def _confidence_to_text(self, confidence: float) -> str:
        """Convert confidence score to human-readable text"""
        if confidence >= 0.9:
            return "very confident"
        elif confidence >= 0.7:
            return "fairly confident"
        elif confidence >= 0.5:
            return "moderately confident"
        else:
            return "less certain"
    
    def _make_friendly(self, text: str) -> str:
        """Make technical text more conversational"""
        if not text:
            return ""
        
        # Convert to more natural language
        text = text.replace("performed analysis on", "looked at")
        text = text.replace("determined that", "found that")
        text = text.replace("implemented", "used")
        text = text.replace("utilized", "used")
        
        return text
    
    def _format_validation_details(self, validation_data: Dict[str, Any]) -> str:
        """Format validation details"""
        details = []
        if validation_data.get('issues'):
            details.append(f"Found {len(validation_data['issues'])} potential issues")
        if validation_data.get('recommendations'):
            details.append(f"Generated {len(validation_data['recommendations'])} recommendations")
        return ' • '.join(details) if details else "No significant issues found"
    
    def _calculate_explanation_confidence(self, request: ExplanationRequest) -> float:
        """Calculate confidence in the generated explanation"""
        base_confidence = 0.8
        
        # Adjust based on content complexity
        if request.content_type == 'reasoning_chain':
            steps = request.content_data.get('steps', [])
            if len(steps) > 5:
                base_confidence -= 0.1  # More complex chains are harder to explain
        
        # Adjust based on style - technical explanations are more reliable
        if request.style_preference == ExplanationStyle.TECHNICAL:
            base_confidence += 0.1
        elif request.style_preference == ExplanationStyle.MINIMAL:
            base_confidence -= 0.05  # Less detail might miss nuances
        
        return max(0.1, min(1.0, base_confidence))
    
    def _truncate_explanation(self, text: str, max_length: int) -> str:
        """Intelligently truncate explanation to fit length limit"""
        if len(text) <= max_length:
            return text
        
        # Try to truncate at sentence boundaries
        sentences = text.split('. ')
        truncated = ""
        
        for sentence in sentences:
            if len(truncated) + len(sentence) + 2 <= max_length - 20:  # Leave room for ellipsis
                truncated += sentence + ". "
            else:
                break
        
        return truncated.strip() + "... [explanation truncated]"
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from explanation text"""
        sections = {}
        
        # Look for markdown-style headers
        lines = text.split('\n')
        current_section = "introduction"
        current_content = []
        
        for line in lines:
            if line.startswith('##'):
                if current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = line.replace('##', '').strip().lower()
                current_content = []
            else:
                current_content.append(line)
        
        # Add final section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        return sections
    
    def _generate_interactive_elements(self, request: ExplanationRequest) -> List[Dict[str, Any]]:
        """Generate interactive elements for the explanation"""
        elements = []
        
        if request.detail_level == ExplanationLevel.INTERACTIVE:
            # Add clarifying questions
            elements.append({
                'type': 'question',
                'content': 'Would you like me to explain any of these steps in more detail?',
                'options': ['Yes, step 1', 'Yes, step 2', 'No, this is clear']
            })
            
            # Add follow-up suggestions
            elements.append({
                'type': 'follow_up',
                'content': 'Related topics you might find interesting:',
                'suggestions': ['Similar reasoning patterns', 'Alternative approaches', 'Real-world applications']
            })
        
        return elements
    
    def _generate_fallback_explanation(self, request: ExplanationRequest, error: str) -> GeneratedExplanation:
        """Generate fallback explanation when main generation fails"""
        fallback_text = f"I apologize, but I encountered an issue generating a detailed explanation. The basic analysis shows: {json.dumps(request.content_data, indent=2)}"
        
        return GeneratedExplanation(
            explanation_id=f"fallback_{int(time.time())}",
            request=request,
            generated_text=fallback_text,
            confidence_score=0.3,
            generation_time=0.0,
            word_count=len(fallback_text.split()),
            reading_time_minutes=0.5,
            clarity_score=0.5
        )
    
    def get_explanation_by_id(self, explanation_id: str) -> Optional[GeneratedExplanation]:
        """Get explanation by ID"""
        for exp in self.generated_explanations:
            if exp.explanation_id == explanation_id:
                return exp
        return None
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get explanation system statistics"""
        if not self.generated_explanations:
            return {"message": "No explanations generated yet"}
        
        total_generated = len(self.generated_explanations)
        avg_generation_time = sum(exp.generation_time for exp in self.generated_explanations) / total_generated
        avg_clarity_score = sum(exp.clarity_score for exp in self.generated_explanations) / total_generated
        
        style_distribution = {}
        for exp in self.generated_explanations:
            style = exp.request.style_preference.value
            style_distribution[style] = style_distribution.get(style, 0) + 1
        
        return {
            'total_explanations_generated': total_generated,
            'average_generation_time': avg_generation_time,
            'average_clarity_score': avg_clarity_score,
            'style_distribution': style_distribution,
            'available_templates': len(self.template_library.templates),
            'most_used_style': max(style_distribution, key=style_distribution.get) if style_distribution else None
        }