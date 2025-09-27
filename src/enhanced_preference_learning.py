"""
Enhanced User Preference Learning System for Deep Tree Echo
Implements Task 2.7: Advanced user preference learning, communication styles, and interaction patterns
"""

import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import numpy as np
from enum import Enum
import json

logger = logging.getLogger(__name__)

class CommunicationStyle(Enum):
    """User communication style preferences"""
    DIRECT = "direct"                    # Concise, to-the-point responses
    DETAILED = "detailed"                # Comprehensive, thorough explanations
    CONVERSATIONAL = "conversational"   # Friendly, casual tone
    FORMAL = "formal"                   # Professional, structured tone
    ANALYTICAL = "analytical"           # Data-driven, logical approach
    CREATIVE = "creative"               # Imaginative, open-ended thinking
    SOCRATIC = "socratic"               # Question-based, guided discovery

class InteractionPattern(Enum):
    """Common user interaction patterns"""
    EXPLORATORY = "exploratory"         # Likes to explore topics deeply
    GOAL_ORIENTED = "goal_oriented"     # Focused on specific outcomes
    ITERATIVE = "iterative"            # Refines queries progressively
    COMPARATIVE = "comparative"         # Compares options and alternatives
    SEQUENTIAL = "sequential"           # Prefers step-by-step processes
    HOLISTIC = "holistic"              # Wants big-picture understanding
    EXPERIMENTAL = "experimental"       # Tests different approaches

class LearningStrategy(Enum):
    """How users prefer to learn and receive information"""
    VISUAL = "visual"                   # Diagrams, examples, illustrations
    AUDITORY = "auditory"               # Explanations, discussions
    KINESTHETIC = "kinesthetic"         # Hands-on, interactive learning
    READING = "reading"                 # Text-based, documentation
    MULTIMODAL = "multimodal"          # Combination of methods

class PreferenceStrength(Enum):
    """Strength of user preferences"""
    WEAK = 0.3      # Slight preference
    MODERATE = 0.6  # Clear preference
    STRONG = 0.8    # Strong preference
    ABSOLUTE = 0.95 # Nearly absolute preference

@dataclass
class EnhancedUserPreference:
    """Enhanced user preference with richer context"""
    preference_id: str
    user_id: str
    preference_category: str  # 'communication_style', 'interaction_pattern', 'learning_strategy'
    preference_value: Any
    confidence: float
    evidence_count: int
    last_reinforced: str
    creation_date: str
    
    # Enhanced attributes
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    usage_frequency: int = 0
    effectiveness_score: float = 0.0  # How well this preference worked
    temporal_patterns: Dict[str, float] = field(default_factory=dict)  # Time-based preferences
    topic_specificity: Dict[str, float] = field(default_factory=dict)  # Topic-specific preferences
    social_context: Dict[str, Any] = field(default_factory=dict)  # Group vs individual preferences
    adaptation_rate: float = 0.1  # How quickly this preference changes
    stability_score: float = 0.5  # How stable this preference is over time

@dataclass
class InteractionAnalysis:
    """Analysis of a single interaction for preference learning"""
    interaction_id: str
    user_id: str
    timestamp: str
    query_text: str
    response_quality_score: float
    user_satisfaction_score: Optional[float]
    
    # Communication analysis
    detected_communication_style: Optional[CommunicationStyle] = None
    communication_confidence: float = 0.0
    
    # Interaction analysis
    detected_interaction_pattern: Optional[InteractionPattern] = None
    interaction_confidence: float = 0.0
    
    # Content analysis
    topic_categories: List[str] = field(default_factory=list)
    complexity_level: float = 0.5
    technical_depth: float = 0.5
    
    # Context analysis
    session_position: int = 1  # Position in conversation
    response_time: float = 0.0
    query_length: int = 0
    follow_up_questions: int = 0
    
    # Feedback signals
    explicit_feedback: Optional[Dict[str, Any]] = None
    implicit_feedback: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PersonalizationProfile:
    """Enhanced personalization profile"""
    user_id: str
    session_id: str
    created_at: str
    last_updated: str
    
    # Core preferences
    communication_styles: Dict[CommunicationStyle, float] = field(default_factory=dict)
    interaction_patterns: Dict[InteractionPattern, float] = field(default_factory=dict)
    learning_strategies: Dict[LearningStrategy, float] = field(default_factory=dict)
    
    # Enhanced characteristics
    cognitive_load_preference: float = 0.5  # How much complexity user prefers
    response_length_preference: float = 0.5  # Preferred response length
    technical_level_preference: float = 0.5  # Preferred technical depth
    formality_preference: float = 0.5  # Formal vs casual preference
    
    # Behavioral patterns
    attention_span_estimate: float = 0.5  # Estimated attention span
    exploration_tendency: float = 0.5  # Tendency to explore vs focus
    patience_level: float = 0.5  # Patience for detailed explanations
    
    # Context-specific preferences
    time_based_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Time of day preferences
    topic_based_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Topic-specific preferences
    mood_based_preferences: Dict[str, Dict[str, float]] = field(default_factory=dict)  # Mood-based preferences
    
    # Learning and adaptation
    learning_velocity: float = 0.1  # How quickly preferences change
    preference_stability: Dict[str, float] = field(default_factory=dict)  # Stability of different preferences
    adaptation_triggers: List[str] = field(default_factory=list)  # What causes preference changes

class CommunicationStyleAnalyzer:
    """Analyzes user communication style from interactions"""
    
    def __init__(self):
        self.style_indicators = {
            CommunicationStyle.DIRECT: {
                'keywords': ['brief', 'short', 'quick', 'simple', 'just tell me'],
                'patterns': ['one word answers', 'short queries', 'imperative mood'],
                'query_length_range': (1, 30)
            },
            CommunicationStyle.DETAILED: {
                'keywords': ['explain', 'detailed', 'comprehensive', 'thorough', 'elaborate'],
                'patterns': ['complex sentences', 'multiple questions', 'specific requirements'],
                'query_length_range': (50, 500)
            },
            CommunicationStyle.CONVERSATIONAL: {
                'keywords': ['hey', 'hi', 'thanks', 'please', 'what do you think'],
                'patterns': ['casual language', 'personal pronouns', 'informal tone'],
                'query_length_range': (10, 100)
            },
            CommunicationStyle.FORMAL: {
                'keywords': ['could you', 'would you', 'I would like', 'please provide'],
                'patterns': ['formal grammar', 'polite language', 'structured requests'],
                'query_length_range': (20, 200)
            },
            CommunicationStyle.ANALYTICAL: {
                'keywords': ['analyze', 'compare', 'evaluate', 'data', 'statistics', 'evidence'],
                'patterns': ['logical structure', 'cause-effect', 'hypothesis'],
                'query_length_range': (30, 300)
            },
            CommunicationStyle.CREATIVE: {
                'keywords': ['imagine', 'creative', 'innovative', 'brainstorm', 'what if'],
                'patterns': ['open-ended questions', 'hypothetical scenarios', 'metaphors'],
                'query_length_range': (20, 200)
            },
            CommunicationStyle.SOCRATIC: {
                'keywords': ['why', 'how', 'what if', 'help me understand', 'guide me'],
                'patterns': ['question chains', 'seeking understanding', 'guided discovery'],
                'query_length_range': (15, 150)
            }
        }
    
    def analyze_communication_style(self, query_text: str, context: Dict[str, Any]) -> Tuple[Optional[CommunicationStyle], float]:
        """Analyze communication style from query text"""
        if not query_text:
            return None, 0.0
        
        query_lower = query_text.lower()
        query_length = len(query_text)
        scores = {}
        
        for style, indicators in self.style_indicators.items():
            score = 0.0
            
            # Check keywords
            keyword_matches = sum(1 for keyword in indicators['keywords'] if keyword in query_lower)
            score += keyword_matches * 0.3
            
            # Check query length
            length_range = indicators['query_length_range']
            if length_range[0] <= query_length <= length_range[1]:
                score += 0.4
            
            # Pattern matching (simplified)
            if 'formal grammar' in indicators['patterns'] and any(phrase in query_lower for phrase in ['could you', 'would you', 'may i']):
                score += 0.3
            
            scores[style] = score
        
        if not scores:
            return None, 0.0
        
        best_style = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_style])
        
        return best_style if confidence > 0.3 else None, confidence

class InteractionPatternAnalyzer:
    """Analyzes user interaction patterns"""
    
    def __init__(self):
        self.pattern_indicators = {
            InteractionPattern.EXPLORATORY: {
                'behaviors': ['multiple follow-ups', 'tangential questions', 'deep dives'],
                'keywords': ['also', 'what about', 'related to', 'furthermore']
            },
            InteractionPattern.GOAL_ORIENTED: {
                'behaviors': ['specific objectives', 'focused questions', 'solution-seeking'],
                'keywords': ['need to', 'how to', 'solve', 'achieve', 'accomplish']
            },
            InteractionPattern.ITERATIVE: {
                'behaviors': ['progressive refinement', 'building on previous answers'],
                'keywords': ['refine', 'adjust', 'improve', 'modify', 'better']
            },
            InteractionPattern.COMPARATIVE: {
                'behaviors': ['option evaluation', 'alternative seeking'],
                'keywords': ['compare', 'versus', 'alternatives', 'options', 'better than']
            },
            InteractionPattern.SEQUENTIAL: {
                'behaviors': ['step-by-step requests', 'ordered processes'],
                'keywords': ['first', 'then', 'next', 'step by step', 'in order']
            },
            InteractionPattern.HOLISTIC: {
                'behaviors': ['big picture questions', 'context seeking'],
                'keywords': ['overall', 'big picture', 'context', 'framework', 'overview']
            }
        }
    
    def analyze_interaction_pattern(self, interactions: List[Dict[str, Any]]) -> Tuple[Optional[InteractionPattern], float]:
        """Analyze interaction pattern from conversation history"""
        if not interactions:
            return None, 0.0
        
        scores = defaultdict(float)
        
        for interaction in interactions[-5:]:  # Look at recent interactions
            query = interaction.get('query', '').lower()
            
            for pattern, indicators in self.pattern_indicators.items():
                # Check keywords
                keyword_matches = sum(1 for keyword in indicators['keywords'] if keyword in query)
                scores[pattern] += keyword_matches * 0.2
                
                # Analyze behavioral patterns (simplified)
                if pattern == InteractionPattern.EXPLORATORY and len(interactions) > 3:
                    scores[pattern] += 0.3
                elif pattern == InteractionPattern.GOAL_ORIENTED and 'solve' in query:
                    scores[pattern] += 0.4
        
        if not scores:
            return None, 0.0
        
        best_pattern = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_pattern] / len(interactions))
        
        return best_pattern if confidence > 0.2 else None, confidence

class PreferenceStabilityTracker:
    """Tracks stability and change patterns in user preferences"""
    
    def __init__(self):
        self.preference_history = defaultdict(list)
        self.change_thresholds = {
            'communication_style': 0.3,
            'interaction_pattern': 0.4,
            'learning_strategy': 0.5
        }
    
    def track_preference_change(self, user_id: str, preference_category: str, 
                              old_value: Any, new_value: Any, confidence: float):
        """Track changes in user preferences"""
        change_record = {
            'timestamp': datetime.now().isoformat(),
            'old_value': old_value,
            'new_value': new_value,
            'confidence': confidence,
            'change_magnitude': self._calculate_change_magnitude(old_value, new_value)
        }
        
        self.preference_history[f"{user_id}_{preference_category}"].append(change_record)
    
    def calculate_stability_score(self, user_id: str, preference_category: str) -> float:
        """Calculate stability score for a preference category"""
        history_key = f"{user_id}_{preference_category}"
        history = self.preference_history.get(history_key, [])
        
        if len(history) < 2:
            return 0.5  # Default stability
        
        # Calculate average change magnitude over time
        recent_changes = history[-10:]  # Look at recent changes
        avg_change = sum(record['change_magnitude'] for record in recent_changes) / len(recent_changes)
        
        # Higher average change = lower stability
        stability = max(0.0, 1.0 - (avg_change * 2))
        return stability
    
    def _calculate_change_magnitude(self, old_value: Any, new_value: Any) -> float:
        """Calculate magnitude of preference change"""
        if old_value == new_value:
            return 0.0
        
        if isinstance(old_value, (int, float)) and isinstance(new_value, (int, float)):
            return abs(new_value - old_value)
        
        # For categorical changes, return fixed magnitude
        return 0.5

class EnhancedPreferenceLearner:
    """Enhanced preference learning with advanced pattern recognition"""
    
    def __init__(self):
        self.communication_analyzer = CommunicationStyleAnalyzer()
        self.interaction_analyzer = InteractionPatternAnalyzer()
        self.stability_tracker = PreferenceStabilityTracker()
        
        self.learning_rates = {
            'communication_style': 0.2,
            'interaction_pattern': 0.15,
            'learning_strategy': 0.1,
            'cognitive_load': 0.05,
            'technical_level': 0.1
        }
    
    def analyze_interaction(self, user_id: str, query: str, response: str,
                          conversation_history: List[Dict[str, Any]],
                          feedback: Optional[Dict[str, Any]] = None) -> InteractionAnalysis:
        """Comprehensive interaction analysis"""
        
        # Basic analysis
        analysis = InteractionAnalysis(
            interaction_id=f"int_{user_id}_{int(time.time())}",
            user_id=user_id,
            timestamp=datetime.now().isoformat(),
            query_text=query,
            response_quality_score=0.8,  # Would be calculated based on actual metrics
            user_satisfaction_score=feedback.get('satisfaction') if feedback else None,
            query_length=len(query),
            session_position=len(conversation_history) + 1
        )
        
        # Communication style analysis
        comm_style, comm_conf = self.communication_analyzer.analyze_communication_style(
            query, {'history': conversation_history}
        )
        analysis.detected_communication_style = comm_style
        analysis.communication_confidence = comm_conf
        
        # Interaction pattern analysis
        full_history = conversation_history + [{'query': query, 'response': response}]
        interaction_pattern, pattern_conf = self.interaction_analyzer.analyze_interaction_pattern(
            full_history
        )
        analysis.detected_interaction_pattern = interaction_pattern
        analysis.interaction_confidence = pattern_conf
        
        # Content analysis
        analysis.topic_categories = self._extract_topics(query)
        analysis.complexity_level = self._assess_complexity(query)
        analysis.technical_depth = self._assess_technical_depth(query)
        
        # Implicit feedback analysis
        analysis.implicit_feedback = self._analyze_implicit_feedback(
            query, response, conversation_history
        )
        
        return analysis
    
    def learn_preferences_from_analysis(self, analysis: InteractionAnalysis,
                                      existing_profile: Optional[PersonalizationProfile] = None) -> List[EnhancedUserPreference]:
        """Learn preferences from interaction analysis"""
        learned_preferences = []
        
        # Learn communication style preference
        if analysis.detected_communication_style and analysis.communication_confidence > 0.3:
            comm_pref = self._create_enhanced_preference(
                analysis.user_id,
                'communication_style',
                analysis.detected_communication_style,
                analysis.communication_confidence,
                analysis
            )
            learned_preferences.append(comm_pref)
        
        # Learn interaction pattern preference
        if analysis.detected_interaction_pattern and analysis.interaction_confidence > 0.2:
            pattern_pref = self._create_enhanced_preference(
                analysis.user_id,
                'interaction_pattern',
                analysis.detected_interaction_pattern,
                analysis.interaction_confidence,
                analysis
            )
            learned_preferences.append(pattern_pref)
        
        # Learn technical level preference
        if analysis.technical_depth != 0.5:  # Non-default value
            tech_pref = self._create_enhanced_preference(
                analysis.user_id,
                'technical_level',
                analysis.technical_depth,
                0.6,
                analysis
            )
            learned_preferences.append(tech_pref)
        
        # Learn response length preference based on query length
        length_preference = self._infer_length_preference(analysis)
        if length_preference:
            learned_preferences.append(length_preference)
        
        return learned_preferences
    
    def _create_enhanced_preference(self, user_id: str, category: str, value: Any,
                                  confidence: float, analysis: InteractionAnalysis) -> EnhancedUserPreference:
        """Create an enhanced user preference"""
        return EnhancedUserPreference(
            preference_id=f"pref_{user_id}_{category}_{int(time.time())}",
            user_id=user_id,
            preference_category=category,
            preference_value=value,
            confidence=confidence,
            evidence_count=1,
            last_reinforced=datetime.now().isoformat(),
            creation_date=datetime.now().isoformat(),
            context_conditions={
                'session_position': analysis.session_position,
                'topic_categories': analysis.topic_categories,
                'complexity_level': analysis.complexity_level
            },
            effectiveness_score=analysis.user_satisfaction_score or 0.7,
            temporal_patterns={
                'time_of_day': datetime.now().hour / 24.0
            },
            adaptation_rate=self.learning_rates.get(category, 0.1)
        )
    
    def update_preference_with_feedback(self, preference: EnhancedUserPreference,
                                      feedback: Dict[str, Any]) -> EnhancedUserPreference:
        """Update preference based on feedback"""
        # Update effectiveness score
        satisfaction = feedback.get('satisfaction', 0.5)
        current_effectiveness = preference.effectiveness_score
        
        # Weighted average update
        alpha = preference.adaptation_rate
        preference.effectiveness_score = (1 - alpha) * current_effectiveness + alpha * satisfaction
        
        # Update confidence based on feedback
        if satisfaction > 0.7:
            preference.confidence = min(1.0, preference.confidence + 0.1)
        elif satisfaction < 0.3:
            preference.confidence = max(0.1, preference.confidence - 0.1)
        
        # Update evidence count
        preference.evidence_count += 1
        preference.last_reinforced = datetime.now().isoformat()
        
        return preference
    
    def _extract_topics(self, query: str) -> List[str]:
        """Extract topic categories from query"""
        # Simplified topic extraction
        topics = []
        query_lower = query.lower()
        
        topic_keywords = {
            'technical': ['code', 'programming', 'algorithm', 'system', 'api'],
            'business': ['strategy', 'market', 'customer', 'revenue', 'profit'],
            'creative': ['design', 'art', 'creative', 'innovative', 'brainstorm'],
            'analytical': ['analyze', 'data', 'statistics', 'metrics', 'performance'],
            'educational': ['learn', 'understand', 'explain', 'teach', 'concept']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                topics.append(topic)
        
        return topics or ['general']
    
    def _assess_complexity(self, query: str) -> float:
        """Assess cognitive complexity of query"""
        complexity_indicators = [
            ('multiple questions', 0.3),
            ('technical terms', 0.2),
            ('conditional logic', 0.2),
            ('long query', 0.1)
        ]
        
        complexity = 0.3  # Base complexity
        query_lower = query.lower()
        
        # Multiple questions
        question_marks = query.count('?')
        if question_marks > 1:
            complexity += 0.2
        
        # Technical terms (simplified check)
        tech_words = ['algorithm', 'implementation', 'optimization', 'architecture', 'framework']
        tech_count = sum(1 for word in tech_words if word in query_lower)
        complexity += min(0.3, tech_count * 0.1)
        
        # Query length
        if len(query) > 200:
            complexity += 0.2
        elif len(query) > 100:
            complexity += 0.1
        
        return min(1.0, complexity)
    
    def _assess_technical_depth(self, query: str) -> float:
        """Assess preferred technical depth"""
        query_lower = query.lower()
        
        # Technical depth indicators
        high_tech = ['implementation', 'algorithm', 'architecture', 'optimization', 'performance']
        low_tech = ['simple', 'basic', 'beginner', 'easy', 'overview']
        
        high_score = sum(1 for word in high_tech if word in query_lower)
        low_score = sum(1 for word in low_tech if word in query_lower)
        
        if high_score > low_score:
            return 0.7 + min(0.3, high_score * 0.1)
        elif low_score > high_score:
            return 0.3 - min(0.2, low_score * 0.1)
        else:
            return 0.5  # Neutral
    
    def _infer_length_preference(self, analysis: InteractionAnalysis) -> Optional[EnhancedUserPreference]:
        """Infer response length preference from query characteristics"""
        query_length = analysis.query_length
        
        # Short queries might indicate preference for concise responses
        if query_length < 20:
            return self._create_enhanced_preference(
                analysis.user_id,
                'response_length',
                0.3,  # Prefer shorter responses
                0.4,
                analysis
            )
        # Long, detailed queries might indicate preference for detailed responses
        elif query_length > 100:
            return self._create_enhanced_preference(
                analysis.user_id,
                'response_length',
                0.8,  # Prefer longer responses
                0.5,
                analysis
            )
        
        return None
    
    def _analyze_implicit_feedback(self, query: str, response: str,
                                 history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze implicit feedback signals"""
        feedback = {}
        
        # Engagement indicators
        if len(history) > 1:
            # User is continuing the conversation - positive signal
            feedback['engagement'] = 0.7
        else:
            feedback['engagement'] = 0.5
        
        # Query complexity vs response appropriateness
        query_complexity = self._assess_complexity(query)
        feedback['complexity_match'] = min(1.0, 0.5 + abs(query_complexity - 0.5))
        
        # Response length appropriateness
        response_length = len(response.split())
        if 50 <= response_length <= 200:
            feedback['length_appropriateness'] = 0.8
        else:
            feedback['length_appropriateness'] = 0.6
        
        return feedback

class EnhancedPersonalizationEngine:
    """Enhanced personalization engine with advanced preference modeling"""
    
    def __init__(self):
        self.preference_learner = EnhancedPreferenceLearner()
        self.profiles = {}
        self.preference_weights = {
            'communication_style': 0.3,
            'interaction_pattern': 0.25,
            'technical_level': 0.2,
            'response_length': 0.15,
            'learning_strategy': 0.1
        }
    
    def process_interaction_for_learning(self, user_id: str, session_id: str,
                                       query: str, response: str,
                                       conversation_history: List[Dict[str, Any]],
                                       feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process interaction for enhanced preference learning"""
        
        # Analyze the interaction
        analysis = self.preference_learner.analyze_interaction(
            user_id, query, response, conversation_history, feedback
        )
        
        # Learn preferences from analysis
        learned_prefs = self.preference_learner.learn_preferences_from_analysis(
            analysis, self.profiles.get(user_id)
        )
        
        # Update or create profile
        profile = self._update_profile_with_preferences(
            user_id, session_id, learned_prefs, analysis
        )
        
        return {
            'success': True,
            'interaction_analysis': asdict(analysis),
            'preferences_learned': [asdict(pref) for pref in learned_prefs],
            'profile_updated': True,
            'personalization_confidence': self._calculate_personalization_confidence(profile)
        }
    
    def get_personalized_response_context(self, user_id: str, session_id: str,
                                        base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized context for response generation"""
        profile = self.profiles.get(user_id)
        if not profile:
            return base_context
        
        personalized_context = base_context.copy()
        
        # Apply communication style preferences
        dominant_comm_style = self._get_dominant_preference(profile.communication_styles)
        if dominant_comm_style:
            personalized_context['preferred_communication_style'] = dominant_comm_style.value
            personalized_context['communication_confidence'] = profile.communication_styles[dominant_comm_style]
        
        # Apply interaction pattern preferences
        dominant_pattern = self._get_dominant_preference(profile.interaction_patterns)
        if dominant_pattern:
            personalized_context['preferred_interaction_pattern'] = dominant_pattern.value
        
        # Apply technical level preferences
        personalized_context['preferred_technical_level'] = profile.technical_level_preference
        personalized_context['preferred_response_length'] = profile.response_length_preference
        personalized_context['cognitive_load_preference'] = profile.cognitive_load_preference
        
        # Context-specific adjustments
        current_time = datetime.now().hour
        time_key = f"hour_{current_time}"
        if time_key in profile.time_based_preferences:
            time_prefs = profile.time_based_preferences[time_key]
            personalized_context['time_adjusted_preferences'] = time_prefs
        
        return personalized_context
    
    def _update_profile_with_preferences(self, user_id: str, session_id: str,
                                       preferences: List[EnhancedUserPreference],
                                       analysis: InteractionAnalysis) -> PersonalizationProfile:
        """Update user profile with new preferences"""
        
        # Get or create profile
        profile = self.profiles.get(user_id) or PersonalizationProfile(
            user_id=user_id,
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            last_updated=datetime.now().isoformat()
        )
        
        # Update preferences
        for pref in preferences:
            self._integrate_preference_into_profile(profile, pref)
        
        # Update behavioral characteristics
        self._update_behavioral_characteristics(profile, analysis)
        
        # Update context-specific preferences
        self._update_contextual_preferences(profile, analysis)
        
        profile.last_updated = datetime.now().isoformat()
        self.profiles[user_id] = profile
        
        return profile
    
    def _integrate_preference_into_profile(self, profile: PersonalizationProfile,
                                         preference: EnhancedUserPreference):
        """Integrate a preference into the user profile"""
        category = preference.preference_category
        value = preference.preference_value
        confidence = preference.confidence
        
        if category == 'communication_style' and isinstance(value, CommunicationStyle):
            current = profile.communication_styles.get(value, 0.0)
            # Weighted update based on confidence and evidence
            alpha = min(0.5, confidence * preference.adaptation_rate)
            profile.communication_styles[value] = (1 - alpha) * current + alpha * confidence
            
        elif category == 'interaction_pattern' and isinstance(value, InteractionPattern):
            current = profile.interaction_patterns.get(value, 0.0)
            alpha = min(0.5, confidence * preference.adaptation_rate)
            profile.interaction_patterns[value] = (1 - alpha) * current + alpha * confidence
            
        elif category == 'technical_level':
            current = profile.technical_level_preference
            alpha = preference.adaptation_rate
            profile.technical_level_preference = (1 - alpha) * current + alpha * value
            
        elif category == 'response_length':
            current = profile.response_length_preference
            alpha = preference.adaptation_rate
            profile.response_length_preference = (1 - alpha) * current + alpha * value
    
    def _update_behavioral_characteristics(self, profile: PersonalizationProfile,
                                         analysis: InteractionAnalysis):
        """Update behavioral characteristics based on analysis"""
        # Update attention span estimate based on query complexity and length
        complexity_factor = analysis.complexity_level
        length_factor = min(1.0, len(analysis.query_text) / 200.0)
        estimated_attention = (complexity_factor + length_factor) / 2.0
        
        alpha = 0.1  # Learning rate for behavioral characteristics
        profile.attention_span_estimate = (1 - alpha) * profile.attention_span_estimate + alpha * estimated_attention
        
        # Update exploration tendency based on interaction patterns
        if analysis.detected_interaction_pattern == InteractionPattern.EXPLORATORY:
            profile.exploration_tendency = min(1.0, profile.exploration_tendency + 0.1)
        elif analysis.detected_interaction_pattern == InteractionPattern.GOAL_ORIENTED:
            profile.exploration_tendency = max(0.0, profile.exploration_tendency - 0.05)
    
    def _update_contextual_preferences(self, profile: PersonalizationProfile,
                                     analysis: InteractionAnalysis):
        """Update context-specific preferences"""
        current_hour = datetime.now().hour
        time_key = f"hour_{current_hour}"
        
        if time_key not in profile.time_based_preferences:
            profile.time_based_preferences[time_key] = {}
        
        # Update time-based communication style preferences
        if analysis.detected_communication_style:
            style_key = f"comm_style_{analysis.detected_communication_style.value}"
            current_score = profile.time_based_preferences[time_key].get(style_key, 0.5)
            alpha = 0.1
            new_score = (1 - alpha) * current_score + alpha * analysis.communication_confidence
            profile.time_based_preferences[time_key][style_key] = new_score
        
        # Update topic-based preferences
        for topic in analysis.topic_categories:
            if topic not in profile.topic_based_preferences:
                profile.topic_based_preferences[topic] = {}
            
            # Store technical level preference for this topic
            topic_tech_level = profile.topic_based_preferences[topic].get('technical_level', 0.5)
            alpha = 0.15
            new_tech_level = (1 - alpha) * topic_tech_level + alpha * analysis.technical_depth
            profile.topic_based_preferences[topic]['technical_level'] = new_tech_level
    
    def _get_dominant_preference(self, preferences: Dict[Any, float]) -> Optional[Any]:
        """Get the dominant preference from a preference dictionary"""
        if not preferences:
            return None
        return max(preferences, key=preferences.get)
    
    def _calculate_personalization_confidence(self, profile: PersonalizationProfile) -> float:
        """Calculate confidence in personalization quality"""
        confidence_factors = []
        
        # Communication style confidence
        if profile.communication_styles:
            max_comm_confidence = max(profile.communication_styles.values())
            confidence_factors.append(max_comm_confidence)
        
        # Interaction pattern confidence
        if profile.interaction_patterns:
            max_pattern_confidence = max(profile.interaction_patterns.values())
            confidence_factors.append(max_pattern_confidence)
        
        # Behavioral characteristic confidence (based on how far from default they are)
        behavioral_confidence = abs(profile.technical_level_preference - 0.5) * 2
        confidence_factors.append(behavioral_confidence)
        
        if not confidence_factors:
            return 0.0
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def get_profile_insights(self, user_id: str) -> Dict[str, Any]:
        """Get insights about user profile"""
        profile = self.profiles.get(user_id)
        if not profile:
            return {'message': 'No profile found'}
        
        insights = {
            'user_id': user_id,
            'profile_created': profile.created_at,
            'last_interaction': profile.last_updated,
            'personalization_confidence': self._calculate_personalization_confidence(profile),
            
            'communication_preferences': {
                style.value: round(confidence, 3)
                for style, confidence in profile.communication_styles.items()
            },
            
            'interaction_preferences': {
                pattern.value: round(confidence, 3)
                for pattern, confidence in profile.interaction_patterns.items()
            },
            
            'behavioral_characteristics': {
                'technical_level_preference': round(profile.technical_level_preference, 3),
                'response_length_preference': round(profile.response_length_preference, 3),
                'cognitive_load_preference': round(profile.cognitive_load_preference, 3),
                'attention_span_estimate': round(profile.attention_span_estimate, 3),
                'exploration_tendency': round(profile.exploration_tendency, 3),
                'patience_level': round(profile.patience_level, 3)
            },
            
            'contextual_adaptations': {
                'time_based_patterns': len(profile.time_based_preferences),
                'topic_specific_patterns': len(profile.topic_based_preferences),
                'mood_based_patterns': len(profile.mood_based_preferences)
            }
        }
        
        return insights