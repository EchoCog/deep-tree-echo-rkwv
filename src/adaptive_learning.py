"""
Adaptive Learning Mechanisms for Deep Tree Echo
Implements user preference learning, strategy optimization, and personalization
"""

import json
import time
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class UserPreference:
    """Represents a learned user preference"""
    preference_id: str
    user_id: str
    preference_type: str  # 'response_style', 'reasoning_type', 'detail_level', etc.
    preference_value: Any
    confidence: float
    evidence_count: int
    last_reinforced: str
    creation_date: str
    context_conditions: Dict[str, Any] = field(default_factory=dict)
    usage_frequency: int = 0

@dataclass
class LearningEvent:
    """Represents a learning event from user interaction"""
    event_id: str
    session_id: str
    user_id: str
    event_type: str  # 'positive_feedback', 'negative_feedback', 'preference_indication'
    event_data: Dict[str, Any]
    timestamp: str
    context: Dict[str, Any]
    processed: bool = False

@dataclass
class PersonalizationProfile:
    """User's personalization profile"""
    user_id: str
    session_id: str
    preferences: Dict[str, UserPreference]
    interaction_patterns: Dict[str, Any]
    cognitive_style: Dict[str, float]
    adaptation_history: List[str]
    learning_progress: Dict[str, float]
    created_at: str
    updated_at: str

@dataclass
class FeedbackEntry:
    """User feedback entry"""
    feedback_id: str
    session_id: str
    user_id: str
    interaction_id: str
    feedback_type: str  # 'rating', 'correction', 'preference'
    feedback_value: Any
    context: Dict[str, Any]
    timestamp: str
    processed: bool = False

class PreferenceLearner(ABC):
    """Abstract base class for preference learning strategies"""
    
    @abstractmethod
    def learn_from_interaction(self, interaction: Dict[str, Any], 
                             feedback: Optional[Dict[str, Any]] = None) -> List[UserPreference]:
        """Learn preferences from user interaction"""
        pass
    
    @abstractmethod
    def update_preference_confidence(self, preference: UserPreference, 
                                   reinforcement: float) -> UserPreference:
        """Update preference confidence based on reinforcement"""
        pass

class ResponseStyleLearner(PreferenceLearner):
    """Learns user preferences for response styles"""
    
    def __init__(self):
        self.style_indicators = {
            'concise': ['short', 'brief', 'quick', 'summary', 'tl;dr'],
            'detailed': ['explain', 'detail', 'elaborate', 'comprehensive', 'thorough'],
            'formal': ['formal', 'professional', 'academic', 'technical'],
            'casual': ['casual', 'simple', 'easy', 'friendly', 'conversational'],
            'analytical': ['analyze', 'break down', 'structure', 'systematic'],
            'creative': ['creative', 'imaginative', 'innovative', 'alternative']
        }
    
    def learn_from_interaction(self, interaction: Dict[str, Any], 
                             feedback: Optional[Dict[str, Any]] = None) -> List[UserPreference]:
        """Learn response style preferences from interaction"""
        preferences = []
        
        user_input = interaction.get('user_input', '').lower()
        response_quality = feedback.get('rating', 0.5) if feedback else 0.5
        
        # Analyze user input for style indicators
        for style, indicators in self.style_indicators.items():
            if any(indicator in user_input for indicator in indicators):
                # Create or update preference
                preference = UserPreference(
                    preference_id=f"style_{style}_{int(time.time())}",
                    user_id=interaction.get('user_id', 'anonymous'),
                    preference_type='response_style',
                    preference_value=style,
                    confidence=0.7 if response_quality > 0.7 else 0.4,
                    evidence_count=1,
                    last_reinforced=datetime.now().isoformat(),
                    creation_date=datetime.now().isoformat(),
                    context_conditions={'input_length': len(user_input.split())}
                )
                preferences.append(preference)
        
        return preferences
    
    def update_preference_confidence(self, preference: UserPreference, 
                                   reinforcement: float) -> UserPreference:
        """Update style preference confidence"""
        # Use exponential moving average
        alpha = 0.2
        preference.confidence = alpha * reinforcement + (1 - alpha) * preference.confidence
        preference.confidence = max(0.1, min(1.0, preference.confidence))
        preference.evidence_count += 1
        preference.last_reinforced = datetime.now().isoformat()
        
        return preference

class CognitiveStrategyLearner(PreferenceLearner):
    """Learns user preferences for cognitive strategies"""
    
    def __init__(self):
        self.strategy_indicators = {
            'memory_focused': ['remember', 'recall', 'previous', 'before', 'earlier'],
            'reasoning_focused': ['analyze', 'logic', 'reasoning', 'think through'],
            'creative_focused': ['imagine', 'creative', 'brainstorm', 'innovative'],
            'structured_approach': ['step by step', 'systematic', 'organized', 'methodical']
        }
    
    def learn_from_interaction(self, interaction: Dict[str, Any], 
                             feedback: Optional[Dict[str, Any]] = None) -> List[UserPreference]:
        """Learn cognitive strategy preferences"""
        preferences = []
        
        user_input = interaction.get('user_input', '').lower()
        strategy_used = interaction.get('strategy_used', '')
        success_rate = feedback.get('success', 0.5) if feedback else 0.5
        
        # Learn from explicit strategy indicators
        for strategy, indicators in self.strategy_indicators.items():
            if any(indicator in user_input for indicator in indicators):
                preference = UserPreference(
                    preference_id=f"strategy_{strategy}_{int(time.time())}",
                    user_id=interaction.get('user_id', 'anonymous'),
                    preference_type='cognitive_strategy',
                    preference_value=strategy,
                    confidence=0.6 + (success_rate * 0.3),
                    evidence_count=1,
                    last_reinforced=datetime.now().isoformat(),
                    creation_date=datetime.now().isoformat(),
                    context_conditions={'query_complexity': interaction.get('complexity', 'medium')}
                )
                preferences.append(preference)
        
        # Learn from successful strategy usage
        if strategy_used and success_rate > 0.7:
            preference = UserPreference(
                preference_id=f"strategy_success_{strategy_used}_{int(time.time())}",
                user_id=interaction.get('user_id', 'anonymous'),
                preference_type='successful_strategy',
                preference_value=strategy_used,
                confidence=success_rate,
                evidence_count=1,
                last_reinforced=datetime.now().isoformat(),
                creation_date=datetime.now().isoformat()
            )
            preferences.append(preference)
        
        return preferences
    
    def update_preference_confidence(self, preference: UserPreference, 
                                   reinforcement: float) -> UserPreference:
        """Update strategy preference confidence"""
        # Higher learning rate for strategy preferences
        alpha = 0.3
        preference.confidence = alpha * reinforcement + (1 - alpha) * preference.confidence
        preference.confidence = max(0.1, min(1.0, preference.confidence))
        preference.evidence_count += 1
        preference.last_reinforced = datetime.now().isoformat()
        
        return preference

class InteractionPatternAnalyzer:
    """Analyzes user interaction patterns to identify behavioral preferences"""
    
    def __init__(self):
        self.pattern_history = defaultdict(deque)  # maxlen will be set per user
        self.temporal_patterns = defaultdict(dict)
        
    def analyze_interaction_patterns(self, user_id: str, 
                                   interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in user interactions"""
        if not interactions:
            return {}
        
        patterns = {
            'session_length_preference': self._analyze_session_lengths(interactions),
            'query_complexity_preference': self._analyze_query_complexity(interactions),
            'response_time_tolerance': self._analyze_response_time_patterns(interactions),
            'feedback_patterns': self._analyze_feedback_patterns(interactions),
            'topic_interests': self._analyze_topic_interests(interactions),
            'temporal_patterns': self._analyze_temporal_patterns(interactions)
        }
        
        return patterns
    
    def _analyze_session_lengths(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze preferred session lengths"""
        session_lengths = {}
        for interaction in interactions:
            session_id = interaction.get('session_id')
            if session_id not in session_lengths:
                session_lengths[session_id] = 0
            session_lengths[session_id] += 1
        
        if session_lengths:
            avg_length = np.mean(list(session_lengths.values()))
            return {
                'average_session_length': avg_length,
                'preference': 'long_sessions' if avg_length > 10 else 'short_sessions'
            }
        return {}
    
    def _analyze_query_complexity(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's query complexity preferences"""
        complexities = []
        for interaction in interactions:
            user_input = interaction.get('user_input', '')
            word_count = len(user_input.split())
            if word_count > 20:
                complexities.append('high')
            elif word_count > 10:
                complexities.append('medium')
            else:
                complexities.append('low')
        
        if complexities:
            most_common = max(set(complexities), key=complexities.count)
            return {
                'preferred_complexity': most_common,
                'complexity_distribution': {
                    level: complexities.count(level) / len(complexities) 
                    for level in ['low', 'medium', 'high']
                }
            }
        return {}
    
    def _analyze_response_time_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user tolerance for response times"""
        response_times = [
            interaction.get('response_time', 0) 
            for interaction in interactions
            if interaction.get('user_satisfaction', 0.5) > 0.6
        ]
        
        if response_times:
            return {
                'average_acceptable_time': np.mean(response_times),
                'max_acceptable_time': np.percentile(response_times, 90),
                'tolerance_level': 'high' if np.mean(response_times) > 2.0 else 'normal'
            }
        return {}
    
    def _analyze_feedback_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user feedback patterns"""
        feedback_types = []
        for interaction in interactions:
            if 'feedback' in interaction:
                feedback_types.append(interaction['feedback'].get('type', 'implicit'))
        
        if feedback_types:
            return {
                'feedback_frequency': len(feedback_types) / len(interactions),
                'feedback_types': {
                    ftype: feedback_types.count(ftype) / len(feedback_types)
                    for ftype in set(feedback_types)
                }
            }
        return {}
    
    def _analyze_topic_interests(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze user's topic interests"""
        # Simple keyword-based topic detection
        topic_keywords = {
            'technology': ['computer', 'software', 'AI', 'algorithm', 'programming'],
            'science': ['research', 'study', 'theory', 'experiment', 'hypothesis'],
            'philosophy': ['meaning', 'existence', 'ethics', 'consciousness', 'morality'],
            'practical': ['how to', 'steps', 'guide', 'tutorial', 'help'],
            'creative': ['creative', 'art', 'design', 'imagination', 'inspiration']
        }
        
        topic_scores = defaultdict(int)
        total_interactions = len(interactions)
        
        for interaction in interactions:
            user_input = interaction.get('user_input', '').lower()
            for topic, keywords in topic_keywords.items():
                if any(keyword in user_input for keyword in keywords):
                    topic_scores[topic] += 1
        
        if topic_scores:
            return {
                topic: score / total_interactions 
                for topic, score in topic_scores.items()
            }
        return {}
    
    def _analyze_temporal_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal interaction patterns"""
        timestamps = []
        for interaction in interactions:
            try:
                timestamp = datetime.fromisoformat(interaction.get('timestamp', ''))
                timestamps.append(timestamp)
            except:
                continue
        
        if timestamps:
            hours = [ts.hour for ts in timestamps]
            days = [ts.strftime('%A') for ts in timestamps]
            
            return {
                'most_active_hours': {
                    hour: hours.count(hour) / len(hours) 
                    for hour in set(hours)
                },
                'most_active_days': {
                    day: days.count(day) / len(days) 
                    for day in set(days)
                },
                'session_frequency': len(set(interaction.get('session_id') for interaction in interactions))
            }
        return {}

class PersonalizationEngine:
    """Engine for creating and managing user personalization profiles"""
    
    def __init__(self):
        self.profiles = {}
        self.preference_learners = {
            'response_style': ResponseStyleLearner(),
            'cognitive_strategy': CognitiveStrategyLearner()
        }
        self.pattern_analyzer = InteractionPatternAnalyzer()
        self.learning_events = deque(maxlen=10000)
        self.lock = threading.RLock()
        
    def create_or_update_profile(self, user_id: str, session_id: str,
                               interactions: List[Dict[str, Any]]) -> PersonalizationProfile:
        """Create or update user personalization profile"""
        with self.lock:
            profile_key = f"{user_id}_{session_id}"
            
            if profile_key in self.profiles:
                profile = self.profiles[profile_key]
                profile.updated_at = datetime.now().isoformat()
            else:
                profile = PersonalizationProfile(
                    user_id=user_id,
                    session_id=session_id,
                    preferences={},
                    interaction_patterns={},
                    cognitive_style={},
                    adaptation_history=[],
                    learning_progress={},
                    created_at=datetime.now().isoformat(),
                    updated_at=datetime.now().isoformat()
                )
            
            # Update interaction patterns
            profile.interaction_patterns = self.pattern_analyzer.analyze_interaction_patterns(
                user_id, interactions
            )
            
            # Update cognitive style assessment
            profile.cognitive_style = self._assess_cognitive_style(interactions)
            
            self.profiles[profile_key] = profile
            logger.debug(f"Updated personalization profile for user {user_id}")
            
            return profile
    
    def learn_preferences_from_interaction(self, interaction: Dict[str, Any],
                                         feedback: Optional[Dict[str, Any]] = None) -> List[UserPreference]:
        """Learn preferences from a single interaction"""
        learned_preferences = []
        
        for learner_type, learner in self.preference_learners.items():
            try:
                prefs = learner.learn_from_interaction(interaction, feedback)
                learned_preferences.extend(prefs)
            except Exception as e:
                logger.error(f"Error in {learner_type} learning: {e}")
        
        # Update profiles with new preferences
        user_id = interaction.get('user_id', 'anonymous')
        session_id = interaction.get('session_id', 'default')
        profile_key = f"{user_id}_{session_id}"
        
        if profile_key in self.profiles:
            profile = self.profiles[profile_key]
            for pref in learned_preferences:
                profile.preferences[pref.preference_id] = pref
        
        return learned_preferences
    
    def apply_personalization(self, user_id: str, session_id: str,
                            processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Apply personalization to processing context"""
        profile_key = f"{user_id}_{session_id}"
        
        if profile_key not in self.profiles:
            return processing_context  # No personalization available
        
        profile = self.profiles[profile_key]
        personalized_context = processing_context.copy()
        
        # Apply response style preferences
        style_prefs = [p for p in profile.preferences.values() 
                      if p.preference_type == 'response_style' and p.confidence > 0.6]
        if style_prefs:
            best_style = max(style_prefs, key=lambda p: p.confidence)
            personalized_context['preferred_response_style'] = best_style.preference_value
        
        # Apply cognitive strategy preferences
        strategy_prefs = [p for p in profile.preferences.values() 
                         if p.preference_type == 'cognitive_strategy' and p.confidence > 0.6]
        if strategy_prefs:
            best_strategy = max(strategy_prefs, key=lambda p: p.confidence)
            personalized_context['preferred_cognitive_strategy'] = best_strategy.preference_value
        
        # Apply interaction pattern preferences
        if profile.interaction_patterns:
            personalized_context['user_patterns'] = profile.interaction_patterns
            
            # Adjust response detail based on complexity preference
            complexity_pref = profile.interaction_patterns.get('query_complexity_preference', {})
            if complexity_pref:
                personalized_context['detail_level'] = complexity_pref.get('preferred_complexity', 'medium')
        
        # Apply cognitive style adaptations
        if profile.cognitive_style:
            personalized_context['cognitive_style'] = profile.cognitive_style
        
        logger.debug(f"Applied personalization for user {user_id}")
        return personalized_context
    
    def process_feedback(self, feedback: FeedbackEntry) -> None:
        """Process user feedback for learning"""
        profile_key = f"{feedback.user_id}_{feedback.session_id}"
        
        if profile_key not in self.profiles:
            return
        
        profile = self.profiles[profile_key]
        
        # Update relevant preferences based on feedback
        if feedback.feedback_type == 'rating' and isinstance(feedback.feedback_value, (int, float)):
            self._update_preferences_from_rating(profile, feedback)
        elif feedback.feedback_type == 'correction':
            self._learn_from_correction(profile, feedback)
        elif feedback.feedback_type == 'preference':
            self._learn_from_explicit_preference(profile, feedback)
        
        # Record learning event
        event = LearningEvent(
            event_id=f"feedback_{int(time.time())}",
            session_id=feedback.session_id,
            user_id=feedback.user_id,
            event_type=feedback.feedback_type,
            event_data=asdict(feedback),
            timestamp=datetime.now().isoformat(),
            context={'profile_updated': True}
        )
        self.learning_events.append(event)
        
        logger.debug(f"Processed feedback for user {feedback.user_id}")
    
    def get_adaptation_recommendations(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get recommendations for system adaptation"""
        profile_key = f"{user_id}_{session_id}"
        
        if profile_key not in self.profiles:
            return {'message': 'No profile available for recommendations'}
        
        profile = self.profiles[profile_key]
        recommendations = {
            'response_style_adaptations': [],
            'strategy_adaptations': [],
            'interaction_adaptations': [],
            'confidence_scores': {}
        }
        
        # Analyze preferences for recommendations
        for pref in profile.preferences.values():
            if pref.confidence > 0.7:
                if pref.preference_type == 'response_style':
                    recommendations['response_style_adaptations'].append({
                        'adaptation': f"Use {pref.preference_value} response style",
                        'confidence': pref.confidence,
                        'evidence_count': pref.evidence_count
                    })
                elif pref.preference_type == 'cognitive_strategy':
                    recommendations['strategy_adaptations'].append({
                        'adaptation': f"Prefer {pref.preference_value} strategy",
                        'confidence': pref.confidence,
                        'evidence_count': pref.evidence_count
                    })
        
        # Analyze interaction patterns for recommendations
        if profile.interaction_patterns:
            patterns = profile.interaction_patterns
            
            if 'session_length_preference' in patterns:
                session_pref = patterns['session_length_preference']
                if session_pref.get('preference') == 'long_sessions':
                    recommendations['interaction_adaptations'].append(
                        "User prefers longer, more detailed conversations"
                    )
                else:
                    recommendations['interaction_adaptations'].append(
                        "User prefers shorter, more focused interactions"
                    )
            
            if 'response_time_tolerance' in patterns:
                time_tolerance = patterns['response_time_tolerance']
                if time_tolerance.get('tolerance_level') == 'high':
                    recommendations['interaction_adaptations'].append(
                        "User is patient with longer processing times"
                    )
                else:
                    recommendations['interaction_adaptations'].append(
                        "Prioritize fast response times for this user"
                    )
        
        # Calculate overall adaptation confidence
        all_prefs = list(profile.preferences.values())
        if all_prefs:
            avg_confidence = sum(p.confidence for p in all_prefs) / len(all_prefs)
            recommendations['confidence_scores']['overall_adaptation'] = avg_confidence
            recommendations['confidence_scores']['data_quality'] = min(1.0, len(all_prefs) / 10.0)
        
        return recommendations
    
    def _assess_cognitive_style(self, interactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Assess user's cognitive style based on interactions"""
        if not interactions:
            return {}
        
        style_scores = {
            'analytical': 0.0,
            'creative': 0.0,
            'practical': 0.0,
            'social': 0.0
        }
        
        analytical_keywords = ['analyze', 'logic', 'systematic', 'data', 'evidence', 'proof']
        creative_keywords = ['creative', 'imagine', 'brainstorm', 'innovative', 'artistic']
        practical_keywords = ['how to', 'step', 'practical', 'useful', 'application']
        social_keywords = ['people', 'relationship', 'community', 'social', 'collaboration']
        
        for interaction in interactions:
            user_input = interaction.get('user_input', '').lower()
            
            # Count keyword matches
            for keyword in analytical_keywords:
                if keyword in user_input:
                    style_scores['analytical'] += 1
            for keyword in creative_keywords:
                if keyword in user_input:
                    style_scores['creative'] += 1
            for keyword in practical_keywords:
                if keyword in user_input:
                    style_scores['practical'] += 1
            for keyword in social_keywords:
                if keyword in user_input:
                    style_scores['social'] += 1
        
        # Normalize scores
        total_interactions = len(interactions)
        for style in style_scores:
            style_scores[style] = min(1.0, style_scores[style] / total_interactions)
        
        return style_scores
    
    def _update_preferences_from_rating(self, profile: PersonalizationProfile, 
                                      feedback: FeedbackEntry) -> None:
        """Update preferences based on user rating"""
        rating = float(feedback.feedback_value)
        normalized_rating = rating / 5.0 if rating <= 5 else rating  # Assume 5-star rating
        
        # Update preferences related to the interaction
        interaction_id = feedback.interaction_id
        for pref in profile.preferences.values():
            if pref.creation_date and interaction_id in pref.creation_date:
                learner_type = pref.preference_type
                if learner_type in self.preference_learners:
                    learner = self.preference_learners[learner_type]
                    updated_pref = learner.update_preference_confidence(pref, normalized_rating)
                    profile.preferences[pref.preference_id] = updated_pref
    
    def _learn_from_correction(self, profile: PersonalizationProfile, 
                             feedback: FeedbackEntry) -> None:
        """Learn from user corrections"""
        correction_data = feedback.feedback_value
        if isinstance(correction_data, dict):
            # Create negative preference for current approach
            if 'rejected_approach' in correction_data:
                rejected = correction_data['rejected_approach']
                negative_pref = UserPreference(
                    preference_id=f"negative_{rejected}_{int(time.time())}",
                    user_id=profile.user_id,
                    preference_type='negative_preference',
                    preference_value=rejected,
                    confidence=0.8,
                    evidence_count=1,
                    last_reinforced=datetime.now().isoformat(),
                    creation_date=datetime.now().isoformat()
                )
                profile.preferences[negative_pref.preference_id] = negative_pref
            
            # Create positive preference for preferred approach
            if 'preferred_approach' in correction_data:
                preferred = correction_data['preferred_approach']
                positive_pref = UserPreference(
                    preference_id=f"positive_{preferred}_{int(time.time())}",
                    user_id=profile.user_id,
                    preference_type='corrected_preference',
                    preference_value=preferred,
                    confidence=0.9,
                    evidence_count=1,
                    last_reinforced=datetime.now().isoformat(),
                    creation_date=datetime.now().isoformat()
                )
                profile.preferences[positive_pref.preference_id] = positive_pref
    
    def _learn_from_explicit_preference(self, profile: PersonalizationProfile, 
                                      feedback: FeedbackEntry) -> None:
        """Learn from explicit user preference statements"""
        pref_data = feedback.feedback_value
        if isinstance(pref_data, dict):
            explicit_pref = UserPreference(
                preference_id=f"explicit_{pref_data.get('type', 'general')}_{int(time.time())}",
                user_id=profile.user_id,
                preference_type=pref_data.get('type', 'explicit_preference'),
                preference_value=pref_data.get('value'),
                confidence=1.0,  # Explicit preferences have high confidence
                evidence_count=1,
                last_reinforced=datetime.now().isoformat(),
                creation_date=datetime.now().isoformat(),
                context_conditions=pref_data.get('conditions', {})
            )
            profile.preferences[explicit_pref.preference_id] = explicit_pref
    
    def get_learning_progress(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get learning progress for a user"""
        profile_key = f"{user_id}_{session_id}"
        
        if profile_key not in self.profiles:
            return {'message': 'No learning data available'}
        
        profile = self.profiles[profile_key]
        
        # Calculate learning metrics
        total_preferences = len(profile.preferences)
        high_confidence_prefs = len([p for p in profile.preferences.values() if p.confidence > 0.7])
        recent_adaptations = len([p for p in profile.preferences.values() 
                                if (datetime.now() - datetime.fromisoformat(p.creation_date)).days < 7])
        
        # Calculate learning velocity (preferences learned per day)
        if profile.created_at:
            days_since_creation = (datetime.now() - datetime.fromisoformat(profile.created_at)).days + 1
            learning_velocity = total_preferences / days_since_creation
        else:
            learning_velocity = 0.0
        
        return {
            'total_preferences_learned': total_preferences,
            'high_confidence_preferences': high_confidence_prefs,
            'recent_adaptations': recent_adaptations,
            'learning_velocity': learning_velocity,
            'preference_types': {
                pref_type: len([p for p in profile.preferences.values() if p.preference_type == pref_type])
                for pref_type in set(p.preference_type for p in profile.preferences.values())
            },
            'cognitive_style_confidence': {
                style: score for style, score in profile.cognitive_style.items() if score > 0.1
            },
            'adaptation_opportunities': len(self.get_adaptation_recommendations(user_id, session_id))
        }
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get adaptive learning system statistics"""
        total_profiles = len(self.profiles)
        total_preferences = sum(len(p.preferences) for p in self.profiles.values())
        total_events = len(self.learning_events)
        
        return {
            'total_profiles': total_profiles,
            'total_preferences_learned': total_preferences,
            'total_learning_events': total_events,
            'average_preferences_per_profile': total_preferences / max(1, total_profiles),
            'learner_types': list(self.preference_learners.keys()),
            'recent_activity': len([e for e in self.learning_events 
                                  if (datetime.now() - datetime.fromisoformat(e.timestamp)).hours < 24])
        }

class AdaptiveLearningSystem:
    """Main system coordinating adaptive learning capabilities"""
    
    def __init__(self):
        self.personalization_engine = PersonalizationEngine()
        self.feedback_queue = deque(maxlen=1000)
        self.learning_enabled = True
        
        logger.info("Adaptive learning system initialized")
    
    def process_interaction_for_learning(self, interaction: Dict[str, Any],
                                       feedback: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process interaction for adaptive learning"""
        if not self.learning_enabled:
            return {'learning_disabled': True}
        
        try:
            # Learn preferences from interaction
            learned_prefs = self.personalization_engine.learn_preferences_from_interaction(
                interaction, feedback
            )
            
            # Update or create profile
            user_id = interaction.get('user_id', 'anonymous')
            session_id = interaction.get('session_id', 'default')
            
            # Get recent interactions for pattern analysis (this would come from a database in practice)
            recent_interactions = [interaction]  # Simplified - would normally fetch from storage
            
            profile = self.personalization_engine.create_or_update_profile(
                user_id, session_id, recent_interactions
            )
            
            return {
                'preferences_learned': len(learned_prefs),
                'profile_updated': True,
                'new_preferences': [asdict(p) for p in learned_prefs],
                'total_profile_preferences': len(profile.preferences)
            }
            
        except Exception as e:
            logger.error(f"Error in learning process: {e}")
            return {'error': str(e)}
    
    def get_personalization_context(self, user_id: str, session_id: str,
                                  base_context: Dict[str, Any]) -> Dict[str, Any]:
        """Get personalized processing context"""
        try:
            personalized_context = self.personalization_engine.apply_personalization(
                user_id, session_id, base_context
            )
            
            return {
                'success': True,
                'personalized_context': personalized_context,
                'personalization_applied': len(personalized_context) > len(base_context)
            }
            
        except Exception as e:
            logger.error(f"Error applying personalization: {e}")
            return {
                'success': False,
                'error': str(e),
                'fallback_context': base_context
            }
    
    def submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user feedback for learning"""
        try:
            feedback = FeedbackEntry(
                feedback_id=f"fb_{int(time.time())}",
                session_id=feedback_data.get('session_id', 'unknown'),
                user_id=feedback_data.get('user_id', 'anonymous'),
                interaction_id=feedback_data.get('interaction_id', ''),
                feedback_type=feedback_data.get('type', 'rating'),
                feedback_value=feedback_data.get('value'),
                context=feedback_data.get('context', {}),
                timestamp=datetime.now().isoformat()
            )
            
            self.feedback_queue.append(feedback)
            self.personalization_engine.process_feedback(feedback)
            
            return {
                'success': True,
                'feedback_id': feedback.feedback_id,
                'processing_status': 'completed'
            }
            
        except Exception as e:
            logger.error(f"Error processing feedback: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_user_insights(self, user_id: str, session_id: str) -> Dict[str, Any]:
        """Get insights about user learning and preferences"""
        try:
            progress = self.personalization_engine.get_learning_progress(user_id, session_id)
            recommendations = self.personalization_engine.get_adaptation_recommendations(user_id, session_id)
            
            return {
                'learning_progress': progress,
                'adaptation_recommendations': recommendations,
                'insights_available': True
            }
            
        except Exception as e:
            logger.error(f"Error getting user insights: {e}")
            return {'insights_available': False, 'error': str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get adaptive learning system status"""
        return {
            'learning_enabled': self.learning_enabled,
            'pending_feedback': len(self.feedback_queue),
            'system_stats': self.personalization_engine.get_system_stats(),
            'last_update': datetime.now().isoformat()
        }