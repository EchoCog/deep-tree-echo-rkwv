"""
Consensus Management System
Handles consensus decisions from Virtual Marduk simulations
"""

import logging
import time
import uuid
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum

from .memory import SimulationResult, ConsensusRecord

logger = logging.getLogger(__name__)

class ConsensusStrategy(Enum):
    """Different consensus strategies"""
    HIGHEST_CONFIDENCE = "highest_confidence"
    MAJORITY_VOTE = "majority_vote"
    WEIGHTED_AVERAGE = "weighted_average"
    RISK_ADJUSTED = "risk_adjusted"

@dataclass
class Vote:
    """A vote from a Virtual Marduk"""
    voter_id: str
    choice: str
    confidence: float
    reasoning: str
    timestamp: float

class ConsensusManager:
    """
    Manages consensus mechanisms for Virtual Marduk simulations.
    Aggregates results and produces unified strategies.
    """
    
    def __init__(self, default_strategy: ConsensusStrategy = ConsensusStrategy.HIGHEST_CONFIDENCE):
        self.default_strategy = default_strategy
        self.consensus_history: List[ConsensusRecord] = []
        self.active_consensus_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info(f"ConsensusManager initialized with {default_strategy.value} strategy")
    
    async def reach_consensus(
        self,
        simulation_results: List[SimulationResult],
        strategy: Optional[ConsensusStrategy] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ConsensusRecord:
        """
        Reach consensus from simulation results using specified strategy
        """
        
        if not simulation_results:
            raise ValueError("Cannot reach consensus without simulation results")
        
        strategy = strategy or self.default_strategy
        consensus_id = str(uuid.uuid4())
        
        logger.info(f"Reaching consensus {consensus_id} with {len(simulation_results)} results using {strategy.value}")
        
        # Apply consensus strategy
        if strategy == ConsensusStrategy.HIGHEST_CONFIDENCE:
            decision = await self._highest_confidence_consensus(simulation_results)
        elif strategy == ConsensusStrategy.MAJORITY_VOTE:
            decision = await self._majority_vote_consensus(simulation_results)
        elif strategy == ConsensusStrategy.WEIGHTED_AVERAGE:
            decision = await self._weighted_average_consensus(simulation_results)
        elif strategy == ConsensusStrategy.RISK_ADJUSTED:
            decision = await self._risk_adjusted_consensus(simulation_results)
        else:
            raise ValueError(f"Unknown consensus strategy: {strategy}")
        
        # Calculate overall confidence
        successful_results = [r for r in simulation_results if r.success]
        overall_confidence = (
            sum(r.confidence for r in successful_results) / len(successful_results)
            if successful_results else 0.0
        )
        
        # Create consensus record
        consensus_record = ConsensusRecord(
            consensus_id=consensus_id,
            simulation_results=[r.simulation_id for r in simulation_results],
            decision=decision,
            confidence=overall_confidence,
            timestamp=time.time(),
            participants=[r.virtual_marduk_id for r in simulation_results]
        )
        
        # Store record
        self.consensus_history.append(consensus_record)
        
        # Log decision
        logger.info(f"Consensus {consensus_id} reached: {decision.get('strategy', 'unknown')} (confidence: {overall_confidence:.2f})")
        
        return consensus_record
    
    async def _highest_confidence_consensus(
        self, 
        simulation_results: List[SimulationResult]
    ) -> Dict[str, Any]:
        """Choose the result with highest confidence"""
        
        successful_results = [r for r in simulation_results if r.success]
        
        if not successful_results:
            return {
                'strategy': 'no_action',
                'reason': 'all_simulations_failed',
                'confidence': 0.0,
                'alternatives_considered': len(simulation_results)
            }
        
        best_result = max(successful_results, key=lambda r: r.confidence)
        
        return {
            'strategy': 'execute_simulation',
            'chosen_simulation': best_result.simulation_id,
            'chosen_hypothesis': best_result.hypothesis,
            'simulation_result': best_result.result,
            'confidence': best_result.confidence,
            'virtual_marduk': best_result.virtual_marduk_id,
            'alternatives_considered': len(simulation_results),
            'success_rate': len(successful_results) / len(simulation_results)
        }
    
    async def _majority_vote_consensus(
        self, 
        simulation_results: List[SimulationResult]
    ) -> Dict[str, Any]:
        """Use majority voting among successful simulations"""
        
        successful_results = [r for r in simulation_results if r.success]
        
        if not successful_results:
            return {
                'strategy': 'no_action',
                'reason': 'all_simulations_failed',
                'confidence': 0.0
            }
        
        # Group results by approach
        approach_votes = {}
        for result in successful_results:
            approach = result.result.get('approach', 'unknown')
            if approach not in approach_votes:
                approach_votes[approach] = []
            approach_votes[approach].append(result)
        
        # Find majority approach
        majority_approach = max(approach_votes.keys(), key=lambda k: len(approach_votes[k]))
        majority_results = approach_votes[majority_approach]
        
        # Choose best result from majority approach
        best_result = max(majority_results, key=lambda r: r.confidence)
        
        return {
            'strategy': 'execute_majority_choice',
            'chosen_approach': majority_approach,
            'chosen_simulation': best_result.simulation_id,
            'simulation_result': best_result.result,
            'confidence': sum(r.confidence for r in majority_results) / len(majority_results),
            'vote_count': len(majority_results),
            'total_votes': len(successful_results),
            'consensus_strength': len(majority_results) / len(successful_results)
        }
    
    async def _weighted_average_consensus(
        self, 
        simulation_results: List[SimulationResult]
    ) -> Dict[str, Any]:
        """Create weighted average decision based on confidence scores"""
        
        successful_results = [r for r in simulation_results if r.success]
        
        if not successful_results:
            return {
                'strategy': 'no_action',
                'reason': 'all_simulations_failed',
                'confidence': 0.0
            }
        
        # Calculate weights based on confidence
        total_confidence = sum(r.confidence for r in successful_results)
        
        # Aggregate approaches with weights
        approach_weights = {}
        approach_details = {}
        
        for result in successful_results:
            approach = result.result.get('approach', 'unknown')
            weight = result.confidence / total_confidence
            
            if approach not in approach_weights:
                approach_weights[approach] = 0.0
                approach_details[approach] = []
            
            approach_weights[approach] += weight
            approach_details[approach].append(result)
        
        # Choose approach with highest weighted score
        best_approach = max(approach_weights.keys(), key=lambda k: approach_weights[k])
        best_weight = approach_weights[best_approach]
        
        # Select representative result for the best approach
        best_approach_results = approach_details[best_approach]
        representative_result = max(best_approach_results, key=lambda r: r.confidence)
        
        return {
            'strategy': 'execute_weighted_choice',
            'chosen_approach': best_approach,
            'approach_weight': best_weight,
            'chosen_simulation': representative_result.simulation_id,
            'simulation_result': representative_result.result,
            'confidence': approach_weights[best_approach],
            'weight_distribution': approach_weights,
            'contributing_simulations': len(best_approach_results)
        }
    
    async def _risk_adjusted_consensus(
        self, 
        simulation_results: List[SimulationResult]
    ) -> Dict[str, Any]:
        """Adjust consensus based on risk factors"""
        
        successful_results = [r for r in simulation_results if r.success]
        
        if not successful_results:
            return {
                'strategy': 'no_action',
                'reason': 'all_simulations_failed',
                'confidence': 0.0
            }
        
        # Calculate risk-adjusted scores
        risk_adjusted_results = []
        
        for result in successful_results:
            # Extract risk factors from simulation result
            stability = result.result.get('stability', 'medium')
            time_taken = result.result.get('time_taken', 'medium')
            resources_used = result.result.get('resources_used', 50)
            
            # Calculate risk multiplier
            stability_factor = {
                'very_high': 1.2, 'high': 1.1, 'medium': 1.0, 
                'low': 0.8, 'very_low': 0.6
            }.get(stability, 1.0)
            
            time_factor = {
                'short': 1.1, 'medium': 1.0, 'long': 0.9, 'very_long': 0.8
            }.get(time_taken, 1.0)
            
            resource_factor = max(0.5, 1.0 - (resources_used - 50) / 100.0)
            
            risk_multiplier = stability_factor * time_factor * resource_factor
            risk_adjusted_score = result.confidence * risk_multiplier
            
            risk_adjusted_results.append({
                'result': result,
                'risk_adjusted_score': risk_adjusted_score,
                'risk_multiplier': risk_multiplier,
                'risk_factors': {
                    'stability': stability,
                    'time_taken': time_taken,
                    'resources_used': resources_used
                }
            })
        
        # Choose result with highest risk-adjusted score
        best_adjusted = max(risk_adjusted_results, key=lambda r: r['risk_adjusted_score'])
        best_result = best_adjusted['result']
        
        return {
            'strategy': 'execute_risk_adjusted',
            'chosen_simulation': best_result.simulation_id,
            'chosen_hypothesis': best_result.hypothesis,
            'simulation_result': best_result.result,
            'original_confidence': best_result.confidence,
            'risk_adjusted_score': best_adjusted['risk_adjusted_score'],
            'risk_multiplier': best_adjusted['risk_multiplier'],
            'risk_factors': best_adjusted['risk_factors'],
            'confidence': best_adjusted['risk_adjusted_score'],  # Use adjusted score as final confidence
            'alternatives_considered': len(simulation_results)
        }
    
    def create_vote_session(
        self,
        session_id: str,
        participants: List[str],
        question: str,
        options: List[str],
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Create a voting session for more complex consensus decisions"""
        
        if session_id in self.active_consensus_sessions:
            raise ValueError(f"Vote session {session_id} already exists")
        
        session = {
            'session_id': session_id,
            'created_at': time.time(),
            'participants': participants,
            'question': question,
            'options': options,
            'votes': {},
            'metadata': metadata or {},
            'status': 'active'
        }
        
        self.active_consensus_sessions[session_id] = session
        logger.info(f"Vote session created: {session_id} with {len(participants)} participants")
        
        return session.copy()
    
    def cast_vote(
        self,
        session_id: str,
        voter_id: str,
        choice: str,
        confidence: float,
        reasoning: str = ""
    ) -> bool:
        """Cast a vote in an active session"""
        
        if session_id not in self.active_consensus_sessions:
            logger.error(f"Vote session not found: {session_id}")
            return False
        
        session = self.active_consensus_sessions[session_id]
        
        if session['status'] != 'active':
            logger.error(f"Vote session {session_id} is not active")
            return False
        
        if voter_id not in session['participants']:
            logger.error(f"Voter {voter_id} not in session {session_id} participants")
            return False
        
        if choice not in session['options']:
            logger.error(f"Invalid choice {choice} for session {session_id}")
            return False
        
        # Cast vote
        vote = Vote(
            voter_id=voter_id,
            choice=choice,
            confidence=confidence,
            reasoning=reasoning,
            timestamp=time.time()
        )
        
        session['votes'][voter_id] = vote
        logger.info(f"Vote cast in session {session_id}: {voter_id} -> {choice} (confidence: {confidence})")
        
        return True
    
    def finalize_vote_session(self, session_id: str) -> Dict[str, Any]:
        """Finalize a vote session and return results"""
        
        if session_id not in self.active_consensus_sessions:
            raise ValueError(f"Vote session not found: {session_id}")
        
        session = self.active_consensus_sessions[session_id]
        session['status'] = 'finalized'
        session['finalized_at'] = time.time()
        
        # Tally votes
        vote_counts = {option: 0 for option in session['options']}
        confidence_sums = {option: 0.0 for option in session['options']}
        
        for vote in session['votes'].values():
            vote_counts[vote.choice] += 1
            confidence_sums[vote.choice] += vote.confidence
        
        # Determine winner
        winner = max(vote_counts.keys(), key=lambda option: vote_counts[option])
        winner_votes = vote_counts[winner]
        total_votes = len(session['votes'])
        
        # Calculate average confidence for winner
        winner_avg_confidence = (
            confidence_sums[winner] / winner_votes if winner_votes > 0 else 0.0
        )
        
        results = {
            'session_id': session_id,
            'winner': winner,
            'winner_votes': winner_votes,
            'total_votes': total_votes,
            'winner_percentage': winner_votes / max(1, total_votes),
            'winner_avg_confidence': winner_avg_confidence,
            'vote_counts': vote_counts,
            'confidence_averages': {
                option: confidence_sums[option] / max(1, vote_counts[option])
                for option in session['options']
            },
            'participation_rate': total_votes / len(session['participants']),
            'finalized_at': session['finalized_at']
        }
        
        logger.info(f"Vote session {session_id} finalized: {winner} wins with {winner_votes}/{total_votes} votes")
        return results
    
    def get_consensus_history(self, limit: int = 10) -> List[ConsensusRecord]:
        """Get recent consensus decisions"""
        return sorted(
            self.consensus_history, 
            key=lambda r: r.timestamp, 
            reverse=True
        )[:limit]
    
    def get_consensus_stats(self) -> Dict[str, Any]:
        """Get consensus system statistics"""
        
        if not self.consensus_history:
            return {
                'total_consensus_decisions': 0,
                'average_confidence': 0.0,
                'active_vote_sessions': len(self.active_consensus_sessions)
            }
        
        avg_confidence = sum(r.confidence for r in self.consensus_history) / len(self.consensus_history)
        
        # Analyze decision patterns
        strategy_counts = {}
        for record in self.consensus_history:
            strategy = record.decision.get('strategy', 'unknown')
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            'total_consensus_decisions': len(self.consensus_history),
            'average_confidence': avg_confidence,
            'active_vote_sessions': len(self.active_consensus_sessions),
            'strategy_distribution': strategy_counts,
            'recent_decisions': len([r for r in self.consensus_history if time.time() - r.timestamp < 3600]),
            'default_strategy': self.default_strategy.value
        }