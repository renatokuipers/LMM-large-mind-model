# TODO: Implement the SequenceLearning class to learn patterns over time
# This component should be able to:
# - Detect recurring patterns in temporal sequences
# - Learn sequential statistical regularities
# - Recognize variations of learned sequences
# - Predict upcoming elements in sequences

# TODO: Implement developmental progression in sequence learning:
# - Simple repetition detection in early stages
# - Short sequence learning in childhood
# - Hierarchical sequence structures in adolescence
# - Complex, multi-level sequential patterns in adulthood

# TODO: Create mechanisms for:
# - Pattern detection: Identify recurring temporal patterns
# - Statistical learning: Extract probabilistic sequence rules
# - Sequence abstraction: Recognize underlying patterns despite variations
# - Hierarchical organization: Structure sequences into meaningful units

# TODO: Implement different sequence types:
# - Action sequences: Ordered behavioral patterns
# - Perceptual sequences: Ordered sensory patterns
# - Conceptual sequences: Ordered abstract elements
# - Social sequences: Ordered interaction patterns

# TODO: Connect to memory and prediction modules
# Sequence learning should store patterns in memory
# and feed into predictive processes

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
import numpy as np
import torch
from datetime import datetime
import uuid

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient

from lmm_project.modules.temporal.models import SequencePattern, HierarchicalSequence
from lmm_project.modules.temporal.neural_net import SequenceEncoder, HierarchicalSequenceNetwork

logger = logging.getLogger(__name__)

class SequenceLearning(BaseModule):
    """
    Learns patterns over time
    
    This module detects, learns, and organizes temporal
    sequences, enabling the recognition of recurring
    patterns and prediction of future elements.
    """
    
    # Override developmental milestones with sequence learning-specific milestones
    development_milestones = {
        0.0: "Basic pattern recognition",
        0.2: "Simple sequence learning",
        0.4: "Multi-element sequence learning",
        0.6: "Statistical pattern extraction",
        0.8: "Hierarchical sequence organization",
        1.0: "Complex sequence abstraction"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the sequence learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="sequence_learning", event_bus=event_bus)
        
        # Initialize sequence representation structures
        self.sequence_patterns: Dict[str, SequencePattern] = {}
        self.hierarchical_sequences: Dict[str, HierarchicalSequence] = {}
        
        # Pattern detection statistics
        self.element_frequencies: Dict[Any, int] = defaultdict(int)
        self.transition_counts: Dict[Any, Dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.sequence_buffer: List[Any] = []
        self.max_buffer_size = 50
        
        # Neural networks for sequence processing
        self.sequence_encoder = SequenceEncoder()
        self.hierarchical_network = HierarchicalSequenceNetwork()
        
        # Embedding client for semantic processing if needed
        self.embedding_client = LLMClient()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Set up context tracking
        self.current_context: Dict[str, Any] = {}
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self.subscribe_to_message("event_sequence", self._handle_event_sequence)
            self.subscribe_to_message("pattern_query", self._handle_pattern_query)
            self.subscribe_to_message("sequence_context", self._handle_context_update)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to learn temporal sequences
        
        Args:
            input_data: Dictionary containing temporal pattern information
            
        Returns:
            Dictionary with learned sequence information
        """
        # Determine what type of input we're processing
        input_type = input_data.get("input_type", "")
        
        if input_type == "learn_sequence":
            return self._process_learn_sequence(input_data)
        elif input_type == "recognize_pattern":
            return self._process_recognize_pattern(input_data)
        elif input_type == "predict_next":
            return self._process_predict_next(input_data)
        elif input_type == "detect_hierarchy":
            return self._process_detect_hierarchy(input_data)
        else:
            # Default to sequence learning if sequence is provided
            if "sequence" in input_data:
                return self._process_learn_sequence(input_data)
            else:
                return {
                    "error": "Unknown input type or insufficient parameters",
                    "valid_types": ["learn_sequence", "recognize_pattern", "predict_next", "detect_hierarchy"]
                }
    
    def _process_learn_sequence(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and learn a new sequence"""
        sequence = input_data.get("sequence", [])
        context = input_data.get("context", {})
        
        if not sequence:
            return {"error": "Sequence data is required"}
        
        # Update context
        self.current_context.update(context)
        
        # Update buffer with new elements
        self.sequence_buffer.extend(sequence)
        if len(self.sequence_buffer) > self.max_buffer_size:
            self.sequence_buffer = self.sequence_buffer[-self.max_buffer_size:]
        
        # Update frequency and transition statistics
        self._update_statistics(sequence)
        
        # Learn patterns based on developmental level
        patterns_learned = []
        
        if self.development_level < 0.3:
            # Basic pattern detection - only simple repetitions
            simple_patterns = self._detect_simple_patterns(sequence)
            for pattern in simple_patterns:
                pattern_id = self._store_pattern(pattern, context)
                patterns_learned.append(pattern_id)
                
        elif self.development_level < 0.6:
            # Intermediate sequence learning - multi-element sequences
            multi_patterns = self._detect_multi_element_patterns(sequence)
            for pattern in multi_patterns:
                pattern_id = self._store_pattern(pattern, context)
                patterns_learned.append(pattern_id)
                
        else:
            # Advanced sequence learning - statistical and hierarchical
            stat_patterns = self._detect_statistical_patterns(sequence)
            for pattern in stat_patterns:
                pattern_id = self._store_pattern(pattern, context)
                patterns_learned.append(pattern_id)
            
            # At high development levels, also analyze hierarchy
            if self.development_level >= 0.8 and len(self.sequence_patterns) > 5:
                self._organize_hierarchical_patterns()
        
        # Return information about learned patterns
        return {
            "sequence_length": len(sequence),
            "patterns_learned": len(patterns_learned),
            "pattern_ids": patterns_learned,
            "buffer_size": len(self.sequence_buffer)
        }
    
    def _process_recognize_pattern(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Recognize patterns in a sequence"""
        sequence = input_data.get("sequence", [])
        
        if not sequence:
            return {"error": "Sequence data is required"}
        
        # Find matching patterns
        matches = []
        
        for pattern_id, pattern in self.sequence_patterns.items():
            match_score = self._compute_pattern_match(sequence, pattern.elements)
            if match_score > 0.7:  # Threshold for recognition
                matches.append({
                    "pattern_id": pattern_id,
                    "match_score": match_score,
                    "pattern_name": pattern.context.get("name", "unnamed_pattern"),
                    "frequency": pattern.frequency
                })
        
        # Sort by match score
        matches.sort(key=lambda x: x["match_score"], reverse=True)
        
        return {
            "sequence_length": len(sequence),
            "matches_found": len(matches),
            "matches": matches[:5]  # Return top 5 matches
        }
    
    def _process_predict_next(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Predict the next element(s) in a sequence"""
        sequence = input_data.get("sequence", [])
        num_predictions = input_data.get("num_predictions", 1)
        
        if not sequence:
            return {"error": "Sequence data is required"}
        
        # Calculate predictions based on transition probabilities
        transition_predictions = self._predict_next_elements(sequence, num_predictions)
        
        # For more advanced prediction at higher development levels, use neural network
        neural_predictions = []
        confidence = 0.0
        
        if self.development_level >= 0.5 and len(sequence) >= 3:
            try:
                # Convert sequence to tensor format
                # This is simplified - in a full implementation, would need proper embedding
                seq_tensor = self._sequence_to_tensor(sequence)
                
                # Use neural network for prediction
                with torch.no_grad():
                    next_element = self.sequence_encoder.predict_next(seq_tensor)
                
                # Convert back to appropriate format
                # This would depend on actual implementation details
                neural_predictions = [{"element": "neural_prediction", "probability": 0.8}]
                confidence = 0.7
            except Exception as e:
                logger.warning(f"Neural prediction failed: {str(e)}")
        
        # Combine statistical and neural predictions
        # In a full implementation, would use a more sophisticated approach
        predictions = transition_predictions
        if neural_predictions and self.development_level >= 0.7:
            predictions = neural_predictions
        
        return {
            "sequence_length": len(sequence),
            "predictions": predictions,
            "confidence": confidence
        }
    
    def _process_detect_hierarchy(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect hierarchical organization in sequences"""
        if self.development_level < 0.7:
            return {
                "error": "Hierarchical detection not available at current development level",
                "development_needed": "This capability requires development level of at least 0.7"
            }
        
        target_patterns = input_data.get("pattern_ids", [])
        
        # If no specific patterns provided, use all available patterns
        if not target_patterns and self.sequence_patterns:
            target_patterns = list(self.sequence_patterns.keys())
        
        if not target_patterns:
            return {"error": "No patterns available for hierarchical analysis"}
        
        hierarchical_structures = []
        
        # Check existing hierarchies
        for h_id, hierarchy in self.hierarchical_sequences.items():
            # Check if hierarchy contains target patterns
            if any(p_id in hierarchy.sub_sequences for p_id in target_patterns):
                hierarchical_structures.append({
                    "hierarchy_id": h_id,
                    "name": hierarchy.name,
                    "level": hierarchy.abstraction_level,
                    "patterns": hierarchy.sub_sequences
                })
        
        # Create new hierarchies if needed
        if not hierarchical_structures and len(target_patterns) >= 2:
            new_hierarchy = self._create_hierarchy(target_patterns)
            if new_hierarchy:
                hierarchical_structures.append({
                    "hierarchy_id": new_hierarchy.id,
                    "name": new_hierarchy.name,
                    "level": new_hierarchy.abstraction_level,
                    "patterns": new_hierarchy.sub_sequences
                })
        
        return {
            "hierarchical_structures": hierarchical_structures,
            "pattern_count": len(target_patterns)
        }
    
    def _update_statistics(self, sequence: List[Any]) -> None:
        """Update statistical information about sequences"""
        if not sequence:
            return
            
        # Update element frequencies
        for element in sequence:
            self.element_frequencies[element] += 1
            
        # Update transition counts
        for i in range(len(sequence) - 1):
            current = sequence[i]
            next_element = sequence[i + 1]
            self.transition_counts[current][next_element] += 1
    
    def _detect_simple_patterns(self, sequence: List[Any]) -> List[List[Any]]:
        """Detect simple repetitive patterns in a sequence"""
        patterns = []
        
        # Look for direct repetitions (AAAA)
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                patterns.append([sequence[i], sequence[i]])
        
        # Look for alternating patterns (ABABA)
        for i in range(len(sequence) - 3):
            if sequence[i] == sequence[i + 2] and sequence[i + 1] == sequence[i + 3]:
                patterns.append([sequence[i], sequence[i + 1], sequence[i + 2], sequence[i + 3]])
        
        return patterns
    
    def _detect_multi_element_patterns(self, sequence: List[Any]) -> List[List[Any]]:
        """Detect multi-element patterns in a sequence"""
        patterns = self._detect_simple_patterns(sequence)
        
        # Look for longer repeating subsequences
        for length in range(3, min(6, len(sequence) // 2 + 1)):
            for i in range(len(sequence) - length * 2 + 1):
                subseq1 = tuple(sequence[i:i + length])
                subseq2 = tuple(sequence[i + length:i + length * 2])
                
                if subseq1 == subseq2:
                    patterns.append(list(subseq1) + list(subseq2))
        
        return patterns
    
    def _detect_statistical_patterns(self, sequence: List[Any]) -> List[List[Any]]:
        """Detect patterns based on statistical regularities"""
        patterns = self._detect_multi_element_patterns(sequence)
        
        # Look for high probability transitions
        high_prob_chains = []
        
        # Start with elements that have significant frequency
        common_elements = [elem for elem, freq in self.element_frequencies.items() 
                         if freq > 3]
        
        # For each common element, find chains of high probability transitions
        for start_elem in common_elements:
            chain = [start_elem]
            current = start_elem
            
            # Build chain of 4 elements with high transition probability
            for _ in range(3):
                if not self.transition_counts[current]:
                    break
                    
                # Find the most likely next element
                next_elements = [(next_elem, count) for next_elem, count 
                              in self.transition_counts[current].items()]
                
                if not next_elements:
                    break
                    
                # Sort by transition count
                next_elements.sort(key=lambda x: x[1], reverse=True)
                
                # Add the most likely transition if probability is high enough
                total_transitions = sum(count for _, count in next_elements)
                if total_transitions > 0:
                    probability = next_elements[0][1] / total_transitions
                    if probability > 0.5:  # High probability threshold
                        current = next_elements[0][0]
                        chain.append(current)
                    else:
                        break
                else:
                    break
            
            # Only keep chains of at least 3 elements
            if len(chain) >= 3:
                high_prob_chains.append(chain)
        
        patterns.extend(high_prob_chains)
        return patterns
    
    def _store_pattern(self, pattern: List[Any], context: Dict[str, Any]) -> str:
        """Store a detected pattern"""
        # Check if pattern already exists
        for pattern_id, existing in self.sequence_patterns.items():
            if self._compute_pattern_match(pattern, existing.elements) > 0.9:
                # Update existing pattern
                existing.frequency += 1
                existing.last_observed = datetime.now()
                
                # Update context if provided
                if context:
                    existing.context.update(context)
                    
                return pattern_id
        
        # Create a new pattern
        transitions = {}
        
        # Calculate transitions within pattern
        for i in range(len(pattern) - 1):
            current = str(pattern[i])
            next_elem = str(pattern[i + 1])
            
            if current not in transitions:
                transitions[current] = {}
                
            if next_elem in transitions[current]:
                transitions[current][next_elem] += 1
            else:
                transitions[current][next_elem] = 1
        
        # Normalize transitions to probabilities
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            for next_elem in nexts:
                transitions[current][next_elem] /= total
        
        # Create the pattern object
        new_pattern = SequencePattern(
            elements=pattern,
            transitions=transitions,
            frequency=1,
            last_observed=datetime.now(),
            context=context,
            confidence=min(0.5 + self.development_level * 0.3, 0.9)  # Confidence increases with development
        )
        
        # Store the pattern
        self.sequence_patterns[new_pattern.id] = new_pattern
        
        return new_pattern.id
    
    def _compute_pattern_match(self, sequence1: List[Any], sequence2: List[Any]) -> float:
        """Compute match score between two sequences (0.0 to 1.0)"""
        # Check for empty sequences
        if not sequence1 or not sequence2:
            return 0.0
            
        # Exact match
        if sequence1 == sequence2:
            return 1.0
            
        # Different lengths - find best alignment
        if len(sequence1) != len(sequence2):
            if len(sequence1) > len(sequence2):
                # Try to find sequence2 in sequence1
                best_match = 0.0
                for i in range(len(sequence1) - len(sequence2) + 1):
                    subseq = sequence1[i:i + len(sequence2)]
                    match = self._compute_pattern_match(subseq, sequence2)
                    best_match = max(best_match, match)
                return best_match
            else:
                # Try to find sequence1 in sequence2
                best_match = 0.0
                for i in range(len(sequence2) - len(sequence1) + 1):
                    subseq = sequence2[i:i + len(sequence1)]
                    match = self._compute_pattern_match(subseq, sequence1)
                    best_match = max(best_match, match)
                return best_match
        
        # Same length but different elements - calculate element-wise similarity
        matches = sum(1 for a, b in zip(sequence1, sequence2) if a == b)
        return matches / len(sequence1)
    
    def _predict_next_elements(self, sequence: List[Any], num_predictions: int = 1) -> List[Dict[str, Any]]:
        """Predict the next elements based on observed transitions"""
        if not sequence:
            return []
            
        last_element = sequence[-1]
        
        # Check if we have transition data for this element
        if last_element not in self.transition_counts or not self.transition_counts[last_element]:
            return []
            
        # Get transition probabilities
        transitions = self.transition_counts[last_element]
        total_transitions = sum(transitions.values())
        
        # Sort by probability
        next_elements = [(next_elem, count / total_transitions) 
                        for next_elem, count in transitions.items()]
        next_elements.sort(key=lambda x: x[1], reverse=True)
        
        # Return top predictions
        return [{"element": elem, "probability": prob} 
                for elem, prob in next_elements[:num_predictions]]
    
    def _sequence_to_tensor(self, sequence: List[Any]) -> torch.Tensor:
        """Convert a sequence to a tensor representation for neural processing"""
        # This is a simplified implementation - in a real system, would use embeddings
        # For non-numeric data would need to create embeddings first
        
        # For simplicity, convert elements to integers if they aren't already
        if sequence and not isinstance(sequence[0], (int, float)):
            # Create simple mapping from elements to indices
            element_to_idx = {elem: i for i, elem in enumerate(set(sequence))}
            sequence = [element_to_idx[elem] for elem in sequence]
        
        # Create a simple one-hot encoding
        # In a real implementation, would use proper embeddings
        seq_array = np.array(sequence, dtype=np.float32)
        
        # Reshape for batch processing (batch_size=1, seq_len, features=1)
        seq_tensor = torch.tensor(seq_array).float().reshape(1, -1, 1)
        
        # Expand to expected input dimension
        input_dim = self.sequence_encoder.input_dim
        batch_size, seq_len, _ = seq_tensor.shape
        expanded = torch.zeros((batch_size, seq_len, input_dim))
        
        # Place the values in the first dimension
        expanded[:, :, 0] = seq_tensor.squeeze(-1)
        
        return expanded
    
    def _organize_hierarchical_patterns(self) -> None:
        """Organize patterns into hierarchical structures"""
        # Skip if we don't have enough patterns
        if len(self.sequence_patterns) < 5:
            return
            
        # Look for patterns that frequently occur together
        pattern_co_occurrences = defaultdict(lambda: defaultdict(int))
        
        # For simplicity, consider patterns to co-occur if they were observed
        # within a similar timeframe (comparing last_observed timestamps)
        patterns = list(self.sequence_patterns.values())
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                # Calculate time difference between observations
                time_diff = abs((pattern1.last_observed - pattern2.last_observed).total_seconds())
                
                # Consider as co-occurring if observed within a short time window
                if time_diff < 3600:  # 1 hour window as an example
                    pattern_co_occurrences[pattern1.id][pattern2.id] += 1
                    pattern_co_occurrences[pattern2.id][pattern1.id] += 1
        
        # Find clusters of co-occurring patterns
        clusters = self._cluster_patterns(pattern_co_occurrences)
        
        # Create hierarchies from clusters
        for cluster in clusters:
            if len(cluster) >= 2:
                self._create_hierarchy(cluster)
    
    def _cluster_patterns(self, co_occurrences: Dict[str, Dict[str, int]]) -> List[List[str]]:
        """Cluster patterns based on co-occurrence"""
        # Simple clustering algorithm
        # In a full implementation, would use a more sophisticated approach
        
        clusters = []
        visited = set()
        
        for pattern_id in co_occurrences:
            if pattern_id in visited:
                continue
                
            # Start a new cluster
            cluster = [pattern_id]
            visited.add(pattern_id)
            
            # Find connected patterns
            queue = [p for p, count in co_occurrences[pattern_id].items() 
                    if count >= 2 and p not in visited]
            
            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue
                    
                cluster.append(current)
                visited.add(current)
                
                # Add connected patterns
                queue.extend([p for p, count in co_occurrences[current].items() 
                            if count >= 2 and p not in visited])
            
            if len(cluster) >= 2:
                clusters.append(cluster)
        
        return clusters
    
    def _create_hierarchy(self, pattern_ids: List[str]) -> Optional[HierarchicalSequence]:
        """Create a hierarchical sequence from patterns"""
        # Ensure all pattern IDs exist
        valid_patterns = [p_id for p_id in pattern_ids if p_id in self.sequence_patterns]
        if len(valid_patterns) < 2:
            return None
            
        # Calculate transitions between patterns
        transitions = defaultdict(lambda: defaultdict(int))
        
        # Check buffer for pattern occurrences
        # This is a simplified approach - in a real system, would analyze actual sequences
        for i in range(len(valid_patterns) - 1):
            current = valid_patterns[i]
            next_pattern = valid_patterns[i + 1]
            transitions[current][next_pattern] += 1
        
        # Normalize transitions
        normalized_transitions = {}
        for current, nexts in transitions.items():
            total = sum(nexts.values())
            normalized_transitions[current] = {}
            for next_pattern, count in nexts.items():
                normalized_transitions[current][next_pattern] = count / total
        
        # Create hierarchical sequence
        hierarchy = HierarchicalSequence(
            name=f"Hierarchy_{str(uuid.uuid4())[:8]}",
            sub_sequences=valid_patterns,
            transitions=normalized_transitions,
            abstraction_level=2  # Higher levels can be created in a full implementation
        )
        
        # Store the hierarchy
        self.hierarchical_sequences[hierarchy.id] = hierarchy
        
        return hierarchy
    
    def _handle_event_sequence(self, message: Message) -> None:
        """Handle event sequence messages from the event bus"""
        content = message.content
        
        if "sequence" in content:
            self._process_learn_sequence({
                "sequence": content["sequence"],
                "context": content.get("context", {})
            })
    
    def _handle_pattern_query(self, message: Message) -> None:
        """Handle pattern query messages from the event bus"""
        content = message.content
        
        if "sequence" in content:
            recognition_result = self._process_recognize_pattern({
                "sequence": content["sequence"]
            })
            
            # Publish recognition results if requested
            if content.get("return_result", False) and self.event_bus:
                self.publish_message("pattern_recognition_result", recognition_result)
    
    def _handle_context_update(self, message: Message) -> None:
        """Handle context update messages from the event bus"""
        content = message.content
        
        if "context" in content:
            self.current_context.update(content["context"])
    
    def get_pattern_by_id(self, pattern_id: str) -> Optional[SequencePattern]:
        """Get a pattern by its ID"""
        return self.sequence_patterns.get(pattern_id)
    
    def get_hierarchy_by_id(self, hierarchy_id: str) -> Optional[HierarchicalSequence]:
        """Get a hierarchical sequence by its ID"""
        return self.hierarchical_sequences.get(hierarchy_id)
    
    def get_all_patterns(self) -> List[SequencePattern]:
        """Get all learned patterns"""
        return list(self.sequence_patterns.values())
    
    def clear_buffer(self) -> None:
        """Clear the sequence buffer"""
        self.sequence_buffer = []
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Call the parent's implementation
        new_level = super().update_development(amount)
        
        # If development crossed a threshold, enhance capabilities
        if self.development_level >= 0.8 and new_level > self.development_level:
            # At high development, reorganize hierarchical patterns
            self._organize_hierarchical_patterns()
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add sequence learning-specific state information
        state.update({
            "pattern_count": len(self.sequence_patterns),
            "hierarchy_count": len(self.hierarchical_sequences),
            "buffer_size": len(self.sequence_buffer)
        })
        
        return state
