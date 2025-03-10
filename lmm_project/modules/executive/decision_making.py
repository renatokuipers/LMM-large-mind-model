# TODO: Implement the DecisionMaking class to evaluate options and make choices
# This component should be able to:
# - Evaluate multiple options based on various criteria
# - Calculate expected outcomes and utilities
# - Manage risk and uncertainty in decisions
# - Balance short-term and long-term consequences

# TODO: Implement developmental progression in decision making:
# - Simple immediate-reward decisions in early stages
# - Growing consideration of multiple factors in childhood
# - Inclusion of long-term outcomes in adolescence
# - Complex trade-off analysis in adulthood

# TODO: Create mechanisms for:
# - Option generation: Identify possible choices
# - Value assignment: Determine the worth of potential outcomes
# - Probability estimation: Assess likelihood of outcomes
# - Outcome integration: Combine multiple factors into decisions

# TODO: Implement different decision strategies:
# - Maximizing: Select the option with highest expected utility
# - Satisficing: Select first option meeting minimum criteria
# - Elimination by aspects: Sequentially remove options failing criteria
# - Recognition-primed: Use past experience to make rapid decisions

# TODO: Connect to emotion and memory systems
# Decision making should be influenced by emotional responses
# and informed by memories of past decisions and outcomes

import logging
import time
import uuid
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import torch
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.executive.models import Decision, ExecutiveNeuralState
from lmm_project.modules.executive.neural_net import DecisionNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class DecisionMaking(BaseModule):
    """
    Evaluates options and makes choices
    
    This module weighs alternatives and selects actions based on
    expected outcomes, values, and contextual factors.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Simple reward-based decisions",
        0.2: "Multi-factor decisions",
        0.4: "Short-term risk assessment",
        0.6: "Long-term outcome consideration",
        0.8: "Complex trade-off analysis",
        1.0: "Strategic decision-making"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the decision making module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="decision_making", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device
        self.device = get_device()
        
        # Initialize neural network
        self.decision_network = DecisionNetwork(
            option_dim=64,
            criteria_dim=16,
            hidden_dim=128
        ).to(self.device)
        
        # Set development level for network
        self.decision_network.set_development_level(development_level)
        
        # Create neural state for tracking
        self.neural_state = ExecutiveNeuralState()
        self.neural_state.decision_development = development_level
        
        # Recent decisions history
        self.decision_history = deque(maxlen=20)
        
        # Decision making parameters
        self.params = {
            "max_options": 3,  # Maximum number of options to consider
            "max_criteria": 2,  # Maximum number of criteria to consider
            "time_allocation": 0.5,  # Relative time allocation (0-1)
            "risk_aversion": 0.5,  # Risk aversion level (0-1)
            "consider_long_term": False,  # Whether to consider long-term outcomes
            "confidence_threshold": 0.6  # Threshold for high-confidence decisions
        }
        
        # Update parameters based on development
        self._adjust_parameters_for_development()
        
        logger.info(f"Decision making module initialized at development level {development_level:.2f}")
    
    def _adjust_parameters_for_development(self):
        """Adjust decision making parameters based on developmental level"""
        if self.development_level < 0.2:
            # Simple decision making at early stages
            self.params.update({
                "max_options": 2,
                "max_criteria": 1,
                "time_allocation": 0.3,
                "risk_aversion": 0.7,  # High risk aversion (conservative)
                "consider_long_term": False,
                "confidence_threshold": 0.7  # Require high confidence
            })
        elif self.development_level < 0.4:
            # Multi-factor decisions
            self.params.update({
                "max_options": 3,
                "max_criteria": 2,
                "time_allocation": 0.4,
                "risk_aversion": 0.6,
                "consider_long_term": False,
                "confidence_threshold": 0.65
            })
        elif self.development_level < 0.6:
            # Short-term risk assessment
            self.params.update({
                "max_options": 4,
                "max_criteria": 3,
                "time_allocation": 0.5,
                "risk_aversion": 0.5,
                "consider_long_term": False,
                "confidence_threshold": 0.6
            })
        elif self.development_level < 0.8:
            # Long-term outcome consideration
            self.params.update({
                "max_options": 5,
                "max_criteria": 4,
                "time_allocation": 0.6,
                "risk_aversion": 0.4,
                "consider_long_term": True,
                "confidence_threshold": 0.55
            })
        else:
            # Strategic decision-making
            self.params.update({
                "max_options": 7,
                "max_criteria": 5,
                "time_allocation": 0.7,
                "risk_aversion": 0.3,
                "consider_long_term": True,
                "confidence_threshold": 0.5
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to make decisions
        
        Args:
            input_data: Dictionary containing decision problem information
                Required keys:
                - 'options': Dict[str, Dict[str, Any]] - Options to choose from with their attributes
                - 'criteria': Dict[str, float] - Decision criteria and their weights
                Optional keys:
                - 'context': Dict[str, Any] - Contextual information
                - 'time_allocation': float - How much time to allocate (overrides default)
                - 'deadline': float - Time by which decision is needed
                - 'operation': str - specific operation ('decide', 'explain', 'revise')
            
        Returns:
            Dictionary with the results of decision making
        """
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        operation = input_data.get("operation", "decide")
        
        # Different operations based on the request
        if operation == "decide":
            return self._make_decision(input_data, process_id)
        elif operation == "explain":
            return self._explain_decision(input_data, process_id)
        elif operation == "revise":
            return self._revise_decision(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _make_decision(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Make a decision based on options and criteria"""
        # Extract required data
        if "options" not in input_data:
            return {"status": "error", "message": "No options provided", "process_id": process_id}
        if "criteria" not in input_data:
            return {"status": "error", "message": "No criteria provided", "process_id": process_id}
        
        options = input_data.get("options", {})
        criteria = input_data.get("criteria", {})
        context = input_data.get("context", {})
        decision_context = input_data.get("decision_context", "General decision")
        
        # Apply development-based limits
        options_limited = dict(list(options.items())[:self.params["max_options"]])
        criteria_limited = dict(list(criteria.items())[:self.params["max_criteria"]])
        
        # Record decision start time
        start_time = time.time()
        
        # Format options and criteria for neural processing
        options_list = list(options_limited.keys())
        option_features = self._extract_option_features(options_limited)
        criteria_features = self._extract_criteria_features(criteria_limited)
        context_features = self._extract_context_features(context) if context else None
        
        # Process through neural network
        with torch.no_grad():
            decision_result = self.decision_network(
                options=option_features.to(self.device),
                criteria=criteria_features.to(self.device),
                context=context_features.to(self.device) if context_features is not None else None
            )
        
        # Extract results
        scores = decision_result["scores"].cpu().numpy()[0]
        probabilities = decision_result["probabilities"].cpu().numpy()[0]
        confidence = decision_result["confidence"].cpu().item()
        best_option_idx = decision_result["best_option_idx"].cpu().item()
        
        # Map back to option names
        option_scores = {opt: float(scores[i]) for i, opt in enumerate(options_list)}
        
        # Record activation in neural state
        self.neural_state.add_activation('decision', {
            'options_count': len(options_limited),
            'criteria_count': len(criteria_limited),
            'confidence': confidence,
            'best_option_idx': best_option_idx
        })
        
        # Select best option
        selected_option = options_list[best_option_idx]
        
        # Calculate decision time
        decision_time = time.time() - start_time
        
        # Create evaluations dictionary (how each option scores on each criterion)
        evaluations = {}
        for option_name in options_list:
            option_eval = {}
            for criterion_name, weight in criteria_limited.items():
                # In a real implementation, this would use actual evaluation logic
                # Here we generate plausible values based on option and criterion
                option_attrs = options_limited[option_name]
                if criterion_name in option_attrs:
                    option_eval[criterion_name] = option_attrs[criterion_name]
                else:
                    # Generate a value that's consistent with the final scores
                    base_score = option_scores[option_name] / len(criteria_limited)
                    # Add some noise to make it realistic
                    noise = np.random.normal(0, 0.1)
                    option_eval[criterion_name] = max(0.0, min(1.0, base_score + noise))
            
            evaluations[option_name] = option_eval
        
        # Create decision record
        decision = Decision(
            decision_id=str(uuid.uuid4()),
            context=decision_context,
            options=options_limited,
            criteria=criteria_limited,
            evaluations=evaluations,
            option_scores=option_scores,
            selected_option=selected_option,
            confidence=confidence,
            decision_time=decision_time
        )
        
        # Add to history
        self.decision_history.append(decision)
        
        # Return decision result
        return {
            "status": "success",
            "decision": decision.dict(),
            "process_id": process_id,
            "development_level": self.development_level
        }
    
    def _explain_decision(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Explain the reasoning behind a decision"""
        # Extract decision ID
        decision_id = input_data.get("decision_id")
        if not decision_id:
            return {"status": "error", "message": "No decision ID provided", "process_id": process_id}
        
        # Find the decision in history
        decision = next((d for d in self.decision_history if d.decision_id == decision_id), None)
        if not decision:
            return {"status": "error", "message": "Decision not found", "process_id": process_id}
        
        # Generate explanation based on development level
        if self.development_level < 0.3:
            # Simple explanation
            explanation = f"Selected {decision.selected_option} because it scored highest."
            factors = [f"{decision.selected_option} had a score of {decision.option_scores[decision.selected_option]:.2f}"]
            
        elif self.development_level < 0.6:
            # More detailed explanation with criteria
            explanation = f"Selected {decision.selected_option} based on evaluation of criteria."
            
            # Find strongest criteria for selected option
            option_evals = decision.evaluations[decision.selected_option]
            top_criteria = sorted(option_evals.items(), key=lambda x: x[1], reverse=True)
            
            factors = [
                f"{criterion}: {score:.2f}" 
                for criterion, score in top_criteria
            ]
            
        else:
            # Comprehensive explanation with comparison
            explanation = f"Selected {decision.selected_option} after comprehensive analysis of all options and criteria."
            
            # Compare selected option with runners-up
            sorted_options = sorted(decision.option_scores.items(), key=lambda x: x[1], reverse=True)
            top_options = sorted_options[:min(3, len(sorted_options))]
            
            # Find distinguishing criteria
            factors = []
            if len(top_options) > 1:
                selected = decision.selected_option
                runner_up = top_options[1][0]
                
                factors.append(f"{selected} scored {decision.option_scores[selected]:.2f} overall, compared to {runner_up}'s {decision.option_scores[runner_up]:.2f}")
                
                # Compare on specific criteria
                for criterion, weight in decision.criteria.items():
                    selected_score = decision.evaluations[selected].get(criterion, 0)
                    runner_up_score = decision.evaluations[runner_up].get(criterion, 0)
                    
                    if abs(selected_score - runner_up_score) > 0.1:
                        factors.append(f"On {criterion} (weight: {weight:.2f}), {selected} scored {selected_score:.2f} vs. {runner_up}'s {runner_up_score:.2f}")
            
            # Add confidence information
            factors.append(f"Decision confidence: {decision.confidence:.2f}")
        
        return {
            "status": "success",
            "decision_id": decision_id,
            "explanation": explanation,
            "factors": factors,
            "decision": decision.dict(),
            "process_id": process_id
        }
    
    def _revise_decision(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Revise a previous decision with new information"""
        # Extract decision ID
        decision_id = input_data.get("decision_id")
        if not decision_id:
            return {"status": "error", "message": "No decision ID provided", "process_id": process_id}
        
        # Find the decision in history
        original_decision = next((d for d in self.decision_history if d.decision_id == decision_id), None)
        if not original_decision:
            return {"status": "error", "message": "Decision not found", "process_id": process_id}
        
        # Get updates to options, criteria, or context
        updated_options = input_data.get("options", original_decision.options)
        updated_criteria = input_data.get("criteria", original_decision.criteria)
        updated_context = input_data.get("context", {})
        
        # Create new input data with updates
        new_input = {
            "options": updated_options,
            "criteria": updated_criteria,
            "context": updated_context,
            "decision_context": f"Revision of: {original_decision.context}",
            "process_id": process_id
        }
        
        # Make a new decision
        new_decision_result = self._make_decision(new_input, process_id)
        
        # Add revision information
        if new_decision_result["status"] == "success":
            revision_info = {
                "original_decision_id": decision_id,
                "changed": new_decision_result["decision"]["selected_option"] != original_decision.selected_option,
                "original_option": original_decision.selected_option
            }
            new_decision_result["revision_info"] = revision_info
        
        return new_decision_result
    
    def _extract_option_features(self, options: Dict[str, Dict[str, Any]]) -> torch.Tensor:
        """
        Extract features from options for neural processing
        
        Args:
            options: Dictionary of options with their attributes
            
        Returns:
            Tensor of options features [1, num_options, feature_dim]
        """
        num_options = len(options)
        feature_dim = 64  # Must match network's option_dim
        
        # Initialize feature tensor
        features = np.zeros((1, num_options, feature_dim))
        
        # For each option, extract features
        for i, (option_name, option_attrs) in enumerate(options.items()):
            if i >= num_options:
                break
                
            # For demonstration, create simple features
            # In a real implementation, this would do proper feature extraction
            
            # Create feature vector based on option attributes
            if option_attrs:
                # Use attribute values as features where possible
                attr_values = []
                for attr, value in option_attrs.items():
                    if isinstance(value, (int, float)):
                        attr_values.append(value)
                    elif isinstance(value, bool):
                        attr_values.append(1.0 if value else 0.0)
                
                # Fill initial positions with actual values
                for j, val in enumerate(attr_values):
                    if j < feature_dim:
                        features[0, i, j] = val
            
            # Fill remaining with random values seeded by option name
            seed = hash(option_name) % 10000
            np.random.seed(seed)
            
            # Calculate how many positions remain to be filled
            remaining = feature_dim - min(feature_dim, len(option_attrs))
            if remaining > 0:
                random_features = np.random.randn(remaining)
                features[0, i, -remaining:] = random_features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_criteria_features(self, criteria: Dict[str, float]) -> torch.Tensor:
        """
        Extract features from criteria for neural processing
        
        Args:
            criteria: Dictionary of criteria with their weights
            
        Returns:
            Tensor of criteria features [1, feature_dim]
        """
        feature_dim = 16  # Must match network's criteria_dim
        
        # Initialize feature tensor
        features = np.zeros((1, feature_dim))
        
        # Use criteria weights as features where possible
        for i, (criterion, weight) in enumerate(criteria.items()):
            if i < feature_dim:
                features[0, i] = weight
        
        # Fill remainder with values derived from criteria names
        criteria_names = list(criteria.keys())
        if criteria_names:
            seed = hash("".join(criteria_names)) % 10000
            np.random.seed(seed)
            
            # Calculate how many positions remain to be filled
            remaining = feature_dim - min(feature_dim, len(criteria))
            if remaining > 0:
                random_features = np.random.randn(remaining)
                features[0, -remaining:] = random_features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def _extract_context_features(self, context: Dict[str, Any]) -> torch.Tensor:
        """
        Extract features from context for neural processing
        
        Args:
            context: Dictionary of contextual information
            
        Returns:
            Tensor of context features [1, feature_dim]
        """
        feature_dim = 64  # Must match network's option_dim for the context encoder
        
        # Initialize feature tensor
        features = np.zeros((1, feature_dim))
        
        if context:
            # Use context values as features where possible
            context_values = []
            for key, value in context.items():
                if isinstance(value, (int, float)):
                    context_values.append(value)
                elif isinstance(value, bool):
                    context_values.append(1.0 if value else 0.0)
            
            # Fill initial positions with actual values
            for i, val in enumerate(context_values):
                if i < feature_dim:
                    features[0, i] = val
            
            # Fill remainder with values derived from context
            seed = hash(str(sorted(context.items()))) % 10000
            np.random.seed(seed)
            
            # Calculate how many positions remain to be filled
            remaining = feature_dim - min(feature_dim, len(context_values))
            if remaining > 0:
                random_features = np.random.randn(remaining)
                features[0, -remaining:] = random_features
        
        return torch.tensor(features, dtype=torch.float32)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update network development level
        self.decision_network.set_development_level(new_level)
        
        # Update neural state
        self.neural_state.decision_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # Adjust parameters based on new development level
        self._adjust_parameters_for_development()
        
        logger.info(f"Decision making module development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Get base state from parent
        base_state = super().get_state()
        
        # Add decision-specific state
        decision_state = {
            "params": self.params,
            "recent_decisions": [d.dict() for d in self.decision_history],
            "decision_count": len(self.decision_history)
        }
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.decision_development,
            "accuracy": self.neural_state.decision_accuracy,
            "recent_activations_count": len(self.neural_state.recent_decision_activations)
        }
        
        # Combine states
        combined_state = {**base_state, **decision_state, **neural_state}
        
        return combined_state
