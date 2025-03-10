import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging

from lmm_project.modules.self_regulation.models import RegulationNeuralState

logger = logging.getLogger("lmm.self_regulation.neural")

def get_device() -> torch.device:
    """Get the appropriate device (GPU if available, otherwise CPU)"""
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU")
        return torch.device("cuda")
    logger.info("CUDA is not available, using CPU")
    return torch.device("cpu")

class RegulationNetwork(nn.Module):
    """
    Base network for self-regulation functions
    
    This class provides common functionality for all self-regulation
    neural networks, including developmental scaling and adaptation.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        network_type: str = "generic",
        developmental_level: float = 0.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.network_type = network_type
        self.developmental_level = developmental_level
        
        # Basic network architecture
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden_layer2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.output_layer = nn.Linear(hidden_dim // 2, output_dim)
        
        # Add dropout for regularization (adjusted by development level)
        self.dropout_rate = max(0.1, 0.5 - (self.developmental_level * 0.4))
        self.dropout = nn.Dropout(self.dropout_rate)
        
        # Track device
        self.device = get_device()
        self.to(self.device)
        
        logger.info(f"Initialized {network_type} network with development level {developmental_level:.2f}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network"""
        # Ensure input is on the correct device
        if x.device != self.device:
            x = x.to(self.device)
            
        # Apply network layers with appropriate activation functions
        x = F.relu(self.input_layer(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer1(x))
        x = self.dropout(x)
        x = F.relu(self.hidden_layer2(x))
        
        # Output activation depends on type
        if self.network_type == "emotional_regulation":
            # For emotion regulation, outputs are probabilities of different strategies
            return F.softmax(self.output_layer(x), dim=-1)
        elif self.network_type == "impulse_control":
            # For impulse control, output is inhibition strength (0-1)
            return torch.sigmoid(self.output_layer(x))
        else:
            # For other types, use tanh to get outputs in the range [-1, 1]
            return torch.tanh(self.output_layer(x))

    def update_developmental_level(self, level: float):
        """Update the network based on new developmental level"""
        self.developmental_level = level
        
        # Adjust dropout based on development
        self.dropout_rate = max(0.1, 0.5 - (self.developmental_level * 0.4))
        self.dropout.p = self.dropout_rate
        
        logger.debug(f"Updated {self.network_type} development to {level:.2f}, dropout: {self.dropout_rate:.2f}")

class EmotionalRegulationNetwork(RegulationNetwork):
    """
    Neural network for emotional regulation
    
    This network learns to select appropriate regulation strategies
    for different emotional states, with effectiveness improving
    with development.
    """
    def __init__(
        self, 
        input_dim: int = 16, 
        hidden_dim: int = 32, 
        num_strategies: int = 8,
        developmental_level: float = 0.0
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_strategies,
            network_type="emotional_regulation",
            developmental_level=developmental_level
        )
        
        # Additional layers specific to emotion regulation
        self.emotion_embedding = nn.Linear(input_dim, hidden_dim)
        self.strategy_preference = nn.Parameter(torch.ones(num_strategies) / num_strategies)
        
        # Strategy effectiveness based on development
        self.strategy_complexity = nn.Parameter(
            torch.linspace(0.1, 0.9, num_strategies),  # Increasing complexity
            requires_grad=False
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process an emotional state to select regulation strategies
        
        Args:
            x: Tensor representing emotional state [batch_size, input_dim]
            
        Returns:
            Dictionary with strategy activations and effectiveness
        """
        # Basic forward pass
        base_activations = super().forward(x)
        
        # Apply developmental scaling to strategy selection
        # Less developed minds can only effectively use simpler strategies
        dev_scaling = torch.clamp(
            self.developmental_level - self.strategy_complexity,
            min=0.1,
            max=1.0
        )
        
        # Scale activations by development-appropriate effectiveness
        scaled_activations = base_activations * dev_scaling.unsqueeze(0)
        
        # Normalize to get final strategy selection probabilities
        strategy_selection = F.softmax(scaled_activations, dim=-1)
        
        return {
            "strategy_selection": strategy_selection,
            "base_activations": base_activations,
            "effectiveness": dev_scaling
        }
    
    def select_strategy(self, emotion_vector: torch.Tensor) -> Tuple[int, float]:
        """
        Select the best regulation strategy for an emotional state
        
        Args:
            emotion_vector: Tensor representing emotional state
            
        Returns:
            Tuple of (strategy_index, effectiveness)
        """
        with torch.no_grad():
            result = self.forward(emotion_vector)
            strategy_selection = result["strategy_selection"]
            
            # Get the highest probability strategy
            if len(strategy_selection.shape) > 1:
                strategy_idx = torch.argmax(strategy_selection[0]).item()
            else:
                strategy_idx = torch.argmax(strategy_selection).item()
                
            # Get effectiveness for this strategy
            effectiveness = result["effectiveness"][strategy_idx].item()
            
            return strategy_idx, effectiveness

class ImpulseControlNetwork(RegulationNetwork):
    """
    Neural network for impulse control
    
    This network learns to inhibit impulsive responses based on
    contextual factors and developmental capability.
    """
    def __init__(
        self, 
        input_dim: int = 12, 
        hidden_dim: int = 24, 
        developmental_level: float = 0.0
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=3,  # [inhibition_strength, delay_capacity, alternative_activation]
            network_type="impulse_control",
            developmental_level=developmental_level
        )
        
        # Impulse-specific representations
        self.impulse_embedding = nn.Linear(input_dim, hidden_dim)
        
        # Inhibition capacity grows with development
        self.base_inhibition_capacity = 0.2
        self.developmental_scaling = max(0.0, self.developmental_level * 0.8)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process impulse and context to produce control signals
        
        Args:
            x: Tensor representing impulse and context [batch_size, input_dim]
            
        Returns:
            Dictionary with control signals and capacity
        """
        # Get base activations from parent
        base_output = super().forward(x)
        
        # Extract specific control signals
        inhibition_strength = base_output[:, 0]
        delay_capacity = base_output[:, 1]
        alternative_activation = base_output[:, 2]
        
        # Scale by developmental capacity
        inhibition_capacity = self.base_inhibition_capacity + self.developmental_scaling
        
        # Final control signals with developmental scaling
        scaled_inhibition = inhibition_strength * inhibition_capacity
        
        return {
            "inhibition_strength": scaled_inhibition,
            "delay_capacity": delay_capacity,
            "alternative_activation": alternative_activation,
            "control_capacity": torch.tensor(inhibition_capacity, device=self.device)
        }
    
    def evaluate_control(self, impulse_vector: torch.Tensor, impulse_strength: float) -> Dict[str, float]:
        """
        Evaluate whether an impulse can be controlled
        
        Args:
            impulse_vector: Tensor representing impulse and context
            impulse_strength: Strength of the impulse (0-1)
            
        Returns:
            Dictionary with control evaluation
        """
        with torch.no_grad():
            result = self.forward(impulse_vector)
            
            # Compare impulse strength with inhibition capacity
            inhibition = result["inhibition_strength"].item() if hasattr(result["inhibition_strength"], "item") else result["inhibition_strength"][0].item()
            
            # Determine if control is successful
            is_controlled = inhibition > impulse_strength
            
            # Calculate control success (0-1)
            control_success = min(1.0, inhibition / max(0.1, impulse_strength))
            
            return {
                "is_controlled": is_controlled,
                "control_success": control_success,
                "inhibition_strength": inhibition,
                "delay_capacity": result["delay_capacity"].item() if hasattr(result["delay_capacity"], "item") else result["delay_capacity"][0].item()
            }
    
    def update_developmental_level(self, level: float):
        """Update inhibition capacity based on development"""
        super().update_developmental_level(level)
        self.developmental_scaling = max(0.0, level * 0.8)

class SelfMonitoringNetwork(RegulationNetwork):
    """
    Neural network for self-monitoring
    
    This network detects discrepancies between current states and goals,
    monitors errors, and provides feedback for regulation.
    """
    def __init__(
        self, 
        input_dim: int = 20, 
        hidden_dim: int = 32, 
        developmental_level: float = 0.0
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=5,  # [discrepancy, error_detection, progress, attention_allocation, correction_signal]
            network_type="self_monitoring",
            developmental_level=developmental_level
        )
        
        # For comparing current state with goal state
        self.comparison_layer = nn.Bilinear(hidden_dim // 2, hidden_dim // 2, hidden_dim // 2)
        
        # Monitoring sensitivity increases with development
        self.detection_threshold = max(0.7, 0.9 - (self.developmental_level * 0.6))
    
    def forward(self, current_state: torch.Tensor, goal_state: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process current state and goal state to monitor discrepancies
        
        Args:
            current_state: Tensor representing current state
            goal_state: Optional tensor representing goal state
            
        Returns:
            Dictionary with monitoring signals
        """
        # Process current state
        current_encoded = F.relu(self.input_layer(current_state))
        current_encoded = self.dropout(current_encoded)
        current_hidden = F.relu(self.hidden_layer1(current_encoded))
        
        # If no goal state is provided, use a default "ideal" state based on development
        if goal_state is None:
            # Create a simple target state (this would be more sophisticated in a real implementation)
            goal_state = torch.zeros_like(current_state)
            goal_hidden = torch.zeros_like(current_hidden)
        else:
            # Process goal state
            goal_encoded = F.relu(self.input_layer(goal_state))
            goal_encoded = self.dropout(goal_encoded)
            goal_hidden = F.relu(self.hidden_layer1(goal_encoded))
        
        # Compare current state with goal state
        state_diff = current_hidden - goal_hidden
        comparison = self.comparison_layer(
            current_hidden[:, :self.hidden_dim // 2], 
            goal_hidden[:, :self.hidden_dim // 2]
        )
        
        # Process comparison through final layers
        combined = torch.cat([comparison, torch.abs(state_diff)], dim=-1)
        monitoring_signals = self.output_layer(combined)
        
        # Extract specific monitoring outputs
        discrepancy = torch.sigmoid(monitoring_signals[:, 0])
        error_detection = torch.sigmoid(monitoring_signals[:, 1])
        progress = torch.sigmoid(monitoring_signals[:, 2])
        attention_allocation = F.softmax(monitoring_signals[:, 3:], dim=-1)
        
        return {
            "discrepancy": discrepancy,
            "error_detection": error_detection,
            "progress": progress,
            "attention_allocation": attention_allocation,
            "detection_threshold": torch.tensor(self.detection_threshold, device=self.device)
        }
    
    def detect_issues(self, current_state: torch.Tensor, goal_state: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """
        Detect issues by comparing current state with goal state
        
        Args:
            current_state: Tensor representing current state
            goal_state: Optional tensor representing goal state
            
        Returns:
            Dictionary with detection results
        """
        with torch.no_grad():
            result = self.forward(current_state, goal_state)
            
            # Get monitoring signals
            discrepancy = result["discrepancy"].item() if hasattr(result["discrepancy"], "item") else result["discrepancy"][0].item()
            error_detection = result["error_detection"].item() if hasattr(result["error_detection"], "item") else result["error_detection"][0].item()
            
            # Determine if issues are detected based on threshold
            discrepancy_detected = discrepancy > self.detection_threshold
            error_detected = error_detection > self.detection_threshold
            
            return {
                "discrepancy_detected": discrepancy_detected,
                "discrepancy_magnitude": discrepancy,
                "error_detected": error_detected,
                "error_magnitude": error_detection,
                "progress": result["progress"].item() if hasattr(result["progress"], "item") else result["progress"][0].item()
            }
    
    def update_developmental_level(self, level: float):
        """Update detection sensitivity based on development"""
        super().update_developmental_level(level)
        # More developed minds can detect smaller discrepancies (lower threshold)
        self.detection_threshold = max(0.2, 0.9 - (level * 0.7))

class RegulationController:
    """
    Controller for the self-regulation neural networks
    
    This class coordinates the different neural networks for
    emotional regulation, impulse control, and self-monitoring.
    """
    def __init__(self, developmental_level: float = 0.0):
        """
        Initialize the regulation controller
        
        Args:
            developmental_level: Initial developmental level (0-1)
        """
        self.developmental_level = developmental_level
        self.device = get_device()
        
        # Initialize networks
        self.emotion_network = EmotionalRegulationNetwork(
            developmental_level=developmental_level
        )
        
        self.impulse_network = ImpulseControlNetwork(
            developmental_level=developmental_level
        )
        
        self.monitoring_network = SelfMonitoringNetwork(
            developmental_level=developmental_level
        )
        
        # Regulation strategies indexed by ID
        self.regulation_strategies = {
            0: "distraction",
            1: "reappraisal",
            2: "suppression",
            3: "situation_modification",
            4: "attention_deployment",
            5: "response_modulation",
            6: "cognitive_change",
            7: "acceptance"
        }
        
        # Track neural state
        self.neural_state = RegulationNeuralState(
            developmental_level=developmental_level
        )
        
        logger.info(f"Regulation controller initialized at level {developmental_level:.2f}")
    
    def regulate_emotion(self, emotion_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply emotional regulation to an emotional state
        
        Args:
            emotion_data: Dictionary with emotion information
                Required keys: emotion_type, intensity, valence, arousal
                
        Returns:
            Dictionary with regulation results
        """
        # Create vector representation of emotion
        emotion_vector = self._create_emotion_vector(emotion_data)
        
        # Get regulation strategy from network
        strategy_idx, effectiveness = self.emotion_network.select_strategy(emotion_vector)
        strategy_name = self.regulation_strategies.get(strategy_idx, "unknown")
        
        # Calculate regulation effects
        # In a more complex implementation, each strategy would have different effects
        # For now, we'll use a simple model where effectiveness scales with development
        
        # Calculate regulated intensity
        original_intensity = emotion_data.get("intensity", 0.8)
        regulated_intensity = max(0.1, original_intensity * (1.0 - effectiveness))
        
        # Calculate regulation success
        regulation_success = 1.0 - (regulated_intensity / original_intensity)
        
        # Update neural state
        self.neural_state.emotional_regulation_activation = {
            strategy_name: effectiveness,
            "intensity_reduction": regulation_success
        }
        
        return {
            "original_emotion": emotion_data,
            "regulation_strategy": strategy_name,
            "regulated_intensity": regulated_intensity,
            "regulation_success": regulation_success,
            "regulation_effectiveness": effectiveness
        }
    
    def control_impulse(self, impulse_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply impulse control to an impulse
        
        Args:
            impulse_data: Dictionary with impulse information
                Required keys: impulse_type, strength, valence
                
        Returns:
            Dictionary with control results
        """
        # Create vector representation of impulse
        impulse_vector = self._create_impulse_vector(impulse_data)
        
        # Get control evaluation from network
        impulse_strength = impulse_data.get("strength", 0.7)
        control_result = self.impulse_network.evaluate_control(impulse_vector, impulse_strength)
        
        # Determine control strategy based on impulse type
        impulse_type = impulse_data.get("impulse_type", "approach")
        if impulse_type in ["approach", "consumption"]:
            control_strategy = "inhibition" if control_result["is_controlled"] else "indulgence"
        else:
            control_strategy = "alternative_action" if control_result["is_controlled"] else "avoidance"
        
        # Update neural state
        self.neural_state.impulse_control_activation = {
            "inhibition": control_result["inhibition_strength"],
            "delay": control_result["delay_capacity"],
            "success": float(control_result["is_controlled"])
        }
        
        return {
            "original_impulse": impulse_data,
            "is_controlled": control_result["is_controlled"],
            "control_strategy": control_strategy,
            "control_success": control_result["control_success"],
            "inhibition_strength": control_result["inhibition_strength"]
        }
    
    def monitor_state(self, current_state: Dict[str, Any], goal_state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Monitor current state relative to goal state
        
        Args:
            current_state: Dictionary with current state information
            goal_state: Optional dictionary with goal state information
                
        Returns:
            Dictionary with monitoring results
        """
        # Create vector representations
        current_vector = self._create_state_vector(current_state)
        goal_vector = self._create_state_vector(goal_state) if goal_state else None
        
        # Get monitoring results from network
        detection_result = self.monitoring_network.detect_issues(current_vector, goal_vector)
        
        # Determine monitoring type based on content
        if "emotion" in current_state:
            monitoring_type = "emotional"
        elif "behavior" in current_state:
            monitoring_type = "behavioral"
        elif "goal" in current_state:
            monitoring_type = "goal_progress"
        else:
            monitoring_type = "cognitive"
        
        # Update neural state
        self.neural_state.self_monitoring_activation = {
            "discrepancy": detection_result["discrepancy_magnitude"],
            "error": detection_result["error_magnitude"],
            "progress": detection_result["progress"]
        }
        
        return {
            "monitoring_type": monitoring_type,
            "observed_state": current_state,
            "goal_state": goal_state,
            "discrepancy_detected": detection_result["discrepancy_detected"],
            "discrepancy_magnitude": detection_result["discrepancy_magnitude"],
            "error_detected": detection_result["error_detected"],
            "error_magnitude": detection_result["error_magnitude"],
            "progress": detection_result["progress"]
        }
    
    def update_development(self, amount: float) -> None:
        """
        Update developmental level of all networks
        
        Args:
            amount: Amount to increase development by (can be negative)
        """
        new_level = min(1.0, max(0.0, self.developmental_level + amount))
        if new_level == self.developmental_level:
            return
            
        # Update internal level
        self.developmental_level = new_level
        
        # Update each network
        self.emotion_network.update_developmental_level(new_level)
        self.impulse_network.update_developmental_level(new_level)
        self.monitoring_network.update_developmental_level(new_level)
        
        # Update neural state parameters
        self.neural_state.update_developmental_parameters(new_level)
        
        logger.info(f"Updated regulation controller development to {new_level:.2f}")
    
    def get_neural_state(self) -> RegulationNeuralState:
        """Get the current neural state"""
        return self.neural_state
    
    def _create_emotion_vector(self, emotion_data: Dict[str, Any]) -> torch.Tensor:
        """
        Create a vector representation of an emotional state
        
        Args:
            emotion_data: Dictionary with emotion information
            
        Returns:
            Tensor representation of emotional state
        """
        # Extract emotion features
        emotion_type = emotion_data.get("emotion_type", "neutral")
        intensity = emotion_data.get("intensity", 0.5)
        valence = emotion_data.get("valence", 0.0)
        arousal = emotion_data.get("arousal", 0.5)
        
        # One-hot encode emotion type (simplified)
        emotion_types = ["anger", "fear", "joy", "sadness", "disgust", "surprise", "neutral"]
        emotion_idx = emotion_types.index(emotion_type) if emotion_type in emotion_types else 6
        emotion_onehot = [0] * len(emotion_types)
        emotion_onehot[emotion_idx] = 1
        
        # Combine features
        features = emotion_onehot + [intensity, valence, arousal, self.developmental_level]
        
        # Pad to input dimension if needed
        while len(features) < self.emotion_network.input_dim:
            features.append(0.0)
            
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _create_impulse_vector(self, impulse_data: Dict[str, Any]) -> torch.Tensor:
        """
        Create a vector representation of an impulse
        
        Args:
            impulse_data: Dictionary with impulse information
            
        Returns:
            Tensor representation of impulse
        """
        # Extract impulse features
        impulse_type = impulse_data.get("impulse_type", "approach")
        strength = impulse_data.get("strength", 0.5)
        valence = impulse_data.get("valence", 0.0)
        
        # One-hot encode impulse type
        impulse_types = ["approach", "avoidance", "consumption", "expression"]
        impulse_idx = impulse_types.index(impulse_type) if impulse_type in impulse_types else 0
        impulse_onehot = [0] * len(impulse_types)
        impulse_onehot[impulse_idx] = 1
        
        # Combine features
        features = impulse_onehot + [strength, valence, self.developmental_level]
        
        # Pad to input dimension if needed
        while len(features) < self.impulse_network.input_dim:
            features.append(0.0)
            
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
    
    def _create_state_vector(self, state_data: Optional[Dict[str, Any]]) -> torch.Tensor:
        """
        Create a vector representation of a state
        
        Args:
            state_data: Dictionary with state information
            
        Returns:
            Tensor representation of state
        """
        if state_data is None:
            # Return zero vector for null state
            return torch.zeros(self.monitoring_network.input_dim, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Extract features based on what's available
        features = []
        
        # Add emotion features if present
        if "emotion" in state_data:
            emotion = state_data["emotion"]
            features.extend([
                emotion.get("intensity", 0.5),
                emotion.get("valence", 0.0),
                emotion.get("arousal", 0.5)
            ])
        else:
            features.extend([0.0, 0.0, 0.0])
            
        # Add behavior features if present
        if "behavior" in state_data:
            behavior = state_data["behavior"]
            features.extend([
                behavior.get("activation", 0.5),
                behavior.get("impulsiveness", 0.0)
            ])
        else:
            features.extend([0.0, 0.0])
            
        # Add goal features if present
        if "goal" in state_data:
            goal = state_data["goal"]
            features.extend([
                goal.get("progress", 0.0),
                goal.get("importance", 0.5)
            ])
        else:
            features.extend([0.0, 0.0])
        
        # Add developmental level
        features.append(self.developmental_level)
        
        # Add general state values
        general_keys = ["activation", "coherence", "stability", "control"]
        for key in general_keys:
            features.append(state_data.get(key, 0.0))
            
        # Pad to input dimension if needed
        while len(features) < self.monitoring_network.input_dim:
            features.append(0.0)
            
        # Convert to tensor
        return torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)
