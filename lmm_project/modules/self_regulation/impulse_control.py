# TODO: Implement the ImpulseControl class to inhibit inappropriate impulses
# This component should be able to:
# - Detect impulses requiring inhibition
# - Delay immediate responses when appropriate
# - Redirect action tendencies toward appropriate alternatives
# - Regulate behavior to align with goals and values

# TODO: Implement developmental progression in impulse control:
# - Minimal impulse control in early stages
# - Growing ability to delay gratification in childhood
# - Increased self-restraint in adolescence
# - Sophisticated impulse regulation in adulthood

# TODO: Create mechanisms for:
# - Impulse detection: Identify action tendencies requiring control
# - Response inhibition: Suppress inappropriate impulses
# - Delay capacity: Wait for appropriate timing
# - Alternative generation: Redirect energy to better options

# TODO: Implement different control strategies:
# - Proactive inhibition: Prepare to suppress responses before triggers
# - Reactive inhibition: Suppress responses after triggers
# - Attentional control: Direct attention away from temptations
# - Implementation intentions: Plan specific responses to challenges

# TODO: Connect to executive function and consciousness modules
# Impulse control should utilize executive inhibition
# and be informed by conscious goals and priorities

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import numpy as np
import uuid
import logging

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus, Message
from lmm_project.modules.self_regulation.models import ImpulseEvent
from lmm_project.modules.self_regulation.neural_net import ImpulseControlNetwork

class ImpulseControl(BaseModule):
    """
    Handles impulse control functionality
    
    This module implements mechanisms for detecting and managing impulses,
    developing from basic awareness to sophisticated control strategies.
    """
    
    # Developmental milestones for impulse control
    development_milestones = {
        0.1: "Basic impulse awareness",
        0.2: "Simple delay of immediate urges",
        0.3: "Basic inhibition of motor responses",
        0.4: "Impulse modulation in familiar contexts",
        0.5: "Delayed gratification for moderate periods",
        0.6: "Cognitive strategies for impulse control",
        0.7: "Context-appropriate impulse inhibition",
        0.8: "Advanced response selection and inhibition",
        0.9: "Integrated control across multiple domains",
        1.0: "Sophisticated, flexible impulse management"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the impulse control module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial development level (0.0 to 1.0)
        """
        super().__init__(
            module_id=module_id,
            module_type="impulse_control",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.logger = logging.getLogger(f"lmm.self_regulation.impulse_control.{module_id}")
        
        # Initialize neural network
        self.network = ImpulseControlNetwork(
            developmental_level=development_level
        )
        
        # Recent impulses
        self.recent_impulses: List[ImpulseEvent] = []
        self.max_impulse_history = 20
        
        # Control history
        self.control_attempts = 0
        self.successful_controls = 0
        
        # Current impulse being processed
        self.current_impulse: Optional[ImpulseEvent] = None
        
        # Delay capability (increases with development)
        self.max_delay_seconds = self._calculate_max_delay()
        
        # Inhibition threshold (decreases with development)
        self.inhibition_threshold = self._calculate_inhibition_threshold()
        
        # Subscribe to relevant messages
        if event_bus:
            self.subscribe_to_message("impulse_event")
            self.subscribe_to_message("control_request")
            self.subscribe_to_message("impulse_query")
        
        self.logger.info(f"Impulse control module initialized at development level {development_level:.2f}")
    
    def _calculate_max_delay(self) -> float:
        """Calculate maximum delay capability based on development level"""
        # Starts at 3 seconds, increases to 5 minutes (300 seconds)
        base_delay = 3.0
        max_delay = 300.0
        return base_delay + (max_delay - base_delay) * self.development_level
        
    def _calculate_inhibition_threshold(self) -> float:
        """Calculate impulse inhibition threshold based on development level"""
        # Starts high (0.8), decreases to low (0.2) with development
        max_threshold = 0.8
        min_threshold = 0.2
        return max_threshold - (max_threshold - min_threshold) * self.development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to control impulses
        
        Args:
            input_data: Dictionary with input data
                Required keys depend on input type:
                - "type": Type of input ("impulse", "control_request", "query")
                - For impulse input: "impulse_type", "strength", etc.
                - For control request: "impulse_id"
                - For query: "query_type"
            
        Returns:
            Dictionary with process results
        """
        input_type = input_data.get("type", "unknown")
        self.logger.debug(f"Processing {input_type} input")
        
        result = {
            "success": False,
            "message": f"Unknown input type: {input_type}"
        }
        
        # Process different input types
        if input_type == "impulse":
            result = self._process_impulse(input_data)
            
        elif input_type == "control_request":
            result = self._process_control_request(input_data)
            
        elif input_type == "query":
            result = self._process_query(input_data)
            
        return result
    
    def _process_impulse(self, impulse_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process an impulse event"""
        # Create an impulse event from the data
        try:
            # Extract required fields
            impulse_type = impulse_data.get("impulse_type")
            if not impulse_type:
                return {"success": False, "message": "Missing impulse_type"}
                
            strength = impulse_data.get("strength", 0.5)
            urgency = impulse_data.get("urgency", 0.5)
            
            # Create the impulse event
            impulse = ImpulseEvent(
                impulse_type=impulse_type,
                strength=strength,
                urgency=urgency,
                trigger=impulse_data.get("trigger"),
                context=impulse_data.get("context", {})
            )
            
            # Add to recent impulses
            self.recent_impulses.append(impulse)
            if len(self.recent_impulses) > self.max_impulse_history:
                self.recent_impulses = self.recent_impulses[-self.max_impulse_history:]
                
            # Set as current impulse
            self.current_impulse = impulse
            
            # Automatically control if strength is high enough
            should_control = strength > self.inhibition_threshold or urgency > 0.7
            
            if should_control and self.development_level >= 0.2:
                control_result = self._control_impulse(impulse)
                return {
                    "success": True,
                    "impulse_id": impulse.id,
                    "impulse": impulse.dict(),
                    "was_controlled": True,
                    "control_result": control_result
                }
            else:
                return {
                    "success": True,
                    "impulse_id": impulse.id,
                    "impulse": impulse.dict(),
                    "was_controlled": False
                }
                
        except Exception as e:
            self.logger.error(f"Error processing impulse: {e}")
            return {"success": False, "message": f"Error processing impulse: {str(e)}"}
    
    def _process_control_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a request to control an impulse"""
        impulse_id = request_data.get("impulse_id")
        
        # Find the impulse
        target_impulse = None
        for impulse in self.recent_impulses:
            if impulse.id == impulse_id:
                target_impulse = impulse
                break
                
        if not target_impulse:
            return {
                "success": False,
                "message": f"Impulse with ID {impulse_id} not found"
            }
            
        # Control the impulse
        control_result = self._control_impulse(target_impulse)
        
        return {
            "success": True,
            "impulse_id": impulse_id,
            "control_result": control_result
        }
    
    def _process_query(self, query_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a query about impulse control"""
        query_type = query_data.get("query_type", "current")
        
        if query_type == "current":
            if self.current_impulse:
                return {
                    "success": True,
                    "current_impulse": self.current_impulse.dict()
                }
            else:
                return {
                    "success": True,
                    "current_impulse": None
                }
                
        elif query_type == "recent":
            limit = query_data.get("limit", 5)
            recent = self.recent_impulses[-limit:] if self.recent_impulses else []
            return {
                "success": True,
                "recent_impulses": [i.dict() for i in recent]
            }
            
        elif query_type == "capabilities":
            # Return control capabilities based on development
            return {
                "success": True,
                "max_delay_seconds": self.max_delay_seconds,
                "inhibition_threshold": self.inhibition_threshold,
                "development_level": self.development_level
            }
            
        elif query_type == "stats":
            # Return control statistics
            success_rate = 0.0
            if self.control_attempts > 0:
                success_rate = self.successful_controls / self.control_attempts
                
            return {
                "success": True,
                "control_attempts": self.control_attempts,
                "successful_controls": self.successful_controls,
                "success_rate": success_rate,
                "development_level": self.development_level
            }
            
        else:
            return {
                "success": False,
                "message": f"Unknown query type: {query_type}"
            }
    
    def _control_impulse(self, impulse: ImpulseEvent) -> Dict[str, Any]:
        """
        Apply control to an impulse
        
        Args:
            impulse: The impulse event to control
            
        Returns:
            Dictionary with control results
        """
        # Increment control attempt counter
        self.control_attempts += 1
        
        # Convert impulse to format expected by neural network
        impulse_data = {
            "impulse_type": impulse.impulse_type,
            "strength": impulse.strength,
            "urgency": impulse.urgency
        }
        
        # Get control strategy from neural network
        try:
            # Prepare impulse vector for neural network
            impulse_vector = self._create_impulse_vector(impulse_data)
            
            # Evaluate control ability with neural network
            can_control, inhibition_strength = self.network.evaluate_control(impulse_vector)
            
            # Determine control success
            # Higher strength impulses are harder to control
            control_difficulty = impulse.strength * (1.0 + impulse.urgency * 0.5)
            
            # Success is more likely with higher development and lower difficulty
            control_success = inhibition_strength > control_difficulty
            
            # Calculate actual success probability
            success_probability = inhibition_strength / max(control_difficulty, 0.1)
            success_probability = min(0.95, max(0.05, success_probability))
            
            # Apply some randomness to the outcome
            # Early development levels have more randomness
            randomness_factor = max(0.1, 0.5 - self.development_level * 0.4)
            adjusted_success_prob = success_probability * (1.0 - randomness_factor)
            
            # Determine actual success
            actual_success = np.random.random() < adjusted_success_prob
            
            # Calculate delay time if control was successful
            delay_time = 0.0
            if actual_success:
                # Higher development allows longer delays
                max_possible_delay = self.max_delay_seconds
                # Stronger impulses are harder to delay for long
                delay_reduction_factor = impulse.strength * impulse.urgency
                delay_time = max_possible_delay * (1.0 - delay_reduction_factor * 0.8)
                delay_time = max(1.0, delay_time)
                
                # Mark as controlled
                impulse.is_controlled = True
                impulse.control_strategy = "inhibition" if inhibition_strength > 0.7 else "delay"
                impulse.control_success = True
                
                # Update success counter
                self.successful_controls += 1
            else:
                # Failed to control
                impulse.is_controlled = False
                impulse.control_strategy = "failed_attempt"
                impulse.control_success = False
            
            # Return control results
            result = {
                "success": True,
                "control_successful": actual_success,
                "inhibition_strength": float(inhibition_strength),
                "control_difficulty": float(control_difficulty),
                "success_probability": float(success_probability),
                "delay_time_seconds": float(delay_time) if actual_success else 0.0
            }
            
            # Publish control event
            if self.event_bus:
                self.publish_message("impulse_controlled", {
                    "impulse": impulse.dict(),
                    "control_result": result
                })
                
            return result
            
        except Exception as e:
            self.logger.error(f"Error in impulse control: {e}")
            return {
                "success": False,
                "message": f"Control error: {str(e)}"
            }
    
    def _create_impulse_vector(self, impulse_data: Dict[str, Any]) -> np.ndarray:
        """
        Create a vector representation of an impulse
        
        Args:
            impulse_data: Dictionary with impulse data
            
        Returns:
            Numpy array representation of the impulse
        """
        # Extract impulse attributes
        impulse_type = impulse_data.get("impulse_type", "neutral")
        strength = impulse_data.get("strength", 0.5)
        urgency = impulse_data.get("urgency", 0.5)
        
        # One-hot encode impulse type (simplified)
        impulse_types = ["approach", "avoidance", "consumption", "action", "social", "exploration", "other"]
        impulse_idx = impulse_types.index(impulse_type) if impulse_type in impulse_types else 6
        impulse_onehot = [0] * len(impulse_types)
        impulse_onehot[impulse_idx] = 1
        
        # Combine features
        features = impulse_onehot + [strength, urgency, self.development_level]
        
        # Convert to numpy array
        return np.array(features, dtype=np.float32)
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        message_type = message.message_type
        content = message.content
        
        if message_type == "impulse_event":
            # Process incoming impulse event
            result = self.process_input({
                "type": "impulse",
                **content
            })
            
            # Send response if requested
            if message.reply_to and self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="impulse_event_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "control_request":
            # Process control request
            result = self.process_input({
                "type": "control_request",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="control_response",
                    content=result,
                    reply_to=message.id
                ))
                
        elif message_type == "impulse_query":
            # Process query
            result = self.process_input({
                "type": "query",
                **content
            })
            
            # Send response
            if self.event_bus:
                self.event_bus.publish(Message(
                    sender=self.module_id,
                    recipient=message.sender,
                    message_type="impulse_query_response",
                    content=result,
                    reply_to=message.id
                ))
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        super().update_development(amount)
        
        # Update neural network development
        self.network.update_developmental_level(self.development_level)
        
        # Update development-dependent parameters
        self.max_delay_seconds = self._calculate_max_delay()
        self.inhibition_threshold = self._calculate_inhibition_threshold()
        
        # Check for developmental milestones
        self._check_development_milestones(old_level)
        
        self.logger.info(f"Updated impulse control development to {self.development_level:.2f}")
        return self.development_level
    
    def _check_development_milestones(self, previous_level: float) -> None:
        """
        Check if any developmental milestones have been reached
        
        Args:
            previous_level: The previous development level
        """
        # Check each milestone to see if we've crossed the threshold
        for level, description in self.development_milestones.items():
            # If we've crossed a milestone threshold
            if previous_level < level <= self.development_level:
                self.logger.info(f"Impulse control milestone reached at {level:.1f}: {description}")
                
                # Adjust control strategies based on the new milestone
                if level == 0.1:
                    self.logger.info("Now capable of basic impulse awareness")
                elif level == 0.2:
                    self.logger.info("Now capable of simple delay of immediate urges")
                elif level == 0.3:
                    self.logger.info("Now capable of basic inhibition of motor responses")
                elif level == 0.4:
                    self.logger.info("Now capable of impulse modulation in familiar contexts")
                elif level == 0.5:
                    self.logger.info("Now capable of delayed gratification for moderate periods")
                elif level == 0.6:
                    self.logger.info("Now capable of cognitive strategies for impulse control")
                elif level == 0.7:
                    self.logger.info("Now capable of context-appropriate impulse inhibition")
                elif level == 0.8:
                    self.logger.info("Now capable of advanced response selection and inhibition")
                elif level == 0.9:
                    self.logger.info("Now capable of integrated control across multiple domains")
                elif level == 1.0:
                    self.logger.info("Now capable of sophisticated, flexible impulse management")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary with the module state
        """
        base_state = super().get_state()
        
        # Add impulse control specific state
        control_state = {
            "current_impulse": self.current_impulse.dict() if self.current_impulse else None,
            "recent_impulses_count": len(self.recent_impulses),
            "control_attempts": self.control_attempts,
            "successful_controls": self.successful_controls,
            "max_delay_seconds": self.max_delay_seconds,
            "inhibition_threshold": self.inhibition_threshold
        }
        
        return {**base_state, **control_state}
    
    def save_state(self, state_dir: str) -> str:
        """
        Save the module state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Create module state directory
        module_dir = os.path.join(state_dir, self.module_type, self.module_id)
        os.makedirs(module_dir, exist_ok=True)
        
        # Save basic module state
        state_path = os.path.join(module_dir, "module_state.json")
        with open(state_path, 'w') as f:
            # Get state with serializable impulse data
            state = self.get_state()
            state["recent_impulses"] = [i.dict() for i in self.recent_impulses[-10:]] # Save last 10
            json.dump(state, f, indent=2, default=str)
            
        self.logger.info(f"Saved impulse control state to {module_dir}")
        return state_path
    
    def load_state(self, state_path: str) -> bool:
        """
        Load the module state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            with open(state_path, 'r') as f:
                state = json.load(f)
                
            # Update base state
            self.development_level = state.get("development_level", 0.0)
            self.achieved_milestones = set(state.get("achieved_milestones", []))
            
            # Update impulse control specific state
            self.control_attempts = state.get("control_attempts", 0)
            self.successful_controls = state.get("successful_controls", 0)
            self.max_delay_seconds = state.get("max_delay_seconds", self._calculate_max_delay())
            self.inhibition_threshold = state.get("inhibition_threshold", self._calculate_inhibition_threshold())
            
            # Recreate current impulse
            from lmm_project.modules.self_regulation.models import ImpulseEvent
            current_impulse_data = state.get("current_impulse")
            if current_impulse_data:
                self.current_impulse = ImpulseEvent(**current_impulse_data)
                
            # Recreate recent impulses
            self.recent_impulses = []
            for impulse_data in state.get("recent_impulses", []):
                try:
                    self.recent_impulses.append(ImpulseEvent(**impulse_data))
                except Exception as e:
                    self.logger.warning(f"Could not recreate impulse: {e}")
            
            # Update neural network
            self.network.update_developmental_level(self.development_level)
            
            self.logger.info(f"Loaded impulse control state from {state_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading state: {e}")
            return False
