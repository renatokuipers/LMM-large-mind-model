"""
Learning Module

This module is responsible for different learning mechanisms,
knowledge acquisition, and skill development. It integrates
multiple learning approaches to enable the mind to learn from
experiences and adapt its behavior.
"""

import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

from lmm_project.modules.learning.associative_learning import AssociativeLearning
from lmm_project.modules.learning.reinforcement_learning import ReinforcementLearning
from lmm_project.modules.learning.procedural_learning import ProceduralLearning
from lmm_project.modules.learning.meta_learning import MetaLearning

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "learning",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "LearningSystem":
    """
    Factory function to create a learning module
    
    This function is responsible for creating a learning system that can:
    - Acquire new knowledge through various learning mechanisms
    - Develop skills through practice and experience
    - Adapt learning strategies based on context and results
    - Integrate different types of learning for optimal knowledge acquisition
    - Monitor and regulate the learning process itself
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level for the system
        
    Returns:
        An instance of the LearningSystem class
    """
    return LearningSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class LearningSystem(BaseModule):
    """
    Integrated learning system with multiple learning mechanisms
    
    The learning system develops from simple associative learning in early stages
    to complex integrated learning approaches in later stages.
    """
    
    # Development milestones for learning
    development_milestones = {
        0.0: "Basic associative learning",
        0.2: "Simple reinforcement learning",
        0.4: "Procedural skill acquisition",
        0.6: "Multi-modal learning integration",
        0.8: "Strategic learning optimization",
        1.0: "Advanced meta-learning capabilities"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the learning system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="learning",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Create learning sub-modules
        self.associative_learning = AssociativeLearning(
            module_id=f"{module_id}_associative",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.reinforcement_learning = ReinforcementLearning(
            module_id=f"{module_id}_reinforcement",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.procedural_learning = ProceduralLearning(
            module_id=f"{module_id}_procedural",
            event_bus=event_bus,
            development_level=development_level
        )
        
        self.meta_learning = MetaLearning(
            module_id=f"{module_id}_meta",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Track learning events and outcomes
        self.learning_events = []
        self.max_events = 100
        
        # Map learning operations to their handlers
        self.operation_handlers = {
            "associative": self._handle_associative_learning,
            "reinforcement": self._handle_reinforcement_learning,
            "procedural": self._handle_procedural_learning,
            "meta": self._handle_meta_learning,
            "integrate": self._handle_integrated_learning
        }
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("perception_input", self._handle_perception)
            self.subscribe_to_message("memory_retrieval", self._handle_memory_retrieval)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for learning
        
        Args:
            input_data: Dictionary containing learning parameters and data
            
        Returns:
            Dictionary with learning results
        """
        # Extract learning type
        learning_type = input_data.get("learning_type", "associative")
        
        # Get appropriate handler for this learning type
        handler = self.operation_handlers.get(learning_type)
        
        if not handler:
            return {
                "status": "error",
                "message": f"Unknown learning type: {learning_type}",
                "module_id": self.module_id
            }
        
        # Process with the appropriate learning mechanism
        result = handler(input_data)
        
        # Record learning event if successful
        if result.get("status") == "success":
            self._record_learning_event(learning_type, input_data, result)
        
        return result
    
    def _handle_associative_learning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle associative learning operations"""
        result = self.associative_learning.process_input(input_data)
        return result
    
    def _handle_reinforcement_learning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle reinforcement learning operations"""
        result = self.reinforcement_learning.process_input(input_data)
        return result
    
    def _handle_procedural_learning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle procedural learning operations"""
        result = self.procedural_learning.process_input(input_data)
        return result
    
    def _handle_meta_learning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle meta-learning operations"""
        result = self.meta_learning.process_input(input_data)
        return result
    
    def _handle_integrated_learning(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle integrated learning that combines multiple approaches
        
        This function becomes more effective at higher developmental levels
        """
        # At lower developmental levels, just use the primary learning type
        if self.development_level < 0.5:
            primary_type = input_data.get("primary_type", "associative")
            handler = self.operation_handlers.get(primary_type)
            
            if not handler:
                return {
                    "status": "error",
                    "message": f"Unknown primary learning type: {primary_type}",
                    "module_id": self.module_id
                }
                
            return handler(input_data)
        
        # At higher developmental levels, integrate multiple learning types
        results = {}
        learning_types = input_data.get("learning_types", ["associative", "reinforcement"])
        
        for learning_type in learning_types:
            handler = self.operation_handlers.get(learning_type)
            if handler and learning_type != "integrate":  # Avoid recursion
                # Create type-specific input by copying and updating
                type_input = input_data.copy()
                type_input["learning_type"] = learning_type
                
                # Process with this learning type
                result = handler(type_input)
                results[learning_type] = result
        
        # Get learning strategy if development is high enough
        learning_strategy = None
        if self.development_level >= 0.7:
            strategy_input = {
                "domain": input_data.get("domain", "general"),
                "content_type": input_data.get("content_type", "general"),
                "operation": "select_strategy"
            }
            strategy_result = self.meta_learning.process_input(strategy_input)
            if strategy_result.get("status") == "success":
                learning_strategy = strategy_result.get("selected_strategy")
        
        return {
            "status": "success",
            "integrated_results": results,
            "learning_strategy": learning_strategy,
            "integration_level": min(1.0, self.development_level * 1.2)  # Higher dev = better integration
        }
    
    def _handle_perception(self, message: Message):
        """Handle perception inputs for learning opportunities"""
        if not message.content:
            return
            
        perception_data = message.content
        
        # Only process if perception has pattern information
        if "pattern" in perception_data:
            pattern = perception_data.get("pattern")
            salience = perception_data.get("salience", 0.5)
            
            # Only learn from salient perceptions
            if salience >= 0.3:
                # Simple associative learning from perception
                if "previous" in perception_data and "pattern" in perception_data["previous"]:
                    previous_pattern = perception_data["previous"]["pattern"]
                    
                    # Learn association between consecutive patterns
                    self.associative_learning.process_input({
                        "operation": "learn",
                        "stimulus": previous_pattern,
                        "response": pattern,
                        "strength": salience,
                        "source": "perception"
                    })
    
    def _handle_memory_retrieval(self, message: Message):
        """Handle memory retrievals for learning enhancement"""
        if not message.content:
            return
            
        memory_data = message.content
        
        # Use retrieved memories to enhance learning
        if "memory" in memory_data and "retrieval_context" in memory_data:
            memory = memory_data["memory"]
            context = memory_data["retrieval_context"]
            
            # If memory retrieval was for learning purposes, update meta-learning
            if "learning" in context or "strategy" in context:
                # Update strategy effectiveness if applicable
                if "strategy_id" in context and "success_level" in memory:
                    self.meta_learning.process_input({
                        "operation": "evaluate_outcome",
                        "strategy_id": context["strategy_id"],
                        "success_level": memory.get("success_level", 0.5),
                        "domain": context.get("domain", "general")
                    })
    
    def _record_learning_event(self, learning_type: str, input_data: Dict[str, Any], result: Dict[str, Any]):
        """Record a learning event for future reference"""
        event = {
            "timestamp": datetime.now(),
            "learning_type": learning_type,
            "content": input_data.get("content", ""),
            "domain": input_data.get("domain", "general"),
            "success": result.get("status") == "success",
            "developmental_level": self.development_level
        }
        
        self.learning_events.append(event)
        
        # Trim history if needed
        if len(self.learning_events) > self.max_events:
            self.learning_events = self.learning_events[-self.max_events:]
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module and all sub-modules
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        previous_level = self.development_level
        new_level = super().update_development(amount)
        
        # Update sub-modules with appropriate amounts
        # Different learning mechanisms develop at slightly different rates
        self.associative_learning.update_development(amount * 1.1)  # Develops slightly faster
        self.reinforcement_learning.update_development(amount * 1.0)
        self.procedural_learning.update_development(amount * 0.9)
        
        # Meta-learning develops more slowly until later stages
        meta_modifier = 0.7 if self.development_level < 0.5 else 1.2
        self.meta_learning.update_development(amount * meta_modifier)
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the learning system"""
        base_state = super().get_state()
        
        # Get state from all sub-modules
        module_state = {
            "associative_learning": self.associative_learning.get_state(),
            "reinforcement_learning": self.reinforcement_learning.get_state(),
            "procedural_learning": self.procedural_learning.get_state(),
            "meta_learning": self.meta_learning.get_state(),
            "learning_events_count": len(self.learning_events),
            "recent_learning_types": [e["learning_type"] for e in self.learning_events[-5:]] if self.learning_events else []
        }
        
        base_state.update(module_state)
        return base_state
        
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state of the learning system
        
        Returns:
            Dictionary containing the serialized state
        """
        # Get base state
        state = self.get_state()
        
        # Add full learning events history (limited to max events)
        state["learning_events"] = self.learning_events
        
        # Save states of all submodules
        state["associative_learning_full"] = self.associative_learning.save_state()
        state["reinforcement_learning_full"] = self.reinforcement_learning.save_state()
        state["procedural_learning_full"] = self.procedural_learning.save_state()
        state["meta_learning_full"] = self.meta_learning.save_state()
        
        # Add operation handlers mapping (just names, not functions)
        state["operations"] = list(self.operation_handlers.keys())
        
        # Add timestamp
        state["saved_at"] = datetime.now().isoformat()
        
        return state
        
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: Dictionary containing the state to load
        """
        # Load development level first
        if "development_level" in state:
            self.development_level = state["development_level"]
            
        # Load learning events history
        if "learning_events" in state:
            self.learning_events = state["learning_events"]
            # Ensure we don't exceed max events
            if len(self.learning_events) > self.max_events:
                self.learning_events = self.learning_events[-self.max_events:]
        
        # Load states for all submodules
        if "associative_learning_full" in state:
            self.associative_learning.load_state(state["associative_learning_full"])
            
        if "reinforcement_learning_full" in state:
            self.reinforcement_learning.load_state(state["reinforcement_learning_full"])
            
        if "procedural_learning_full" in state:
            self.procedural_learning.load_state(state["procedural_learning_full"])
            
        if "meta_learning_full" in state:
            self.meta_learning.load_state(state["meta_learning_full"])
            
        # Re-register event handlers if event bus exists
        if self.event_bus:
            self.subscribe_to_message("perception_input", self._handle_perception)
            self.subscribe_to_message("memory_retrieval", self._handle_memory_retrieval)
            
        logger.info(f"Loaded learning system state with {len(self.learning_events)} learning events")
        
    def subscribe_to_message(self, message_type: str, callback: callable) -> None:
        """
        Subscribe to a message type on the event bus
        
        Args:
            message_type: Type of message to subscribe to
            callback: Function to call when a message is received
        """
        if self.event_bus:
            self.event_bus.subscribe(message_type, callback)
            
    def publish_message(self, message_type: str, content: Any) -> None:
        """
        Publish a message to the event bus
        
        Args:
            message_type: Type of message to publish
            content: Content of the message
        """
        if self.event_bus:
            message = Message(
                sender=self.module_id,
                message_type=message_type,
                content=content
            )
            self.event_bus.publish(message)
            
    def get_learning_stats(self) -> Dict[str, Any]:
        """
        Get statistics about learning progress
        
        Returns:
            Dictionary with learning statistics
        """
        # Count learning events by type
        event_counts = {}
        for event in self.learning_events:
            learning_type = event.get("learning_type", "unknown")
            if learning_type not in event_counts:
                event_counts[learning_type] = 0
            event_counts[learning_type] += 1
            
        # Calculate success rates
        success_rates = {}
        for learning_type, count in event_counts.items():
            successes = sum(1 for e in self.learning_events if e.get("learning_type") == learning_type and e.get("success", False))
            success_rates[learning_type] = successes / count if count > 0 else 0
            
        # Get development levels across components
        development_levels = {
            "associative": self.associative_learning.development_level,
            "reinforcement": self.reinforcement_learning.development_level,
            "procedural": self.procedural_learning.development_level,
            "meta": self.meta_learning.development_level,
            "overall": self.development_level
        }
        
        return {
            "event_counts": event_counts,
            "success_rates": success_rates,
            "development_levels": development_levels,
            "total_events": len(self.learning_events),
            "milestone": self.get_current_milestone()
        }
        
    def get_current_milestone(self) -> str:
        """Get the current developmental milestone for learning"""
        for level in sorted(self.development_milestones.keys(), reverse=True):
            if self.development_level >= level:
                return self.development_milestones[level]
        return self.development_milestones[0.0]
