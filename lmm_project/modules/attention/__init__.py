"""
Attention Module

This module handles the focusing of cognitive resources on specific 
aspects of perception, memory, and thought. It determines what information
is prioritized and brought into working memory for further processing.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Tuple
import time
import uuid
from datetime import datetime
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message

logger = logging.getLogger(__name__)

def get_module(
    module_id: str = "attention",
    event_bus: Optional[EventBus] = None,
    development_level: float = 0.0
) -> "AttentionSystem":
    """
    Factory function to create and return an attention module
    
    This function initializes and returns a complete attention system with
    focus control and salience detection capabilities.
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication
        development_level: Initial developmental level for the system
        
    Returns:
        Initialized AttentionSystem
    """
    return AttentionSystem(
        module_id=module_id,
        event_bus=event_bus,
        development_level=development_level
    )

class AttentionSystem(BaseModule):
    """
    Attention system responsible for directing cognitive focus
    
    The attention system develops from basic attention capture by stimulus
    intensity to sophisticated volitional control of attention and multitasking.
    """
    # Development milestones
    development_milestones = {
        0.0: "Basic attention capture",
        0.2: "Sustained attention",
        0.4: "Selective attention",
        0.6: "Divided attention",
        0.8: "Executive attention control",
        1.0: "Sophisticated attention management"
    }
    
    def __init__(
        self,
        module_id: str,
        event_bus: Optional[EventBus] = None,
        development_level: float = 0.0
    ):
        """
        Initialize the attention system
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication
            development_level: Initial developmental level
        """
        super().__init__(
            module_id=module_id,
            module_type="attention_system",
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Current focus of attention
        self.current_focus = None
        
        # Attention history
        self.focus_history = deque(maxlen=20)
        
        # Attentional control parameters
        self._attention_params = {
            "intensity_weight": 0.8,    # Weight for stimulus intensity
            "novelty_weight": 0.6,      # Weight for stimulus novelty
            "relevance_weight": 0.3,    # Weight for task relevance
            "volitional_weight": 0.2,   # Weight for intentional control
            "sustained_decay": 0.1,     # How quickly sustained attention decays
            "distraction_threshold": 0.7, # Threshold for attention capture
        }
        
        # Attention capacity - increases with development
        self._capacity = 1
        
        # Active focuses (for divided attention)
        self._active_focuses = []
        
        # Task context (what we're trying to focus on)
        self._task_context = {}
        
        # Adjust parameters based on development level
        self._adjust_parameters_for_development()
        
        # Subscribe to relevant events
        if self.event_bus:
            self.subscribe_to_message("perception_result")
            self.subscribe_to_message("attention_request")
            self.subscribe_to_message("attention_query")
    
    def _adjust_parameters_for_development(self):
        """Adjust attention parameters based on developmental level"""
        # Attention capacity grows with development
        self._capacity = max(1, int(1 + self.development_level * 3))
        
        if self.development_level < 0.2:
            # Early development - mainly stimulus-driven
            self._attention_params.update({
                "intensity_weight": 0.9,
                "novelty_weight": 0.7,
                "relevance_weight": 0.1,
                "volitional_weight": 0.0,
                "sustained_decay": 0.3,
                "distraction_threshold": 0.3,
            })
        elif self.development_level < 0.4:
            # Developing sustained attention
            self._attention_params.update({
                "intensity_weight": 0.8,
                "novelty_weight": 0.7,
                "relevance_weight": 0.3,
                "volitional_weight": 0.1,
                "sustained_decay": 0.2,
                "distraction_threshold": 0.4,
            })
        elif self.development_level < 0.6:
            # Developing selective attention
            self._attention_params.update({
                "intensity_weight": 0.7,
                "novelty_weight": 0.6,
                "relevance_weight": 0.5,
                "volitional_weight": 0.3,
                "sustained_decay": 0.15,
                "distraction_threshold": 0.5,
            })
        elif self.development_level < 0.8:
            # Developing divided attention
            self._attention_params.update({
                "intensity_weight": 0.6,
                "novelty_weight": 0.5,
                "relevance_weight": 0.7,
                "volitional_weight": 0.5,
                "sustained_decay": 0.1,
                "distraction_threshold": 0.6,
            })
        else:
            # Advanced executive attention
            self._attention_params.update({
                "intensity_weight": 0.4,
                "novelty_weight": 0.4,
                "relevance_weight": 0.8,
                "volitional_weight": 0.8,
                "sustained_decay": 0.05,
                "distraction_threshold": 0.8,
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to determine attentional focus
        
        Args:
            input_data: Data to evaluate for attention
                Required keys: 'content', 'source'
                Optional keys: 'intensity', 'novelty', 'relevance'
                
        Returns:
            Dictionary with attention results
        """
        # Generate ID for this attention process
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Ensure timestamp is a float
        if "timestamp" in input_data:
            if isinstance(input_data["timestamp"], datetime):
                timestamp = input_data["timestamp"].timestamp()
            else:
                timestamp = float(input_data["timestamp"])
        else:
            timestamp = time.time()
        
        # Extract key attention parameters or use defaults
        intensity = input_data.get("intensity", 0.5)
        novelty = input_data.get("novelty", 0.5)
        relevance = input_data.get("relevance", 0.5)
        volitional = input_data.get("volitional", False)
        
        # Calculate salience score
        salience = self._calculate_salience(intensity, novelty, relevance, volitional)
        
        # Determine if this input captures attention based on
        # development level and current focus
        captures_attention = self._evaluate_attention_capture(salience, input_data)
        
        # Create focus object if attention is captured
        if captures_attention:
            focus = {
                "focus_id": f"focus_{uuid.uuid4().hex[:8]}",
                "content": input_data.get("content", {}),
                "source": input_data.get("source", "unknown"),
                "salience": salience,
                "timestamp": timestamp,
                "process_id": process_id,
                "sustained_until": timestamp + (10 * (1 - self._attention_params["sustained_decay"]))
            }
            
            # Update current focus - behavior depends on development level
            self._update_focus(focus)
            
            # Record in history
            self.focus_history.append(focus)
        
        # Prepare result
        result = {
            "process_id": process_id,
            "timestamp": timestamp,
            "development_level": self.development_level,
            "module_id": self.module_id,
            "captures_attention": captures_attention,
            "salience": salience,
            "current_focus": self.current_focus,
            "capacity": self._capacity,
            "active_focuses": len(self._active_focuses)
        }
        
        # Add developmental-appropriate additional information
        if self.development_level >= 0.4:
            # Add attention components at higher development levels
            result["attention_components"] = {
                "intensity_contribution": intensity * self._attention_params["intensity_weight"],
                "novelty_contribution": novelty * self._attention_params["novelty_weight"],
                "relevance_contribution": relevance * self._attention_params["relevance_weight"],
                "volitional_control": volitional * self._attention_params["volitional_weight"]
            }
            
        if self.development_level >= 0.6:
            # Add information about divided attention capabilities
            result["divided_attention"] = {
                "capacity": self._capacity,
                "active_focuses": [f["focus_id"] for f in self._active_focuses],
                "capacity_available": self._capacity - len(self._active_focuses)
            }
        
        # Publish attention result
        if self.event_bus:
            self.publish_message(
                "attention_focus_update",
                {"result": result, "process_id": process_id}
            )
            
        return result
    
    def _calculate_salience(
        self, 
        intensity: float,
        novelty: float,
        relevance: float,
        volitional: bool
    ) -> float:
        """
        Calculate the salience score of an input
        
        Args:
            intensity: Intensity of the stimulus (0-1)
            novelty: Novelty of the stimulus (0-1)
            relevance: Task relevance of the stimulus (0-1)
            volitional: Whether this is a deliberate attention shift
            
        Returns:
            Salience score (0-1)
        """
        # Weighted combination of factors
        salience = (
            intensity * self._attention_params["intensity_weight"] +
            novelty * self._attention_params["novelty_weight"] +
            relevance * self._attention_params["relevance_weight"]
        )
        
        # Add volitional control if developed enough
        if volitional and self._attention_params["volitional_weight"] > 0:
            salience += self._attention_params["volitional_weight"]
            
        # Normalize to 0-1 range
        salience = min(1.0, salience / (
            self._attention_params["intensity_weight"] + 
            self._attention_params["novelty_weight"] + 
            self._attention_params["relevance_weight"] +
            (self._attention_params["volitional_weight"] if volitional else 0)
        ))
        
        return salience
    
    def _evaluate_attention_capture(self, salience: float, input_data: Dict[str, Any]) -> bool:
        """
        Determine if an input captures attention
        
        Args:
            salience: Calculated salience score
            input_data: Input data
            
        Returns:
            Whether attention is captured
        """
        # Very early development - attention easily captured
        if self.development_level < 0.2:
            return salience > 0.3
            
        # Check current attention state
        if not self.current_focus:
            # No current focus - easier to capture
            return salience > 0.4
            
        # Volitional control overrides at higher development levels
        if (self.development_level >= 0.6 and 
            input_data.get("volitional", False) and 
            self._attention_params["volitional_weight"] > 0.4):
            return True
            
        # Check against distraction threshold - threshold increases with development
        distraction_threshold = self._attention_params["distraction_threshold"]
        
        # If we have capacity for divided attention, the threshold is lower
        if self.development_level >= 0.6 and len(self._active_focuses) < self._capacity:
            distraction_threshold *= 0.7
            
        return salience > distraction_threshold
    
    def _update_focus(self, new_focus: Dict[str, Any]):
        """
        Update the current focus of attention
        
        This behaves differently depending on developmental level:
        - Early: Single focus, easily displaced
        - Middle: More stable single focus
        - Advanced: Potential for multiple simultaneous focuses
        
        Args:
            new_focus: New focus object
        """
        # Early development - simply replace current focus
        if self.development_level < 0.6:
            self.current_focus = new_focus
            self._active_focuses = [new_focus]
            return
            
        # Divided attention capability
        if len(self._active_focuses) < self._capacity:
            # We have capacity for another focus
            self._active_focuses.append(new_focus)
            # Most salient becomes current focus
            self._active_focuses.sort(key=lambda x: x["salience"], reverse=True)
            self.current_focus = self._active_focuses[0]
        else:
            # Replace least salient focus if new one is more salient
            self._active_focuses.sort(key=lambda x: x["salience"])
            if new_focus["salience"] > self._active_focuses[0]["salience"]:
                self._active_focuses[0] = new_focus
                # Resort and update current
                self._active_focuses.sort(key=lambda x: x["salience"], reverse=True)
                self.current_focus = self._active_focuses[0]
    
    def set_task_context(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set the current task context to guide attention
        
        This allows task-relevant stimuli to be prioritized
        
        Args:
            task_data: Information about current task
            
        Returns:
            Updated task context
        """
        self._task_context = task_data
        
        # Publish task context update
        if self.event_bus:
            self.publish_message(
                "attention_context_update",
                {"task_context": task_data}
            )
            
        return self._task_context
    
    def _handle_message(self, message: Message):
        """Handle incoming messages"""
        if message.message_type == "perception_result":
            # Process perception results for potential attention
            if message.content and "result" in message.content:
                result = message.content["result"]
                
                # Extract attention-relevant info
                attention_input = {
                    "content": result,
                    "source": "perception",
                    "process_id": result.get("process_id", message.id),
                    "timestamp": message.timestamp
                }
                
                # Try to extract salience factors
                if "patterns" in result:
                    patterns = result["patterns"]
                    # More patterns indicates higher intensity
                    attention_input["intensity"] = min(1.0, len(patterns) / 10)
                    
                    # Check for certain pattern types that suggest novelty
                    novel_patterns = ["complex_text", "question", "exclamation"]
                    if any(p["pattern_type"] in novel_patterns for p in patterns):
                        attention_input["novelty"] = 0.8
                    
                # Task relevance - if we have task context, check for relevance
                if self._task_context and "keywords" in self._task_context:
                    # Simple keyword matching for relevance
                    text = result.get("text", "")
                    keywords = self._task_context["keywords"]
                    matches = sum(1 for kw in keywords if kw.lower() in text.lower())
                    attention_input["relevance"] = min(1.0, matches / len(keywords)) if keywords else 0.5
                
                # Process for attention
                self.process_input(attention_input)
                
        elif message.message_type == "attention_request":
            # Direct request for attention
            if message.content:
                # Mark as volitional
                message.content["volitional"] = True
                self.process_input(message.content)
                
        elif message.message_type == "attention_query":
            # Handle query about attention state
            self._handle_attention_query(message)
    
    def _handle_attention_query(self, message: Message):
        """Handle queries about attention state"""
        query_type = message.content.get("query_type")
        query_id = message.content.get("query_id", str(uuid.uuid4()))
        
        response = {
            "query_id": query_id,
            "query_type": query_type,
            "module_id": self.module_id
        }
        
        if query_type == "current_focus":
            # Return current focus of attention
            response["current_focus"] = self.current_focus
            
        elif query_type == "focus_history":
            # Return recent focus history
            count = message.content.get("count", 5)
            response["focus_history"] = list(self.focus_history)[-count:]
            
        elif query_type == "capacity":
            # Return attention capacity information
            response["capacity"] = {
                "total_capacity": self._capacity,
                "used_capacity": len(self._active_focuses),
                "available_capacity": self._capacity - len(self._active_focuses),
                "development_level": self.development_level
            }
            
        # Publish response
        if self.event_bus:
            self.publish_message(
                "attention_query_response",
                response
            )
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update development level
        prev_level = self.development_level
        new_level = super().update_development(amount)
        
        # If development level changed significantly, adjust parameters
        if int(prev_level * 10) != int(new_level * 10):
            logger.info(f"Attention system upgraded to development level {new_level:.1f}")
            self._adjust_parameters_for_development()
            
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add attention-specific state
        state.update({
            "current_focus": self.current_focus,
            "focus_history_size": len(self.focus_history),
            "active_focuses": len(self._active_focuses),
            "capacity": self._capacity,
            "attention_parameters": self._attention_params
        })
        
        return state
        
    def get_current_focus(self) -> Dict[str, Any]:
        """Get the current focus of attention"""
        return self.current_focus
        
    def get_focus_history(self, count: int = 5) -> List[Dict[str, Any]]:
        """Get the recent focus history"""
        return list(self.focus_history)[-count:]