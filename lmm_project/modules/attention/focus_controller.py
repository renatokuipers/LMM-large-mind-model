import logging
from collections import deque
from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID, uuid4
from datetime import datetime
import time
import numpy as np

from lmm_project.utils.logging_utils import get_module_logger
from lmm_project.core.event_bus import EventBus, Event
from lmm_project.core.message import Message, MessageType, Recipient

from .models import (
    AttentionMode,
    FocusLevel,
    AttentionTarget,
    AttentionFocus,
    AttentionEvent,
    AttentionConfig
)

# Initialize logger
logger = get_module_logger("modules.attention.focus_controller")

class FocusController:
    """
    Controls what the attention system focuses on. Manages the allocation
    of attention resources, maintains focus history, and handles shifts
    of attention based on salience and priority.
    """
    
    def __init__(
        self,
        event_bus: EventBus,
        config: Optional[AttentionConfig] = None,
        developmental_age: float = 0.0
    ):
        """
        Initialize the focus controller.
        
        Args:
            event_bus: The event bus for communication
            config: Configuration for the controller
            developmental_age: Current developmental age of the mind
        """
        self._config = config or AttentionConfig()
        self._event_bus = event_bus
        self._developmental_age = developmental_age
        
        # Current focus of attention
        self._current_focus: Optional[AttentionFocus] = None
        self._focus_start_time = 0
        
        # History of recent focus points
        self._focus_history = deque(maxlen=self._config.focus_history_length)
        
        # Active attention targets
        self._active_targets: Dict[UUID, AttentionTarget] = {}
        
        # Register event handlers
        self._register_event_handlers()
        
        logger.info(f"Focus controller initialized with age {developmental_age}")
    
    def _register_event_handlers(self) -> None:
        """Register handlers for relevant events"""
        self._event_bus.subscribe("salience_assessment_created", self._handle_salience_assessment)
        self._event_bus.subscribe("focus_request", self._handle_focus_request)
        self._event_bus.subscribe("development_age_updated", self._handle_age_update)
        
        # Timer events for attention span management
        self._event_bus.subscribe("timer_tick", self._handle_timer_tick)
    
    def _handle_salience_assessment(self, event: Event) -> None:
        """
        Handle a salience assessment event.
        
        Args:
            event: The event containing salience assessment data
        """
        try:
            # Extract assessment data
            assessment_data = event.data.get("assessment")
            if not assessment_data:
                logger.warning("Received assessment event with no assessment data")
                return
            
            # Check if salience is high enough to potentially shift attention
            salience_score = assessment_data.get("salience_score", 0.0)
            input_id = assessment_data.get("input_id")
            input_type = assessment_data.get("input_type")
            
            if not (input_id and input_type):
                logger.warning("Assessment missing required fields")
                return
            
            # If salience exceeds threshold, consider for attention
            if salience_score >= self._get_current_salience_threshold():
                logger.debug(f"High salience detected: {salience_score} for {input_type} {input_id}")
                
                # Create a potential attention target
                target = AttentionTarget(
                    target_type=input_type,
                    target_id=UUID(input_id) if isinstance(input_id, str) else input_id,
                    description=f"Salient {input_type}",
                    priority=min(1.0, salience_score),  # Use salience as priority
                    relevance_score=salience_score
                )
                
                # Consider shifting attention
                self._consider_attention_shift(target)
        except Exception as e:
            logger.error(f"Error handling salience assessment: {e}")
    
    def _handle_focus_request(self, event: Event) -> None:
        """
        Handle a request to focus attention.
        
        Args:
            event: The event containing focus request data
        """
        try:
            # Extract request data
            request_data = event.data.get("request")
            if not request_data:
                logger.warning("Received focus request with no request data")
                return
            
            target_type = request_data.get("target_type")
            target_id = request_data.get("target_id")
            description = request_data.get("description")
            priority = request_data.get("priority", FocusLevel.MEDIUM)
            mode = request_data.get("mode", AttentionMode.FOCUSED)
            force_focus = request_data.get("force_focus", False)
            
            if not (target_type and target_id and description):
                logger.warning("Focus request missing required fields")
                return
            
            # Convert target_id to UUID if it's a string
            if isinstance(target_id, str):
                target_id = UUID(target_id)
            
            # Create a focus target
            target = AttentionTarget(
                target_type=target_type,
                target_id=target_id,
                description=description,
                priority=priority,
                relevance_score=1.0 if force_focus else 0.8
            )
            
            # Process the focus request
            new_focus = self.focus_attention(
                target_type, target_id, description, priority, mode, force_focus
            )
            
            # Publish response event
            self._event_bus.publish(
                "focus_request_processed",
                {
                    "success": new_focus is not None,
                    "focus": new_focus.dict() if new_focus else None,
                    "request_id": request_data.get("request_id")
                }
            )
        except Exception as e:
            logger.error(f"Error handling focus request: {e}")
    
    def _handle_timer_tick(self, event: Event) -> None:
        """
        Handle timer tick events to manage attention span.
        
        Args:
            event: The timer event
        """
        if not self._current_focus:
            return
        
        # Update focus duration
        current_time = time.time() * 1000  # ms
        duration = int(current_time - self._focus_start_time)
        
        if self._current_focus:
            self._current_focus.duration_ms = duration
        
        # Check if attention span exceeded
        attention_span = self._get_attention_span()
        if duration > attention_span:
            logger.debug(f"Attention span exceeded: {duration}ms > {attention_span}ms")
            
            # Consider natural attention shift
            self._consider_attention_decay()
    
    def _handle_age_update(self, event: Event) -> None:
        """
        Handle development age update event.
        
        Args:
            event: The event containing the new age
        """
        new_age = event.data.get("new_age")
        if new_age is not None:
            self.update_developmental_age(new_age)
    
    def focus_attention(
        self, 
        target_type: str,
        target_id: UUID,
        description: str,
        priority: float = FocusLevel.MEDIUM,
        mode: AttentionMode = AttentionMode.FOCUSED,
        force_focus: bool = False
    ) -> Optional[AttentionFocus]:
        """
        Focus attention on a specific target.
        
        Args:
            target_type: Type of the target (e.g., 'sensory_input', 'pattern')
            target_id: ID of the target
            description: Description of the target
            priority: Priority level for this focus request
            mode: Attention mode to use
            force_focus: Whether to force focus even if below threshold
            
        Returns:
            The new attention focus if successful, None otherwise
        """
        # Create an attention target
        target = AttentionTarget(
            target_type=target_type,
            target_id=target_id,
            description=description,
            priority=priority,
            relevance_score=1.0 if force_focus else 0.8  # High relevance for explicit requests
        )
        
        # Check if we should shift attention
        if not self._should_shift_attention(target) and not force_focus:
            logger.debug(f"Focus shift rejected for {target_type} {target_id}")
            return None
        
        # If we have a current focus, archive it to history
        if self._current_focus:
            # Update duration before archiving
            current_time = time.time() * 1000  # ms
            duration = int(current_time - self._focus_start_time)
            self._current_focus.duration_ms = duration
            
            # Archive to history
            self._focus_history.append(self._current_focus)
        
        # Create new focus
        new_focus = AttentionFocus(
            mode=mode,
            primary_target=target,
            secondary_targets=[],
            focus_intensity=priority,
            duration_ms=0,
            context_data={"creator": "focus_controller"}
        )
        
        # Record start time
        self._focus_start_time = time.time() * 1000  # ms
        
        # Update current focus
        self._current_focus = new_focus
        
        # Add to active targets
        self._active_targets[target.id] = target
        
        # Publish focus shift event
        self._publish_focus_shift_event(new_focus)
        
        logger.info(f"Attention focused on {target_type} {target_id} with priority {priority}")
        return new_focus
    
    def add_secondary_target(
        self, 
        target_type: str,
        target_id: UUID,
        description: str,
        priority: float = FocusLevel.LOW
    ) -> bool:
        """
        Add a secondary target to the current focus.
        
        Args:
            target_type: Type of the target (e.g., 'sensory_input', 'pattern')
            target_id: ID of the target
            description: Description of the target
            priority: Priority level for this target
            
        Returns:
            Whether the target was successfully added
        """
        # Check if we have a current focus
        if not self._current_focus:
            logger.warning("Cannot add secondary target without primary focus")
            return False
        
        # Check if we're in a mode that supports secondary targets
        if self._current_focus.mode not in [AttentionMode.DIVIDED, AttentionMode.ALTERNATING]:
            logger.debug(f"Current mode {self._current_focus.mode} doesn't support secondary targets")
            return False
        
        # Check if we've reached max secondary targets
        if len(self._current_focus.secondary_targets) >= self._config.max_secondary_targets:
            logger.debug(f"Max secondary targets reached: {self._config.max_secondary_targets}")
            return False
        
        # Create target
        target = AttentionTarget(
            target_type=target_type,
            target_id=target_id,
            description=description,
            priority=priority,
            relevance_score=0.6  # Lower relevance for secondary targets
        )
        
        # Add to current focus
        self._current_focus.secondary_targets.append(target)
        
        # Add to active targets
        self._active_targets[target.id] = target
        
        logger.debug(f"Added secondary target: {target_type} {target_id}")
        return True
    
    def clear_focus(self) -> None:
        """Clear the current focus of attention."""
        if self._current_focus:
            # Update duration before archiving
            current_time = time.time() * 1000  # ms
            duration = int(current_time - self._focus_start_time)
            self._current_focus.duration_ms = duration
            
            # Archive to history
            self._focus_history.append(self._current_focus)
            
            # Clear current focus
            self._current_focus = None
            
            # Publish focus clear event
            self._publish_focus_clear_event()
            
            logger.info("Attention focus cleared")
    
    def get_current_focus(self) -> Optional[AttentionFocus]:
        """
        Get the current focus of attention.
        
        Returns:
            Current attention focus or None if no focus
        """
        if self._current_focus:
            # Update duration
            current_time = time.time() * 1000  # ms
            duration = int(current_time - self._focus_start_time)
            self._current_focus.duration_ms = duration
        
        return self._current_focus
    
    def get_focus_history(self, count: int = 5) -> List[AttentionFocus]:
        """
        Get recent attention focuses.
        
        Args:
            count: Maximum number of focuses to return
            
        Returns:
            List of recent attention focuses
        """
        return list(self._focus_history)[-count:]
    
    def get_active_targets(self) -> Dict[UUID, AttentionTarget]:
        """
        Get currently active attention targets.
        
        Returns:
            Dictionary of active attention targets by ID
        """
        return self._active_targets.copy()
    
    def update_developmental_age(self, new_age: float) -> None:
        """
        Update the developmental age of the focus controller.
        
        Args:
            new_age: The new developmental age
        """
        self._developmental_age = new_age
        logger.info(f"Focus controller age updated to {new_age}")
    
    def _should_shift_attention(self, target: AttentionTarget) -> bool:
        """
        Determine if attention should shift to a new target.
        
        Args:
            target: The potential new target
            
        Returns:
            Whether attention should shift
        """
        # If no current focus, always shift
        if not self._current_focus:
            return True
        
        # Get the current focus target
        current_target = self._current_focus.primary_target
        
        # If target is the same as current, no need to shift
        if current_target and current_target.target_id == target.target_id:
            return False
        
        # If force focusing (very high priority), always shift
        if target.priority >= FocusLevel.VERY_HIGH:
            return True
        
        # Calculate attention holding power of current focus
        current_time = time.time() * 1000  # ms
        duration = current_time - self._focus_start_time
        attention_span = self._get_attention_span()
        
        # Holding power diminishes as we approach attention span limit
        holding_ratio = max(0, 1.0 - (duration / attention_span))
        holding_power = current_target.priority * holding_ratio
        
        # Calculate distractibility factor (decreases with age)
        distractibility = max(0.1, 1.0 - min(1.0, self._developmental_age * 2))
        
        # Calculate attention shift threshold
        shift_threshold = holding_power * (1.0 - distractibility)
        
        # Determine if the new target can overcome the current focus
        return target.priority > shift_threshold
    
    def _consider_attention_shift(self, target: AttentionTarget) -> None:
        """
        Consider shifting attention to a new salient target.
        
        Args:
            target: The potential new target
        """
        # Check if attention should shift
        if self._should_shift_attention(target):
            # Create a new focus
            new_focus = AttentionFocus(
                mode=AttentionMode.ALERT if self._developmental_age < 0.3 else AttentionMode.FOCUSED,
                primary_target=target,
                secondary_targets=[],
                focus_intensity=target.priority,
                duration_ms=0,
                context_data={"creator": "salience_shift"}
            )
            
            # If we have a current focus, archive it to history
            if self._current_focus:
                # Update duration before archiving
                current_time = time.time() * 1000  # ms
                duration = int(current_time - self._focus_start_time)
                self._current_focus.duration_ms = duration
                
                # Archive to history
                self._focus_history.append(self._current_focus)
            
            # Update current focus
            self._current_focus = new_focus
            self._focus_start_time = time.time() * 1000  # ms
            
            # Add to active targets
            self._active_targets[target.id] = target
            
            # Publish focus shift event
            self._publish_focus_shift_event(new_focus)
            
            logger.info(f"Attention shifted to {target.target_type} {target.target_id} due to salience")
        else:
            # Consider as secondary target for divided attention
            if (self._current_focus and 
                self._current_focus.mode == AttentionMode.DIVIDED and
                self._config.enable_divided_attention and
                self._developmental_age >= 0.4):  # Divided attention develops later
                
                # Check if we've reached max secondary targets
                if len(self._current_focus.secondary_targets) < self._config.max_secondary_targets:
                    # Add as secondary target
                    self._current_focus.secondary_targets.append(target)
                    
                    # Add to active targets
                    self._active_targets[target.id] = target
                    
                    logger.debug(f"Added secondary target: {target.target_type} {target.target_id}")
    
    def _consider_attention_decay(self) -> None:
        """Consider natural decay of attention after span is exceeded."""
        # Probability of attention decay increases with time beyond span
        current_time = time.time() * 1000  # ms
        duration = current_time - self._focus_start_time
        attention_span = self._get_attention_span()
        
        # Calculate decay probability
        excess_ratio = (duration - attention_span) / attention_span
        decay_probability = min(0.95, excess_ratio * 0.5)
        
        # Decrease probability based on developmental age
        # (more mature minds can sustain attention longer)
        decay_probability *= max(0.1, 1.0 - min(0.9, self._developmental_age))
        
        # Apply random chance for natural attention decay
        if np.random.random() < decay_probability:
            logger.debug(f"Natural attention decay after {duration}ms")
            self.clear_focus()
    
    def _get_attention_span(self) -> int:
        """
        Get the current attention span based on developmental age.
        
        Returns:
            Attention span in milliseconds
        """
        # Base span from config
        base_span = self._config.base_attention_span_ms
        
        # Developmental factor (attention span increases with age)
        dev_factor = 1.0 + (self._developmental_age * 2.0)
        
        # Calculate span
        return int(base_span * dev_factor)
    
    def _get_current_salience_threshold(self) -> float:
        """
        Get the current salience threshold based on developmental age and focus.
        
        Returns:
            Current salience threshold
        """
        # Base threshold from config
        base_threshold = self._config.base_salience_threshold
        
        # If there's a current focus, it raises the threshold
        focus_factor = 0.0
        if self._current_focus:
            focus_factor = self._current_focus.focus_intensity * 0.25
        
        # Developmental factor (threshold increases with age as
        # the mind gets better at maintaining focus)
        dev_factor = min(0.3, self._developmental_age * 0.15)
        
        # Calculate threshold
        return min(0.95, base_threshold + focus_factor + dev_factor)
    
    def _publish_focus_shift_event(self, focus: AttentionFocus) -> None:
        """
        Publish an event for a focus shift.
        
        Args:
            focus: The new focus
        """
        event = AttentionEvent(
            event_type="focus_shifted",
            payload={
                "focus": focus.dict(),
                "developmental_age": self._developmental_age
            }
        )
        
        self._event_bus.publish("attention_focus_shifted", event.dict())
        
        # Also publish a message for other modules
        message = Message(
            type=MessageType.ATTENTION_FOCUS,
            source="attention.focus_controller",
            recipient=Recipient.BROADCAST,
            content={
                "focus": focus.dict(),
                "action": "focus_shifted"
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
    
    def _publish_focus_clear_event(self) -> None:
        """Publish an event for focus clearing."""
        event = AttentionEvent(
            event_type="focus_cleared",
            payload={
                "developmental_age": self._developmental_age
            }
        )
        
        self._event_bus.publish("attention_focus_cleared", event.dict())
        
        # Also publish a message for other modules
        message = Message(
            type=MessageType.ATTENTION_FOCUS,
            source="attention.focus_controller",
            recipient=Recipient.BROADCAST,
            content={
                "action": "focus_cleared"
            }
        )
        
        self._event_bus.publish("message", {"message": message.dict()})
