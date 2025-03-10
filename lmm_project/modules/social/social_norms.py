# TODO: Implement the SocialNorms class to learn and apply social rules
# This component should be able to:
# - Learn implicit and explicit social rules from observation
# - Detect violations of social norms
# - Apply appropriate social conventions in different contexts
# - Update norm understanding based on feedback

# TODO: Implement developmental progression in social norms:
# - Basic rule following in early stages
# - Concrete norm adherence in childhood
# - Understanding norm flexibility in adolescence
# - Complex contextual norm application in adulthood

# TODO: Create mechanisms for:
# - Norm acquisition: Learn rules from observation and instruction
# - Violation detection: Recognize when norms are broken
# - Context recognition: Identify which norms apply in different settings
# - Norm updating: Revise understanding based on experience

# TODO: Implement different norm categories:
# - Etiquette norms: Polite behavior conventions
# - Moral norms: Ethical principles for behavior
# - Conventional norms: Arbitrary cultural standards
# - Descriptive norms: Common behavioral patterns

# TODO: Connect to theory of mind and memory modules
# Social norm understanding should use theory of mind to understand
# others' norm expectations and store norms in semantic memory

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
import numpy as np
import torch
from datetime import datetime
from uuid import uuid4

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient

from lmm_project.modules.social.models import SocialNorm, NormViolation
from lmm_project.modules.social.neural_net import NormProcessingNetwork

logger = logging.getLogger(__name__)

class SocialNorms(BaseModule):
    """
    Learns and applies social rules
    
    This module acquires social conventions, detects norm violations,
    applies appropriate rules in different contexts, and updates
    norm understanding based on feedback.
    """
    
    # Override developmental milestones with social norm-specific milestones
    development_milestones = {
        0.0: "Basic rule recognition",
        0.2: "Simple norm following",
        0.4: "Context-sensitive norm application",
        0.6: "Norm flexibility understanding",
        0.8: "Complex norm system navigation",
        1.0: "Cultural norm adaptation and creation"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the social norms module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="social_norms", event_bus=event_bus)
        
        # Initialize norm systems
        self.norms: Dict[str, SocialNorm] = {}
        
        # Context-norm associations for fast lookup
        self.context_norms = defaultdict(list)
        
        # Context recognition
        self.known_contexts = set()
        
        # Initialize basic norms
        self._initialize_basic_norms()
        
        # Norm violation tracking
        self.norm_violations: List[NormViolation] = []
        
        # Neural networks for norm processing
        self.norm_network = NormProcessingNetwork()
        
        # Embedding client for semantic processing
        self.embedding_client = LLMClient()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Tracking norm learning
        self.observed_behaviors: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self.subscribe_to_message("social_observation", self._handle_observation)
            self.subscribe_to_message("norm_feedback", self._handle_feedback)
            self.subscribe_to_message("context_change", self._handle_context_change)
    
    def _initialize_basic_norms(self) -> None:
        """Initialize a starter set of basic social norms"""
        # Greeting norm
        greeting = SocialNorm(
            name="greeting",
            description="Acknowledge others when meeting",
            contexts=["social_meeting", "introduction", "workplace"],
            norm_type="etiquette",
            strength=0.7,
            flexibility=0.6,
            behaviors={
                "verbal_greeting": 0.8,
                "wave": 0.5,
                "smile": 0.6,
                "handshake": 0.4
            }
        )
        self.norms[greeting.id] = greeting
        for context in greeting.contexts:
            self.context_norms[context].append(greeting.id)
            self.known_contexts.add(context)
        
        # Turn-taking norm
        turn_taking = SocialNorm(
            name="turn_taking",
            description="Take turns in conversation without interrupting",
            contexts=["conversation", "discussion", "meeting"],
            norm_type="conventional",
            strength=0.8,
            flexibility=0.5,
            behaviors={
                "wait_for_pause": 0.9,
                "signal_desire_to_speak": 0.6,
                "avoid_interruption": 0.8,
                "acknowledge_others": 0.7
            }
        )
        self.norms[turn_taking.id] = turn_taking
        for context in turn_taking.contexts:
            self.context_norms[context].append(turn_taking.id)
            self.known_contexts.add(context)
        
        # Personal space norm
        personal_space = SocialNorm(
            name="personal_space",
            description="Maintain appropriate physical distance",
            contexts=["public_space", "conversation", "queue"],
            norm_type="conventional",
            strength=0.8,
            flexibility=0.3,
            behaviors={
                "maintain_distance": 0.9,
                "respect_boundaries": 0.8,
                "ask_before_touching": 0.7
            }
        )
        self.norms[personal_space.id] = personal_space
        for context in personal_space.contexts:
            self.context_norms[context].append(personal_space.id)
            self.known_contexts.add(context)
        
        # Honesty norm
        honesty = SocialNorm(
            name="honesty",
            description="Tell the truth and avoid deception",
            contexts=["all"],  # Applies broadly
            norm_type="moral",
            strength=0.9,
            flexibility=0.2,
            behaviors={
                "tell_truth": 0.9,
                "avoid_deception": 0.8,
                "correct_mistakes": 0.7
            }
        )
        self.norms[honesty.id] = honesty
        for context in honesty.contexts:
            self.context_norms[context].append(honesty.id)
            self.known_contexts.add(context)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to apply social norms
        
        Args:
            input_data: Dictionary containing social situation information
            
        Returns:
            Dictionary with norm-relevant analyses and responses
        """
        # Determine what type of input we're processing
        input_type = input_data.get("input_type", "")
        
        if input_type == "identify_norms":
            return self._process_identify_norms(input_data)
        elif input_type == "check_behavior":
            return self._process_check_behavior(input_data)
        elif input_type == "detect_violations":
            return self._process_detect_violations(input_data)
        elif input_type == "learn_norm":
            return self._process_learn_norm(input_data)
        else:
            # Default processing identifies appropriate norms for a context
            context = input_data.get("context")
            if context:
                return self._process_identify_norms({"context": context})
            
            return {
                "error": "Unknown input type or insufficient parameters",
                "valid_types": ["identify_norms", "check_behavior", "detect_violations", "learn_norm"]
            }
    
    def _process_identify_norms(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Identify appropriate norms for a given context"""
        context = input_data.get("context")
        if not context:
            return {"error": "Context is required"}
        
        # Add context to known contexts
        self.known_contexts.add(context)
        
        # Get directly associated norms
        direct_norm_ids = self.context_norms.get(context, [])
        
        # Also get norms that apply to "all" contexts
        universal_norm_ids = self.context_norms.get("all", [])
        
        # Combine norm IDs and remove duplicates
        norm_ids = list(set(direct_norm_ids + universal_norm_ids))
        
        # Get norm details
        norm_details = []
        for norm_id in norm_ids:
            norm = self.norms.get(norm_id)
            if norm:
                # Only include norms the agent can understand at current development level
                # More complex or flexible norms require higher development
                if self._can_understand_norm(norm):
                    norm_detail = {
                        "id": norm_id,
                        "name": norm.name,
                        "description": norm.description,
                        "type": norm.norm_type,
                        "strength": norm.strength,
                        "behaviors": norm.behaviors
                    }
                    norm_details.append(norm_detail)
        
        # Sort by strength (strongest norms first)
        norm_details.sort(key=lambda x: x["strength"], reverse=True)
        
        return {
            "context": context,
            "applicable_norms": norm_details,
            "norm_count": len(norm_details)
        }
    
    def _process_check_behavior(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check if a behavior is appropriate in a given context"""
        context = input_data.get("context")
        behavior = input_data.get("behavior")
        agent_id = input_data.get("agent_id")
        
        if not context or not behavior:
            return {"error": "Context and behavior are required"}
        
        # Add to observed behaviors
        if agent_id:
            self.observed_behaviors[context][behavior] += 1
        
        # Get applicable norms
        norm_result = self._process_identify_norms({"context": context})
        
        if "error" in norm_result:
            return norm_result
        
        applicable_norms = norm_result.get("applicable_norms", [])
        
        # Check behavior against norms
        behavior_appropriate = True
        relevant_norms = []
        violations = []
        
        for norm_data in applicable_norms:
            norm_id = norm_data["id"]
            norm = self.norms.get(norm_id)
            
            if not norm:
                continue
            
            # Check if this norm has expectations about this behavior
            if behavior in norm.behaviors:
                relevant_norms.append({
                    "id": norm_id,
                    "name": norm.name,
                    "expected_level": norm.behaviors[behavior],
                    "importance": norm.strength
                })
                
                # If the behavior is prohibited (importance high, expected level low)
                if norm.behaviors[behavior] < 0.3 and norm.strength > 0.6:
                    behavior_appropriate = False
                    violations.append({
                        "norm_id": norm_id,
                        "norm_name": norm.name,
                        "severity": norm.strength * (0.3 - norm.behaviors[behavior]) * 3.33  # Scale to 0-1
                    })
        
        # Record violation if detected and agent is specified
        if not behavior_appropriate and agent_id:
            for violation in violations:
                violation_record = NormViolation(
                    norm_id=violation["norm_id"],
                    agent_id=agent_id,
                    context=context,
                    severity=violation["severity"],
                    description=f"Agent {agent_id} violated {violation['norm_name']} norm with behavior: {behavior}"
                )
                self.norm_violations.append(violation_record)
                
                # Limit stored violations to prevent memory issues
                if len(self.norm_violations) > 100:
                    self.norm_violations.pop(0)
                
                # Publish a violation event if we have an event bus
                if self.event_bus:
                    self.publish_message("norm_violation", {
                        "norm_id": violation["norm_id"],
                        "agent_id": agent_id,
                        "context": context,
                        "behavior": behavior,
                        "severity": violation["severity"]
                    })
        
        return {
            "context": context,
            "behavior": behavior,
            "appropriate": behavior_appropriate,
            "relevant_norms": relevant_norms,
            "violations": violations if not behavior_appropriate else []
        }
    
    def _process_detect_violations(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Detect norm violations in a social situation"""
        context = input_data.get("context")
        behaviors = input_data.get("behaviors", [])
        agent_id = input_data.get("agent_id")
        
        if not context or not behaviors:
            return {"error": "Context and behaviors are required"}
        
        # Process each behavior
        violations = []
        for behavior in behaviors:
            result = self._process_check_behavior({
                "context": context,
                "behavior": behavior,
                "agent_id": agent_id
            })
            
            if not result.get("appropriate", True):
                # A violation was detected
                for violation in result.get("violations", []):
                    violations.append({
                        "behavior": behavior,
                        "norm_id": violation["norm_id"],
                        "norm_name": violation["norm_name"],
                        "severity": violation["severity"]
                    })
        
        return {
            "context": context,
            "agent_id": agent_id,
            "behaviors": behaviors,
            "violations_detected": len(violations) > 0,
            "violations": violations
        }
    
    def _process_learn_norm(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Learn a new norm or update an existing one"""
        # Check development level for learning complex norms
        if self.development_level < 0.3:
            return {
                "error": "Norm learning not available at current development level",
                "development_needed": "This capability requires development level of at least 0.3"
            }
        
        # Extract norm information
        name = input_data.get("name")
        description = input_data.get("description")
        contexts = input_data.get("contexts", [])
        norm_type = input_data.get("norm_type")
        behaviors = input_data.get("behaviors", {})
        
        if not name or not description or not contexts or not norm_type or not behaviors:
            return {"error": "Incomplete norm information provided"}
        
        # Check if a similar norm already exists
        existing_norm_id = None
        for norm_id, norm in self.norms.items():
            if norm.name.lower() == name.lower():
                existing_norm_id = norm_id
                break
        
        if existing_norm_id:
            # Update existing norm
            norm = self.norms[existing_norm_id]
            
            # Update contexts
            new_contexts = [c for c in contexts if c not in norm.contexts]
            if new_contexts:
                norm.contexts.extend(new_contexts)
                for context in new_contexts:
                    self.context_norms[context].append(existing_norm_id)
                    self.known_contexts.add(context)
            
            # Update behaviors
            for behavior, importance in behaviors.items():
                if behavior in norm.behaviors:
                    # Average with existing value to smooth updates
                    norm.behaviors[behavior] = (norm.behaviors[behavior] + importance) / 2
                else:
                    # Add new behavior
                    norm.behaviors[behavior] = importance
            
            logger.info(f"Updated existing norm: {name}")
            
            return {
                "status": "updated",
                "norm_id": existing_norm_id,
                "name": norm.name,
                "contexts": norm.contexts,
                "behaviors": norm.behaviors
            }
        else:
            # Create new norm
            strength = input_data.get("strength", 0.5)
            flexibility = input_data.get("flexibility", 0.5)
            
            # Development level affects how nuanced learned norms can be
            if self.development_level < 0.5:
                # Simpler norms at lower development levels
                flexibility = 0.3  # Less flexibility
                
                # Simplify behaviors to binary (important vs. not important)
                simplified_behaviors = {}
                for behavior, importance in behaviors.items():
                    simplified_behaviors[behavior] = 1.0 if importance > 0.5 else 0.0
                behaviors = simplified_behaviors
            
            # Create the norm
            new_norm = SocialNorm(
                name=name,
                description=description,
                contexts=contexts,
                norm_type=norm_type,
                strength=strength,
                flexibility=flexibility,
                behaviors=behaviors
            )
            
            # Store the norm
            self.norms[new_norm.id] = new_norm
            
            # Update context-norm associations
            for context in contexts:
                self.context_norms[context].append(new_norm.id)
                self.known_contexts.add(context)
                
            logger.info(f"Learned new norm: {name}")
            
            return {
                "status": "created",
                "norm_id": new_norm.id,
                "name": new_norm.name,
                "contexts": new_norm.contexts,
                "behaviors": new_norm.behaviors
            }
    
    def _can_understand_norm(self, norm: SocialNorm) -> bool:
        """Determine if current development level allows understanding of this norm"""
        # Very simple norms can be understood at all levels
        if norm.flexibility < 0.2 and len(norm.behaviors) <= 2:
            return True
            
        # Context-sensitive norms require some development
        if len(norm.contexts) > 1 and self.development_level < 0.3:
            return False
            
        # Highly flexible norms require more development
        if norm.flexibility > 0.6 and self.development_level < 0.5:
            return False
            
        # Complex norms with many behaviors require high development
        if len(norm.behaviors) > 5 and self.development_level < 0.6:
            return False
            
        return True
    
    def _handle_observation(self, message: Message) -> None:
        """Handle social observation events from the event bus"""
        content = message.content
        
        # Extract context and behavior
        context = content.get("context")
        behavior = content.get("behavior")
        agent_id = content.get("agent_id")
        
        if context and behavior:
            # Check the behavior against norms
            self._process_check_behavior({
                "context": context,
                "behavior": behavior,
                "agent_id": agent_id
            })
            
            # Learn from observation if at sufficient development level
            if self.development_level >= 0.3:
                # Update frequency counts
                self.observed_behaviors[context][behavior] += 1
                
                # If we've seen this behavior frequently in this context, consider it normative
                if self.observed_behaviors[context][behavior] >= 5:
                    # Check if this behavior is already covered by an existing norm
                    norm_result = self._process_identify_norms({"context": context})
                    covered = False
                    
                    for norm_data in norm_result.get("applicable_norms", []):
                        norm_id = norm_data["id"]
                        norm = self.norms.get(norm_id)
                        if norm and behavior in norm.behaviors:
                            covered = True
                            break
                    
                    # If not covered, create a new descriptive norm
                    if not covered:
                        self._process_learn_norm({
                            "name": f"{context}_{behavior}",
                            "description": f"Common behavior observed in {context}",
                            "contexts": [context],
                            "norm_type": "descriptive",
                            "strength": 0.5,  # Moderate strength for learned norms
                            "flexibility": 0.7,  # Descriptive norms are flexible
                            "behaviors": {behavior: 0.8}  # High expectation since frequently observed
                        })
    
    def _handle_feedback(self, message: Message) -> None:
        """Handle feedback about norm violations or adherence"""
        content = message.content
        
        # Extract relevant information
        norm_id = content.get("norm_id")
        context = content.get("context")
        behavior = content.get("behavior")
        is_violation = content.get("is_violation", False)
        feedback_strength = content.get("strength", 0.5)
        
        if norm_id and norm_id in self.norms and behavior:
            norm = self.norms[norm_id]
            
            # Update behavior importance based on feedback
            if behavior in norm.behaviors:
                current_value = norm.behaviors[behavior]
                
                if is_violation:
                    # If this was marked as a violation, decrease the expected level
                    # The stronger the feedback, the more the value is decreased
                    new_value = current_value - (feedback_strength * 0.2)
                else:
                    # If this was proper adherence, increase the expected level
                    new_value = current_value + (feedback_strength * 0.1)
                
                # Ensure value stays within bounds
                norm.behaviors[behavior] = min(1.0, max(0.0, new_value))
            else:
                # Add new behavior
                initial_value = 0.2 if is_violation else 0.8
                norm.behaviors[behavior] = initial_value
    
    def _handle_context_change(self, message: Message) -> None:
        """Handle context change events"""
        content = message.content
        
        # Extract context
        new_context = content.get("context")
        
        if new_context:
            # Identify appropriate norms for this context
            self._process_identify_norms({"context": new_context})
    
    def get_norm_by_id(self, norm_id: str) -> Optional[SocialNorm]:
        """Get a norm by its ID"""
        return self.norms.get(norm_id)
    
    def get_norms_by_type(self, norm_type: str) -> List[SocialNorm]:
        """Get all norms of a specific type"""
        return [norm for norm in self.norms.values() if norm.norm_type == norm_type]
    
    def get_norms_by_context(self, context: str) -> List[SocialNorm]:
        """Get all norms applicable in a specific context"""
        norm_ids = set(self.context_norms.get(context, []) + self.context_norms.get("all", []))
        return [self.norms[norm_id] for norm_id in norm_ids if norm_id in self.norms]
    
    def get_all_contexts(self) -> List[str]:
        """Get all known contexts"""
        return list(self.known_contexts)
    
    def get_recent_violations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent norm violations"""
        violations = []
        
        for violation in self.norm_violations[-limit:]:
            norm = self.norms.get(violation.norm_id)
            if norm:
                violations.append({
                    "agent_id": violation.agent_id,
                    "context": violation.context,
                    "norm_name": norm.name,
                    "severity": violation.severity,
                    "timestamp": violation.timestamp.isoformat(),
                    "description": violation.description
                })
        
        return violations
    
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
        
        # No additional behavior needed as milestones are checked in parent class
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add norm-specific state information
        state.update({
            "norm_count": len(self.norms),
            "context_count": len(self.known_contexts),
            "violation_count": len(self.norm_violations)
        })
        
        return state
