# TODO: Implement the TheoryOfMind class to understand others' mental states
# This component should be able to:
# - Represent others' beliefs, desires, and intentions
# - Infer mental states from observed behavior
# - Understand false beliefs and different perspectives
# - Track multiple agents' mental models simultaneously

# TODO: Implement developmental progression in theory of mind:
# - Simple agency detection in early stages
# - Understanding desires before beliefs in early childhood
# - First-order belief representation in childhood
# - Higher-order mental state representation in adolescence/adulthood

# TODO: Create mechanisms for:
# - Perspective taking: Simulate others' viewpoints
# - Belief inference: Deduce what others believe
# - Intention recognition: Infer goals from actions
# - Mental state tracking: Monitor changes in others' knowledge

# TODO: Implement different levels of mental state representation:
# - First-order: What others believe
# - Second-order: What others believe about others' beliefs
# - Higher-order: More complex nested mental states
# - Shared mental models: Common ground in interaction

# TODO: Connect to language and memory modules
# Theory of mind should utilize language processing
# and draw on memories of past social interactions

from typing import Dict, List, Any, Optional, Tuple, Union, Set
from collections import defaultdict
import logging
import numpy as np
import torch
from datetime import datetime

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.utils.llm_client import LLMClient

from lmm_project.modules.social.models import MentalState, AgentModel
from lmm_project.modules.social.neural_net import MentalStateEncoder

logger = logging.getLogger(__name__)

class TheoryOfMind(BaseModule):
    """
    Understands others' mental states
    
    This module represents, infers, and tracks the beliefs,
    desires, intentions, and emotions of other agents,
    enabling the prediction of their behavior.
    """
    
    # Override developmental milestones with ToM-specific milestones
    development_milestones = {
        0.0: "Basic agency detection",
        0.2: "Simple desire recognition",
        0.4: "First-order belief representation",
        0.6: "False-belief understanding",
        0.8: "Second-order belief representation",
        1.0: "Complex perspective taking and counterfactual reasoning"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None):
        """
        Initialize the theory of mind module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
        """
        super().__init__(module_id=module_id, module_type="theory_of_mind", event_bus=event_bus)
        
        # Initialize mental state representation structures
        self.agent_models: Dict[str, AgentModel] = {}
        self.mental_state_history: Dict[str, List[MentalState]] = defaultdict(list)
        self.common_ground: Dict[str, Any] = {}
        
        # Initialize neural networks
        self.mental_state_encoder = MentalStateEncoder()
        
        # Tracking for seen agents
        self.seen_agents: Set[str] = set()
        
        # Perspective tracking
        self.perspective_cache: Dict[str, Dict[str, Any]] = {}
        
        # Embedding client for semantic processing
        self.embedding_client = LLMClient()
        self.embedding_cache: Dict[str, List[float]] = {}
        
        # Subscribe to relevant events if event bus is provided
        if self.event_bus:
            self.subscribe_to_message("perception_input", self._handle_perception)
            self.subscribe_to_message("agent_action", self._handle_agent_action)
            self.subscribe_to_message("communication_event", self._handle_communication)
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to understand others' mental states
        
        Args:
            input_data: Dictionary containing social interaction information
            
        Returns:
            Dictionary with inferred mental states
        """
        # Determine what type of input we're processing
        input_type = input_data.get("input_type", "")
        
        if input_type == "agent_perception":
            return self._process_agent_perception(input_data)
        elif input_type == "action_inference":
            return self._process_action_inference(input_data)
        elif input_type == "belief_update":
            return self._process_belief_update(input_data)
        elif input_type == "perspective_taking":
            return self._process_perspective_taking(input_data)
        else:
            # Default processing handles general ToM inference
            return self._process_default(input_data)
    
    def _process_agent_perception(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process perception of an agent to track their mental state"""
        agent_id = input_data.get("agent_id")
        if not agent_id:
            return {"error": "Missing agent_id in input data"}
        
        # Create or update agent model
        if agent_id not in self.agent_models:
            # Create a new agent model
            name = input_data.get("name", f"Agent-{agent_id}")
            
            # Create initial mental state
            mental_state = MentalState(
                agent_id=agent_id,
                beliefs={},
                desires={},
                intentions={},
                emotions=input_data.get("emotions", {})
            )
            
            # Create agent model
            self.agent_models[agent_id] = AgentModel(
                agent_id=agent_id,
                name=name,
                mental_state=mental_state
            )
            
            self.seen_agents.add(agent_id)
            logger.info(f"Created new agent model for {name} (ID: {agent_id})")
        else:
            # Update existing agent model
            agent_model = self.agent_models[agent_id]
            
            # Update emotions if provided
            if "emotions" in input_data:
                for emotion, intensity in input_data["emotions"].items():
                    agent_model.mental_state.emotions[emotion] = intensity
                
                agent_model.mental_state.last_updated = datetime.now()
        
        # Return the current mental state
        return {
            "agent_id": agent_id,
            "mental_state": self.agent_models[agent_id].mental_state.model_dump(),
            "confidence": min(1.0, self.development_level + 0.2)
        }
    
    def _process_action_inference(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Infer beliefs and intentions from observed actions"""
        agent_id = input_data.get("agent_id")
        action = input_data.get("action")
        
        if not agent_id or not action:
            return {"error": "Missing agent_id or action in input data"}
        
        # Ensure we have an agent model
        if agent_id not in self.agent_models:
            # Create a basic agent model if we don't have one
            mental_state = MentalState(
                agent_id=agent_id,
                beliefs={},
                desires={},
                intentions={},
                emotions={}
            )
            
            self.agent_models[agent_id] = AgentModel(
                agent_id=agent_id,
                name=f"Agent-{agent_id}",
                mental_state=mental_state
            )
            
            self.seen_agents.add(agent_id)
        
        # Get agent model
        agent_model = self.agent_models[agent_id]
        
        # Infer intentions based on action
        # The sophistication of this inference increases with development level
        inferred_intentions = {}
        inferred_beliefs = {}
        
        # Simple intention inference (available at all development levels)
        action_verb = action.get("verb", "")
        action_object = action.get("object", "")
        
        if action_verb and action_object:
            # Create intention key from action
            intention_key = f"{action_verb}_{action_object}"
            inferred_intentions[intention_key] = 0.8  # High confidence in direct intention
            
            # Infer basic belief that action object exists
            inferred_beliefs[f"exists_{action_object}"] = 0.9
        
        # More sophisticated inference at higher development levels
        if self.development_level >= 0.4:
            # Infer beliefs about action prerequisites
            if "prerequisites" in action:
                for prereq in action["prerequisites"]:
                    inferred_beliefs[prereq] = 0.7
            
            # Infer likely goals/outcomes
            if "outcomes" in action:
                for outcome, probability in action["outcomes"].items():
                    desire_key = f"achieve_{outcome}"
                    inferred_intentions[desire_key] = probability * 0.8
        
        # Update agent model with inferred mental states
        for belief, confidence in inferred_beliefs.items():
            agent_model.mental_state.beliefs[belief] = confidence
            
        for intention, commitment in inferred_intentions.items():
            agent_model.mental_state.intentions[intention] = commitment
            
        agent_model.mental_state.last_updated = datetime.now()
        
        # Record history (limited to keep memory footprint reasonable)
        self.mental_state_history[agent_id].append(agent_model.mental_state.model_copy(deep=True))
        if len(self.mental_state_history[agent_id]) > 10:
            self.mental_state_history[agent_id].pop(0)
        
        # Return inferred mental state
        return {
            "agent_id": agent_id,
            "inferred_beliefs": inferred_beliefs,
            "inferred_intentions": inferred_intentions,
            "mental_state": agent_model.mental_state.model_dump(),
            "confidence": min(0.9, self.development_level + 0.1)
        }
    
    def _process_belief_update(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update beliefs for an agent based on new information"""
        agent_id = input_data.get("agent_id")
        beliefs = input_data.get("beliefs", {})
        observability = input_data.get("observability", 1.0)  # How observable the information is
        
        if not agent_id:
            return {"error": "Missing agent_id in input data"}
        
        # Ensure we have an agent model
        if agent_id not in self.agent_models:
            return {"error": f"No agent model exists for {agent_id}"}
        
        # Get agent model
        agent_model = self.agent_models[agent_id]
        
        # Update beliefs based on development level
        # At low development levels, can't track beliefs different from reality
        applied_beliefs = {}
        
        for belief, value in beliefs.items():
            # Determine if agent would know this based on observability
            # More developed ToM can reason about information barriers
            if self.development_level < 0.4:
                # At low development, can only track beliefs that match reality
                if observability > 0.7:  # Only highly observable information
                    agent_model.mental_state.beliefs[belief] = value
                    applied_beliefs[belief] = value
            elif self.development_level < 0.6:
                # Can track some differences between reality and beliefs
                if observability > 0.3:
                    agent_model.mental_state.beliefs[belief] = value * observability
                    applied_beliefs[belief] = value * observability
            else:
                # Full false-belief understanding
                # Can track completely separate belief states
                agent_model.mental_state.beliefs[belief] = value * observability
                applied_beliefs[belief] = value * observability
        
        agent_model.mental_state.last_updated = datetime.now()
        
        # Return updated beliefs
        return {
            "agent_id": agent_id,
            "updated_beliefs": applied_beliefs,
            "mental_state": agent_model.mental_state.model_dump(),
            "confidence": min(0.9, self.development_level + 0.1)
        }
    
    def _process_perspective_taking(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Take the perspective of another agent"""
        agent_id = input_data.get("agent_id")
        situation = input_data.get("situation", {})
        
        if not agent_id:
            return {"error": "Missing agent_id in input data"}
        
        # Check development level for perspective taking
        if self.development_level < 0.5:
            return {
                "error": "Perspective taking not available at current development level",
                "development_needed": "This capability requires development level of at least 0.5"
            }
        
        # Ensure we have an agent model
        if agent_id not in self.agent_models:
            return {"error": f"No agent model exists for {agent_id}"}
        
        # Get agent model
        agent_model = self.agent_models[agent_id]
        
        # Create perspective view
        perspective = {
            "agent_id": agent_id,
            "visible_objects": [],
            "known_facts": [],
            "interpretation": {}
        }
        
        # Calculate what would be visible from agent's location
        if "agent_location" in situation and "objects" in situation:
            agent_location = situation["agent_location"]
            visible_objects = []
            
            for obj in situation["objects"]:
                # Simple distance-based visibility
                if "location" in obj:
                    obj_location = obj["location"]
                    # Calculate distance (simplified 2D calculation)
                    if len(agent_location) >= 2 and len(obj_location) >= 2:
                        distance = np.sqrt((agent_location[0] - obj_location[0])**2 + 
                                          (agent_location[1] - obj_location[1])**2)
                        
                        # Check visibility
                        if distance < situation.get("visibility_range", 10):
                            visible_objects.append(obj)
            
            perspective["visible_objects"] = visible_objects
        
        # Determine what facts the agent would know
        for fact, is_known in agent_model.mental_state.knowledge.items():
            if is_known:
                perspective["known_facts"].append(fact)
        
        # Add beliefs as part of the perspective
        for belief, confidence in agent_model.mental_state.beliefs.items():
            if confidence > 0.5:  # Only include beliefs with sufficient confidence
                perspective["interpretation"][belief] = confidence
        
        # Cache the perspective for future reference
        self.perspective_cache[agent_id] = perspective
        
        return {
            "agent_id": agent_id,
            "perspective": perspective,
            "confidence": min(0.8, self.development_level)
        }
    
    def _process_default(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Default processing for general ToM inference"""
        # Extract relevant information
        agent_id = input_data.get("agent_id")
        
        if not agent_id:
            return {"error": "Missing agent_id in input data"}
        
        # Return current mental model if we have one
        if agent_id in self.agent_models:
            return {
                "agent_id": agent_id,
                "mental_state": self.agent_models[agent_id].mental_state.model_dump(),
                "confidence": min(0.8, self.development_level)
            }
        
        # Otherwise report that we don't have a model
        return {
            "agent_id": agent_id,
            "error": f"No mental model exists for agent {agent_id}",
            "suggestion": "Send agent_perception data to create model"
        }
    
    def _handle_perception(self, message: Message) -> None:
        """Handle perception input from the event bus"""
        content = message.content
        if "agents" in content:
            for agent in content["agents"]:
                if "id" in agent:
                    self._process_agent_perception({
                        "input_type": "agent_perception",
                        "agent_id": agent["id"],
                        "name": agent.get("name"),
                        "emotions": agent.get("emotions", {})
                    })
    
    def _handle_agent_action(self, message: Message) -> None:
        """Handle agent action observation from the event bus"""
        content = message.content
        if "agent_id" in content and "action" in content:
            self._process_action_inference({
                "input_type": "action_inference",
                "agent_id": content["agent_id"],
                "action": content["action"]
            })
    
    def _handle_communication(self, message: Message) -> None:
        """Handle communication events from the event bus"""
        content = message.content
        if "sender_id" in content and "content" in content:
            # Communication can reveal beliefs
            beliefs = {}
            
            # Very simple belief extraction - would be more sophisticated in full implementation
            if "believes" in content:
                beliefs = content["believes"]
            
            # Update sender's mental model with these beliefs
            if beliefs:
                self._process_belief_update({
                    "input_type": "belief_update",
                    "agent_id": content["sender_id"],
                    "beliefs": beliefs,
                    "observability": 0.9  # High observability for directly communicated beliefs
                })
    
    def get_agent_model(self, agent_id: str) -> Optional[AgentModel]:
        """Get the mental model for a specific agent"""
        return self.agent_models.get(agent_id)
    
    def get_agent_mental_state(self, agent_id: str) -> Optional[MentalState]:
        """Get the current mental state for a specific agent"""
        agent_model = self.get_agent_model(agent_id)
        if agent_model:
            return agent_model.mental_state
        return None
    
    def get_all_agent_ids(self) -> List[str]:
        """Get the IDs of all tracked agents"""
        return list(self.agent_models.keys())
    
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
        
        # No additional behavior is needed here, as development milestones
        # are checked in the parent class implementation
        
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """Get the current state of the module"""
        state = super().get_state()
        
        # Add ToM-specific state information
        state.update({
            "agent_count": len(self.agent_models),
            "agent_ids": list(self.agent_models.keys()),
            "seen_agents_count": len(self.seen_agents)
        })
        
        return state
