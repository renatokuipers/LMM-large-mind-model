# Social module 

# TODO: Implement the social module factory function to return an integrated SocialSystem
# This module should be responsible for social cognition, relationship modeling,
# moral reasoning, and understanding social norms.

# TODO: Create SocialSystem class that integrates all social sub-components:
# - theory_of_mind: understanding others' mental states
# - social_norms: learning and applying social rules
# - moral_reasoning: making ethical judgments
# - relationship_models: representing social relationships

# TODO: Implement development tracking for social cognition
# Social capabilities should develop from basic social responsiveness in early stages
# to sophisticated social understanding and nuanced moral reasoning in later stages

# TODO: Connect social module to emotion, language, and memory modules
# Social cognition should be informed by emotional understanding,
# utilize language representations, and draw on social memories

# TODO: Implement perspective-taking capabilities
# Include the ability to represent others' viewpoints, understand
# how situations appear to others, and imagine others' experiences

from typing import Dict, List, Any, Optional
from lmm_project.core.event_bus import EventBus
import logging

from lmm_project.modules.social.theory_of_mind import TheoryOfMind
from lmm_project.modules.social.social_norms import SocialNorms
from lmm_project.modules.social.moral_reasoning import MoralReasoning
from lmm_project.modules.social.relationship_models import RelationshipModels

logger = logging.getLogger(__name__)

class SocialSystem:
    """
    Integrated social cognition system
    
    This class combines theory of mind, social norms, 
    moral reasoning, and relationship modeling components
    into a unified social cognition system.
    """
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the social system with all components
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of the module
        """
        self.module_id = module_id
        self.event_bus = event_bus
        self.module_type = "social_system"
        
        # Initialize all social components
        self.theory_of_mind = TheoryOfMind(
            module_id=f"{module_id}/theory_of_mind",
            event_bus=event_bus
        )
        
        self.social_norms = SocialNorms(
            module_id=f"{module_id}/social_norms",
            event_bus=event_bus
        )
        
        self.moral_reasoning = MoralReasoning(
            module_id=f"{module_id}/moral_reasoning",
            event_bus=event_bus
        )
        
        self.relationship_models = RelationshipModels(
            module_id=f"{module_id}/relationship_models",
            event_bus=event_bus
        )
        
        # Track overall system development
        self.development_level = development_level
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input data through appropriate components
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing processing results
        """
        input_type = input_data.get("input_type", "")
        
        # Route input to appropriate component
        if "mental_state" in input_type or "theory_of_mind" in input_type:
            return self.theory_of_mind.process_input(input_data)
        elif "norm" in input_type or "social_rule" in input_type:
            return self.social_norms.process_input(input_data)
        elif "moral" in input_type or "ethical" in input_type:
            return self.moral_reasoning.process_input(input_data)
        elif "relationship" in input_type or "social_interaction" in input_type:
            return self.relationship_models.process_input(input_data)
        else:
            # For unspecified inputs, try each component in priority order
            # based on content keywords
            content = str(input_data).lower()
            
            if any(kw in content for kw in ["belief", "desire", "intention", "perspective"]):
                return self.theory_of_mind.process_input(input_data)
            elif any(kw in content for kw in ["norm", "rule", "convention", "violation"]):
                return self.social_norms.process_input(input_data)
            elif any(kw in content for kw in ["moral", "ethical", "right", "wrong", "good", "bad"]):
                return self.moral_reasoning.process_input(input_data)
            elif any(kw in content for kw in ["relationship", "friend", "interaction", "social"]):
                return self.relationship_models.process_input(input_data)
            
            # If no clear routing, return error
            return {
                "error": "Unrecognized input type",
                "module_id": self.module_id,
                "module_type": self.module_type
            }
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        # Update overall development level
        prev_level = self.development_level
        self.development_level = min(1.0, self.development_level + amount)
        
        # Update sub-modules with proportional development
        self.theory_of_mind.update_development(amount)
        self.social_norms.update_development(amount)
        self.moral_reasoning.update_development(amount)
        self.relationship_models.update_development(amount)
        
        # Log development milestone transitions
        if int(prev_level * 10) != int(self.development_level * 10):
            logger.info(f"Social system reached development level {self.development_level:.2f}")
            
        return self.development_level
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of this module
        
        Args:
            level: New development level (0.0 to 1.0)
        """
        # Set overall development level
        self.development_level = max(0.0, min(1.0, level))
        
        # Set sub-modules to the same level
        self.theory_of_mind.development_level = self.development_level
        self.social_norms.development_level = self.development_level
        self.moral_reasoning.development_level = self.development_level
        self.relationship_models.development_level = self.development_level
        
        logger.info(f"Social system development level set to {self.development_level:.2f}")
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the social system
        
        Returns:
            Dictionary containing system state
        """
        return {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "development_level": self.development_level,
            "theory_of_mind": self.theory_of_mind.get_state(),
            "social_norms": self.social_norms.get_state(),
            "moral_reasoning": self.moral_reasoning.get_state(),
            "relationship_models": self.relationship_models.get_state()
        }
    
    def save_state(self, state_dir: str) -> Dict[str, str]:
        """
        Save the state of all components
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Dictionary mapping component names to saved state paths
        """
        saved_paths = {}
        saved_paths["theory_of_mind"] = self.theory_of_mind.save_state(state_dir)
        saved_paths["social_norms"] = self.social_norms.save_state(state_dir)
        saved_paths["moral_reasoning"] = self.moral_reasoning.save_state(state_dir)
        saved_paths["relationship_models"] = self.relationship_models.save_state(state_dir)
        return saved_paths
    
    def load_state(self, state_paths: Dict[str, str]) -> Dict[str, bool]:
        """
        Load the state of all components
        
        Args:
            state_paths: Dictionary mapping component names to state file paths
            
        Returns:
            Dictionary mapping component names to load success status
        """
        load_status = {}
        if "theory_of_mind" in state_paths:
            load_status["theory_of_mind"] = self.theory_of_mind.load_state(state_paths["theory_of_mind"])
        if "social_norms" in state_paths:
            load_status["social_norms"] = self.social_norms.load_state(state_paths["social_norms"])
        if "moral_reasoning" in state_paths:
            load_status["moral_reasoning"] = self.moral_reasoning.load_state(state_paths["moral_reasoning"])
        if "relationship_models" in state_paths:
            load_status["relationship_models"] = self.relationship_models.load_state(state_paths["relationship_models"])
        
        # Recalculate overall development level
        self.development_level = (
            self.theory_of_mind.development_level + 
            self.social_norms.development_level + 
            self.moral_reasoning.development_level + 
            self.relationship_models.development_level
        ) / 4.0
        
        return load_status


def get_module(module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0) -> Any:
    """
    Factory function to create a social module
    
    This function is responsible for creating a social system that can:
    - Understand others' beliefs, intentions, and emotions
    - Learn and apply social norms and conventions
    - Make moral judgments about actions and situations
    - Model relationships and social dynamics
    - Adapt behavior to different social contexts
    
    Args:
        module_id: Unique identifier for the module
        event_bus: Event bus for communication with other modules
        development_level: Initial developmental level of the module
        
    Returns:
        An instance of the SocialSystem class
    """
    return SocialSystem(module_id=module_id, event_bus=event_bus, development_level=development_level)
