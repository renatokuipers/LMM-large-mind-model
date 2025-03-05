#!/usr/bin/env python3
"""
Custom component example for the NeuralChild framework.

This example demonstrates how to create a custom neural component
and add it to the Neural Child's mind.
"""

import os
import sys
import time
import random
import numpy as np
from typing import Dict, List, Any, Optional, Union

# Add the parent directory to the path so we can import NeuralChild
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from NeuralChild import NeuralChild
from NeuralChild.components.base import NeuralComponent, ComponentState
from NeuralChild.core.mother import MotherResponse, ChildInput


class SocialAwarenessComponent(NeuralComponent):
    """
    A custom component that handles social awareness and interaction.
    
    This component models the child's ability to understand social cues,
    develop theory of mind, and interact with others.
    """
    
    def __init__(self, config=None):
        """Initialize the social awareness component."""
        super().__init__(config or {})
        
        # Social awareness metrics
        self.social_awareness = 0.1  # Initial value
        self.theory_of_mind = 0.0    # Understanding that others have different thoughts/feelings
        self.social_relationships = {}  # Track relationships with others
        
        # Development params
        self.params = {
            'social_development_rate': config.get('social_development_rate', 0.5),
            'theory_of_mind_threshold': config.get('theory_of_mind_threshold', 120)  # Days until theory of mind begins
        }
    
    def process_mother_response(self, response: MotherResponse, child_age_days: int) -> Dict[str, Any]:
        """
        Process a response from the mother to update social awareness.
        
        Args:
            response: The mother's response
            child_age_days: Child's current age in days
            
        Returns:
            Dict with processed social information
        """
        # Extract social cues from mother's response
        social_cues = self._extract_social_cues(response)
        
        # Update social awareness based on interaction
        self._update_social_awareness(social_cues, child_age_days)
        
        # Update relationship with mother
        self._update_relationship('mother', response, child_age_days)
        
        # Process theory of mind development
        self._process_theory_of_mind(response, child_age_days)
        
        # Return processed social information
        return {
            'social_cues_detected': social_cues,
            'social_awareness': self.social_awareness,
            'theory_of_mind': self.theory_of_mind,
            'relationship_with_mother': self.social_relationships.get('mother', {})
        }
    
    def prepare_child_input(self, current_state: Dict[str, Any], child_age_days: int) -> Dict[str, Any]:
        """
        Contribute to the child's input based on social awareness.
        
        Args:
            current_state: Current internal state
            child_age_days: Child's current age in days
            
        Returns:
            Dict with social contribution to child input
        """
        # Add social awareness elements to the child's response
        social_elements = {}
        
        # Add social greeting if developed enough
        if self.social_awareness > 0.3 and random.random() < self.social_awareness:
            social_elements['greeting'] = self._generate_greeting(child_age_days)
        
        # Add social engagement elements based on development
        if self.social_awareness > 0.5:
            social_elements['engagement'] = self._generate_social_engagement(child_age_days)
        
        # If theory of mind is developing, add perspective elements
        if self.theory_of_mind > 0.2:
            social_elements['perspective_taking'] = self._generate_perspective_taking(child_age_days)
        
        return {
            'social_elements': social_elements,
            'social_awareness_level': self.social_awareness,
            'theory_of_mind_level': self.theory_of_mind
        }
    
    def update_state(self, internal_state: Dict[str, Any], external_input: Dict[str, Any], 
                     child_age_days: int) -> ComponentState:
        """
        Update component state based on internal state and external input.
        
        Args:
            internal_state: Complete internal state of the child
            external_input: External input (usually from mother)
            child_age_days: Child's current age in days
            
        Returns:
            Updated component state
        """
        # Natural development based on age
        self._natural_development(child_age_days)
        
        # Get emotional state from internal state
        emotion_state = internal_state.get('component_states', {}).get('Emotion', {})
        primary_emotion = emotion_state.get('primary_emotion', 'neutral')
        
        # Social awareness is influenced by emotional state
        if primary_emotion in ['joy', 'trust', 'anticipation']:
            self.social_awareness = min(1.0, self.social_awareness + 0.01)
        elif primary_emotion in ['fear', 'sadness']:
            self.social_awareness = max(0.1, self.social_awareness - 0.005)
        
        # Create component state
        state = ComponentState(
            activation=min(1.0, self.social_awareness * 2),  # Scale for activation
            confidence=min(child_age_days / 500, 0.9),  # Confidence increases with age
            social_awareness=self.social_awareness,
            theory_of_mind=self.theory_of_mind,
            relationships=self.social_relationships,
            current_social_focus=self._get_current_social_focus(internal_state)
        )
        
        return state
    
    def _extract_social_cues(self, response: MotherResponse) -> List[str]:
        """Extract social cues from mother's response."""
        social_cues = []
        
        # Extract from verbal response
        verbal = response.verbal_response or ""
        
        # Check for social greetings
        greeting_words = ["hello", "hi", "good morning", "good afternoon", "hey"]
        if any(word in verbal.lower() for word in greeting_words):
            social_cues.append("greeting")
        
        # Check for questions (social engagement)
        if "?" in verbal:
            social_cues.append("question")
        
        # Check for social praise
        praise_words = ["good", "great", "wonderful", "excellent", "amazing"]
        if any(word in verbal.lower() for word in praise_words):
            social_cues.append("praise")
        
        # Extract from emotional state
        if response.emotional_state:
            emotion = response.emotional_state.get("primary_emotion")
            if emotion:
                social_cues.append(f"emotional:{emotion}")
        
        # Extract from non-verbal cues
        if response.non_verbal_cues:
            for cue in response.non_verbal_cues:
                social_cues.append(f"non_verbal:{cue}")
        
        return social_cues
    
    def _update_social_awareness(self, social_cues: List[str], child_age_days: int):
        """Update social awareness based on detected cues."""
        # Base development rate depends on age and configured rate
        base_rate = self.params['social_development_rate'] * (0.1 + min(child_age_days / 1000, 0.9))
        
        # Each social cue contributes to development
        cue_contribution = 0.01 * len(social_cues)
        
        # Update social awareness
        self.social_awareness = min(1.0, self.social_awareness + (base_rate * cue_contribution))
    
    def _update_relationship(self, person: str, response: MotherResponse, child_age_days: int):
        """Update relationship with a specific person."""
        # Initialize relationship if not exists
        if person not in self.social_relationships:
            self.social_relationships[person] = {
                'familiarity': 0.5,  # For mother, start with some familiarity
                'trust': 0.5,        # Initial trust
                'attachment': 0.3,   # Initial attachment
                'interactions': 0    # Count of interactions
            }
        
        # Get the relationship
        relationship = self.social_relationships[person]
        
        # Increment interaction count
        relationship['interactions'] += 1
        
        # Update familiarity (increases with interactions)
        relationship['familiarity'] = min(1.0, relationship['familiarity'] + 0.01)
        
        # Update trust based on emotional state
        if hasattr(response, 'emotional_state') and response.emotional_state:
            emotion = response.emotional_state.get("primary_emotion")
            if emotion in ["joy", "trust", "anticipation"]:
                relationship['trust'] = min(1.0, relationship['trust'] + 0.02)
            elif emotion in ["anger", "disgust"]:
                relationship['trust'] = max(0.1, relationship['trust'] - 0.03)
        
        # Update attachment based on age and interaction quality
        attachment_growth = 0.005  # Base growth
        if "praise" in self._extract_social_cues(response):
            attachment_growth *= 2  # Praise strengthens attachment
        
        relationship['attachment'] = min(1.0, relationship['attachment'] + attachment_growth)
    
    def _process_theory_of_mind(self, response: MotherResponse, child_age_days: int):
        """Process theory of mind development."""
        # Theory of mind starts developing after a certain age
        if child_age_days < self.params['theory_of_mind_threshold']:
            return
        
        # Base development rate
        base_rate = 0.0005  # Very slow development
        
        # Teaching about others' feelings accelerates development
        verbal = response.verbal_response or ""
        teaching_keywords = ["feel", "think", "want", "need", "believe", "know"]
        
        if any(word in verbal.lower() for word in teaching_keywords):
            base_rate *= 2
        
        # Update theory of mind
        self.theory_of_mind = min(1.0, self.theory_of_mind + base_rate)
    
    def _natural_development(self, child_age_days: int):
        """Natural development of social awareness based on age."""
        # Very slow natural development
        natural_growth = 0.0001 * min(child_age_days / 365, 1.0)
        self.social_awareness = min(1.0, self.social_awareness + natural_growth)
    
    def _get_current_social_focus(self, internal_state: Dict[str, Any]) -> Dict[str, Any]:
        """Determine the current social focus."""
        # Default focus
        focus = {"type": "none", "target": None, "intensity": 0.0}
        
        # If highly activated, focus on the mother
        if self.social_awareness > 0.3:
            focus = {
                "type": "person",
                "target": "mother",
                "intensity": self.social_awareness,
                "relationship": self.social_relationships.get("mother", {})
            }
        
        return focus
    
    def _generate_greeting(self, child_age_days: int) -> str:
        """Generate a social greeting appropriate for the child's age."""
        if child_age_days < 180:  # Less than 6 months
            return "ba"  # Simple vocalization
        elif child_age_days < 365:  # Less than 1 year
            return "mama"  # Simple word
        elif child_age_days < 730:  # Less than 2 years
            return "hi mama"  # Simple phrase
        else:
            greetings = ["hi", "hello", "hi mama", "hello mama"]
            return random.choice(greetings)
    
    def _generate_social_engagement(self, child_age_days: int) -> Dict[str, Any]:
        """Generate social engagement elements appropriate for age."""
        engagement = {"type": "basic"}
        
        if child_age_days < 365:  # Less than 1 year
            engagement["behavior"] = random.choice(["reach", "smile", "babble"])
        elif child_age_days < 730:  # Less than 2 years
            engagement["behavior"] = random.choice(["point", "show toy", "request"])
        elif child_age_days < 1095:  # Less than 3 years
            engagement["behavior"] = random.choice(["simple question", "share", "play together"])
        else:
            engagement["behavior"] = random.choice(["question", "conversation", "joint activity"])
        
        return engagement
    
    def _generate_perspective_taking(self, child_age_days: int) -> Dict[str, Any]:
        """Generate perspective-taking elements based on theory of mind."""
        perspective = {"level": "basic"}
        
        # Scale complexity with theory of mind development
        if self.theory_of_mind < 0.3:
            perspective["content"] = "notice emotion"
        elif self.theory_of_mind < 0.5:
            perspective["content"] = "basic empathy"
        elif self.theory_of_mind < 0.7:
            perspective["content"] = "understand desires"
        else:
            perspective["content"] = "understand beliefs"
        
        return perspective


def run_custom_component_example():
    """Run the custom component example."""
    print("=" * 50)
    print("NeuralChild Custom Component Example")
    print("=" * 50)
    
    # Create a neural child
    child = NeuralChild()
    
    # Create a custom social awareness component
    social_component = SocialAwarenessComponent()
    
    # Add the component to the child's mind
    child.add_component("SocialAwareness", social_component)
    
    # Perform some interactions to demonstrate the component
    print("Performing 10 interactions with the new SocialAwareness component...\n")
    
    for i in range(10):
        print(f"Interaction {i+1}:")
        
        # Interact with mother
        result = child.interact_with_mother()
        
        # Get social component state
        component_states = child.get_component_states()
        social_state = component_states.get("SocialAwareness", {})
        
        # Print social awareness metrics
        print(f"Social awareness: {social_state.get('social_awareness', 0):.2f}")
        print(f"Theory of mind: {social_state.get('theory_of_mind', 0):.2f}")
        
        # Print mother-child interaction
        child_input = result.get('child_input', {})
        mother_response = result.get('mother_response', {})
        
        child_content = child_input.get('content', '')
        mother_content = mother_response.get('verbal_response', '')
        
        if child_content:
            print(f"Child: {child_content}")
        
        if mother_content:
            print(f"Mother: {mother_content}")
        
        print()
    
    print("=" * 50)
    print("Custom Component Example Complete!")
    print("=" * 50)


if __name__ == "__main__":
    run_custom_component_example()