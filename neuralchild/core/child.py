"""
Child Module

This module defines the Child class, which is the central component of the NeuralChild framework.
It represents the neural child's mind and manages all its psychological components.
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

import numpy as np

from ..utils.data_types import (
    ChildState, ChildResponse, MotherResponse, Emotion, 
    EmotionType, Word, EpisodicMemory, SemanticMemory, MemoryType,
    DevelopmentalStage, DevelopmentalSubstage, ComponentState,
    StageTransition, get_substage_from_age, get_stage_from_substage,
    STAGE_TO_SUBSTAGES
)
from ..utils.component_integration import ComponentIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Child:
    """
    The Child class represents the neural child's mind.
    
    It integrates all psychological components and manages the child's overall state,
    including developmental stage, memories, emotions, and language capabilities.
    """
    
    def __init__(self, initial_state: Optional[ChildState] = None):
        """
        Initialize the Child's mind.
        
        Args:
            initial_state: Optional starting state for the child
        """
        # Initialize state
        self.state = initial_state if initial_state else ChildState()
        
        # Component registry
        self.components = {}
        
        # Component integration system
        self.integration = ComponentIntegration()
        
        # Initialize base emotions if needed
        if not self.state.current_emotional_state:
            self._init_base_emotions()
            
        logging.info(f"Child initialized at {self.state.developmental_stage.value} stage")
    
    def _init_base_emotions(self):
        """Initialize the basic emotions that are present from birth."""
        # Even newborns have basic emotions like distress, contentment
        basic_emotions = [
            Emotion(type=EmotionType.JOY, intensity=0.3),
            Emotion(type=EmotionType.SADNESS, intensity=0.3),
            Emotion(type=EmotionType.FEAR, intensity=0.2),
            Emotion(type=EmotionType.SURPRISE, intensity=0.4),
        ]
        self.state.current_emotional_state = basic_emotions
    
    def register_component(self, component_id: str, component_type: str, component: Any):
        """
        Register a neural component with the child's mind.
        
        Args:
            component_id: Unique identifier for the component
            component_type: Type of component (memory, language, etc.)
            component: The component object
        """
        # Add component to registry
        self.components[component_id] = component
        
        # Create component state
        component_state = ComponentState(
            component_id=component_id,
            component_type=component_type,
            activation_level=0.5,
            confidence=0.5,
            last_updated=datetime.now()
        )
        
        # Add to child state
        self.state.component_states[component_id] = component_state
        
        # Get initial metrics from component if available
        initial_metrics = {}
        if hasattr(component, 'get_development_metrics'):
            initial_metrics = component.get_development_metrics()
        elif hasattr(component, 'get_cognitive_metrics'):
            initial_metrics = component.get_cognitive_metrics()
        elif hasattr(component, 'get_social_metrics'):
            initial_metrics = component.get_social_metrics()
        elif hasattr(component, 'get_consciousness_metrics'):
            initial_metrics = component.get_consciousness_metrics()
        elif hasattr(component, 'get_memory_stats'):
            initial_metrics = component.get_memory_stats()
            
        # Register with integration system
        self.integration.register_component_state(component_id, component_type, initial_metrics)
        
        logging.info(f"Registered component: {component_id} ({component_type})")
    
    def process_mother_response(self, response: MotherResponse) -> ChildResponse:
        """
        Process a response from the Mother component and generate a child response.
        
        This method updates the child's state based on the mother's response,
        incorporating the effects on language, emotions, memories, etc.
        
        Args:
            response: The mother's response object
            
        Returns:
            The child's response object
        """
        # Process the response based on developmental stage and substage
        stage = self.state.developmental_stage
        substage = self.state.developmental_substage
        
        # Check if we're in a transition between stages
        in_transition = self.state.stage_transition is not None
        
        # Process emotions (all stages)
        self._process_emotions(response)
        
        # Process language (stage-appropriate)
        if stage == DevelopmentalStage.INFANCY:
            self._process_early_language(response)
        else:
            self._process_language(response)
        
        # Process memory (stage-appropriate)
        if stage in [DevelopmentalStage.INFANCY, DevelopmentalStage.EARLY_CHILDHOOD]:
            self._form_simple_memory(response)
        else:
            self._form_complex_memories(response)
        
        # Process abstract concepts (more advanced stages)
        if stage in [DevelopmentalStage.ADOLESCENCE, DevelopmentalStage.EARLY_ADULTHOOD]:
            self._process_abstract_concepts(response)
            
        # Apply cross-component integration effects
        active_component = None
        # Determine which component is most relevant to this interaction
        if "consciousness" in response.text.lower():
            active_component = "consciousness_component"
        elif any(emotion.type == EmotionType.ANGER or emotion.type == EmotionType.SADNESS for emotion in response.emotional_state):
            active_component = "emotional_component"
        elif "remember" in response.text.lower() or "memory" in response.text.lower():
            active_component = "memory_system"
        elif "think" in response.text.lower() or "problem" in response.text.lower():
            active_component = "cognitive_component"
        elif "friend" in response.text.lower() or "people" in response.text.lower():
            active_component = "social_component"
        
        # Apply integration effects
        integration_effects = self.integration.apply_cross_component_effects(
            developmental_stage=stage,
            active_component_id=active_component
        )
        
        # Update component states with integration effects
        for component_id, effects in integration_effects.items():
            if component_id in self.components and effects:
                # Sum up influence factors
                total_influence = sum(effects.values())
                
                # Update component state in child's state
                if component_id in self.state.component_states:
                    self.state.component_states[component_id].activation_level = min(1.0, 0.5 + total_influence * 0.3)
                    self.state.component_states[component_id].last_updated = datetime.now()
        
        # Generate response based on stage and substage
        # If in transition, blend characteristics of current and next stage
        child_response = None
        if in_transition:
            transition = self.state.stage_transition
            progress = transition.transition_progress
            
            # Blend responses if transitioning between main stages
            if transition.current_stage != transition.next_stage:
                # Get responses from both current and next stage
                if stage == DevelopmentalStage.INFANCY:
                    current_response = self._generate_infant_response(response)
                    next_response = self._generate_early_childhood_response(response)
                elif stage == DevelopmentalStage.EARLY_CHILDHOOD:
                    current_response = self._generate_early_childhood_response(response)
                    next_response = self._generate_middle_childhood_response(response)
                elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
                    current_response = self._generate_middle_childhood_response(response)
                    next_response = self._generate_adolescent_response(response)
                elif stage == DevelopmentalStage.ADOLESCENCE:
                    current_response = self._generate_adolescent_response(response)
                    next_response = self._generate_adult_response(response)
                else:  # EARLY_ADULTHOOD
                    current_response = self._generate_adult_response(response)
                    next_response = current_response  # No further stage
                
                # Decide which response to use based on transition progress
                if np.random.random() < progress:
                    # Use next stage response more often as transition progresses
                    child_response = next_response
                else:
                    child_response = current_response
            else:
                # Default to current stage response for substage transitions
                if stage == DevelopmentalStage.INFANCY:
                    child_response = self._generate_infant_response(response)
                elif stage == DevelopmentalStage.EARLY_CHILDHOOD:
                    child_response = self._generate_early_childhood_response(response)
                elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
                    child_response = self._generate_middle_childhood_response(response)
                elif stage == DevelopmentalStage.ADOLESCENCE:
                    child_response = self._generate_adolescent_response(response)
                else:  # EARLY_ADULTHOOD
                    child_response = self._generate_adult_response(response)
        else:
            # Normal response generation based on current stage
            if stage == DevelopmentalStage.INFANCY:
                child_response = self._generate_infant_response(response)
            elif stage == DevelopmentalStage.EARLY_CHILDHOOD:
                child_response = self._generate_early_childhood_response(response)
            elif stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
                child_response = self._generate_middle_childhood_response(response)
            elif stage == DevelopmentalStage.ADOLESCENCE:
                child_response = self._generate_adolescent_response(response)
            else:  # EARLY_ADULTHOOD
                child_response = self._generate_adult_response(response)
        
        # Update developmental metrics
        self.update_developmental_metrics()
        
        # Check for stage progression
        self.check_stage_progression()
        
        # Update component states in integration system
        for component_id, component in self.components.items():
            updated_metrics = {}
            if hasattr(component, 'get_development_metrics'):
                updated_metrics = component.get_development_metrics()
            elif hasattr(component, 'get_cognitive_metrics'):
                updated_metrics = component.get_cognitive_metrics()
            elif hasattr(component, 'get_social_metrics'):
                updated_metrics = component.get_social_metrics()
            elif hasattr(component, 'get_consciousness_metrics'):
                updated_metrics = component.get_consciousness_metrics()
            elif hasattr(component, 'get_memory_stats'):
                updated_metrics = component.get_memory_stats()
                
            if updated_metrics:
                self.integration.update_component_state(component_id, updated_metrics)
        
        return child_response
    
    def _process_emotions(self, response: MotherResponse):
        """
        Process emotional content from the Mother's response.
        
        This includes emotional contagion and regulation depending on developmental stage.
        
        Args:
            response: The Mother's response containing emotional content
        """
        # Emotional contagion - child's emotions are influenced by Mother's
        # The degree of influence decreases with age/development
        contagion_factor = 0.8  # High in infancy
        if self.state.developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            contagion_factor = 0.6
        elif self.state.developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            contagion_factor = 0.4
        elif self.state.developmental_stage == DevelopmentalStage.ADOLESCENCE:
            contagion_factor = 0.2
        elif self.state.developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
            contagion_factor = 0.1
        
        # Apply emotional contagion
        new_emotions = []
        for mother_emotion in response.emotional_state:
            # Check if child already has this emotion
            existing = next((e for e in self.state.current_emotional_state 
                            if e.type == mother_emotion.type), None)
            
            if existing:
                # Blend existing with mother's emotion
                new_intensity = (existing.intensity + 
                                (mother_emotion.intensity * contagion_factor)) / 2
                new_emotions.append(Emotion(
                    type=existing.type,
                    intensity=min(1.0, new_intensity),
                    cause=f"Response from Mother: {response.text[:50]}..."
                ))
            else:
                # Add new emotion from mother (at reduced intensity)
                new_emotions.append(Emotion(
                    type=mother_emotion.type,
                    intensity=mother_emotion.intensity * contagion_factor,
                    cause=f"Response from Mother: {response.text[:50]}..."
                ))
        
        # Add emotions that weren't in mother's response (with decay)
        for child_emotion in self.state.current_emotional_state:
            if not any(e.type == child_emotion.type for e in new_emotions):
                # Emotions decay over time if not reinforced
                decay_rate = 0.2  # Base rate
                if self.state.developmental_stage >= DevelopmentalStage.MIDDLE_CHILDHOOD:
                    # Better emotional regulation with age
                    decay_rate = 0.3
                
                new_intensity = child_emotion.intensity * (1 - decay_rate)
                if new_intensity > 0.1:  # Only keep emotions above threshold
                    new_emotions.append(Emotion(
                        type=child_emotion.type,
                        intensity=new_intensity,
                        cause=child_emotion.cause
                    ))
        
        self.state.current_emotional_state = new_emotions
        
        # Update emotional regulation metric
        if self.state.developmental_stage >= DevelopmentalStage.EARLY_CHILDHOOD:
            # Measure how well child regulates emotions (less contagion = better regulation)
            self.state.metrics.emotional_regulation = min(
                1.0, self.state.metrics.emotional_regulation + 0.01
            )
    
    def _process_early_language(self, response: MotherResponse):
        """
        Process language at early childhood stage.
        
        At this stage, the child is building vocabulary but has limited grammar.
        
        Args:
            response: The Mother's response to process
        """
        # Simple word extraction and association
        words = response.text.lower().split()
        
        # Process only a few words (limited attention span)
        attention_span = min(5, len(words))
        for word in words[:attention_span]:
            # Clean the word of punctuation
            word = ''.join(c for c in word if c.isalnum())
            if not word:
                continue
                
            # Associate word with current emotional state
            emotional_context = {
                str(emotion.type): emotion.intensity 
                for emotion in self.state.current_emotional_state
            }
            
            if word in self.state.vocabulary:
                # Reinforce existing word
                existing_word = self.state.vocabulary[word]
                existing_word.usage_count += 1
                existing_word.last_used = datetime.now()
                existing_word.understanding_level = min(
                    0.7,  # Cap for early childhood
                    existing_word.understanding_level + 0.05
                )
                
                # Update associations
                for emotion_type, intensity in emotional_context.items():
                    if emotion_type in existing_word.associations:
                        # Blend existing association with new context
                        existing_word.associations[emotion_type] = (
                            existing_word.associations[emotion_type] * 0.8 +
                            intensity * 0.2
                        )
                    else:
                        existing_word.associations[emotion_type] = intensity
            else:
                # Learn new word
                self.state.vocabulary[word] = Word(
                    word=word,
                    associations=emotional_context,
                    understanding_level=0.2  # Basic understanding
                )
        
        # Update vocabulary size metric
        self.state.metrics.vocabulary_size = len(self.state.vocabulary)
    
    def _process_language(self, response: MotherResponse):
        """
        Process language at middle childhood and beyond.
        
        At these stages, the child understands grammar and more complex language.
        
        Args:
            response: The Mother's response to process
        """
        # More sophisticated language processing
        # This would involve:
        # 1. Parsing grammar
        # 2. Understanding context
        # 3. Detecting sentiment
        # 4. Learning new concepts
        
        # For simplicity, we'll focus on vocabulary building with deeper understanding
        words = response.text.lower().split()
        
        # Process more words (better attention span)
        for word in words:
            # Clean the word of punctuation
            word = ''.join(c for c in word if c.isalnum())
            if not word:
                continue
                
            # Associate word with current emotional state
            emotional_context = {
                str(emotion.type): emotion.intensity 
                for emotion in self.state.current_emotional_state
            }
            
            if word in self.state.vocabulary:
                # Reinforce existing word with deeper understanding
                existing_word = self.state.vocabulary[word]
                existing_word.usage_count += 1
                existing_word.last_used = datetime.now()
                
                # Understanding improves with development stage
                if self.state.developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
                    max_understanding = 0.85
                elif self.state.developmental_stage == DevelopmentalStage.ADOLESCENCE:
                    max_understanding = 0.95
                else:  # EARLY_ADULTHOOD
                    max_understanding = 1.0
                
                existing_word.understanding_level = min(
                    max_understanding,
                    existing_word.understanding_level + 0.02
                )
                
                # Update associations (more stable at advanced stages)
                for emotion_type, intensity in emotional_context.items():
                    if emotion_type in existing_word.associations:
                        # More stable associations with age
                        existing_factor = 0.9  # Existing associations have more weight
                        existing_word.associations[emotion_type] = (
                            existing_word.associations[emotion_type] * existing_factor +
                            intensity * (1 - existing_factor)
                        )
                    else:
                        existing_word.associations[emotion_type] = intensity
            else:
                # Learn new word with higher initial understanding
                initial_understanding = 0.4  # Better than early childhood
                if self.state.developmental_stage == DevelopmentalStage.ADOLESCENCE:
                    initial_understanding = 0.5
                elif self.state.developmental_stage == DevelopmentalStage.EARLY_ADULTHOOD:
                    initial_understanding = 0.6
                
                self.state.vocabulary[word] = Word(
                    word=word,
                    associations=emotional_context,
                    understanding_level=initial_understanding
                )
        
        # Update vocabulary size metric
        self.state.metrics.vocabulary_size = len(self.state.vocabulary)
        
        # Update grammatical complexity metric (simplified)
        # In a real implementation, this would involve actual grammar analysis
        if len(words) > 0:
            unique_words = len(set(words))
            sentence_length = len(words)
            complexity_factor = unique_words / sentence_length
            
            # Update grammatical complexity (slow improvement)
            self.state.metrics.grammatical_complexity = min(
                1.0, 
                self.state.metrics.grammatical_complexity + (complexity_factor * 0.01)
            )
    
    def _form_simple_memory(self, response: MotherResponse):
        """
        Form a simple episodic memory from the interaction.
        
        Args:
            response: The Mother's response to remember
        """
        # Calculate emotional valence (positive/negative) of the memory
        valence = 0.0
        for emotion in self.state.current_emotional_state:
            if emotion.type in [EmotionType.JOY, EmotionType.TRUST]:
                valence += emotion.intensity
            elif emotion.type in [EmotionType.SADNESS, EmotionType.FEAR, EmotionType.ANGER]:
                valence -= emotion.intensity
        
        # Normalize valence to [-1, 1]
        valence = max(-1.0, min(1.0, valence))
        
        # Create the memory
        memory_id = str(uuid.uuid4())
        memory = EpisodicMemory(
            id=memory_id,
            type=MemoryType.EPISODIC,
            event_description=f"Mother said: {response.text[:100]}...",
            emotional_valence=valence,
            associated_emotions=self.state.current_emotional_state.copy(),
            context={"interaction_type": "verbal"}
        )
        
        self.state.episodic_memories[memory_id] = memory
    
    def _form_complex_memories(self, response: MotherResponse):
        """
        Form more complex memories including both episodic and semantic.
        
        Args:
            response: The Mother's response to process into memories
        """
        # Form episodic memory
        self._form_simple_memory(response)
        
        # Form semantic memories from teaching elements
        for concept, details in response.teaching_elements.items():
            if isinstance(details, str):
                # Create or update semantic memory
                existing_memory = next(
                    (mem for mem in self.state.semantic_memories.values() if mem.concept == concept),
                    None
                )
                
                if existing_memory:
                    # Update existing memory
                    existing_memory.definition = details
                    existing_memory.confidence = min(1.0, existing_memory.confidence + 0.1)
                    existing_memory.last_accessed = datetime.now()
                    existing_memory.strength = min(1.0, existing_memory.strength + 0.1)
                else:
                    # Create new semantic memory
                    memory_id = str(uuid.uuid4())
                    memory = SemanticMemory(
                        id=memory_id,
                        type=MemoryType.SEMANTIC,
                        concept=concept,
                        definition=details,
                        confidence=0.7,  # Higher initial confidence for taught concepts
                    )
                    self.state.semantic_memories[memory_id] = memory
    
    def _process_abstract_concepts(self, response: MotherResponse):
        """
        Process abstract concepts in the Mother's response.
        
        This capability develops in adolescence and adulthood.
        
        Args:
            response: The Mother's response to process
        """
        # This would involve more sophisticated analysis of abstract ideas
        # For now, just update the abstract thinking metric
        if "teaching_elements" in response.__dict__ and response.teaching_elements:
            abstract_concepts = 0
            for concept in response.teaching_elements:
                # Consider concepts like "justice", "freedom", etc. as abstract
                abstract_words = ["justice", "freedom", "truth", "beauty", "ethics", 
                                 "morality", "philosophy", "democracy", "consciousness", 
                                 "theory", "concept", "abstract", "principle"]
                if any(word in concept.lower() for word in abstract_words):
                    abstract_concepts += 1
            
            # Update abstract thinking metric
            if abstract_concepts > 0:
                improvement = 0.01 * abstract_concepts
                self.state.metrics.abstract_thinking = min(
                    1.0, self.state.metrics.abstract_thinking + improvement
                )
    
    def _generate_infant_response(self, mother_response: MotherResponse) -> ChildResponse:
        """
        Generate a response appropriate for the infancy stage.
        
        At this stage, responses are pre-linguistic vocalizations rather than words.
        
        Args:
            mother_response: The Mother's response to react to
            
        Returns:
            The Child's response
        """
        # Infants respond with vocalizations, not words
        # The type of vocalization depends on emotional state
        
        # Get dominant emotion
        dominant_emotion = None
        max_intensity = 0
        for emotion in self.state.current_emotional_state:
            if emotion.intensity > max_intensity:
                dominant_emotion = emotion
                max_intensity = emotion.intensity
        
        # Generate vocalization based on dominant emotion
        vocalization = "goo"  # Default neutral sound
        if dominant_emotion:
            if dominant_emotion.type == EmotionType.JOY:
                vocalization = "goo-goo, gah" if np.random.random() < 0.5 else "hehe"
            elif dominant_emotion.type == EmotionType.SADNESS:
                vocalization = "waaah" if np.random.random() < 0.7 else "wah-wah"
            elif dominant_emotion.type == EmotionType.FEAR:
                vocalization = "waaaah!" if np.random.random() < 0.8 else "hic-hic"
            elif dominant_emotion.type == EmotionType.ANGER:
                vocalization = "aah!" if np.random.random() < 0.6 else "grr"
            elif dominant_emotion.type == EmotionType.SURPRISE:
                vocalization = "ooh!" if np.random.random() < 0.5 else "ah!"
        
        # Create simple memory of the interaction
        self._form_simple_memory(mother_response)
        
        return ChildResponse(
            vocalization=vocalization,
            emotional_state=self.state.current_emotional_state,
            attention_focus="Mother's face" if np.random.random() < 0.7 else "surrounding environment"
        )
    
    def _generate_early_childhood_response(self, mother_response: MotherResponse) -> ChildResponse:
        """
        Generate a response appropriate for the early childhood stage.
        
        At this stage, responses are simple words or short phrases.
        
        Args:
            mother_response: The Mother's response to react to
            
        Returns:
            The Child's response
        """
        # Early childhood responses are simple words or short phrases
        # Generate response based on vocabulary and emotional state
        
        # Get words with highest understanding that match emotional context
        usable_words = []
        emotional_context = {e.type: e.intensity for e in self.state.current_emotional_state}
        
        for word_obj in self.state.vocabulary.values():
            # Use words with understanding level above threshold
            if word_obj.understanding_level >= 0.3:
                # Calculate emotional relevance
                relevance = 0
                for emotion_type, intensity in emotional_context.items():
                    if str(emotion_type) in word_obj.associations:
                        relevance += intensity * word_obj.associations[str(emotion_type)]
                
                # Add word to usable list with its relevance score
                usable_words.append((word_obj.word, relevance, word_obj.understanding_level))
        
        # Sort by combined score of relevance and understanding
        usable_words.sort(key=lambda x: x[1] * x[2], reverse=True)
        
        # Generate response of 1-3 words
        response_length = min(3, len(usable_words))
        if response_length == 0:
            # Fall back to vocalization if no usable words
            return self._generate_infant_response(mother_response)
        
        # Randomly select between 1 and response_length words
        actual_length = np.random.randint(1, response_length + 1)
        selected_words = [pair[0] for pair in usable_words[:actual_length]]
        
        # Early childhood grammar is limited - just concatenate words
        response_text = " ".join(selected_words)
        
        return ChildResponse(
            text=response_text,
            emotional_state=self.state.current_emotional_state,
            attention_focus="Mother" if np.random.random() < 0.6 else "objects of interest"
        )
    
    def _generate_middle_childhood_response(self, mother_response: MotherResponse) -> ChildResponse:
        """
        Generate a response appropriate for the middle childhood stage.
        
        At this stage, responses use proper grammar but limited complexity.
        
        Args:
            mother_response: The Mother's response to react to
            
        Returns:
            The Child's response
        """
        # Middle childhood responses use proper grammar but limited complexity
        # This would involve a more sophisticated language generation model
        
        # For the prototype, we'll simulate better responses with templates
        templates = [
            "I think {noun} {verb}.",
            "Can you tell me about {noun}?",
            "I {verb} the {noun}.",
            "Why does {noun} {verb}?",
            "I feel {emotion} about {noun}.",
            "I want to {verb}.",
            "That's interesting about {noun}."
        ]
        
        # Get suitable vocabulary
        nouns = []
        verbs = []
        for word in self.state.vocabulary.values():
            # Only use words with decent understanding
            if word.understanding_level >= 0.5:
                # Very simplified part-of-speech determination
                if word.word.endswith("ing") or word.word.endswith("ed"):
                    verbs.append(word.word)
                else:
                    nouns.append(word.word)
        
        # Get emotional words
        emotions = []
        for emotion in self.state.current_emotional_state:
            if emotion.intensity > 0.5:
                if emotion.type == EmotionType.JOY:
                    emotions.append("happy")
                elif emotion.type == EmotionType.SADNESS:
                    emotions.append("sad")
                elif emotion.type == EmotionType.FEAR:
                    emotions.append("scared")
                elif emotion.type == EmotionType.ANGER:
                    emotions.append("angry")
                elif emotion.type == EmotionType.SURPRISE:
                    emotions.append("surprised")
        
        # Default fallbacks
        if not nouns:
            nouns = ["things", "stuff", "that"]
        if not verbs:
            verbs = ["like", "want", "see"]
        if not emotions:
            emotions = ["okay", "fine", "good"]
        
        # Select template and fill in
        template = np.random.choice(templates)
        
        response_text = template.format(
            noun=np.random.choice(nouns),
            verb=np.random.choice(verbs),
            emotion=np.random.choice(emotions)
        )
        
        return ChildResponse(
            text=response_text,
            emotional_state=self.state.current_emotional_state,
            attention_focus="conversation"
        )
    
    def _generate_adolescent_response(self, mother_response: MotherResponse) -> ChildResponse:
        """
        Generate a response appropriate for the adolescence stage.
        
        At this stage, responses show abstract thinking and complex emotions.
        
        Args:
            mother_response: The Mother's response to react to
            
        Returns:
            The Child's response
        """
        # Adolescent responses show more complex language and abstract thinking
        # This is essentially a more sophisticated version of the middle childhood response
        
        # Simplified implementation for prototype
        templates = [
            "I've been thinking about {abstract_concept} lately, and I believe {opinion}.",
            "Sometimes I wonder why {abstract_concept} is so {adjective}.",
            "Do you think that {opinion} about {abstract_concept}?",
            "I feel {complex_emotion} when I consider {abstract_concept}.",
            "I'm not sure if I agree that {opinion}.",
            "From my perspective, {opinion} because {abstract_concept} is {adjective}.",
            "I've noticed that {opinion}, which makes me feel {complex_emotion}."
        ]
        
        # Abstract concepts (would come from semantic memories in full implementation)
        abstract_concepts = [
            "friendship", "identity", "future", "meaning", "society", 
            "justice", "freedom", "relationships", "belonging"
        ]
        
        # Opinions (simplified)
        opinions = [
            "people should be more honest with each other",
            "it's important to understand different perspectives",
            "personal growth comes from facing challenges",
            "finding your own path is essential",
            "nothing is absolutely certain",
            "everyone has their own truth",
            "relationships require compromise"
        ]
        
        # Adjectives
        adjectives = ["complex", "interesting", "challenging", "important", 
                     "confusing", "meaningful", "frustrating", "inspiring"]
        
        # Complex emotions
        complex_emotions = ["contemplative", "ambivalent", "melancholic", 
                           "hopeful", "anxious", "curious", "conflicted"]
        
        # Select template and fill in
        template = np.random.choice(templates)
        
        response_text = template.format(
            abstract_concept=np.random.choice(abstract_concepts),
            opinion=np.random.choice(opinions),
            adjective=np.random.choice(adjectives),
            complex_emotion=np.random.choice(complex_emotions)
        )
        
        return ChildResponse(
            text=response_text,
            emotional_state=self.state.current_emotional_state,
            attention_focus="deep thoughts" if np.random.random() < 0.4 else "conversation"
        )
    
    def _generate_adult_response(self, mother_response: MotherResponse) -> ChildResponse:
        """
        Generate a response appropriate for the early adulthood stage.
        
        At this stage, responses show fully mature language and thought processes.
        
        Args:
            mother_response: The Mother's response to react to
            
        Returns:
            The Child's response
        """
        # Adult responses show fully mature language and thought
        # Similar to adolescent but with more sophistication and nuance
        
        # In a full implementation, this would involve more advanced NLP
        # For now, we'll use a template approach with more complex templates
        
        templates = [
            "I've been reflecting on {abstract_concept1} in relation to {abstract_concept2}, and I've concluded that {complex_opinion}.",
            "There's an interesting parallel between {abstract_concept1} and {abstract_concept2} that suggests {complex_opinion}.",
            "From my experience, {complex_opinion}, especially when considering the nature of {abstract_concept1}.",
            "I find myself {complex_emotion} about {abstract_concept1}, perhaps because {complex_opinion}.",
            "What are your thoughts on the idea that {complex_opinion}? I've been considering this in terms of {abstract_concept1}.",
            "The interplay between {abstract_concept1} and {abstract_concept2} creates a tension that makes me feel {complex_emotion}.",
            "I've developed a perspective that {complex_opinion}, which has helped me understand {abstract_concept1} better."
        ]
        
        # Abstract concepts (more sophisticated than adolescent)
        abstract_concepts = [
            "philosophical ethics", "cultural identity", "societal structures", 
            "human consciousness", "existential meaning", "theoretical frameworks",
            "comparative worldviews", "emotional intelligence", "cognitive biases",
            "interpersonal dynamics", "psychological development", "moral philosophy"
        ]
        
        # Complex opinions
        complex_opinions = [
            "true understanding requires both emotional and intellectual engagement",
            "personal meaning is constructed through our interactions with others and our environment",
            "flexibility in thinking allows for greater adaptation to life's uncertainties",
            "authentic connections require vulnerability and acceptance of imperfection",
            "balancing individual needs with collective responsibility creates sustainable progress",
            "reflection on our cognitive processes enhances our decision-making capabilities",
            "integrating diverse perspectives leads to more robust understanding of complex issues"
        ]
        
        # Complex emotions
        complex_emotions = [
            "philosophically curious", "empathetically attuned", "reflectively engaged",
            "cautiously optimistic", "pragmatically hopeful", "intellectually stimulated",
            "contemplatively peaceful", "constructively critical", "mindfully present"
        ]
        
        # Select template and fill in
        template = np.random.choice(templates)
        
        # Get two different abstract concepts
        concept1, concept2 = np.random.choice(abstract_concepts, size=2, replace=False)
        
        response_text = template.format(
            abstract_concept1=concept1,
            abstract_concept2=concept2,
            complex_opinion=np.random.choice(complex_opinions),
            complex_emotion=np.random.choice(complex_emotions)
        )
        
        return ChildResponse(
            text=response_text,
            emotional_state=self.state.current_emotional_state,
            attention_focus="deep analysis" if np.random.random() < 0.6 else "meaningful conversation"
        )
    
    def update_developmental_metrics(self):
        """Update the child's developmental metrics based on current state."""
        # Update metrics
        metrics = self.state.metrics
        
        # Update substage based on age
        expected_substage = get_substage_from_age(self.state.simulated_age_months)
        expected_stage = get_stage_from_substage(expected_substage)
        
        # If we're in a transition, don't update the substage automatically
        if self.state.stage_transition is None:
            # If expected stage is ahead of current stage and no transition
            # is in progress, start checking for progression
            if expected_stage > self.state.developmental_stage:
                self.check_stage_progression()
            # Otherwise, if we're not in a transition, update substage based on age
            # but only within the current main stage
            elif expected_stage == self.state.developmental_stage:
                self.state.developmental_substage = expected_substage
        
        # Synchronize component integration system
        self.integration.synchronize_development(self.state.developmental_stage)
        
        # Update history for visualization
        for metric_name, value in vars(self.state.metrics).items():
            if metric_name != "history" and not metric_name.startswith("_"):
                if metric_name not in self.state.metrics.history:
                    self.state.metrics.history[metric_name] = []
                self.state.metrics.history[metric_name].append(value)
    
    def check_stage_progression(self) -> bool:
        """
        Check if the child should progress to the next developmental stage.
        
        This method now implements a more nuanced progression approach with:
        1. Substages within each main stage
        2. Gradual transitions between substages and stages
        3. Updated metric thresholds for each substage
        
        Returns:
            True if progression occurred, False otherwise
        """
        # Get current stage, substage, and metrics
        current_stage = self.state.developmental_stage
        current_substage = self.state.developmental_substage
        age_months = self.state.simulated_age_months
        metrics = self.state.metrics
        
        # Determine expected substage based on age
        expected_substage = get_substage_from_age(age_months)
        expected_stage = get_stage_from_substage(expected_substage)
        
        # If already in expected substage, no progression needed
        if current_substage == expected_substage:
            return False
        
        # Define thresholds for stage transitions
        substage_thresholds = {
            # Infancy substage thresholds
            DevelopmentalSubstage.EARLY_INFANCY: {
                "min_age_months": 0,
                "vocabulary_size": 0,
                "object_permanence": 0.1
            },
            DevelopmentalSubstage.MIDDLE_INFANCY: {
                "min_age_months": 8,
                "vocabulary_size": 5,
                "object_permanence": 0.3
            },
            DevelopmentalSubstage.LATE_INFANCY: {
                "min_age_months": 16,
                "vocabulary_size": 20,
                "emotional_regulation": 0.2,
                "object_permanence": 0.5
            },
            
            # Early childhood substage thresholds
            DevelopmentalSubstage.EARLY_TODDLER: {
                "min_age_months": 24,
                "vocabulary_size": 50,
                "emotional_regulation": 0.3,
                "object_permanence": 0.7
            },
            DevelopmentalSubstage.LATE_TODDLER: {
                "min_age_months": 36,
                "vocabulary_size": 200,
                "grammatical_complexity": 0.2,
                "emotional_regulation": 0.4,
                "social_awareness": 0.2
            },
            DevelopmentalSubstage.PRESCHOOL: {
                "min_age_months": 48,
                "vocabulary_size": 350,
                "grammatical_complexity": 0.3,
                "emotional_regulation": 0.45,
                "social_awareness": 0.3
            },
            
            # Middle childhood substage thresholds
            DevelopmentalSubstage.EARLY_ELEMENTARY: {
                "min_age_months": 60,
                "vocabulary_size": 500,
                "grammatical_complexity": 0.4,
                "emotional_regulation": 0.5,
                "social_awareness": 0.4
            },
            DevelopmentalSubstage.MIDDLE_ELEMENTARY: {
                "min_age_months": 84,
                "vocabulary_size": 1000,
                "grammatical_complexity": 0.5,
                "emotional_regulation": 0.55,
                "social_awareness": 0.5,
                "abstract_thinking": 0.2
            },
            DevelopmentalSubstage.LATE_ELEMENTARY: {
                "min_age_months": 108,
                "vocabulary_size": 1500,
                "grammatical_complexity": 0.55,
                "emotional_regulation": 0.6,
                "social_awareness": 0.55,
                "abstract_thinking": 0.3
            },
            
            # Adolescence substage thresholds
            DevelopmentalSubstage.EARLY_ADOLESCENCE: {
                "min_age_months": 120,
                "vocabulary_size": 2000,
                "grammatical_complexity": 0.6,
                "emotional_regulation": 0.6,
                "social_awareness": 0.6,
                "abstract_thinking": 0.4,
                "self_awareness": 0.3
            },
            DevelopmentalSubstage.MIDDLE_ADOLESCENCE: {
                "min_age_months": 156,
                "vocabulary_size": 5000,
                "grammatical_complexity": 0.7,
                "emotional_regulation": 0.65,
                "social_awareness": 0.65,
                "abstract_thinking": 0.6,
                "self_awareness": 0.5
            },
            DevelopmentalSubstage.LATE_ADOLESCENCE: {
                "min_age_months": 192,
                "vocabulary_size": 8000,
                "grammatical_complexity": 0.75,
                "emotional_regulation": 0.7,
                "social_awareness": 0.7,
                "abstract_thinking": 0.65,
                "self_awareness": 0.6
            },
            
            # Early adulthood substage thresholds
            DevelopmentalSubstage.EMERGING_ADULT: {
                "min_age_months": 216,
                "vocabulary_size": 10000,
                "grammatical_complexity": 0.8,
                "emotional_regulation": 0.7,
                "social_awareness": 0.7,
                "abstract_thinking": 0.7,
                "self_awareness": 0.7
            },
            DevelopmentalSubstage.YOUNG_ADULT: {
                "min_age_months": 252,
                "vocabulary_size": 15000,
                "grammatical_complexity": 0.85,
                "emotional_regulation": 0.8,
                "social_awareness": 0.8,
                "abstract_thinking": 0.8,
                "self_awareness": 0.8
            },
            DevelopmentalSubstage.ESTABLISHED_ADULT: {
                "min_age_months": 300,
                "vocabulary_size": 20000,
                "grammatical_complexity": 0.9,
                "emotional_regulation": 0.9,
                "social_awareness": 0.9,
                "abstract_thinking": 0.9,
                "self_awareness": 0.9
            }
        }
        
        # Get substages for current stage
        current_stage_substages = STAGE_TO_SUBSTAGES[current_stage]
        
        # Find index of current substage in the sequence
        try:
            current_index = current_stage_substages.index(current_substage)
        except ValueError:
            # If current substage isn't in the current stage's substages
            # (shouldn't happen due to validator, but for safety)
            current_index = 0
            self.state.developmental_substage = current_stage_substages[0]
        
        # Check if we're moving to the next substage within the same stage
        # or advancing to a new main stage
        if expected_stage == current_stage:
            # We're progressing within the same main stage
            expected_index = current_stage_substages.index(expected_substage)
            
            # Make sure we're moving forward, not backward
            if expected_index <= current_index:
                return False
            
            # Check if we need to move one substage at a time
            next_substage_index = min(current_index + 1, len(current_stage_substages) - 1)
            next_substage = current_stage_substages[next_substage_index]
            
            # Check thresholds for the next substage
            substage_met = self._check_substage_thresholds(next_substage, substage_thresholds)
            
            if substage_met:
                # Begin or update transition between substages
                if self.state.stage_transition is None:
                    # Start a new transition
                    self.state.stage_transition = StageTransition(
                        current_stage=current_stage,
                        next_stage=current_stage,
                        current_substage=current_substage,
                        next_substage=next_substage,
                        transition_progress=0.1,
                        metrics=self._get_metric_dict(metrics)
                    )
                    logging.info(f"Beginning transition from {current_substage.value} to {next_substage.value}")
                else:
                    # Update existing transition progress
                    self.state.stage_transition.transition_progress += 0.1
                    
                # If transition is complete, update the substage
                if self.state.stage_transition.transition_progress >= 1.0:
                    self.state.developmental_substage = next_substage
                    logging.info(f"Completed transition to {next_substage.value}")
                    self.state.stage_transition = None
                    return True
        else:
            # We're moving to a new main stage
            # Get the first substage of the next stage
            next_stage = DevelopmentalStage.EARLY_CHILDHOOD
            if current_stage == DevelopmentalStage.EARLY_CHILDHOOD:
                next_stage = DevelopmentalStage.MIDDLE_CHILDHOOD
            elif current_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
                next_stage = DevelopmentalStage.ADOLESCENCE
            elif current_stage == DevelopmentalStage.ADOLESCENCE:
                next_stage = DevelopmentalStage.EARLY_ADULTHOOD
            
            # Can't progress beyond EARLY_ADULTHOOD
            if current_stage == DevelopmentalStage.EARLY_ADULTHOOD:
                return False
            
            next_substage = STAGE_TO_SUBSTAGES[next_stage][0]
            
            # Check thresholds for the first substage of the next main stage
            substage_met = self._check_substage_thresholds(next_substage, substage_thresholds)
            
            # Check integration level
            integration_met = self.integration.integration_level >= self.integration.developmental_scaling[current_stage]
            
            if substage_met and integration_met:
                # Begin or update transition between main stages
                if self.state.stage_transition is None:
                    # Start a new transition
                    self.state.stage_transition = StageTransition(
                        current_stage=current_stage,
                        next_stage=next_stage,
                        current_substage=current_substage,
                        next_substage=next_substage,
                        transition_progress=0.05,  # Slower progress for main stage transitions
                        metrics=self._get_metric_dict(metrics)
                    )
                    logging.info(f"Beginning transition from {current_stage.value} to {next_stage.value}")
                else:
                    # Update existing transition progress
                    self.state.stage_transition.transition_progress += 0.05
                    
                # If transition is complete, update the stage and substage
                if self.state.stage_transition.transition_progress >= 1.0:
                    self.state.developmental_stage = next_stage
                    self.state.developmental_substage = next_substage
                    logging.info(f"Completed transition to {next_stage.value} ({next_substage.value})")
                    self.state.stage_transition = None
                    return True
        
        return False
    
    def _check_substage_thresholds(self, substage: DevelopmentalSubstage, thresholds: Dict) -> bool:
        """
        Check if the child meets the thresholds for a specific substage.
        
        Args:
            substage: The substage to check thresholds for
            thresholds: Dictionary of thresholds for all substages
            
        Returns:
            True if all thresholds are met, False otherwise
        """
        if substage not in thresholds:
            return False
            
        substage_thresholds = thresholds[substage]
        
        # Check if minimum age has been reached
        if self.state.simulated_age_months < substage_thresholds.get("min_age_months", 0):
            return False
        
        # Check if all other thresholds are met
        for metric, threshold in substage_thresholds.items():
            if metric == "min_age_months":
                continue  # Already checked
            
            if metric == "vocabulary_size":
                if len(self.state.vocabulary) < threshold:
                    return False
            elif hasattr(self.state.metrics, metric):
                if getattr(self.state.metrics, metric) < threshold:
                    return False
        
        return True
    
    def _get_metric_dict(self, metrics: Any) -> Dict[str, float]:
        """
        Convert metrics object to a dictionary.
        
        Args:
            metrics: The metrics object
            
        Returns:
            Dictionary of metric names and values
        """
        metric_dict = {}
        for metric_name in [
            "vocabulary_size", "grammatical_complexity", "emotional_regulation",
            "social_awareness", "object_permanence", "abstract_thinking", "self_awareness"
        ]:
            if hasattr(metrics, metric_name):
                if metric_name == "vocabulary_size":
                    metric_dict[metric_name] = len(self.state.vocabulary)
                else:
                    metric_dict[metric_name] = getattr(metrics, metric_name)
        
        return metric_dict
    
    def save_state(self, filepath: str):
        """
        Save the current state to a file.
        
        Args:
            filepath: Path to save the state
        """
        # Update last save time
        self.state.last_save_time = datetime.now()
        
        # Use Pydantic's json export with encoding='utf-8'
        state_json = self.state.model_dump_json(indent=2)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(state_json)
            
        logger.info(f"State saved to {filepath}")
    
    @classmethod
    def load_state(cls, filepath: str) -> 'Child':
        """
        Load a state from a file and create a Child instance.
        
        Args:
            filepath: Path to load the state from
            
        Returns:
            A new Child instance with the loaded state
        """
        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            state_json = f.read()
            
        # Parse JSON to ChildState
        state = ChildState.model_validate_json(state_json)
        
        # Create new Child with loaded state
        child = cls(initial_state=state)
        logger.info(f"State loaded from {filepath}")
        
        return child 