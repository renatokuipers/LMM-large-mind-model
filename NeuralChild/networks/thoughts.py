# thoughts.py
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Set
import logging
import random
import numpy as np
from collections import deque
from pydantic import BaseModel

from networks.base_network import BaseNetwork
from networks.network_types import NetworkType, ConnectionType

logger = logging.getLogger("ThoughtsNetwork")

class Thought(BaseModel):
    """A single thought or idea generated by the mind"""
    content: str
    confidence: float  # How confident/certain the thought is (0-1)
    sources: List[str] = []  # What networks/stimuli contributed to this thought
    created_at: datetime = datetime.now()
    type: str = "general"  # Types: question, statement, observation, desire, etc.
    related_concepts: List[str] = []
    emotional_valence: float = 0.0  # -1 (negative) to +1 (positive)
    
    def age_seconds(self) -> float:
        """Get age of this thought in seconds"""
        return (datetime.now() - self.created_at).total_seconds()
    
    def is_fresh(self) -> bool:
        """Check if this thought is still fresh (less than 60 seconds old)"""
        return self.age_seconds() < 60

class ThoughtsNetwork(BaseNetwork):
    """
    Active information processing network using Transformer-like architecture
    
    The thoughts network represents the active, conscious processing of information.
    It generates internal thoughts, questions, and reasoning based on inputs from
    other networks, especially consciousness, perception, and emotion.
    """
    
    def __init__(
        self,
        initial_state=None,
        learning_rate_multiplier: float = 1.0,
        activation_threshold: float = 0.3,  # Thoughts require significant activation
        name: str = "Thoughts"
    ):
        """Initialize the thoughts network"""
        super().__init__(
            network_type=NetworkType.THOUGHTS,
            initial_state=initial_state,
            learning_rate_multiplier=learning_rate_multiplier,
            activation_threshold=activation_threshold,
            name=name
        )
        
        # Thoughts parameters
        self.reasoning_ability = 0.2  # Ability to perform multi-step reasoning (increases with age)
        self.abstraction_level = 0.1  # Ability to form abstract thoughts (increases with age)
        self.creativity = 0.5         # Ability to generate novel thoughts
        self.introspection = 0.2      # Ability to think about own thoughts
        
        # Active thoughts storage
        self.active_thoughts: List[Thought] = []      # Currently active thoughts
        self.recent_thoughts = deque(maxlen=50)   # History of recent thoughts
        self.recurring_thoughts: Dict[str, int] = {}  # Thoughts that appear frequently
        
        # Thought generation patterns
        self.thought_patterns = {
            "observation": 0.5,    # Simple observations about percepts
            "question": 0.3,       # Questions about percepts or concepts
            "desire": 0.2,         # Expressing wants or needs
            "reasoning": 0.1,      # Connecting ideas or causal reasoning
            "imagination": 0.2,    # Creative or novel thoughts
            "social": 0.4,         # Thoughts about others (especially mother)
            "self_reference": 0.2  # Thoughts about self
        }
        
        # Vocabulary for thought generation (enriched during runtime)
        self.thought_vocabulary = {
            "observation": ["see", "look", "is", "there"],
            "question": ["what", "why", "how", "where"],
            "desire": ["want", "need", "like", "more"],
            "reasoning": ["because", "so", "if", "then"],
            "imagination": ["maybe", "could", "pretend", "like"],
            "social": ["mommy", "daddy", "you", "we"],
            "self_reference": ["me", "I", "my", "mine"]
        }
        
        logger.info(f"Initialized thoughts network with reasoning ability {self.reasoning_ability}")
    
    def process_inputs(self) -> Dict[str, Any]:
        """Process inputs to generate thoughts"""
        # Decay old thoughts
        self._decay_old_thoughts()
        
        # Collect inputs from various sources
        perceptions = []      # Things perceived
        consciousness = []    # Contents of consciousness
        attention_focus = []  # What's being attended to
        emotional_state = {}  # Current emotional state
        active_concepts = []  # Active concepts from other networks
        language_items = []   # Words, phrases that might be used
        
        for input_item in self.input_buffer:
            data = input_item.get("data", {})
            source = input_item.get("source", "unknown")
            
            # Process perceptions
            if source == NetworkType.PERCEPTION.value:
                perceptions.extend(data.get("percepts", []))
            
            # Process consciousness contents
            if source == NetworkType.CONSCIOUSNESS.value:
                contents = data.get("active_contents", {})
                if isinstance(contents, dict):
                    for category, items in contents.items():
                        if isinstance(items, list):
                            consciousness.extend(items)
                        elif isinstance(items, dict):
                            consciousness.extend(items.keys())
                self_reps = data.get("self_representations", [])
                consciousness.extend(self_reps)
            
            # Process attention focus
            if source == NetworkType.ATTENTION.value:
                attention_focus.extend(data.get("focus_objects", []))
            
            # Process emotional state
            if source == NetworkType.EMOTIONS.value:
                emotional_state.update(data.get("emotional_state", {}))
            
            # Process unconscious associations
            if source == NetworkType.UNCONSCIOUSNESS.value:
                active_concepts.extend(data.get("triggered_concepts", []))
            
            # Process language inputs
            vocab_data = data.get("vocabulary", {})
            if isinstance(vocab_data, list):
                language_items.extend(vocab_data)
            elif isinstance(vocab_data, dict):
                for category, words in vocab_data.items():
                    if isinstance(words, list):
                        language_items.extend(words)
            
            # Process direct thought triggers
            trigger = data.get("thought_trigger", {})
            if trigger:
                content = trigger.get("content")
                if content:
                    thought = Thought(
                        content=content,
                        confidence=trigger.get("confidence", 0.8),
                        sources=[source],
                        type=trigger.get("type", "general"),
                        emotional_valence=trigger.get("emotional_valence", 0.0)
                    )
                    self.active_thoughts.append(thought)
                    self.recent_thoughts.append(thought)
        
        # Combine all inputs for thought generation
        all_inputs = list(set(perceptions + consciousness + attention_focus + active_concepts + language_items))
        
        # Generate new thoughts based on inputs
        new_thoughts = self._generate_thoughts(
            all_inputs, 
            perceptions, 
            attention_focus, 
            emotional_state
        )
        
        # Add new thoughts to active and recent thoughts
        for thought in new_thoughts:
            self.active_thoughts.append(thought)
            self.recent_thoughts.append(thought)
            
            # Track recurring thoughts
            if thought.content not in self.recurring_thoughts:
                self.recurring_thoughts[thought.content] = 0
            self.recurring_thoughts[thought.content] += 1
        
        # Calculate activation level based on thought generation
        if self.active_thoughts:
            # Higher activation with more thoughts and higher confidence
            total_confidence = sum(t.confidence for t in self.active_thoughts)
            avg_confidence = total_confidence / len(self.active_thoughts)
            activation = min(1.0, 0.3 + avg_confidence * 0.4 + min(0.3, len(self.active_thoughts) * 0.1))
        else:
            activation = 0.1  # Minimal background activation
        
        # Update thought vocabulary with new words
        self._update_thought_vocabulary(language_items)
        
        # Clear input buffer
        self.input_buffer = []
        
        return {
            "network_activation": activation,
            "thoughts": [t.content for t in self.active_thoughts],
            "thought_types": {t.type: 1 for t in self.active_thoughts}
        }
    
    def _decay_old_thoughts(self) -> None:
        """Remove old thoughts from active thoughts"""
        self.active_thoughts = [t for t in self.active_thoughts if t.is_fresh()]
    
    def _generate_thoughts(
        self, 
        all_inputs: List[str], 
        perceptions: List[str],
        attention_focus: List[str],
        emotional_state: Dict[str, float]
    ) -> List<Thought:
        """Generate new thoughts based on inputs"""
        new_thoughts = []
        
        # Skip thought generation if no significant inputs
        if not all_inputs:
            return new_thoughts
        
        # Determine dominant emotion
        dominant_emotion = None
        max_intensity = 0.0
        for emotion, intensity in emotional_state.items():
            if intensity > max_intensity:
                max_intensity = intensity
                dominant_emotion = emotion
        
        # Calculate how many thoughts to generate based on reasoning ability
        thought_capacity = int(1 + self.reasoning_ability * 3)
        num_thoughts = random.randint(1, thought_capacity)
        
        # Generate thoughts of different types
        for _ in range(num_thoughts):
            # Select thought type based on patterns and randomness
            thought_type = self._select_thought_type(dominant_emotion)
            
            # Generate a thought of the selected type
            if thought_type == "observation" and perceptions:
                # Observations are based on perceptions
                stimulus = random.choice(perceptions)
                thought = self._generate_observation(stimulus, emotional_state)
                if thought:
                    new_thoughts.append(thought)
                    
            elif thought_type == "question" and all_inputs:
                # Questions are based on attention focus or perceptions
                if attention_focus:
                    stimulus = random.choice(attention_focus)
                else:
                    stimulus = random.choice(all_inputs)
                thought = self._generate_question(stimulus)
                if thought:
                    new_thoughts.append(thought)
                    
            elif thought_type == "desire":
                # Desires are influenced by emotions and drives
                thought = self._generate_desire(emotional_state)
                if thought:
                    new_thoughts.append(thought)
                    
            elif thought_type == "reasoning" and len(self.active_thoughts) > 0:
                # Reasoning connects existing thoughts
                if random.random() < self.reasoning_ability:
                    thought = self._generate_reasoning(self.active_thoughts)
                    if thought:
                        new_thoughts.append(thought)
                        
            elif thought_type == "imagination":
                # Imagination creates novel combinations
                if random.random() < self.creativity:
                    thought = self._generate_imagination(all_inputs)
                    if thought:
                        new_thoughts.append(thought)
                        
            elif thought_type == "social" and "mother" in ' '.join(all_inputs):
                # Social thoughts focus on important others
                thought = self._generate_social_thought()
                if thought:
                    new_thoughts.append(thought)
                    
            elif thought_type == "self_reference":
                # Self-referential thoughts if self-awareness is developing
                if random.random() < self.introspection:
                    thought = self._generate_self_reference(emotional_state)
                    if thought:
                        new_thoughts.append(thought)
        
        return new_thoughts
    
    def _select_thought_type(self, dominant_emotion: Optional[str]) -> str:
        """Select a thought type based on patterns and emotional state"""
        # Adjust thought patterns based on emotion
        adjusted_patterns = self.thought_patterns.copy()
        
        if dominant_emotion:
            if dominant_emotion == "joy":
                adjusted_patterns["imagination"] += 0.2
                adjusted_patterns["social"] += 0.1
            elif dominant_emotion == "sadness":
                adjusted_patterns["self_reference"] += 0.2
            elif dominant_emotion == "fear":
                adjusted_patterns["question"] += 0.2
            elif dominant_emotion == "anger":
                adjusted_patterns["desire"] += 0.2
            elif dominant_emotion == "surprise":
                adjusted_patterns["question"] += 0.3
                adjusted_patterns["observation"] += 0.1
            elif dominant_emotion == "trust":
                adjusted_patterns["social"] += 0.2
            elif dominant_emotion == "anticipation":
                adjusted_patterns["imagination"] += 0.1
                adjusted_patterns["desire"] += 0.1
        
        # Weighted random selection
        options = list(adjusted_patterns.keys())
        weights = list(adjusted_patterns.values())
        
        return random.choices(options, weights=weights, k=1)[0]
    
    def _generate_observation(self, stimulus: str, emotional_state: Dict[str, float]) -> Optional[Thought]:
        """Generate an observation thought"""
        if not self.thought_vocabulary["observation"]:
            return None
            
        # Get observation words
        obs_words = self.thought_vocabulary["observation"]
        
        # Simple format: "[observation word] [stimulus]"
        obs_word = random.choice(obs_words)
        
        # Format based on abstraction level
        if self.abstraction_level < 0.3:
            # Very simple
            content = f"{stimulus}"
        elif self.abstraction_level < 0.6:
            # Simple sentence
            content = f"{obs_word} {stimulus}"
        else:
            # More complex
            content = f"{obs_word} the {stimulus}"
        
        # Determine emotional valence
        valence = 0.0
        for emotion, intensity in emotional_state.items():
            if emotion in ["joy", "trust", "anticipation"]:
                valence += intensity * 0.2
            elif emotion in ["sadness", "fear", "anger", "disgust"]:
                valence -= intensity * 0.2
        
        return Thought(
            content=content,
            confidence=0.7 + random.random() * 0.2,  # High confidence for observations
            sources=["perception"],
            type="observation",
            related_concepts=[stimulus],
            emotional_valence=valence
        )
    
    def _generate_question(self, stimulus: str) -> Optional[Thought]:
        """Generate a question thought"""
        if not self.thought_vocabulary["question"]:
            return None
            
        # Get question words
        question_words = self.thought_vocabulary["question"]
        
        # Format based on abstraction level
        question_word = random.choice(question_words)
        
        if self.abstraction_level < 0.3:
            # Very simple question
            content = f"{question_word} {stimulus}?"
        elif self.abstraction_level < 0.6:
            # Simple question
            content = f"{question_word} is {stimulus}?"
        else:
            # More complex question
            content = f"{question_word} is this {stimulus}?"
        
        return Thought(
            content=content,
            confidence=0.5 + random.random() * 0.3,  # Moderate confidence for questions
            sources=["curiosity", "perception"],
            type="question",
            related_concepts=[stimulus],
            emotional_valence=0.2  # Questions tend to have positive valence (curiosity)
        )
    
    def _generate_desire(self, emotional_state: Dict[str, float]) -> Optional[Thought]:
        """Generate a desire thought"""
        if not self.thought_vocabulary["desire"]:
            return None
            
        # Get desire words
        desire_words = self.thought_vocabulary["desire"]
        
        # Objects that might be desired
        objects = ["milk", "food", "play", "toy", "mommy", "hug", "ball"]
        # Extend with any objects in vocabulary
        for word_list in self.thought_vocabulary.values():
            for word in word_list:
                if len(word) > 2 and word not in ["want", "need", "like", "more"]:
                    objects.append(word)
        
        # Select desire word and object
        desire_word = random.choice(desire_words)
        desire_object = random.choice(objects)
        
        # Format based on abstraction level
        if self.abstraction_level < 0.3:
            # Very simple
            content = f"{desire_object}"
        elif self.abstraction_level < 0.6:
            # Simple
            content = f"{desire_word} {desire_object}"
        else:
            # More complex
            content = f"I {desire_word} {desire_object}"
        
        # Calculate intensity based on emotions
        intensity = 0.5
        if "anticipation" in emotional_state:
            intensity += emotional_state["anticipation"] * 0.3
        if "joy" in emotional_state:
            intensity += emotional_state["joy"] * 0.2
        
        return Thought(
            content=content,
            confidence=intensity,
            sources=["drives", "emotions"],
            type="desire",
            related_concepts=[desire_object],
            emotional_valence=0.5  # Desires tend to have positive valence
        )
    
    def _generate_reasoning(self, active_thoughts: List[Thought]) -> Optional[Thought]:
        """Generate a reasoning thought connecting existing thoughts"""
        if len(active_thoughts) < 2 or not self.thought_vocabulary["reasoning"]:
            return None
            
        # Select two thoughts to connect
        thought1, thought2 = random.sample(active_thoughts, 2)
        
        # Get reasoning words
        reasoning_words = self.thought_vocabulary["reasoning"]
        connector = random.choice(reasoning_words)
        
        # Format based on reasoning ability
        if self.reasoning_ability < 0.3:
            # Very simple connection
            content = f"{thought1.content} {connector} {thought2.content}"
        else:
            # More structured reasoning
            content = f"{thought1.content} {connector} {thought2.content}"
        
        # Confidence is based on reasoning ability and source thoughts
        confidence = min(0.8, (thought1.confidence + thought2.confidence) / 3 + self.reasoning_ability * 0.2)
        
        # Combine related concepts from both thoughts
        related_concepts = list(set(thought1.related_concepts + thought2.related_concepts))
        
        return Thought(
            content=content,
            confidence=confidence,
            sources=["reasoning"] + thought1.sources + thought2.sources,
            type="reasoning",
            related_concepts=related_concepts,
            emotional_valence=(thought1.emotional_valence + thought2.emotional_valence) / 2
        )
    
    def _generate_imagination(self, concepts: List[str]) -> Optional[Thought]:
        """Generate an imaginative thought combining concepts in novel ways"""
        if len(concepts) < 2 or not self.thought_vocabulary["imagination"]:
            return None
            
        # Select concepts to combine
        selected_concepts = random.sample(concepts, min(2, len(concepts)))
        
        # Get imagination words
        imagination_words = self.thought_vocabulary["imagination"]
        
        # Format based on creativity and abstraction
        if self.abstraction_level < 0.3:
            # Simple juxtaposition
            content = f"{selected_concepts[0]} {selected_concepts[1]}"
        else:
            # More creative combination
            imagination_word = random.choice(imagination_words)
            content = f"{imagination_word} {selected_concepts[0]} {selected_concepts[1]}"
        
        # Confidence is lower for imaginative thoughts
        confidence = 0.3 + self.creativity * 0.3
        
        return Thought(
            content=content,
            confidence=confidence,
            sources=["imagination"],
            type="imagination",
            related_concepts=selected_concepts,
            emotional_valence=0.3  # Imagination tends to be positive
        )
    
    def _generate_social_thought(self) -> Optional[Thought]:
        """Generate a thought about social relationships"""
        if not self.thought_vocabulary["social"]:
            return None
            
        # Get social words
        social_words = self.thought_vocabulary["social"]
        
        # Simple social thought formats
        formats = [
            "{person}",
            "{person} help",
            "{person} love",
            "where {person}",
            "see {person}"
        ]
        
        # Select person and format
        person = random.choice(social_words)
        format_template = random.choice(formats)
        
        # Generate content
        content = format_template.format(person=person)
        
        return Thought(
            content=content,
            confidence=0.6 + random.random() * 0.2,
            sources=["social"],
            type="social",
            related_concepts=[person],
            emotional_valence=0.4  # Social thoughts tend to be positive
        )
    
    def _generate_self_reference(self, emotional_state: Dict[str, float]) -> Optional[Thought]:
        """Generate a self-referential thought"""
        if not self.thought_vocabulary["self_reference"] or self.introspection < 0.2:
            return None
            
        # Get self-reference words
        self_words = self.thought_vocabulary["self_reference"]
        
        # Find dominant emotion
        dominant_emotion = None
        dominant_intensity = 0.0
        for emotion, intensity in emotional_state.items():
            if intensity > dominant_intensity:
                dominant_intensity = intensity
                dominant_emotion = emotion
        
        # Simple formats
        formats = [
            "{self_word}",
            "{self_word} {emotion}",
            "{self_word} want",
            "{self_word} do"
        ]
        
        # Select format and words
        format_template = random.choice(formats)
        self_word = random.choice(self_words)
        
        # Add emotion if applicable
        if "{emotion}" in format_template and dominant_emotion:
            content = format_template.format(self_word=self_word, emotion=dominant_emotion)
        else:
            content = format_template.format(self_word=self_word)
        
        return Thought(
            content=content,
            confidence=0.5 + self.introspection * 0.3,
            sources=["introspection"],
            type="self_reference",
            related_concepts=[self_word],
            emotional_valence=0.0 + (0.5 * self.introspection)  # Neutral to positive
        )
    
    def _update_thought_vocabulary(self, new_words: List[str]) -> None:
        """Update thought vocabulary with new words"""
        for word in new_words:
            word = word.lower().strip()
            if not word or len(word) < 2:
                continue
                
            # Classify word into a category based on simple heuristics
            if word in ["what", "why", "how", "where", "when", "who"]:
                if word not in self.thought_vocabulary["question"]:
                    self.thought_vocabulary["question"].append(word)
            elif word in ["see", "look", "hear", "feel", "smell", "is", "are", "am"]:
                if word not in self.thought_vocabulary["observation"]:
                    self.thought_vocabulary["observation"].append(word)
            elif word in ["want", "need", "like", "more", "give", "get"]:
                if word not in self.thought_vocabulary["desire"]:
                    self.thought_vocabulary["desire"].append(word)
            elif word in ["because", "so", "if", "then", "but", "and"]:
                if word not in self.thought_vocabulary["reasoning"]:
                    self.thought_vocabulary["reasoning"].append(word)
            elif word in ["maybe", "could", "would", "pretend", "imagine"]:
                if word not in self.thought_vocabulary["imagination"]:
                    self.thought_vocabulary["imagination"].append(word)
            elif word in ["mommy", "daddy", "friend", "person", "you", "we", "they"]:
                if word not in self.thought_vocabulary["social"]:
                    self.thought_vocabulary["social"].append(word)
            elif word in ["me", "I", "my", "mine", "self"]:
                if word not in self.thought_vocabulary["self_reference"]:
                    self.thought_vocabulary["self_reference"].append(word)
    
    def update_development(self, age_days: float, vocabulary_size: int) -> None:
        """Update developmental parameters based on age and vocabulary"""
        # Reasoning ability increases with age
        self.reasoning_ability = min(0.9, 0.2 + (age_days / 300))
        
        # Abstraction level increases with age and vocabulary
        vocab_factor = min(0.4, vocabulary_size / 1000)
        age_factor = min(0.5, age_days / 400)
        self.abstraction_level = min(0.9, 0.1 + age_factor + vocab_factor)
        
        # Creativity is initially high but becomes more focused/structured
        self.creativity = max(0.3, min(0.9, 0.5 + (age_days / 600) - (age_days / 1000)))
        
        # Introspection increases with age but requires development
        self.introspection = min(0.8, 0.2 + (age_days / 500))
    
    def _prepare_output_data(self) -> Dict[str, Any]:
        """Prepare data to send to other networks"""
        # Extract current thoughts
        current_thoughts = [t.content for t in self.active_thoughts]
        
        # Get thought types distribution
        thought_types = {}
        for thought in self.active_thoughts:
            if thought.type not in thought_types:
                thought_types[thought.type] = 0
            thought_types[thought.type] += 1
        
        # Get most frequent recurring thoughts
        recurring = sorted(
            self.recurring_thoughts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]
        
        # Calculate average thought complexity (length as proxy)
        if self.active_thoughts:
            avg_length = sum(len(t.content.split()) for t in self.active_thoughts) / len(self.active_thoughts)
        else:
            avg_length = 0
        
        return {
            "activation": self.state.activation,
            "confidence": self.state.confidence,
            "network_type": self.network_type.value,
            "thoughts": current_thoughts,
            "thought_types": thought_types,
            "recurring_thoughts": recurring,
            "reasoning_ability": self.reasoning_ability,
            "abstraction_level": self.abstraction_level,
            "creativity": self.creativity,
            "introspection": self.introspection,
            "thought_complexity": avg_length,
            "vocabulary_richness": sum(len(words) for words in self.thought_vocabulary.values())
        }
    
    def get_current_thoughts(self) -> List[str]:
        """Get list of current thought strings"""
        return [t.content for t in self.active_thoughts]