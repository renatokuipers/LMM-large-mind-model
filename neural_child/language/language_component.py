"""
Language component for the Neural Child's mind.

This module contains the implementation of the language component that handles
language acquisition and processing for the simulated mind.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Set
import random
import re
from pathlib import Path
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter, defaultdict

from neural_child.mind.base import NeuralComponent

class LanguageComponent(NeuralComponent):
    """Language component that handles language acquisition and processing."""
    
    def __init__(
        self,
        input_size: int = 32,
        hidden_size: int = 64,
        output_size: int = 32,
        name: str = "language_component"
    ):
        """Initialize the language component.
        
        Args:
            input_size: Size of the input layer
            hidden_size: Size of the hidden layer
            output_size: Size of the output layer
            name: Name of the component
        """
        super().__init__(input_size, hidden_size, output_size, name)
        
        # Language development metrics
        self.receptive_language_development = 0.0  # Understanding language
        self.expressive_language_development = 0.0  # Producing language
        
        # Vocabulary
        self.vocabulary: Set[str] = set()
        self.word_frequency: Counter = Counter()
        self.word_associations: Dict[str, List[str]] = defaultdict(list)
        self.word_emotional_associations: Dict[str, Dict[str, float]] = {}
        
        # Grammar development
        self.grammar_patterns: List[Dict[str, Any]] = []
        self.grammar_development = 0.0
        
        # Babbling development
        self.babbling_development = 0.0
        self.babbling_sounds: List[str] = ["ba", "da", "ma", "pa", "ta", "ga"]
        
        # Language production capabilities
        self.max_utterance_length = 1  # Start with single-word utterances
        
        # Language comprehension capabilities
        self.comprehension_level = 0.0
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs.
        
        Args:
            inputs: Dictionary of inputs to the component
                - mother_utterance: String representing the mother's utterance
                - teaching_elements: List of teaching elements from the mother
                - developmental_stage: String representing the developmental stage
                - age_months: Float representing the age in months
                - emotional_state: Dictionary of the child's emotional state
                
        Returns:
            Dictionary of outputs from the component
                - child_utterance: String representing the child's utterance
                - comprehension: Float representing the comprehension level
                - vocabulary_size: Integer representing the vocabulary size
                - language_development: Dictionary of language development metrics
        """
        # Extract inputs
        mother_utterance = inputs.get("mother_utterance", "")
        teaching_elements = inputs.get("teaching_elements", [])
        developmental_stage = inputs.get("developmental_stage", "Prenatal")
        age_months = inputs.get("age_months", 0.0)
        emotional_state = inputs.get("emotional_state", {})
        
        # Process mother's utterance
        self._process_mother_utterance(mother_utterance, emotional_state)
        
        # Process teaching elements
        self._process_teaching_elements(teaching_elements)
        
        # Update language development metrics based on age
        self._update_language_development(age_months, developmental_stage)
        
        # Generate child's utterance based on developmental stage
        child_utterance = self._generate_utterance(developmental_stage, emotional_state)
        
        # Prepare outputs
        outputs = {
            "child_utterance": child_utterance,
            "comprehension": self.comprehension_level,
            "vocabulary_size": len(self.vocabulary),
            "language_development": {
                "receptive_language": self.receptive_language_development,
                "expressive_language": self.expressive_language_development,
                "grammar_development": self.grammar_development,
                "babbling_development": self.babbling_development,
                "max_utterance_length": self.max_utterance_length
            }
        }
        
        # Update activation level
        self.update_activation(self.expressive_language_development)
        
        return outputs
    
    def _process_mother_utterance(self, utterance: str, emotional_state: Dict[str, float]):
        """Process the mother's utterance.
        
        Args:
            utterance: String representing the mother's utterance
            emotional_state: Dictionary of the child's emotional state
        """
        if not utterance:
            return
        
        # Tokenize utterance
        try:
            tokens = word_tokenize(utterance.lower())
        except:
            # If NLTK data is not available, use a simple tokenizer
            tokens = re.findall(r'\b\w+\b', utterance.lower())
        
        # Process tokens based on receptive language development
        num_tokens_to_process = int(len(tokens) * self.receptive_language_development) + 1
        tokens_to_process = tokens[:num_tokens_to_process]
        
        # Update word frequency
        self.word_frequency.update(tokens_to_process)
        
        # Add words to vocabulary based on frequency and receptive language development
        for token in tokens_to_process:
            if self.word_frequency[token] >= 3 and random.random() < self.receptive_language_development:
                self.vocabulary.add(token)
        
        # Update word associations
        for i in range(len(tokens_to_process) - 1):
            self.word_associations[tokens_to_process[i]].append(tokens_to_process[i + 1])
            # Limit associations to prevent memory issues
            if len(self.word_associations[tokens_to_process[i]]) > 20:
                self.word_associations[tokens_to_process[i]] = self.word_associations[tokens_to_process[i]][-20:]
        
        # Update word emotional associations
        dominant_emotion = max(emotional_state.items(), key=lambda x: x[1]) if emotional_state else ("neutral", 0.0)
        for token in tokens_to_process:
            if token not in self.word_emotional_associations:
                self.word_emotional_associations[token] = {emotion: 0.0 for emotion in emotional_state}
            
            # Update emotional association for the dominant emotion
            if dominant_emotion[0] in self.word_emotional_associations[token]:
                self.word_emotional_associations[token][dominant_emotion[0]] += 0.1
                
                # Normalize emotional associations
                total = sum(self.word_emotional_associations[token].values())
                if total > 0:
                    for emotion in self.word_emotional_associations[token]:
                        self.word_emotional_associations[token][emotion] /= total
        
        # Extract grammar patterns if grammar development is sufficient
        if self.grammar_development >= 0.3 and len(tokens) >= 2:
            pattern = {
                "pattern": " ".join(tokens[:3]) if len(tokens) >= 3 else " ".join(tokens),
                "frequency": 1
            }
            
            # Check if pattern already exists
            pattern_exists = False
            for existing_pattern in self.grammar_patterns:
                if existing_pattern["pattern"] == pattern["pattern"]:
                    existing_pattern["frequency"] += 1
                    pattern_exists = True
                    break
            
            if not pattern_exists:
                self.grammar_patterns.append(pattern)
                
            # Limit grammar patterns to prevent memory issues
            if len(self.grammar_patterns) > 50:
                self.grammar_patterns = sorted(self.grammar_patterns, key=lambda x: x["frequency"], reverse=True)[:50]
    
    def _process_teaching_elements(self, teaching_elements: List[Dict[str, Any]]):
        """Process teaching elements from the mother.
        
        Args:
            teaching_elements: List of teaching elements from the mother
        """
        for element in teaching_elements:
            if element.get("type") == "vocabulary":
                # Process vocabulary teaching element
                content = element.get("content", "")
                importance = element.get("importance", 0.5)
                
                # Tokenize content
                try:
                    tokens = word_tokenize(content.lower())
                except:
                    # If NLTK data is not available, use a simple tokenizer
                    tokens = re.findall(r'\b\w+\b', content.lower())
                
                # Add words to vocabulary based on importance and receptive language development
                for token in tokens:
                    if random.random() < importance * self.receptive_language_development:
                        self.vocabulary.add(token)
                        self.word_frequency[token] += 3  # Boost frequency for taught words
    
    def _update_language_development(self, age_months: float, developmental_stage: str):
        """Update language development metrics based on age and developmental stage.
        
        Args:
            age_months: Float representing the age in months
            developmental_stage: String representing the developmental stage
        """
        # Receptive language develops earlier than expressive language
        if age_months <= 12:
            # First year: beginning of receptive language
            self.receptive_language_development = min(0.3, age_months * 0.025)
            self.expressive_language_development = min(0.1, max(0.0, (age_months - 6) * 0.016))
        elif age_months <= 36:
            # 1-3 years: rapid development of receptive and expressive language
            self.receptive_language_development = min(0.6, 0.3 + (age_months - 12) * 0.0125)
            self.expressive_language_development = min(0.5, 0.1 + (age_months - 12) * 0.0167)
        elif age_months <= 72:
            # 3-6 years: continued development
            self.receptive_language_development = min(0.8, 0.6 + (age_months - 36) * 0.0055)
            self.expressive_language_development = min(0.7, 0.5 + (age_months - 36) * 0.0055)
        else:
            # After 6 years: refinement
            self.receptive_language_development = min(1.0, 0.8 + (age_months - 72) * 0.001)
            self.expressive_language_development = min(1.0, 0.7 + (age_months - 72) * 0.0015)
        
        # Grammar development
        if age_months <= 18:
            # First 18 months: minimal grammar
            self.grammar_development = min(0.1, age_months * 0.005)
        elif age_months <= 48:
            # 18-48 months: rapid grammar development
            self.grammar_development = min(0.6, 0.1 + (age_months - 18) * 0.0167)
        elif age_months <= 96:
            # 4-8 years: continued grammar development
            self.grammar_development = min(0.9, 0.6 + (age_months - 48) * 0.0063)
        else:
            # After 8 years: refinement
            self.grammar_development = min(1.0, 0.9 + (age_months - 96) * 0.001)
        
        # Babbling development
        if age_months <= 12:
            # First year: babbling development
            self.babbling_development = min(1.0, age_months * 0.083)
        else:
            # After first year: babbling decreases as language develops
            self.babbling_development = max(0.0, 1.0 - (age_months - 12) * 0.05)
        
        # Max utterance length
        if age_months <= 18:
            # First 18 months: single words
            self.max_utterance_length = 1
        elif age_months <= 24:
            # 18-24 months: two-word combinations
            self.max_utterance_length = 2
        elif age_months <= 36:
            # 2-3 years: simple sentences
            self.max_utterance_length = 3 + int((age_months - 24) / 4)
        elif age_months <= 60:
            # 3-5 years: more complex sentences
            self.max_utterance_length = 6 + int((age_months - 36) / 6)
        else:
            # After 5 years: increasingly complex sentences
            self.max_utterance_length = 10 + int((age_months - 60) / 12)
        
        # Comprehension level
        self.comprehension_level = self.receptive_language_development
    
    def _generate_utterance(self, developmental_stage: str, emotional_state: Dict[str, float]) -> str:
        """Generate a child's utterance based on developmental stage and emotional state.
        
        Args:
            developmental_stage: String representing the developmental stage
            emotional_state: Dictionary of the child's emotional state
            
        Returns:
            String representing the child's utterance
        """
        # Prenatal stage: no utterance
        if developmental_stage == "Prenatal":
            return ""
        
        # Infancy: babbling or single words
        if developmental_stage == "Infancy":
            # Determine if the child will babble or use words
            if random.random() < self.babbling_development or len(self.vocabulary) == 0:
                # Generate babbling
                num_syllables = random.randint(1, 3)
                babbling = "".join(random.choice(self.babbling_sounds) for _ in range(num_syllables))
                return babbling
            else:
                # Use a single word from vocabulary
                if self.vocabulary:
                    # Choose a word associated with the dominant emotion
                    dominant_emotion = max(emotional_state.items(), key=lambda x: x[1]) if emotional_state else ("neutral", 0.0)
                    emotion_related_words = []
                    
                    for word in self.vocabulary:
                        if word in self.word_emotional_associations and dominant_emotion[0] in self.word_emotional_associations[word]:
                            if self.word_emotional_associations[word][dominant_emotion[0]] > 0.3:
                                emotion_related_words.append(word)
                    
                    if emotion_related_words and random.random() < 0.7:
                        return random.choice(emotion_related_words)
                    else:
                        # Choose a random word from vocabulary, weighted by frequency
                        words = list(self.vocabulary)
                        weights = [self.word_frequency[word] for word in words]
                        total_weight = sum(weights)
                        if total_weight > 0:
                            weights = [w / total_weight for w in weights]
                            return np.random.choice(words, p=weights)
                        else:
                            return random.choice(words) if words else ""
                else:
                    return ""
        
        # Early Childhood: simple word combinations
        if developmental_stage == "Early Childhood":
            # Generate a simple utterance
            utterance_length = min(self.max_utterance_length, random.randint(1, 3))
            
            if utterance_length == 1 or len(self.vocabulary) < 2:
                # Single word
                if self.vocabulary:
                    words = list(self.vocabulary)
                    weights = [self.word_frequency[word] for word in words]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        return np.random.choice(words, p=weights)
                    else:
                        return random.choice(words) if words else ""
                else:
                    return ""
            else:
                # Word combination
                utterance = []
                
                # Start with a random word
                if self.vocabulary:
                    words = list(self.vocabulary)
                    weights = [self.word_frequency[word] for word in words]
                    total_weight = sum(weights)
                    if total_weight > 0:
                        weights = [w / total_weight for w in weights]
                        current_word = np.random.choice(words, p=weights)
                    else:
                        current_word = random.choice(words) if words else ""
                    
                    utterance.append(current_word)
                    
                    # Add associated words
                    for _ in range(utterance_length - 1):
                        if current_word in self.word_associations and self.word_associations[current_word]:
                            # Choose a word that has been associated with the current word
                            next_word = random.choice(self.word_associations[current_word])
                            if next_word in self.vocabulary:
                                utterance.append(next_word)
                                current_word = next_word
                            else:
                                # If the associated word is not in vocabulary, choose a random word
                                next_word = random.choice(list(self.vocabulary))
                                utterance.append(next_word)
                                current_word = next_word
                        else:
                            # If no associations, choose a random word
                            next_word = random.choice(list(self.vocabulary))
                            utterance.append(next_word)
                            current_word = next_word
                
                return " ".join(utterance)
        
        # Middle Childhood and beyond: more complex language
        # Use grammar patterns if available
        if self.grammar_patterns and random.random() < self.grammar_development:
            # Choose a grammar pattern weighted by frequency
            patterns = [p["pattern"] for p in self.grammar_patterns]
            weights = [p["frequency"] for p in self.grammar_patterns]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                pattern = np.random.choice(patterns, p=weights)
                
                # Replace some words in the pattern with words from vocabulary
                pattern_words = pattern.split()
                for i in range(len(pattern_words)):
                    if random.random() < 0.3 and self.vocabulary:
                        pattern_words[i] = random.choice(list(self.vocabulary))
                
                return " ".join(pattern_words)
            else:
                # Fall back to generating a simple utterance
                return self._generate_simple_utterance(utterance_length=self.max_utterance_length)
        else:
            # Generate a simple utterance
            return self._generate_simple_utterance(utterance_length=self.max_utterance_length)
    
    def _generate_simple_utterance(self, utterance_length: int = 3) -> str:
        """Generate a simple utterance.
        
        Args:
            utterance_length: Maximum length of the utterance
            
        Returns:
            String representing the utterance
        """
        if not self.vocabulary:
            return ""
        
        # Determine actual utterance length (random up to max)
        actual_length = random.randint(1, min(utterance_length, len(self.vocabulary)))
        
        # Generate utterance
        utterance = []
        for _ in range(actual_length):
            # Choose a word from vocabulary, weighted by frequency
            words = list(self.vocabulary)
            weights = [self.word_frequency[word] for word in words]
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
                word = np.random.choice(words, p=weights)
            else:
                word = random.choice(words)
            
            utterance.append(word)
        
        return " ".join(utterance)
    
    def get_vocabulary_size(self) -> int:
        """Get the vocabulary size.
        
        Returns:
            Integer representing the vocabulary size
        """
        return len(self.vocabulary)
    
    def get_language_development_metrics(self) -> Dict[str, float]:
        """Get the language development metrics.
        
        Returns:
            Dictionary of language development metrics
        """
        return {
            "receptive_language": self.receptive_language_development,
            "expressive_language": self.expressive_language_development,
            "grammar_development": self.grammar_development,
            "babbling_development": self.babbling_development,
            "max_utterance_length": self.max_utterance_length,
            "comprehension_level": self.comprehension_level
        }
    
    def save(self, directory: Path):
        """Save the component to a directory.
        
        Args:
            directory: Directory to save the component to
        """
        # Call parent save method
        super().save(directory)
        
        # Save additional state
        additional_state = {
            "receptive_language_development": self.receptive_language_development,
            "expressive_language_development": self.expressive_language_development,
            "vocabulary": list(self.vocabulary),
            "word_frequency": dict(self.word_frequency),
            "word_associations": dict(self.word_associations),
            "word_emotional_associations": self.word_emotional_associations,
            "grammar_patterns": self.grammar_patterns,
            "grammar_development": self.grammar_development,
            "babbling_development": self.babbling_development,
            "babbling_sounds": self.babbling_sounds,
            "max_utterance_length": self.max_utterance_length,
            "comprehension_level": self.comprehension_level
        }
        
        # Save additional state
        additional_state_path = directory / f"{self.name}_additional_state.json"
        with open(additional_state_path, "w") as f:
            import json
            json.dump(additional_state, f, indent=2)
    
    def load(self, directory: Path):
        """Load the component from a directory.
        
        Args:
            directory: Directory to load the component from
        """
        # Call parent load method
        super().load(directory)
        
        # Load additional state
        additional_state_path = directory / f"{self.name}_additional_state.json"
        if additional_state_path.exists():
            with open(additional_state_path, "r") as f:
                import json
                additional_state = json.load(f)
                self.receptive_language_development = additional_state["receptive_language_development"]
                self.expressive_language_development = additional_state["expressive_language_development"]
                self.vocabulary = set(additional_state["vocabulary"])
                self.word_frequency = Counter(additional_state["word_frequency"])
                
                # Convert word associations back to defaultdict
                self.word_associations = defaultdict(list)
                for word, associations in additional_state["word_associations"].items():
                    self.word_associations[word] = associations
                
                self.word_emotional_associations = additional_state["word_emotional_associations"]
                self.grammar_patterns = additional_state["grammar_patterns"]
                self.grammar_development = additional_state["grammar_development"]
                self.babbling_development = additional_state["babbling_development"]
                self.babbling_sounds = additional_state["babbling_sounds"]
                self.max_utterance_length = additional_state["max_utterance_length"]
                self.comprehension_level = additional_state["comprehension_level"] 