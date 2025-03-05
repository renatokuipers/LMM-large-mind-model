"""
Language component for the NeuralChild project.

This module defines the LanguageComponent class that handles the child's
language acquisition, understanding, and production capabilities.
"""

from typing import Dict, List, Any, Optional, Set, Tuple
import re
import random
import math
import string
from enum import Enum
from pydantic import BaseModel, Field
from collections import Counter, defaultdict

from .base import NeuralComponent, ConnectionType
from ..config import DevelopmentalStage


class WordType(str, Enum):
    """Types of words in vocabulary."""
    NOUN = "noun"
    VERB = "verb"
    ADJECTIVE = "adjective"
    ADVERB = "adverb"
    PRONOUN = "pronoun"
    PREPOSITION = "preposition"
    CONJUNCTION = "conjunction"
    ARTICLE = "article"
    INTERJECTION = "interjection"
    UNKNOWN = "unknown"


class GrammaticalCategory(str, Enum):
    """Grammatical categories."""
    NUMBER = "number"  # Singular/plural
    TENSE = "tense"  # Past/present/future
    PERSON = "person"  # First/second/third person
    GENDER = "gender"  # Masculine/feminine/neuter
    CASE = "case"  # Nominative/accusative/etc.
    MOOD = "mood"  # Indicative/subjunctive/etc.
    ASPECT = "aspect"  # Perfect/imperfect/progressive
    VOICE = "voice"  # Active/passive


class Word(BaseModel):
    """Representation of a word in the vocabulary."""
    text: str
    word_type: WordType
    age_acquired: int = 0  # Age in days when acquired
    frequency: int = 1  # How often encountered
    comprehension_level: float = Field(default=0.1, ge=0.0, le=1.0)  # How well understood
    production_level: float = Field(default=0.0, ge=0.0, le=1.0)  # How well produced
    emotional_associations: Dict[str, float] = Field(default_factory=dict)  # Emotional associations
    conceptual_associations: Dict[str, float] = Field(default_factory=dict)  # Concept associations
    contexts: Set[str] = Field(default_factory=set)  # Contexts where word has been used
    
    class Config:
        arbitrary_types_allowed = True


class GrammarRule(BaseModel):
    """Representation of a grammar rule."""
    name: str
    pattern: str  # Pattern description
    comprehension_level: float = Field(default=0.0, ge=0.0, le=1.0)  # How well understood
    production_level: float = Field(default=0.0, ge=0.0, le=1.0)  # How well produced
    age_acquired: int = 0  # Age in days when acquired
    examples: List[str] = Field(default_factory=list)  # Example usages


class LanguageMetrics(BaseModel):
    """Metrics for language development."""
    vocabulary_size: int = 0
    noun_percentage: float = 0.0
    verb_percentage: float = 0.0
    adjective_percentage: float = 0.0
    complex_word_percentage: float = 0.0
    grammar_complexity: float = 0.0
    mlu: float = 0.0  # Mean Length of Utterance
    sentence_complexity: float = 0.0
    question_usage: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True


class LanguageComponent(NeuralComponent):
    """
    Component that handles language acquisition and processing.
    
    This component models how language develops from babbling to complex
    sentences through interaction with the environment.
    """
    
    def __init__(
        self,
        development_stage: DevelopmentalStage = DevelopmentalStage.PRENATAL,
        component_id: Optional[str] = None
    ):
        """
        Initialize the language component.
        
        Args:
            development_stage: Current developmental stage
            component_id: Optional ID (generated if not provided)
        """
        super().__init__(
            name="Language",
            activation_threshold=0.25,
            activation_decay_rate=0.08,
            learning_rate=0.05,
            development_stage=development_stage,
            component_id=component_id
        )
        
        # Vocabulary: map of word text to Word objects
        self.vocabulary: Dict[str, Word] = {}
        
        # Grammar rules: map of rule name to GrammarRule objects
        self.grammar_rules: Dict[str, GrammarRule] = {}
        
        # Word frequency count for all encountered words
        self.word_frequencies: Counter = Counter()
        
        # Recently heard phrases
        self.recent_phrases: List[str] = []
        
        # Language production history
        self.utterance_history: List[str] = []
        
        # Current language metrics
        self.metrics = LanguageMetrics()
        
        # Babbling sounds (consonants and vowels introduced gradually)
        self.babbling_consonants = ['m', 'b', 'p', 'd', 't', 'n']
        self.babbling_vowels = ['a', 'e', 'i', 'o', 'u']
        
        # Stage-specific features
        self.features_by_stage = {
            DevelopmentalStage.PRENATAL: {
                "can_comprehend": False,
                "can_produce": False,
                "max_vocabulary": 0,
                "max_utterance_length": 0,
                "can_babble": False
            },
            DevelopmentalStage.INFANCY: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 50,
                "max_utterance_length": 1,
                "can_babble": True
            },
            DevelopmentalStage.EARLY_CHILDHOOD: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 1000,
                "max_utterance_length": 4,
                "can_babble": False
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 5000,
                "max_utterance_length": 8,
                "can_babble": False
            },
            DevelopmentalStage.ADOLESCENCE: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 10000,
                "max_utterance_length": 15,
                "can_babble": False
            },
            DevelopmentalStage.EARLY_ADULTHOOD: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 20000,
                "max_utterance_length": 25,
                "can_babble": False
            },
            DevelopmentalStage.MID_ADULTHOOD: {
                "can_comprehend": True,
                "can_produce": True,
                "max_vocabulary": 25000,
                "max_utterance_length": 30,
                "can_babble": False
            }
        }
        
        # Set current features based on stage
        self.current_features = self.features_by_stage.get(
            self.development_stage, 
            self.features_by_stage[DevelopmentalStage.PRENATAL]
        )
        
        # Initialize simple grammar rules for early stages
        self._initialize_grammar_rules()
        
        # Additional metadata for language processing
        self.metadata.update({
            "comprehension_factor": 0.6,  # Comprehension develops faster than production
            "production_factor": 0.3,  # Production develops more slowly
            "retention_factor": 0.8,  # How well new words are retained
            "emotional_association_factor": 0.5,  # How strongly words associate with emotions
            "conceptual_association_factor": 0.4,  # How strongly words associate with concepts
            "imitation_factor": 0.7,  # How strongly the child imitates heard language
            "novel_generation_factor": 0.3,  # How much novel language is generated
            "current_mlu": 1.0,  # Mean Length of Utterance (words per utterance)
            "question_probability": 0.1,  # Probability of forming questions
        })
    
    def _initialize_grammar_rules(self) -> None:
        """Initialize basic grammar rules."""
        # Single-word utterances (early stage)
        self.grammar_rules["single_word"] = GrammarRule(
            name="single_word",
            pattern="WORD",
            comprehension_level=0.5,
            production_level=0.4,
            age_acquired=180,  # ~6 months
            examples=["mama", "dada", "ball", "milk"]
        )
        
        # Two-word utterances (early combination stage)
        self.grammar_rules["two_word"] = GrammarRule(
            name="two_word",
            pattern="NOUN + VERB or ADJECTIVE + NOUN",
            comprehension_level=0.2,
            production_level=0.1,
            age_acquired=365,  # ~1 year
            examples=["want milk", "big ball", "daddy go", "more cookie"]
        )
        
        # Simple sentences (early grammar)
        self.grammar_rules["simple_sentence"] = GrammarRule(
            name="simple_sentence",
            pattern="SUBJECT + VERB + OBJECT",
            comprehension_level=0.1,
            production_level=0.0,
            age_acquired=730,  # ~2 years
            examples=["I want milk", "Mommy read book", "Dog is big"]
        )
    
    def _on_stage_transition(
        self, 
        old_stage: DevelopmentalStage, 
        new_stage: DevelopmentalStage
    ) -> None:
        """
        Handle developmental stage transitions.
        
        Args:
            old_stage: Previous developmental stage
            new_stage: New developmental stage
        """
        # Call parent method
        super()._on_stage_transition(old_stage, new_stage)
        
        # Update current features
        self.current_features = self.features_by_stage.get(
            new_stage, 
            self.features_by_stage[DevelopmentalStage.PRENATAL]
        )
        
        # Add stage-specific grammar rules
        if new_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Add more complex grammar rules
            self.grammar_rules["questions"] = GrammarRule(
                name="questions",
                pattern="QUESTION_WORD + VERB + SUBJECT + OBJECT?",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=1095,  # ~3 years
                examples=["Where is ball?", "What doing mommy?", "Why go outside?"]
            )
            
            self.grammar_rules["negative"] = GrammarRule(
                name="negative",
                pattern="SUBJECT + AUXILIARY + NOT + VERB + OBJECT?",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=1095,  # ~3 years
                examples=["I not want that", "Mommy not go", "That not mine"]
            )
        
        elif new_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            # Add more advanced grammar rules
            self.grammar_rules["complex_sentence"] = GrammarRule(
                name="complex_sentence",
                pattern="SUBJECT + VERB + OBJECT + CONJUNCTION + CLAUSE",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=1825,  # ~5 years
                examples=[
                    "I want the toy that is blue", 
                    "We can go to park if it's sunny",
                    "The dog is barking because he's happy"
                ]
            )
            
            self.grammar_rules["passive_voice"] = GrammarRule(
                name="passive_voice",
                pattern="SUBJECT + BE + PAST_PARTICIPLE + BY + AGENT",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=2190,  # ~6 years
                examples=[
                    "The ball was thrown by me",
                    "The book was read by mommy",
                    "The cake was eaten by everyone"
                ]
            )
        
        elif new_stage == DevelopmentalStage.ADOLESCENCE:
            # Add more sophisticated grammar rules
            self.grammar_rules["conditional"] = GrammarRule(
                name="conditional",
                pattern="IF + CLAUSE + THEN + CLAUSE",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=3650,  # ~10 years
                examples=[
                    "If it rains tomorrow, then we will stay inside",
                    "If you finish your homework, then you can play outside",
                    "I would go to the party if I was invited"
                ]
            )
            
            self.grammar_rules["relative_clause"] = GrammarRule(
                name="relative_clause",
                pattern="NOUN + WHO/WHICH/THAT + CLAUSE",
                comprehension_level=0.1,
                production_level=0.0,
                age_acquired=3650,  # ~10 years
                examples=[
                    "The person who called earlier is my teacher",
                    "The book that I'm reading is interesting",
                    "The car which was parked outside is gone"
                ]
            )
    
    def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process language inputs and produce language outputs.
        
        Args:
            inputs: Dictionary containing:
                - 'text': Text to process (optional)
                - 'emotional_state': Current emotional state (optional)
                - 'context': Contextual information (optional)
                - 'age_days': Current age in days (required)
                
        Returns:
            Dictionary containing:
                - 'comprehension': Comprehension results
                - 'production': Production results
                - 'vocabulary_size': Current vocabulary size
                - 'newly_acquired_words': Words acquired in this processing
        """
        # Extract inputs
        text = inputs.get('text', '')
        emotional_state = inputs.get('emotional_state', {})
        context = inputs.get('context', {})
        age_days = inputs.get('age_days', 0)
        
        # Activate component based on text length
        if text:
            # Normalize activation stimulus based on text length and complexity
            words = re.findall(r'\b\w+\b', text.lower())
            complexity = sum(len(word) > 5 for word in words) / max(1, len(words))
            stimulus = min(1.0, 0.3 + (len(words) / 20) * (1 + complexity))
            self.activate(stimulus)
        
        # Initialize result structure
        result = {
            'comprehension': {
                'understood_words': [],
                'understood_phrases': [],
                'understood_meaning': None,
                'comprehension_level': 0.0
            },
            'production': {
                'utterance': None,
                'structure_used': None,
                'intended_meaning': None,
                'production_level': 0.0
            },
            'vocabulary_size': len(self.vocabulary),
            'newly_acquired_words': []
        }
        
        # If not activated enough or can't comprehend yet, return minimal results
        if self.activation < self.activation_threshold or not self.current_features["can_comprehend"]:
            return result
        
        # Process input text for comprehension if provided
        if text:
            comprehension_results = self._process_comprehension(text, context)
            result['comprehension'] = comprehension_results
            
            # Learn from comprehension
            newly_acquired = self._learn_from_comprehension(text, emotional_state, context, age_days)
            result['newly_acquired_words'] = newly_acquired
            result['vocabulary_size'] = len(self.vocabulary)
        
        # Generate language production if capable
        if self.current_features["can_produce"]:
            production_results = self._generate_production(emotional_state, context)
            result['production'] = production_results
        
        # Update language metrics
        self._update_metrics()
        result['metrics'] = self.metrics.dict()
        
        return result
    
    def _process_comprehension(
        self, 
        text: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process text for comprehension.
        
        Args:
            text: Text to comprehend
            context: Contextual information
            
        Returns:
            Comprehension results
        """
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Track understood words
        understood_words = []
        
        # Track comprehension level
        total_comprehension = 0.0
        
        # Check each word against vocabulary
        for word in words:
            if word in self.vocabulary:
                understood_words.append(word)
                total_comprehension += self.vocabulary[word].comprehension_level
                
                # Update word frequency
                self.vocabulary[word].frequency += 1
                self.word_frequencies[word] += 1
                
                # Add context if new
                if context and "context_tag" in context:
                    self.vocabulary[word].contexts.add(context["context_tag"])
            
        # Add to recent phrases
        self.recent_phrases.append(text)
        if len(self.recent_phrases) > 20:
            self.recent_phrases.pop(0)
        
        # Calculate average comprehension
        avg_comprehension = total_comprehension / max(1, len(words))
        
        # Identify phrases/structures based on grammar rules
        understood_phrases = []
        
        # Simple implementation - check if text matches examples in grammar rules
        for rule_name, rule in self.grammar_rules.items():
            for example in rule.examples:
                if example.lower() in text.lower():
                    understood_phrases.append({
                        "text": example,
                        "rule": rule_name,
                        "comprehension_level": rule.comprehension_level
                    })
                    
                    # If rule is understood, improves comprehension
                    avg_comprehension = max(avg_comprehension, rule.comprehension_level)
        
        # Generate a simple interpretation of meaning
        # This would be much more sophisticated in a real system
        understood_meaning = None
        
        # If enough words understood, attempt meaning interpretation
        if understood_words and len(understood_words) / max(1, len(words)) > 0.3:
            # Very simplified meaning extraction
            if "question_words" in context and any(qw in text.lower() for qw in context["question_words"]):
                understood_meaning = "This is asking about something"
            elif any(neg in text.lower() for neg in ["no", "not", "don't", "doesn't"]):
                understood_meaning = "This is saying not to do something"
            elif any(word in self.vocabulary and self.vocabulary[word].word_type == WordType.VERB for word in words):
                understood_meaning = "This is about doing something"
            else:
                understood_meaning = "This is telling me something"
        
        return {
            'understood_words': understood_words,
            'understood_phrases': understood_phrases,
            'understood_meaning': understood_meaning,
            'comprehension_level': avg_comprehension
        }
    
    def _learn_from_comprehension(
        self, 
        text: str, 
        emotional_state: Dict[str, Any],
        context: Dict[str, Any],
        age_days: int
    ) -> List[str]:
        """
        Learn new words and grammar from text.
        
        Args:
            text: Text to learn from
            emotional_state: Current emotional state
            context: Contextual information
            age_days: Current age in days
            
        Returns:
            List of newly acquired words
        """
        # Maximum vocabulary size at current stage
        max_vocab = self.current_features["max_vocabulary"]
        
        # Don't learn if vocabulary maxed out for stage
        if len(self.vocabulary) >= max_vocab:
            return []
        
        # Tokenize text
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Track newly acquired words
        newly_acquired = []
        
        # Extract emotion for associations
        primary_emotion = emotional_state.get('primary_emotion', '')
        emotion_intensity = emotional_state.get('intensity', 0.5)
        
        # Learning is affected by emotional state
        learning_boost = 1.0
        if primary_emotion in ["joy", "surprise", "anticipation"]:
            learning_boost = 1.3  # Positive emotions boost learning
        elif primary_emotion in ["fear", "anger"]:
            learning_boost = 0.7  # Negative emotions reduce learning
        
        # Process each word
        for word in words:
            # Skip if already in vocabulary
            if word in self.vocabulary:
                continue
                
            # Simple word type inference (would be more sophisticated in a real system)
            word_type = WordType.UNKNOWN
            
            # Basic word type inferencing
            if word in ["the", "a", "an"]:
                word_type = WordType.ARTICLE
            elif word in ["i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them"]:
                word_type = WordType.PRONOUN
            elif word in ["and", "but", "or", "nor", "for", "so", "yet"]:
                word_type = WordType.CONJUNCTION
            elif word in ["in", "on", "at", "by", "for", "with", "about", "against", "between"]:
                word_type = WordType.PREPOSITION
            elif word.endswith(("ly")):
                word_type = WordType.ADVERB
            elif word.endswith(("ed", "ing")):
                word_type = WordType.VERB
            elif word.endswith(("s", "es")) and len(word) > 3:
                word_type = WordType.NOUN
            else:
                # Default guesses based on position in sentence
                # This is a very simplified approach
                if len(words) > 1:
                    position = words.index(word)
                    if position == 0:
                        word_type = WordType.NOUN  # First word often a noun
                    elif position == len(words) - 1:
                        word_type = WordType.NOUN  # Last word often a noun
                    elif words[position - 1] in ["a", "an", "the"]:
                        word_type = WordType.NOUN  # After article, likely a noun
                    elif words[position - 1] in ["is", "are", "was", "were"]:
                        word_type = WordType.ADJECTIVE  # After be-verb, might be adjective
                    else:
                        word_type = WordType.NOUN  # Default to noun
            
            # Probability of acquisition depends on learning rate and context
            acquisition_probability = (
                self.learning_rate 
                * self.metadata["retention_factor"] 
                * learning_boost
            )
            
            # Context can increase acquisition probability
            if context:
                # If demonstrated with object, higher probability
                if context.get("demonstrated", False):
                    acquisition_probability *= 1.5
                
                # If repeated multiple times, higher probability
                if context.get("repetition_count", 0) > 1:
                    acquisition_probability *= min(2.0, 1.0 + context["repetition_count"] * 0.1)
                
                # If explicitly taught, higher probability
                if context.get("explicitly_taught", False):
                    acquisition_probability *= 2.0
            
            # Attempt acquisition based on probability
            if random.random() < acquisition_probability:
                # Create emotional associations
                emotional_assoc = {}
                if primary_emotion:
                    emotional_assoc[primary_emotion] = emotion_intensity * self.metadata["emotional_association_factor"]
                
                # Create conceptual associations
                conceptual_assoc = {}
                if "associated_concepts" in context:
                    for concept, strength in context["associated_concepts"].items():
                        conceptual_assoc[concept] = strength * self.metadata["conceptual_association_factor"]
                
                # Create new word entry
                self.vocabulary[word] = Word(
                    text=word,
                    word_type=word_type,
                    age_acquired=age_days,
                    frequency=1,
                    comprehension_level=0.3 * self.metadata["comprehension_factor"],
                    production_level=0.1 * self.metadata["production_factor"],
                    emotional_associations=emotional_assoc,
                    conceptual_associations=conceptual_assoc,
                    contexts=set([context.get("context_tag", "general")] if context else ["general"])
                )
                
                # Add to word frequencies
                self.word_frequencies[word] = 1
                
                # Track acquisition
                newly_acquired.append(word)
                
                # Limit acquisitions per processing
                if len(newly_acquired) >= 3:  # Don't learn too many words at once
                    break
        
        # Also attempt to improve grammar rule comprehension
        for rule_name, rule in self.grammar_rules.items():
            for example in rule.examples:
                if example.lower() in text.lower():
                    # Improve comprehension of this rule
                    comprehension_boost = 0.05 * self.metadata["comprehension_factor"] * learning_boost
                    rule.comprehension_level = min(1.0, rule.comprehension_level + comprehension_boost)
                    
                    # Smaller boost to production
                    production_boost = 0.03 * self.metadata["production_factor"] * learning_boost
                    rule.production_level = min(1.0, rule.production_level + production_boost)
                    break
        
        return newly_acquired
    
    def _generate_production(
        self, 
        emotional_state: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate language production.
        
        Args:
            emotional_state: Current emotional state
            context: Contextual information
            
        Returns:
            Production results
        """
        # If babbling stage, generate babbling
        if self.current_features["can_babble"] and self.development_stage == DevelopmentalStage.INFANCY:
            return self._generate_babbling()
        
        # Check if vocabulary is empty
        if not self.vocabulary:
            return {
                'utterance': None,
                'structure_used': None,
                'intended_meaning': None,
                'production_level': 0.0
            }
        
        # Determine what kind of structure to use based on stage and learned rules
        usable_rules = []
        for rule_name, rule in self.grammar_rules.items():
            # Only use rules with some production capability
            if rule.production_level > 0.2:
                usable_rules.append((rule_name, rule))
        
        # If no usable grammar rules, default to single words
        if not usable_rules and self.vocabulary:
            # Use single word (simple naming or requesting)
            # Choose words with higher production level
            candidates = [(word, info.production_level) for word, info in self.vocabulary.items()
                          if info.production_level > 0.3]
            
            # If no high-production words, use any words
            if not candidates:
                candidates = [(word, 0.1) for word in self.vocabulary.keys()]
            
            # Weight selection by production level
            words, weights = zip(*candidates)
            chosen_word = random.choices(words, weights=weights, k=1)[0]
            
            return {
                'utterance': chosen_word,
                'structure_used': "single_word",
                'intended_meaning': "naming or requesting",
                'production_level': self.vocabulary[chosen_word].production_level
            }
        
        # Select a grammar rule to use
        if usable_rules:
            selected_rule_name, selected_rule = random.choice(usable_rules)
            
            # Generate utterance based on rule
            if selected_rule_name == "single_word":
                # Single word utterance
                candidates = [(word, info.production_level) for word, info in self.vocabulary.items()
                             if info.production_level > 0.3]
                
                if not candidates:
                    candidates = [(word, 0.1) for word in self.vocabulary.keys()]
                
                words, weights = zip(*candidates)
                chosen_word = random.choices(words, weights=weights, k=1)[0]
                
                return {
                    'utterance': chosen_word,
                    'structure_used': selected_rule_name,
                    'intended_meaning': "naming or requesting",
                    'production_level': selected_rule.production_level
                }
            
            elif selected_rule_name == "two_word":
                # Two-word utterance
                # Find verbs or adjectives
                nouns = [word for word, info in self.vocabulary.items() 
                        if info.word_type == WordType.NOUN and info.production_level > 0.2]
                verbs = [word for word, info in self.vocabulary.items() 
                        if info.word_type == WordType.VERB and info.production_level > 0.2]
                adjectives = [word for word, info in self.vocabulary.items() 
                             if info.word_type == WordType.ADJECTIVE and info.production_level > 0.2]
                
                # Default to any words if specific types not available
                if not nouns:
                    nouns = list(self.vocabulary.keys())
                if not verbs:
                    verbs = nouns
                if not adjectives:
                    adjectives = nouns
                
                # Generate structure
                if random.random() < 0.5 and verbs:
                    # NOUN + VERB structure
                    noun = random.choice(nouns)
                    verb = random.choice(verbs)
                    utterance = f"{noun} {verb}"
                    intended_meaning = "action statement"
                else:
                    # ADJECTIVE + NOUN structure
                    adj = random.choice(adjectives)
                    noun = random.choice(nouns)
                    utterance = f"{adj} {noun}"
                    intended_meaning = "description"
                
                return {
                    'utterance': utterance,
                    'structure_used': selected_rule_name,
                    'intended_meaning': intended_meaning,
                    'production_level': selected_rule.production_level
                }
            
            elif selected_rule_name == "simple_sentence":
                # Simple sentence structure
                subjects = ["I", "me", "you", "mommy", "daddy"] + [
                    word for word, info in self.vocabulary.items() 
                    if info.word_type == WordType.NOUN and info.production_level > 0.3
                ]
                
                verbs = ["want", "like", "see", "have", "go"] + [
                    word for word, info in self.vocabulary.items() 
                    if info.word_type == WordType.VERB and info.production_level > 0.3
                ]
                
                objects = ["it", "that", "this"] + [
                    word for word, info in self.vocabulary.items() 
                    if info.word_type == WordType.NOUN and info.production_level > 0.3
                ]
                
                # Generate sentence
                subject = random.choice(subjects)
                verb = random.choice(verbs)
                obj = random.choice(objects)
                
                utterance = f"{subject} {verb} {obj}"
                
                return {
                    'utterance': utterance,
                    'structure_used': selected_rule_name,
                    'intended_meaning': "request or statement",
                    'production_level': selected_rule.production_level
                }
            
            # Add production for other grammar rules as needed
            
        # Default production (if no other cases matched)
        return {
            'utterance': None,
            'structure_used': None,
            'intended_meaning': None,
            'production_level': 0.0
        }
    
    def _generate_babbling(self) -> Dict[str, Any]:
        """
        Generate babbling sounds appropriate for infancy.
        
        Returns:
            Production results with babbling
        """
        # Number of babbling units to generate
        num_units = random.randint(1, 3)
        
        # Generate babbling
        babble_units = []
        for _ in range(num_units):
            consonant = random.choice(self.babbling_consonants)
            vowel = random.choice(self.babbling_vowels)
            
            # Create CV, CVCV, or CV-CV pattern
            if random.random() < 0.3:
                # Single syllable
                unit = f"{consonant}{vowel}"
            elif random.random() < 0.7:
                # Reduplicated syllable
                unit = f"{consonant}{vowel}-{consonant}{vowel}"
            else:
                # Two different syllables
                consonant2 = random.choice(self.babbling_consonants)
                vowel2 = random.choice(self.babbling_vowels)
                unit = f"{consonant}{vowel}-{consonant2}{vowel2}"
            
            babble_units.append(unit)
        
        # Join units
        babbling = " ".join(babble_units)
        
        return {
            'utterance': babbling,
            'structure_used': "babbling",
            'intended_meaning': "exploration/practice",
            'production_level': 0.5  # Babbling is relatively efficient at this stage
        }
    
    def _update_metrics(self) -> None:
        """Update language development metrics."""
        # Vocabulary size
        self.metrics.vocabulary_size = len(self.vocabulary)
        
        # Word type distributions
        total_words = len(self.vocabulary)
        if total_words > 0:
            noun_count = sum(1 for word in self.vocabulary.values() if word.word_type == WordType.NOUN)
            verb_count = sum(1 for word in self.vocabulary.values() if word.word_type == WordType.VERB)
            adj_count = sum(1 for word in self.vocabulary.values() if word.word_type == WordType.ADJECTIVE)
            
            self.metrics.noun_percentage = noun_count / total_words
            self.metrics.verb_percentage = verb_count / total_words
            self.metrics.adjective_percentage = adj_count / total_words
            
            # Complex word percentage (words with >5 characters)
            complex_count = sum(1 for word in self.vocabulary.keys() if len(word) > 5)
            self.metrics.complex_word_percentage = complex_count / total_words
        
        # Grammar complexity
        if self.grammar_rules:
            # Average comprehension level across all rules
            avg_comprehension = sum(rule.comprehension_level for rule in self.grammar_rules.values())
            avg_comprehension /= len(self.grammar_rules)
            
            # Average production level across all rules
            avg_production = sum(rule.production_level for rule in self.grammar_rules.values())
            avg_production /= len(self.grammar_rules)
            
            # Grammar complexity is weighted average of both
            self.metrics.grammar_complexity = (avg_comprehension * 0.6) + (avg_production * 0.4)
        
        # Mean Length of Utterance (from recent production history)
        if self.utterance_history:
            utterance_lengths = [len(re.findall(r'\b\w+\b', u)) for u in self.utterance_history]
            self.metrics.mlu = sum(utterance_lengths) / len(utterance_lengths)
        
        # Sentence complexity (simple estimate)
        self.metrics.sentence_complexity = self.metrics.grammar_complexity * self.metrics.mlu
        
        # Question usage (from recent production history)
        if self.utterance_history:
            question_count = sum(1 for u in self.utterance_history if '?' in u)
            self.metrics.question_usage = question_count / len(self.utterance_history)
        
        # Update component confidence based on metrics
        # Higher vocabulary and grammar complexity indicate better development
        success_rate = (
            (self.metrics.vocabulary_size / max(1, self.current_features["max_vocabulary"])) * 0.6 +
            self.metrics.grammar_complexity * 0.4
        )
        self.update_confidence(min(1.0, success_rate))
    
    def learn_word(
        self, 
        word: str, 
        word_type: WordType, 
        age_days: int,
        emotion: Optional[str] = None,
        emotion_intensity: float = 0.5,
        comprehension_level: float = 0.3,
        production_level: float = 0.1,
        context_tag: Optional[str] = None
    ) -> bool:
        """
        Explicitly learn a new word.
        
        Args:
            word: Word to learn
            word_type: Type of word
            age_days: Current age in days
            emotion: Associated emotion
            emotion_intensity: Intensity of emotional association
            comprehension_level: Initial comprehension level
            production_level: Initial production level
            context_tag: Context tag for the word
            
        Returns:
            Whether the word was successfully learned
        """
        # Skip if already in vocabulary
        if word in self.vocabulary:
            return False
        
        # Skip if at vocabulary limit for stage
        if len(self.vocabulary) >= self.current_features["max_vocabulary"]:
            return False
        
        # Create emotional associations
        emotional_assoc = {}
        if emotion:
            emotional_assoc[emotion] = emotion_intensity * self.metadata["emotional_association_factor"]
        
        # Create new word entry
        self.vocabulary[word] = Word(
            text=word,
            word_type=word_type,
            age_acquired=age_days,
            frequency=1,
            comprehension_level=comprehension_level * self.metadata["comprehension_factor"],
            production_level=production_level * self.metadata["production_factor"],
            emotional_associations=emotional_assoc,
            conceptual_associations={},
            contexts=set([context_tag] if context_tag else ["general"])
        )
        
        # Add to word frequencies
        self.word_frequencies[word] = 1
        
        return True
    
    def get_vocabulary_list(self) -> List[Dict[str, Any]]:
        """
        Get the current vocabulary list.
        
        Returns:
            List of word information
        """
        return [
            {
                "word": word,
                "type": info.word_type,
                "age_acquired": info.age_acquired,
                "comprehension": info.comprehension_level,
                "production": info.production_level,
                "frequency": info.frequency
            }
            for word, info in self.vocabulary.items()
        ]
    
    def get_most_frequent_words(self, limit: int = 10) -> List[Tuple[str, int]]:
        """
        Get the most frequently encountered words.
        
        Args:
            limit: Maximum number of words to return
            
        Returns:
            List of (word, frequency) tuples
        """
        return self.word_frequencies.most_common(limit)
    
    def get_language_metrics(self) -> Dict[str, Any]:
        """
        Get current language metrics.
        
        Returns:
            Dictionary of language metrics
        """
        return self.metrics.dict()