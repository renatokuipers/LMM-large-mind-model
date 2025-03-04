# vocabulary.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
import os
from pathlib import Path
import json
from pydantic import BaseModel, Field, field_validator

from language.lexical_memory import LexicalMemory, LexicalItem
from language.semantic_network import SemanticNetwork, SemanticRelation

logger = logging.getLogger("VocabularyManager")

class VocabularyStatistics(BaseModel):
    """Statistics about the child's vocabulary development"""
    total_words: int = Field(0, ge=0)
    active_vocabulary: int = Field(0, ge=0)  # Words with high recall and production confidence
    passive_vocabulary: int = Field(0, ge=0)  # Words understood but not easily produced
    by_category: Dict[str, int] = Field(default_factory=dict)
    by_pos: Dict[str, int] = Field(default_factory=dict)
    recent_words: List[str] = Field(default_factory=list)
    most_used: List[str] = Field(default_factory=list)
    average_understanding: float = Field(0.0, ge=0.0, le=1.0)
    average_production: float = Field(0.0, ge=0.0, le=1.0)
    learning_acceleration: float = Field(0.0)  # Rate of vocabulary growth

class LearningEvent(BaseModel):
    """Record of a word learning event"""
    word: str
    learned_at: datetime = Field(default_factory=datetime.now)
    context: str
    source: str = Field("passive")  # "passive", "explicit", "inference"
    associated_emotion: Optional[str] = None
    emotion_intensity: float = Field(0.0, ge=0.0, le=1.0)

class VocabularyManager:
    """Manages the child's vocabulary acquisition and usage"""
    
    def __init__(self, 
                data_dir: Path = Path("./data/vocabulary"),
                simulation_speed: float = 1.0):
        """Initialize vocabulary manager"""
        self.data_dir = data_dir
        self.simulation_speed = simulation_speed
        
        # Ensure data directory exists
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        
        # Initialize components
        self.lexical_memory = LexicalMemory()
        self.semantic_network = SemanticNetwork()
        
        # Learning history
        self.learning_history: List[LearningEvent] = []
        self.vocabulary_size_over_time: Dict[str, int] = {}  # date -> size
        
        # Load common early vocabulary to bootstrap the system
        self._load_starter_vocabulary()
        
        logger.info("Vocabulary manager initialized")
    
    def _load_starter_vocabulary(self) -> None:
        """Load a small set of common early vocabulary words to bootstrap the system"""
        starter_words = [
            ("mommy", "mother, caregiver", "person"),
            ("daddy", "father, caregiver", "person"),
            ("baby", "small human child", "person"),
            ("milk", "white drink for babies", "food"),
            ("ball", "round toy that rolls", "toy"),
            ("dog", "animal that barks", "animal"),
            ("cat", "animal that meows", "animal"),
            ("no", "negative response", "function"),
            ("yes", "affirmative response", "function"),
            ("hi", "greeting", "function"),
            ("bye", "farewell", "function"),
            ("more", "additional quantity", "function"),
            ("up", "direction or position", "function"),
            ("down", "direction or position", "function")
        ]
        
        for word, definition, category in starter_words:
            # Add to lexical memory with very low understanding
            self.lexical_memory.add_word(word, definition, None, 0.0, definition)
            
            # Add to semantic network
            self.semantic_network.add_concept(word, category)
            
            # For certain word pairs, add relations
            if word == "mommy":
                self.semantic_network.add_relation("mommy", "person", SemanticRelation.IS_A)
            elif word == "daddy":
                self.semantic_network.add_relation("daddy", "person", SemanticRelation.IS_A)
            elif word == "dog":
                self.semantic_network.add_relation("dog", "animal", SemanticRelation.IS_A)
            elif word == "cat":
                self.semantic_network.add_relation("cat", "animal", SemanticRelation.IS_A)
                self.semantic_network.add_relation("cat", "dog", SemanticRelation.SIMILAR_TO)
        
        logger.info(f"Loaded {len(starter_words)} starter vocabulary words")
    
    def process_heard_speech(self, text: str, context: str = "", 
                            emotional_state: Optional[Dict[str, float]] = None) -> List[str]:
        """Process speech heard by the child, potentially learning new words"""
        if not text:
            return []
            
        # Determine dominant emotion if provided
        dominant_emotion = None
        max_intensity = 0.0
        if emotional_state:
            for emotion, intensity in emotional_state.items():
                if intensity > max_intensity:
                    dominant_emotion = emotion
                    max_intensity = intensity
        
        # Process with lexical memory
        new_words = self.lexical_memory.process_text(
            text, context, dominant_emotion, max_intensity
        )
        
        # Process with semantic network
        self.semantic_network.process_text(text, context, emotional_state)
        
        # Record learning events for new words
        for word in new_words:
            learning_event = LearningEvent(
                word=word,
                context=context,
                source="passive",
                associated_emotion=dominant_emotion,
                emotion_intensity=max_intensity
            )
            self.learning_history.append(learning_event)
            
            logger.info(f"Passively learned new word: '{word}'")
        
        # Update vocabulary size history
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.vocabulary_size_over_time[current_date] = len(self.lexical_memory.words)
        
        return new_words
    
    def explicitly_learn_word(self, 
                            word: str, 
                            definition: str, 
                            example_usage: Optional[str] = None,
                            emotional_state: Optional[Dict[str, float]] = None) -> LexicalItem:
        """Explicitly learn a new word (as when taught by mother)"""
        if not word:
            return None
            
        # Determine dominant emotion if provided
        dominant_emotion = None
        max_intensity = 0.0
        if emotional_state:
            for emotion, intensity in emotional_state.items():
                if intensity > max_intensity:
                    dominant_emotion = emotion
                    max_intensity = intensity
        
        # Add or update word in lexical memory with stronger learning effect
        lexical_item = self.lexical_memory.explicitly_teach_word(
            word=word,
            definition=definition,
            example_usage=example_usage,
            emotion=dominant_emotion,
            emotion_intensity=max_intensity
        )
        
        # Add to semantic network
        concept = self.semantic_network.add_concept(
            word=word, 
            emotions={dominant_emotion: max_intensity} if dominant_emotion else None
        )
        
        # Extract potential relations from definition
        if example_usage and self.lexical_memory.spacy_available and self.lexical_memory.nlp:
            combined_text = f"{definition} {example_usage}"
            doc = self.lexical_memory.nlp(combined_text)
            
            # Look for hypernym patterns like "X is a Y"
            for token in doc:
                if token.dep_ == "attr" and token.head.lemma_ in ["be", "is", "are"]:
                    for subj in [t for t in token.head.children if t.dep_ == "nsubj"]:
                        if subj.text.lower() == word.lower():
                            # This might be a hypernym relation
                            self.semantic_network.add_relation(
                                word, token.text.lower(), SemanticRelation.IS_A
                            )
        
        # Record learning event
        learning_event = LearningEvent(
            word=word,
            context=definition,
            source="explicit",
            associated_emotion=dominant_emotion,
            emotion_intensity=max_intensity
        )
        self.learning_history.append(learning_event)
        
        # Update vocabulary size history
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.vocabulary_size_over_time[current_date] = len(self.lexical_memory.words)
        
        logger.info(f"Explicitly learned word: '{word}'")
        return lexical_item
    
    def get_word(self, word: str) -> Optional[LexicalItem]:
        """Get a specific word from vocabulary if it exists"""
        word = word.lower().strip()
        return self.lexical_memory.words.get(word)
    
    def get_most_accessible_words(self, count: int = 10, 
                                emotional_state: Optional[Dict[str, float]] = None) -> List[LexicalItem]:
        """Get the most accessible words for production based on current state"""
        return self.lexical_memory.get_most_accessible_words(
            count=count, 
            min_recall=0.2,
            emotional_bias=emotional_state
        )
    
    def activate_related_words(self, seed_words: List[str], 
                             emotional_state: Optional[Dict[str, float]] = None) -> List[str]:
        """Activate words related to the seed words through semantic spreading activation"""
        # Convert seed words to lowercase
        seed_words = [w.lower().strip() for w in seed_words]
        
        # Filter to words that exist in our vocabulary
        valid_seeds = [w for w in seed_words if w in self.lexical_memory.words]
        
        if not valid_seeds:
            return []
        
        # Use semantic network for spreading activation
        activated_concepts = self.semantic_network.spread_activation(
            valid_seeds, 
            activation_level=0.8,
            decay_factor=0.6,
            max_depth=2
        )
        
        # For each activated concept, boost its lexical memory activation
        for concept in activated_concepts:
            if concept in self.lexical_memory.words:
                lexical_item = self.lexical_memory.words[concept]
                # Increase recall strength temporarily
                recall_boost = 0.1
                lexical_item.recall_strength = min(1.0, lexical_item.recall_strength + recall_boost)
        
        return activated_concepts
    
    def update_after_child_production(self, words_used: List[str], 
                                    successful_communication: bool = True) -> None:
        """Update vocabulary stats after the child produces speech"""
        if not words_used:
            return
            
        # Update usage statistics for used words
        for word in words_used:
            word = word.lower().strip()
            if word in self.lexical_memory.words:
                self.lexical_memory.words[word].update_after_production(successful_communication)
    
    def apply_memory_decay(self, days_elapsed: float = 1.0) -> None:
        """Apply natural decay to vocabulary over time"""
        # Apply scaled decay based on simulation speed
        effective_days = days_elapsed * self.simulation_speed
        
        # Apply decay to lexical memory
        self.lexical_memory.apply_memory_decay(effective_days)
        
        # Apply decay to semantic network
        self.semantic_network.apply_decay(effective_days)
    
    def get_vocabulary_statistics(self) -> VocabularyStatistics:
        """Get comprehensive statistics about vocabulary development"""
        # Get basic lexical statistics
        lexical_stats = self.lexical_memory.vocabulary_statistics()
        
        # Calculate active vs passive vocabulary
        active_vocab = sum(1 for w in self.lexical_memory.words.values() 
                         if w.production_confidence >= 0.5 and w.recall_strength >= 0.6)
        
        passive_vocab = sum(1 for w in self.lexical_memory.words.values() 
                          if w.understanding >= 0.4 and (w.production_confidence < 0.5 or w.recall_strength < 0.6))
        
        # Calculate vocabulary growth acceleration
        dates = sorted(self.vocabulary_size_over_time.keys())
        if len(dates) >= 3:
            recent_dates = dates[-3:]
            sizes = [self.vocabulary_size_over_time[date] for date in recent_dates]
            if len(sizes) >= 2:
                # Calculate average daily growth over recent periods
                growth_rates = [(sizes[i] - sizes[i-1]) for i in range(1, len(sizes))]
                learning_acceleration = sum(growth_rates) / len(growth_rates)
            else:
                learning_acceleration = 0.0
        else:
            learning_acceleration = 0.0
        
        # Get category distribution
        by_category = {}
        for concept in self.semantic_network.concepts.values():
            if concept.category:
                by_category[concept.category] = by_category.get(concept.category, 0) + 1
        
        return VocabularyStatistics(
            total_words=lexical_stats["total_words"],
            active_vocabulary=active_vocab,
            passive_vocabulary=passive_vocab,
            by_category=by_category,
            by_pos=lexical_stats["by_pos"],
            recent_words=lexical_stats["recent_words"],
            most_used=lexical_stats["most_used"],
            average_understanding=lexical_stats["average_understanding"],
            average_production=lexical_stats["average_production"],
            learning_acceleration=learning_acceleration
        )
    
    def get_recent_learning_events(self, count: int = 10) -> List[LearningEvent]:
        """Get the most recent learning events"""
        sorted_events = sorted(self.learning_history, key=lambda x: x.learned_at, reverse=True)
        return sorted_events[:count]
    
    def save_state(self, filename: Optional[str] = None) -> None:
        """Save the state of the vocabulary manager"""
        if not filename:
            filename = f"vocabulary_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
        filepath = self.data_dir / filename
        
        # Save lexical memory and semantic network separately
        self.lexical_memory.apply_memory_decay(0.1)  # Apply a small decay to reflect save time
        self.semantic_network.save_state(f"semantic_network_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Prepare basic data for serialization
        state = {
            "vocabulary_size": len(self.lexical_memory.words),
            "vocabulary_size_over_time": self.vocabulary_size_over_time,
            "learning_history": [event.model_dump() for event in self.learning_history[-100:]],  # Last 100 events
            "statistics": self.get_vocabulary_statistics().model_dump()
        }
        
        # Convert datetime objects to strings
        state_json = json.dumps(state, default=lambda o: o.isoformat() if isinstance(o, datetime) else o)
        
        with open(filepath, 'w') as f:
            f.write(state_json)
            
        logger.info(f"Vocabulary state saved to {filepath}")
    
    def load_state(self, lexical_filepath: Path, semantic_filepath: Path) -> None:
        """Load the vocabulary state from saved files"""
        # Load lexical memory (would require implementation in LexicalMemory)
        # self.lexical_memory.load_state(lexical_filepath)
        
        # Load semantic network
        self.semantic_network.load_state(semantic_filepath)
        
        # Update vocabulary size history with current size
        current_date = datetime.now().strftime("%Y-%m-%d")
        self.vocabulary_size_over_time[current_date] = len(self.lexical_memory.words)
        
        logger.info(f"Vocabulary state loaded with {len(self.lexical_memory.words)} words")