# lexical_memory.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
import numpy as np
from pydantic import BaseModel, Field, field_validator

from language.developmental_stages import LanguageFeature

logger = logging.getLogger("LexicalMemory")

class LexicalItem(BaseModel):
    """A word in the child's lexical memory with its properties and associations"""
    word: str = Field(..., description="The word itself")
    lemma: str = Field(..., description="Base form of the word")
    pos: str = Field("NOUN", description="Part of speech")
    phonetic: str = Field("", description="Phonetic representation")
    learned_at: datetime = Field(default_factory=datetime.now)
    understanding: float = Field(0.1, ge=0.0, le=1.0, description="How well the word is understood")
    recall_strength: float = Field(0.1, ge=0.0, le=1.0, description="How easily the word can be recalled")
    production_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence in using the word")
    usage_count: int = Field(0, ge=0, description="How many times the word has been used")
    last_used: Optional[datetime] = Field(None, description="When the word was last used")
    
    # Semantic properties
    contexts: List[str] = Field(default_factory=list, description="Contexts where word was encountered")
    collocations: Dict[str, int] = Field(default_factory=dict, description="Words that often appear with this one")
    emotional_associations: Dict[str, float] = Field(default_factory=dict, description="Emotions associated with word")
    definition: Optional[str] = Field(None, description="Child's understanding of the word's meaning")
    
    @property
    def age_days(self) -> float:
        """Return how many days since this word was learned"""
        return (datetime.now() - self.learned_at).total_seconds() / 86400
    
    def update_after_exposure(self, 
                             context: str, 
                             collocated_words: Optional[List[str]] = None,
                             emotion: Optional[str] = None,
                             emotion_intensity: float = 0.0) -> None:
        """Update word properties after exposure in mother's speech or active use"""
        # Add context if new
        if context and context not in self.contexts:
            self.contexts.append(context)
        
        # Update understanding (plateaus as it approaches 1.0)
        learning_rate = 0.05 * (1.0 - self.understanding)
        self.understanding = min(1.0, self.understanding + learning_rate)
        
        # Update recall strength with smaller increment
        recall_increment = 0.03 * (1.0 - self.recall_strength)
        self.recall_strength = min(1.0, self.recall_strength + recall_increment)
        
        # Update collocations
        if collocated_words:
            for word in collocated_words:
                if word == self.word:
                    continue
                self.collocations[word] = self.collocations.get(word, 0) + 1
        
        # Update emotional association if provided
        if emotion:
            if emotion not in self.emotional_associations:
                self.emotional_associations[emotion] = emotion_intensity
            else:
                # Blend existing and new emotional association
                current = self.emotional_associations[emotion]
                self.emotional_associations[emotion] = (current * 0.8) + (emotion_intensity * 0.2)
    
    def update_after_production(self, success: bool = True) -> None:
        """Update word properties after the child uses it in speech"""
        self.usage_count += 1
        self.last_used = datetime.now()
        
        # Update production confidence based on success
        if success:
            # Successful use increases confidence
            confidence_increment = 0.05 * (1.0 - self.production_confidence)
            self.production_confidence = min(1.0, self.production_confidence + confidence_increment)
        else:
            # Unsuccessful use decreases confidence slightly
            self.production_confidence = max(0.0, self.production_confidence - 0.02)
        
        # Successful use also strengthens recall
        if success:
            self.recall_strength = min(1.0, self.recall_strength + 0.02)
    
    def decay_over_time(self, days_since_last_access: float = 1.0) -> None:
        """Apply natural forgetting/decay to the word's properties"""
        # Words decay if not used - inverse exponential decay
        decay_factor = min(0.95, 1.0 - (0.01 * days_since_last_access))
        
        self.recall_strength *= decay_factor
        self.production_confidence *= decay_factor
        
        # Understanding decays more slowly
        understanding_decay = min(0.99, 1.0 - (0.005 * days_since_last_access))
        self.understanding *= understanding_decay

class LexicalMemory:
    """Manages the child's lexical memory (vocabulary)"""
    
    def __init__(self):
        """Initialize lexical memory"""
        self.words: Dict[str, LexicalItem] = {}
        self.total_exposure_count: int = 0
        self.pos_counts: Dict[str, int] = {"NOUN": 0, "VERB": 0, "ADJ": 0, "ADV": 0, "PRON": 0, "DET": 0, "PREP": 0, "CONJ": 0, "INTJ": 0}
        
        try:
            import nltk
            from nltk.corpus import cmudict
            nltk.download('cmudict', quiet=True)
            self.cmudict = cmudict.dict()
            self.nltk_available = True
        except (ImportError, LookupError):
            logger.warning("NLTK or cmudict not available; phonetic features disabled")
            self.nltk_available = False
            self.cmudict = {}
            
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")
            self.spacy_available = True
        except (ImportError, OSError):
            logger.warning("spaCy not available; advanced linguistic features disabled")
            self.spacy_available = False
            self.nlp = None
            
        logger.info(f"Lexical memory initialized. NLTK: {self.nltk_available}, spaCy: {self.spacy_available}")
    
    def get_phonetic(self, word: str) -> str:
        """Get phonetic representation of a word using CMU dict if available"""
        if not self.nltk_available:
            return ""
        
        word = word.lower()
        if word in self.cmudict:
            # Get first pronunciation
            phonemes = self.cmudict[word][0]
            # Remove stress markers
            phonemes = [p.strip("0123456789") for p in phonemes]
            return " ".join(phonemes)
        return ""
    
    def analyze_with_spacy(self, word: str) -> Tuple[str, str]:
        """Analyze a word with spaCy to get lemma and POS"""
        if not self.spacy_available or not self.nlp:
            return word, "NOUN"  # Default
        
        doc = self.nlp(word)
        if len(doc) > 0:
            token = doc[0]
            return token.lemma_, token.pos_
        return word, "NOUN"
    
    def add_word(self, word: str, context: str = "", emotion: Optional[str] = None, 
                emotion_intensity: float = 0.0, definition: Optional[str] = None) -> LexicalItem:
        """Add a new word to lexical memory"""
        word = word.lower().strip()
        if not word:
            return None
        
        # Check if word already exists
        if word in self.words:
            # Update existing word
            self.words[word].update_after_exposure(context, None, emotion, emotion_intensity)
            return self.words[word]
        
        # Process with NLP tools if available
        lemma, pos = self.analyze_with_spacy(word)
        phonetic = self.get_phonetic(word)
        
        # Create new lexical item
        lexical_item = LexicalItem(
            word=word,
            lemma=lemma,
            pos=pos,
            phonetic=phonetic,
            contexts=[context] if context else [],
            definition=definition
        )
        
        # Add emotional association if provided
        if emotion and emotion_intensity > 0:
            lexical_item.emotional_associations[emotion] = emotion_intensity
        
        # Store the item
        self.words[word] = lexical_item
        
        # Update statistics
        self.pos_counts[pos] = self.pos_counts.get(pos, 0) + 1
        logger.info(f"Added new word to lexical memory: '{word}' ({pos})")
        
        return lexical_item
    
    def process_text(self, text: str, context: str = "", emotion: Optional[str] = None, 
                    emotion_intensity: float = 0.0) -> List[str]:
        """Process text to extract and update words"""
        self.total_exposure_count += 1
        learned_words = []
        
        if self.spacy_available and self.nlp:
            # Use spaCy for more accurate processing
            doc = self.nlp(text)
            
            # Extract content words and their collocations
            content_words = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            
            # Process each content word
            for word in content_words:
                if len(word) < 2:  # Skip very short words
                    continue
                    
                # Add or update word
                if word in self.words:
                    # Find collocated words in a 3-word window
                    collocations = []
                    for other_word in content_words:
                        if other_word != word and other_word in self.words:
                            collocations.append(other_word)
                    
                    self.words[word].update_after_exposure(context, collocations, emotion, emotion_intensity)
                else:
                    # Small chance of learning new word passively
                    if random.random() < 0.1:  # 10% chance to learn passively
                        lexical_item = self.add_word(word, context, emotion, emotion_intensity * 0.5)
                        if lexical_item:
                            learned_words.append(word)
            
            # Process named entities separately
            for ent in doc.ents:
                entity_text = ent.text.lower()
                if entity_text not in self.words:
                    self.add_word(entity_text, context, emotion, emotion_intensity, f"A {ent.label_}")
                    learned_words.append(entity_text)
        else:
            # Simple fallback if spaCy is not available
            simple_tokens = text.lower().split()
            for token in simple_tokens:
                # Clean token of punctuation
                clean_token = token.strip(".,;:!?\"'()[]{}").lower()
                if not clean_token or len(clean_token) < 2:
                    continue
                    
                if clean_token in self.words:
                    self.words[clean_token].update_after_exposure(context, None, emotion, emotion_intensity)
                elif random.random() < 0.05:  # 5% chance to learn new word
                    lexical_item = self.add_word(clean_token, context, emotion, emotion_intensity)
                    if lexical_item:
                        learned_words.append(clean_token)
        
        return learned_words
    
    def explicitly_teach_word(self, word: str, definition: str, example_usage: Optional[str] = None, 
                             emotion: Optional[str] = None, emotion_intensity: float = 0.0) -> LexicalItem:
        """Teacher explicitly teaches a word to the child"""
        word = word.lower().strip()
        
        # Process with NLP tools
        lemma, pos = self.analyze_with_spacy(word)
        phonetic = self.get_phonetic(word)
        
        # Build context from definition and example
        context = definition
        if example_usage:
            context += f" Example: {example_usage}"
            
        if word in self.words:
            # Update existing word with stronger learning effect
            self.words[word].update_after_exposure(context, None, emotion, emotion_intensity)
            # Update definition
            self.words[word].definition = definition
            # Learning is stronger for explicit teaching
            understanding_boost = 0.15 * (1.0 - self.words[word].understanding)
            self.words[word].understanding = min(1.0, self.words[word].understanding + understanding_boost)
            
            logger.info(f"Updated existing word through explicit teaching: '{word}'")
            return self.words[word]
        else:
            # Create new word with higher initial understanding
            lexical_item = LexicalItem(
                word=word,
                lemma=lemma,
                pos=pos,
                phonetic=phonetic,
                understanding=0.3,  # Higher initial understanding for explicit teaching
                recall_strength=0.3,
                contexts=[context],
                definition=definition
            )
            
            # Add emotional association if provided
            if emotion and emotion_intensity > 0:
                lexical_item.emotional_associations[emotion] = emotion_intensity
            
            # Store the item
            self.words[word] = lexical_item
            
            # Update statistics
            self.pos_counts[pos] = self.pos_counts.get(pos, 0) + 1
            logger.info(f"Explicitly taught new word: '{word}' ({pos})")
            
            return lexical_item
    
    def recall_word(self, cue: str, emotional_state: Optional[Dict[str, float]] = None, 
                   min_recall_strength: float = 0.0) -> Optional[LexicalItem]:
        """Recall a word based on a cue (e.g., semantic category, sound, etc.)"""
        if cue in self.words and self.words[cue].recall_strength >= min_recall_strength:
            # Direct recall by word
            return self.words[cue]
        
        # Try to find similar words
        candidates = []
        
        for word, item in self.words.items():
            # Skip words with insufficient recall strength
            if item.recall_strength < min_recall_strength:
                continue
                
            # Check for semantic similarity
            if cue in item.contexts or cue in item.collocations:
                score = item.recall_strength * 0.8
                candidates.append((item, score))
                continue
            
            # Check if cue is a part of the word or vice versa
            if cue in word or word in cue:
                score = item.recall_strength * 0.6 * (min(len(cue), len(word)) / max(len(cue), len(word)))
                candidates.append((item, score))
                continue
            
            # Check for emotional association if emotional state provided
            if emotional_state and item.emotional_associations:
                emotion_match = 0.0
                for emotion, intensity in emotional_state.items():
                    if emotion in item.emotional_associations:
                        emotion_match += intensity * item.emotional_associations[emotion]
                
                if emotion_match > 0.3:  # Threshold for emotional relevance
                    score = item.recall_strength * 0.5 * emotion_match
                    candidates.append((item, score))
        
        if candidates:
            # Sort by score and add randomness
            for i in range(len(candidates)):
                item, score = candidates[i]
                # Add a small random factor to simulate imperfect recall
                candidates[i] = (item, score * (0.8 + 0.4 * random.random()))
            
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0][0]
        
        return None
    
    def get_most_accessible_words(self, count: int = 10, min_recall: float = 0.2, 
                                 emotional_bias: Optional[Dict[str, float]] = None) -> List[LexicalItem]:
        """Get the most accessible words for production"""
        candidates = []
        
        for word, item in self.words.items():
            if item.recall_strength >= min_recall:
                score = item.recall_strength * 0.7 + item.production_confidence * 0.3
                
                # Apply emotional bias if provided
                if emotional_bias and item.emotional_associations:
                    emotion_match = 0.0
                    for emotion, intensity in emotional_bias.items():
                        if emotion in item.emotional_associations:
                            emotion_match += intensity * item.emotional_associations[emotion]
                    
                    score += emotion_match * 0.4
                
                # Apply recency effect
                if item.last_used:
                    days_since_used = (datetime.now() - item.last_used).total_seconds() / 86400
                    recency_factor = max(0.5, 1.0 - (days_since_used * 0.1))
                    score *= recency_factor
                
                # Add randomness
                score *= (0.8 + 0.4 * random.random())
                
                candidates.append((item, score))
        
        # Sort by score and return top words
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in candidates[:count]]
    
    def apply_memory_decay(self, days_elapsed: float = 1.0) -> None:
        """Apply natural decay to all words in memory"""
        for word in self.words.values():
            word.decay_over_time(days_elapsed)
    
    def vocabulary_statistics(self) -> Dict[str, Any]:
        """Get statistics about the child's vocabulary"""
        if not self.words:
            return {
                "total_words": 0,
                "average_understanding": 0.0,
                "by_pos": self.pos_counts,
                "recent_words": []
            }
        
        total = len(self.words)
        avg_understanding = sum(w.understanding for w in self.words.values()) / total
        avg_recall = sum(w.recall_strength for w in self.words.values()) / total
        avg_production = sum(w.production_confidence for w in self.words.values()) / total
        
        # Get most recently learned words
        recent = sorted(self.words.values(), key=lambda w: w.learned_at, reverse=True)[:10]
        recent_words = [w.word for w in recent]
        
        # Get most used words
        most_used = sorted(self.words.values(), key=lambda w: w.usage_count, reverse=True)[:10]
        most_used_words = [w.word for w in most_used]
        
        return {
            "total_words": total,
            "average_understanding": avg_understanding,
            "average_recall": avg_recall,
            "average_production": avg_production,
            "by_pos": self.pos_counts,
            "recent_words": recent_words,
            "most_used": most_used_words,
            "total_exposures": self.total_exposure_count
        }