"""
Language Component

This module implements the language acquisition and processing capabilities
of the child's mind. It models how language is learned through interactions,
starting from simple words and progressing to complex grammar.
"""

import re
import logging
import uuid
from datetime import datetime
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple, Optional, Any

import numpy as np
from nltk.tokenize import word_tokenize
import nltk

from ..utils.data_types import (
    Word, DevelopmentalStage, Emotion, EmotionType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure required NLTK data is available
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')


class LanguageComponent:
    """
    The LanguageComponent handles language acquisition and processing.
    
    It models how a child learns language through interactions, from
    simple word recognition to complex grammar and semantics.
    """
    
    def __init__(self, vocabulary_file: str = None):
        """
        Initialize the language component.
        
        Args:
            vocabulary_file: Path to a vocabulary file (optional)
        """
        # Initialize vocabulary storage
        self.vocabulary = {}
        self.word_frequency = Counter()
        self.co_occurrence = defaultdict(Counter)
        
        # Ensure NLTK data is available
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize language development attributes
        self.babbling_sounds = ["ba", "da", "ma", "pa", "ga", "ta"]
        self.sentence_patterns = []
        self.grammar_rules = {}
        self.acquired_phonemes = set()
        
        # Initialize metrics
        self.metrics = {
            "vocabulary_size": 0,
            "word_combinations": 0,
            "grammatical_complexity": 0,
            "pronunciation_accuracy": 0,
            "comprehension_level": 0
        }
        
        # For test compatibility
        self.grammar_complexity = 0.0
        
        # Log initialization
        logger.info("Language component initialized")
    
    def process_input(self, text: str, emotional_context: List[Emotion], 
                     developmental_stage: DevelopmentalStage) -> Dict[str, Any]:
        """
        Process language input based on developmental stage.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            developmental_stage: Current developmental stage
            
        Returns:
            Dictionary of processing results
        """
        # Lower and clean the text
        processed_text = text.lower()
        
        # Process differently based on stage
        if developmental_stage == DevelopmentalStage.INFANCY:
            return self._process_infant_input(processed_text, emotional_context)
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            return self._process_early_childhood_input(processed_text, emotional_context)
        elif developmental_stage == DevelopmentalStage.MIDDLE_CHILDHOOD:
            return self._process_middle_childhood_input(processed_text, emotional_context)
        elif developmental_stage == DevelopmentalStage.ADOLESCENCE:
            return self._process_adolescent_input(processed_text, emotional_context)
        else:  # EARLY_ADULTHOOD
            return self._process_adult_input(processed_text, emotional_context)
    
    def _process_infant_input(self, text: str, emotional_context: List[Emotion]) -> Dict[str, Any]:
        """
        Process input during infancy stage.
        
        At this stage, the focus is on phonological development and early word recognition.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            
        Returns:
            Dictionary of processing results
        """
        # Extract sounds that infant might recognize
        recognized_sounds = []
        
        # Check for simple sounds in input
        for sound in self.babbling_sounds:
            if sound in text:
                recognized_sounds.append(sound)
                
        # For very frequent words, infants may start to recognize them
        familiar_words = []
        for word, count in self.word_frequency.most_common(5):
            if word in text and count > 5:  # Needs repeated exposure
                familiar_words.append(word)
                
                # Strengthen word if already in vocabulary, otherwise add it
                if word in self.vocabulary:
                    self.vocabulary[word].understanding_level = min(
                        0.3,  # Cap for infancy
                        self.vocabulary[word].understanding_level + 0.05
                    )
                else:
                    self.vocabulary[word] = Word(
                        word=word,
                        understanding_level=0.1,  # Very basic understanding
                        associations={str(e.type): e.intensity for e in emotional_context}
                    )
        
        # Update frequency counters of all words in input
        words = re.findall(r'\b\w+\b', text)
        for word in words:
            self.word_frequency[word] += 1
        
        # Acquire easy phonemes
        for phoneme, difficulty in self.phoneme_difficulty.items():
            if difficulty <= 0.3 and phoneme in text:  # Early sounds are easier to acquire
                self.acquired_phonemes.add(phoneme)
        
        # Update metrics
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        self.metrics["pronunciation_accuracy"] = min(
            0.3,  # Cap for infancy
            0.1 + (len(self.acquired_phonemes) / len(self.phoneme_difficulty)) * 0.2
        )
        
        return {
            "recognized_sounds": recognized_sounds,
            "familiar_words": familiar_words,
            "phonological_development": len(self.acquired_phonemes) / len(self.phoneme_difficulty)
        }
    
    def _process_early_childhood_input(self, text: str, emotional_context: List[Emotion]) -> Dict[str, Any]:
        """
        Process input during early childhood.
        
        At this stage, vocabulary grows rapidly and simple word combinations emerge.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            
        Returns:
            Dictionary of processing results
        """
        # Tokenize into words, maintaining order
        try:
            words = word_tokenize(text)
        except:
            # Fallback if NLTK tokenizer fails
            words = text.split()
        
        understood_words = []
        new_words = []
        
        # Process each word
        for word in words:
            # Clean the word
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
                
            # Update frequency counter
            self.word_frequency[clean_word] += 1
            
            # Add or update in vocabulary
            if clean_word in self.vocabulary:
                # Strengthen existing word
                word_obj = self.vocabulary[clean_word]
                word_obj.usage_count += 1
                word_obj.last_used = datetime.now()
                word_obj.understanding_level = min(
                    0.7,  # Cap for early childhood
                    word_obj.understanding_level + 0.05
                )
                
                # Update emotional associations
                for emotion in emotional_context:
                    emotion_type = str(emotion.type)
                    if emotion_type in word_obj.associations:
                        # Blend existing association with new context
                        word_obj.associations[emotion_type] = (
                            word_obj.associations[emotion_type] * 0.8 +
                            emotion.intensity * 0.2
                        )
                    else:
                        word_obj.associations[emotion_type] = emotion.intensity
                
                # If understanding is sufficient, count as understood
                if word_obj.understanding_level >= 0.3:
                    understood_words.append(clean_word)
            else:
                # Learn new word
                self.vocabulary[clean_word] = Word(
                    word=clean_word,
                    associations={str(e.type): e.intensity for e in emotional_context},
                    understanding_level=0.2  # Basic understanding
                )
                new_words.append(clean_word)
        
        # Track word co-occurrences for simple syntax development
        for i, word1 in enumerate(words):
            word1 = re.sub(r'[^\w]', '', word1.lower())
            if not word1:
                continue
                
            # Look at adjacent words for simple co-occurrence patterns
            if i < len(words) - 1:
                word2 = re.sub(r'[^\w]', '', words[i+1].lower())
                if word2:
                    self.co_occurrence[word1][word2] += 1
        
        # Acquire intermediate phonemes
        for phoneme, difficulty in self.phoneme_difficulty.items():
            if difficulty <= 0.5 and phoneme in text:  # More sounds become available
                self.acquired_phonemes.add(phoneme)
        
        # Track simple word combinations (2-3 word phrases)
        simple_phrases = re.findall(r'\b\w+\s+\w+(?:\s+\w+)?\b', text)
        self.metrics["word_combinations"] = len(simple_phrases)
        
        # Update metrics
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        self.metrics["pronunciation_accuracy"] = min(
            0.6,  # Cap for early childhood
            0.3 + (len(self.acquired_phonemes) / len(self.phoneme_difficulty)) * 0.3
        )
        self.metrics["comprehension_level"] = min(
            0.5,  # Cap for early childhood
            self.metrics["comprehension_level"] + 0.02
        )
        
        return {
            "understood_words": understood_words,
            "new_words": new_words,
            "word_combinations": simple_phrases,
            "comprehension_level": self.metrics["comprehension_level"]
        }
    
    def _process_middle_childhood_input(self, text: str, emotional_context: List[Emotion]) -> Dict[str, Any]:
        """
        Process input during middle childhood.
        
        At this stage, grammar develops and language comprehension improves significantly.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            
        Returns:
            Dictionary of processing results
        """
        # Tokenize text
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        # Process vocabulary similarly to early childhood but with deeper understanding
        understood_words = []
        new_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
                
            self.word_frequency[clean_word] += 1
            
            if clean_word in self.vocabulary:
                word_obj = self.vocabulary[clean_word]
                word_obj.usage_count += 1
                word_obj.last_used = datetime.now()
                word_obj.understanding_level = min(
                    0.9,  # Higher cap for middle childhood
                    word_obj.understanding_level + 0.03
                )
                
                # Update emotional associations (more stable now)
                for emotion in emotional_context:
                    emotion_type = str(emotion.type)
                    if emotion_type in word_obj.associations:
                        # More weight to existing associations
                        word_obj.associations[emotion_type] = (
                            word_obj.associations[emotion_type] * 0.9 +
                            emotion.intensity * 0.1
                        )
                    else:
                        word_obj.associations[emotion_type] = emotion.intensity
                
                if word_obj.understanding_level >= 0.5:
                    understood_words.append(clean_word)
            else:
                # Learn new word with better initial understanding
                self.vocabulary[clean_word] = Word(
                    word=clean_word,
                    associations={str(e.type): e.intensity for e in emotional_context},
                    understanding_level=0.4  # Better initial understanding
                )
                new_words.append(clean_word)
        
        # Extract sentence patterns for grammar development
        # Simple pattern extraction based on sentence length and structure
        sentences = re.split(r'[.!?]', text)
        for sentence in sentences:
            if not sentence.strip():
                continue
                
            # Get sentence length as a simple measure of complexity
            sentence_len = len(sentence.split())
            
            # Use a simple representation of the pattern (could be more sophisticated)
            pattern = f"pattern_length_{sentence_len}"
            self.sentence_patterns.append(pattern)
            
            # Identify simple grammar rules
            # This is a very simplified approach - real grammar acquisition is much more complex
            if sentence_len >= 3:
                # Check for subject-verb-object pattern
                if len(words) >= 3:
                    rule = "subject_verb_object"
                    confidence = self.grammar_rules.get(rule, 0.0) + 0.05
                    self.grammar_rules[rule] = min(1.0, confidence)
        
        # Acquire more advanced phonemes
        for phoneme, difficulty in self.phoneme_difficulty.items():
            if difficulty <= 0.8 and phoneme in text:
                self.acquired_phonemes.add(phoneme)
        
        # Calculate grammatical complexity
        if self.sentence_patterns:
            # Average sentence length as a simple proxy for complexity
            total_patterns = len(self.sentence_patterns)
            weighted_sum = sum(int(p.split('_')[-1]) * count 
                              for p, count in Counter(self.sentence_patterns).items())
            avg_length = weighted_sum / total_patterns
            
            # Normalize to 0-1 range (assuming max complexity is ~15 words)
            norm_complexity = min(1.0, avg_length / 15)
            self.metrics["grammatical_complexity"] = min(
                0.8,  # Cap for middle childhood
                norm_complexity
            )
        
        # Update other metrics
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        self.metrics["pronunciation_accuracy"] = min(
            0.9,  # Cap for middle childhood
            0.6 + (len(self.acquired_phonemes) / len(self.phoneme_difficulty)) * 0.3
        )
        self.metrics["comprehension_level"] = min(
            0.8,  # Cap for middle childhood
            self.metrics["comprehension_level"] + 0.02
        )
        
        return {
            "understood_words": understood_words,
            "new_words": new_words,
            "grammatical_complexity": self.metrics["grammatical_complexity"],
            "comprehension_level": self.metrics["comprehension_level"],
            "sentence_patterns": len(self.sentence_patterns)
        }
    
    def _process_adolescent_input(self, text: str, emotional_context: List[Emotion]) -> Dict[str, Any]:
        """
        Process input during adolescence.
        
        At this stage, abstract language concepts develop and linguistic sophistication increases.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            
        Returns:
            Dictionary of processing results
        """
        # Similar vocabulary processing but with near-adult capabilities
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        understood_words = []
        new_words = []
        
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
                
            self.word_frequency[clean_word] += 1
            
            if clean_word in self.vocabulary:
                word_obj = self.vocabulary[clean_word]
                word_obj.usage_count += 1
                word_obj.last_used = datetime.now()
                word_obj.understanding_level = min(
                    0.95,  # Near-adult understanding
                    word_obj.understanding_level + 0.02
                )
                
                # Update associations (stable at this point)
                for emotion in emotional_context:
                    emotion_type = str(emotion.type)
                    if emotion_type in word_obj.associations:
                        word_obj.associations[emotion_type] = (
                            word_obj.associations[emotion_type] * 0.95 +
                            emotion.intensity * 0.05
                        )
                    else:
                        word_obj.associations[emotion_type] = emotion.intensity
                
                if word_obj.understanding_level >= 0.6:
                    understood_words.append(clean_word)
            else:
                # Learn new word with good initial understanding
                self.vocabulary[clean_word] = Word(
                    word=clean_word,
                    associations={str(e.type): e.intensity for e in emotional_context},
                    understanding_level=0.6  # Strong initial understanding
                )
                new_words.append(clean_word)
        
        # Check for advanced language features
        abstract_concept_indicators = [
            "concept", "theory", "principle", "idea", "philosophy", 
            "perspective", "belief", "value", "meaning", "symbolize",
            "represent", "metaphor", "abstract", "complex"
        ]
        
        # Count abstract concepts
        abstract_concepts_found = sum(1 for indicator in abstract_concept_indicators 
                                    if indicator in text.lower())
        
        # Look for complex sentence structures
        complex_structures = 0
        if "if" in text.lower() and "then" in text.lower():
            complex_structures += 1
        if "because" in text.lower() or "since" in text.lower():
            complex_structures += 1
        if "however" in text.lower() or "nevertheless" in text.lower():
            complex_structures += 1
        if "although" in text.lower() or "despite" in text.lower():
            complex_structures += 1
        
        # Acquire all phonemes
        for phoneme in self.phoneme_difficulty:
            if phoneme in text:
                self.acquired_phonemes.add(phoneme)
        
        # Update metrics for adolescent level
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        self.metrics["pronunciation_accuracy"] = min(
            1.0,  # Can reach full accuracy
            0.9 + (len(self.acquired_phonemes) / len(self.phoneme_difficulty)) * 0.1
        )
        self.metrics["grammatical_complexity"] = min(
            0.9,  # Near-adult complexity
            self.metrics["grammatical_complexity"] + 0.01 + (complex_structures * 0.02)
        )
        self.metrics["comprehension_level"] = min(
            0.9,  # Near-adult comprehension
            self.metrics["comprehension_level"] + 0.01 + (abstract_concepts_found * 0.01)
        )
        
        return {
            "understood_words": understood_words,
            "new_words": new_words,
            "abstract_concepts": abstract_concepts_found,
            "complex_structures": complex_structures,
            "grammatical_complexity": self.metrics["grammatical_complexity"],
            "comprehension_level": self.metrics["comprehension_level"]
        }
    
    def _process_adult_input(self, text: str, emotional_context: List[Emotion]) -> Dict[str, Any]:
        """
        Process input during early adulthood.
        
        At this stage, language processing reaches full maturity, with sophisticated
        understanding of nuance, metaphor, and complex linguistic structures.
        
        Args:
            text: The text to process
            emotional_context: Emotional context during processing
            
        Returns:
            Dictionary of processing results
        """
        # Full adult language processing
        try:
            words = word_tokenize(text)
        except:
            words = text.split()
        
        understood_words = []
        new_words = []
        
        # Process with adult-level understanding
        for word in words:
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if not clean_word:
                continue
                
            self.word_frequency[clean_word] += 1
            
            if clean_word in self.vocabulary:
                word_obj = self.vocabulary[clean_word]
                word_obj.usage_count += 1
                word_obj.last_used = datetime.now()
                word_obj.understanding_level = min(
                    1.0,  # Full understanding possible
                    word_obj.understanding_level + 0.01
                )
                
                # Stable associations
                for emotion in emotional_context:
                    emotion_type = str(emotion.type)
                    if emotion_type in word_obj.associations:
                        word_obj.associations[emotion_type] = (
                            word_obj.associations[emotion_type] * 0.98 +
                            emotion.intensity * 0.02
                        )
                    else:
                        word_obj.associations[emotion_type] = emotion.intensity
                
                understood_words.append(clean_word)
            else:
                # Learn new word with strong initial understanding
                self.vocabulary[clean_word] = Word(
                    word=clean_word,
                    associations={str(e.type): e.intensity for e in emotional_context},
                    understanding_level=0.7  # Strong initial understanding
                )
                new_words.append(clean_word)
        
        # Advanced language analysis
        # Look for specialized vocabulary
        specialized_vocab = ["analysis", "hypothesis", "methodology", "theoretical",
                           "integrate", "synthesize", "critique", "paradigm"]
        specialized_count = sum(1 for word in specialized_vocab if word in text.lower())
        
        # Look for nuanced language features
        nuance_indicators = ["subtle", "nuanced", "implies", "suggests", "indicates",
                           "perhaps", "possibly", "relatively", "comparatively"]
        nuance_count = sum(1 for word in nuance_indicators if word in text.lower())
        
        # Look for metaphors and analogies
        metaphor_indicators = ["like", "as", "metaphor", "analogy", "represents", "symbolizes"]
        potential_metaphors = 0
        for indicator in metaphor_indicators:
            if indicator in text.lower():
                potential_metaphors += 1
        
        # Final metrics - can reach full capacity
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        self.metrics["pronunciation_accuracy"] = 1.0  # Full accuracy
        self.metrics["grammatical_complexity"] = min(
            1.0,
            self.metrics["grammatical_complexity"] + 0.005 + (specialized_count * 0.01)
        )
        self.metrics["comprehension_level"] = min(
            1.0,
            self.metrics["comprehension_level"] + 0.005 + (nuance_count * 0.01)
        )
        
        return {
            "understood_words": understood_words,
            "new_words": new_words,
            "specialized_vocabulary": specialized_count,
            "linguistic_nuance": nuance_count,
            "potential_metaphors": potential_metaphors,
            "grammatical_complexity": self.metrics["grammatical_complexity"],
            "comprehension_level": self.metrics["comprehension_level"]
        }
    
    def generate_vocalization(self, 
                             developmental_stage: DevelopmentalStage, 
                             emotional_context: List[Emotion]) -> str:
        """
        Generate appropriate vocalization based on developmental stage.
        
        Args:
            developmental_stage: The current developmental stage
            emotional_context: Current emotional state
            
        Returns:
            Appropriate vocalization string
        """
        # Get dominant emotion
        dominant_emotion = None
        max_intensity = 0
        for emotion in emotional_context:
            if emotion.intensity > max_intensity:
                dominant_emotion = emotion
                max_intensity = emotion.intensity
        
        # Generate based on stage and emotion
        if developmental_stage == DevelopmentalStage.INFANCY:
            # Pre-linguistic babbling
            available_sounds = [s for s in self.babbling_sounds 
                               if any(phoneme in s for phoneme in self.acquired_phonemes)]
            
            if not available_sounds:
                available_sounds = self.babbling_sounds[:2]  # Default to first two sounds
                
            # Combine sounds based on emotion
            if dominant_emotion:
                if dominant_emotion.type == EmotionType.JOY:
                    return f"{np.random.choice(available_sounds)}-{np.random.choice(available_sounds)}, {np.random.choice(available_sounds)}"
                elif dominant_emotion.type == EmotionType.SADNESS:
                    return f"{np.random.choice(available_sounds)}... {np.random.choice(available_sounds)}"
                elif dominant_emotion.type == EmotionType.FEAR:
                    return f"{np.random.choice(available_sounds)}! {np.random.choice(available_sounds)}!"
                elif dominant_emotion.type == EmotionType.ANGER:
                    return f"{np.random.choice(available_sounds)}! {np.random.choice(available_sounds)}!"
                else:
                    return f"{np.random.choice(available_sounds)} {np.random.choice(available_sounds)}"
            else:
                return f"{np.random.choice(available_sounds)} {np.random.choice(available_sounds)}"
                
        elif developmental_stage == DevelopmentalStage.EARLY_CHILDHOOD:
            # Single words or 2-word phrases
            high_freq_words = [word for word, _ in self.word_frequency.most_common(20)]
            understood_words = [word for word, obj in self.vocabulary.items() 
                              if obj.understanding_level >= 0.4]
            
            # Combine available words
            available_words = list(set(high_freq_words + understood_words))
            if not available_words:
                # Fall back to babbling if no words available
                return self.generate_vocalization(DevelopmentalStage.INFANCY, emotional_context)
            
            # Single word or two-word combinations
            if len(available_words) >= 2 and np.random.random() < 0.5:
                return f"{np.random.choice(available_words)} {np.random.choice(available_words)}"
            else:
                return np.random.choice(available_words)
                
        else:
            # For older stages, use the text generation in other components
            # This is a fallback that shouldn't normally be used
            return "I'm trying to say something."
    
    def get_best_response_words(self, 
                               emotion_context: List[Emotion], 
                               topic_context: str,
                               limit: int = 10) -> List[str]:
        """
        Get the most relevant words for a response based on emotional and topic context.
        
        Args:
            emotion_context: Current emotional state
            topic_context: Topic or context for the response
            limit: Maximum number of words to return
            
        Returns:
            List of relevant words
        """
        # Calculate relevance scores for words based on emotion
        emotion_relevance = {}
        for word, word_obj in self.vocabulary.items():
            # Skip poorly understood words
            if word_obj.understanding_level < 0.3:
                continue
                
            # Calculate emotional relevance
            score = 0.0
            for emotion in emotion_context:
                emotion_type = str(emotion.type)
                if emotion_type in word_obj.associations:
                    score += word_obj.associations[emotion_type] * emotion.intensity
            
            emotion_relevance[word] = score
        
        # Generate embedding for topic context to find relevant words
        # This is simplified - in a full implementation, we'd use semantic similarity
        topic_words = set(re.findall(r'\b\w+\b', topic_context.lower()))
        
        # Calculate combined relevance
        combined_relevance = {}
        for word, emotion_score in emotion_relevance.items():
            # Give bonus for words related to topic
            topic_bonus = 0.5 if word in topic_words else 0.0
            
            # Include word frequency as a factor (familiar words are easier to use)
            frequency_factor = min(1.0, self.word_frequency.get(word, 0) / 10)
            
            # Understanding level affects likelihood of use
            understanding_factor = self.vocabulary[word].understanding_level
            
            # Combine factors
            combined_relevance[word] = (
                (emotion_score * 0.4) + 
                (topic_bonus * 0.3) + 
                (frequency_factor * 0.1) + 
                (understanding_factor * 0.2)
            )
        
        # Sort by relevance and return top words
        sorted_words = sorted(combined_relevance.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in sorted_words[:limit]]
    
    def get_development_metrics(self) -> Dict[str, float]:
        """
        Get the current language development metrics.
        
        Returns:
            Dictionary of development metrics
        """
        # Update vocabulary size metric
        self.metrics["vocabulary_size"] = len(self.vocabulary)
        
        return self.metrics.copy()
    
    def get_vocabulary_stats(self) -> Dict[str, Any]:
        """
        Get statistics about vocabulary development.
        
        Returns:
            Dictionary of vocabulary statistics
        """
        if not self.vocabulary:
            return {
                "total_words": 0,
                "avg_understanding": 0.0,
                "most_common_words": [],
                "word_categories": {}
            }
        
        # Calculate average understanding level
        avg_understanding = sum(word.understanding_level for word in self.vocabulary.values()) / len(self.vocabulary)
        
        # Get most common words
        most_common = [word for word, _ in self.word_frequency.most_common(10)]
        
        # Simple categorization of words (very simplified)
        categories = defaultdict(int)
        for word in self.vocabulary:
            if word in ["i", "me", "my", "mine"]:
                categories["personal"] += 1
            elif word in ["mom", "dad", "mommy", "daddy", "mama", "dada"]:
                categories["family"] += 1
            elif word in ["want", "need", "give", "take", "get"]:
                categories["desires"] += 1
            elif word in ["happy", "sad", "mad", "angry", "scared"]:
                categories["emotions"] += 1
            elif word in ["dog", "cat", "bird", "animal"]:
                categories["animals"] += 1
            elif word in ["car", "ball", "toy", "book"]:
                categories["objects"] += 1
            else:
                categories["other"] += 1
        
        return {
            "total_words": len(self.vocabulary),
            "avg_understanding": avg_understanding,
            "most_common_words": most_common,
            "word_categories": dict(categories)
        } 
    def process_language_input(self, text: str, emotional_context: List[Emotion] = None, 
                          developmental_stage: DevelopmentalStage = DevelopmentalStage.INFANCY,
                          age_months: int = 0) -> Dict[str, Any]:
        """
        Process language input - compatibility method for tests.
        
        Args:
            text: The input text
            emotional_context: Emotions associated with the input
            developmental_stage: Current developmental stage
            age_months: Age in months
            
        Returns:
            Dict with processing results
        """
        if emotional_context is None:
            emotional_context = []
            
        # Call the newer process_input method
        result = self.process_input(text, emotional_context, developmental_stage)
        
        # Update development metrics for test compatibility
        metrics = self.get_development_metrics()
        self.vocabulary_size = metrics["vocabulary_size"]
        self.grammatical_complexity = metrics["grammatical_complexity"]
        self.grammar_complexity = metrics["grammatical_complexity"]  # Alias
        
        return result
    
    def get_vocabulary_size(self) -> int:
        """Return the current vocabulary size for test compatibility."""
        return self.get_development_metrics()["vocabulary_size"]
        
    def get_grammar_complexity(self) -> float:
        """Return the current grammar complexity for test compatibility."""
        return self.get_development_metrics()["grammatical_complexity"]