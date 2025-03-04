# production.py
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Union, Any
import logging
import random
import numpy as np
from pydantic import BaseModel, Field, field_validator

from language.developmental_stages import LanguageDevelopmentStage, LanguageCapabilities
from language.lexical_memory import LexicalItem, LexicalMemory
from language.syntactic_processor import SyntacticProcessor

logger = logging.getLogger("LanguageProduction")

class UtteranceTemplate(BaseModel):
    """Template for generating utterances at different developmental stages"""
    template: str
    min_stage: LanguageDevelopmentStage
    emotion_bias: Optional[Dict[str, float]] = None
    requires_words: List[str] = Field(default_factory=list)
    
    def is_applicable(self, stage: LanguageDevelopmentStage, 
                     available_words: List[str], 
                     dominant_emotion: Optional[str] = None) -> bool:
        """Check if this template can be used in the current context"""
        # Check stage requirement
        stage_values = list(LanguageDevelopmentStage)
        if stage_values.index(stage) < stage_values.index(self.min_stage):
            return False
        
        # Check word requirements
        if self.requires_words:
            if not all(word in available_words for word in self.requires_words):
                return False
        
        # Check emotion bias if dominant emotion provided
        if dominant_emotion and self.emotion_bias:
            if dominant_emotion not in self.emotion_bias or self.emotion_bias[dominant_emotion] < 0.3:
                return False
        
        return True

class LanguageProduction:
    """Manages the production of utterances based on the child's development stage"""
    
    def __init__(self, lexical_memory: LexicalMemory, syntactic_processor: SyntacticProcessor):
        """Initialize language production with dependencies"""
        self.lexical_memory = lexical_memory
        self.syntactic_processor = syntactic_processor
        self.templates = self._initialize_templates()
        self.babble_sounds = ["ba", "da", "ma", "ga", "na", "pa", "ta", "ah", "oh", "ee", "oo"]
        
        # Try to load NLTK's CMU pronouncing dictionary for phonetics
        try:
            import nltk
            from nltk.corpus import cmudict
            nltk.download('cmudict', quiet=True)
            self.cmudict = cmudict.dict()
            self.phonetics_available = True
        except (ImportError, LookupError):
            logger.warning("NLTK CMUdict not available; phonetic features limited")
            self.phonetics_available = False
            self.cmudict = {}
        
        logger.info("Language production module initialized")
    
    def _initialize_templates(self) -> List[UtteranceTemplate]:
        """Initialize utterance templates for different stages and emotions"""
        templates = []
        
        # Holophrastic stage (single words)
        templates.append(UtteranceTemplate(
            template="{noun}",
            min_stage=LanguageDevelopmentStage.HOLOPHRASTIC,
            emotion_bias={"joy": 0.8, "surprise": 0.7}
        ))
        
        templates.append(UtteranceTemplate(
            template="{noun}!",
            min_stage=LanguageDevelopmentStage.HOLOPHRASTIC,
            emotion_bias={"joy": 0.9, "surprise": 0.9}
        ))
        
        templates.append(UtteranceTemplate(
            template="{noun}...",
            min_stage=LanguageDevelopmentStage.HOLOPHRASTIC,
            emotion_bias={"sadness": 0.8, "fear": 0.7}
        ))
        
        templates.append(UtteranceTemplate(
            template="{noun}?",
            min_stage=LanguageDevelopmentStage.HOLOPHRASTIC,
            emotion_bias={"surprise": 0.8, "anticipation": 0.7}
        ))
        
        # Telegraphic stage (2-3 words, no grammar)
        templates.append(UtteranceTemplate(
            template="{noun} {verb}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} {verb}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            requires_words=["i", "me", "you"]
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} {noun}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            requires_words=["my", "your"]
        ))
        
        templates.append(UtteranceTemplate(
            template="more {noun}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            requires_words=["more"]
        ))
        
        templates.append(UtteranceTemplate(
            template="no {noun}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC,
            requires_words=["no"],
            emotion_bias={"anger": 0.7, "disgust": 0.7}
        ))
        
        templates.append(UtteranceTemplate(
            template="{adj} {noun}",
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC
        ))
        
        templates.append(UtteranceTemplate(
            template="{noun} {adj}",  # Postposed adjective, childlike
            min_stage=LanguageDevelopmentStage.TELEGRAPHIC
        ))
        
        # Simple syntax stage
        templates.append(UtteranceTemplate(
            template="{pronoun} {verb} {noun}",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["i", "me", "you"]
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} {verb} {adj}",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["i", "me", "you", "am", "is", "are"]
        ))
        
        templates.append(UtteranceTemplate(
            template="can {pronoun} {verb} {noun}?",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["can", "i", "you"]
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} want {noun}",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["i", "want"]
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} like {noun}",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["i", "like"],
            emotion_bias={"joy": 0.7, "trust": 0.7}
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} don't like {noun}",
            min_stage=LanguageDevelopmentStage.SIMPLE_SYNTAX,
            requires_words=["i", "don't", "like"],
            emotion_bias={"disgust": 0.7, "anger": 0.6}
        ))
        
        # Complex syntax stage
        templates.append(UtteranceTemplate(
            template="{pronoun} {verb} {noun} because {pronoun} {verb} {adj}",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            requires_words=["because"]
        ))
        
        templates.append(UtteranceTemplate(
            template="if {pronoun} {verb} {noun}, {pronoun} will {verb} {adj}",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            requires_words=["if", "will"]
        ))
        
        templates.append(UtteranceTemplate(
            template="{pronoun} think that {noun} {verb} {adj}",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            requires_words=["think", "that"]
        ))
        
        templates.append(UtteranceTemplate(
            template="when {pronoun} {verb}, {pronoun} feel {adj}",
            min_stage=LanguageDevelopmentStage.COMPLEX_SYNTAX,
            requires_words=["when", "feel"]
        ))
        
        # Advanced stage
        templates.append(UtteranceTemplate(
            template="{pronoun} would like to {verb} {noun} if {pronoun} could",
            min_stage=LanguageDevelopmentStage.ADVANCED,
            requires_words=["would", "like", "to", "if", "could"]
        ))
        
        templates.append(UtteranceTemplate(
            template="do {pronoun} think {pronoun} could {verb} {noun} together?",
            min_stage=LanguageDevelopmentStage.ADVANCED,
            requires_words=["do", "think", "could", "together"]
        ))
        
        return templates
    
    def generate_babble(self) -> str:
        """Generate pre-linguistic babbling sounds"""
        # Number of syllables increases with age
        syllable_count = random.randint(1, 3)
        syllables = random.choices(self.babble_sounds, k=syllable_count)
        return "".join(syllables)
    
    def _get_words_by_category(self, known_words: List[LexicalItem]) -> Dict[str, List[str]]:
        """Organize known words by grammatical category"""
        categories = {
            "noun": [],
            "verb": [],
            "adj": [],
            "adv": [],
            "pronoun": [],
            "function": []  # Articles, prepositions, etc.
        }
        
        # Basic pronouns that might be hard-coded in early development
        basic_pronouns = ["i", "me", "my", "mine", "you", "your", "it", "this", "that"]
        
        # Function words that might be available earlier than their complexity suggests
        basic_function = ["the", "a", "an", "in", "on", "with", "and", "but", "or", "not", 
                          "no", "yes", "can", "will", "to", "for", "of", "at", "by", "more"]
        
        for word in known_words:
            # Skip words with very low production confidence
            if word.production_confidence < 0.1:
                continue
                
            # Check basic pre-defined categories first
            if word.word.lower() in basic_pronouns:
                categories["pronoun"].append(word.word.lower())
                continue
                
            if word.word.lower() in basic_function:
                categories["function"].append(word.word.lower())
                continue
            
            # Then use POS information
            if word.pos == "NOUN":
                categories["noun"].append(word.word.lower())
            elif word.pos == "VERB":
                categories["verb"].append(word.word.lower())
            elif word.pos in ["ADJ", "JJ"]:
                categories["adj"].append(word.word.lower())
            elif word.pos in ["ADV", "RB"]:
                categories["adv"].append(word.word.lower())
            elif word.pos in ["PRON", "DET", "ADP", "CONJ", "PART"]:
                categories["function"].append(word.word.lower())
        
        return categories
    
    def generate_utterance(self, 
                          capabilities: LanguageCapabilities, 
                          emotional_state: Dict[str, float],
                          context: Optional[str] = None,
                          response_to: Optional[str] = None) -> str:
        """Generate an utterance based on current language capabilities and emotional state"""
        stage = capabilities.stage
        
        # Pre-linguistic stage just produces babbling
        if stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            return self.generate_babble()
        
        # Determine dominant emotion
        dominant_emotion = None
        max_intensity = 0.0
        for emotion, intensity in emotional_state.items():
            if intensity > max_intensity:
                dominant_emotion = emotion
                max_intensity = intensity
        
        # Get available words
        accessible_words = self.lexical_memory.get_most_accessible_words(
            count=50,  # Get a good selection to choose from
            min_recall=0.2,
            emotional_bias=emotional_state
        )
        
        # Skip if no words available
        if not accessible_words:
            return self.generate_babble()
        
        # Organize words by category
        word_categories = self._get_words_by_category(accessible_words)
        
        # Get a flat list of all known words
        all_known_words = []
        for category in word_categories.values():
            all_known_words.extend(category)
        
        # Find applicable templates
        applicable_templates = []
        for template in self.templates:
            if template.is_applicable(stage, all_known_words, dominant_emotion):
                applicable_templates.append(template)
        
        # If response_to is provided, try to incorporate referenced words
        if response_to and self.syntactic_processor.spacy_available:
            response_doc = self.syntactic_processor.nlp(response_to)
            important_words = [token.text.lower() for token in response_doc 
                              if token.is_alpha and not token.is_stop and len(token.text) > 1]
            
            # Filter to words the child knows
            known_response_words = [word for word in important_words if word in all_known_words]
            
            # Prioritize templates that can use these words
            if known_response_words and applicable_templates:
                # This will be used later when filling the template
                context_words = known_response_words
        
        # Fall back to holophrastic if no applicable templates
        if not applicable_templates and stage != LanguageDevelopmentStage.HOLOPHRASTIC:
            # Try holophrastic templates
            for template in self.templates:
                if template.min_stage == LanguageDevelopmentStage.HOLOPHRASTIC:
                    applicable_templates.append(template)
        
        # Fall back to babbling if still no templates
        if not applicable_templates:
            return self.generate_babble()
        
        # Apply emotion bias to template selection
        if dominant_emotion:
            emotion_weighted = []
            for template in applicable_templates:
                weight = 1.0
                if template.emotion_bias and dominant_emotion in template.emotion_bias:
                    weight = 1.0 + template.emotion_bias[dominant_emotion]
                emotion_weighted.append((template, weight))
            
            # Weighted random selection
            weights = [w for _, w in emotion_weighted]
            template = random.choices(
                [t for t, _ in emotion_weighted],
                weights=weights,
                k=1
            )[0]
        else:
            # Random selection
            template = random.choice(applicable_templates)
        
        # Fill the template
        filled_template = self._fill_template(
            template.template, 
            word_categories,
            capabilities.grammar_complexity,
            context_words if 'context_words' in locals() else []
        )
        
        # Update production stats for used words
        words_in_utterance = filled_template.lower().split()
        for word in words_in_utterance:
            clean_word = word.strip(".,;:!?\"'()[]{}").lower()
            if clean_word in self.lexical_memory.words:
                self.lexical_memory.words[clean_word].update_after_production(success=True)
        
        # Add age-appropriate speech errors
        if stage in [LanguageDevelopmentStage.HOLOPHRASTIC, LanguageDevelopmentStage.TELEGRAPHIC]:
            filled_template = self._add_speech_errors(filled_template, stage)
        
        return filled_template
    
    def _fill_template(self, 
                      template: str, 
                      word_categories: Dict[str, List[str]],
                      grammar_complexity: float,
                      context_words: List[str] = None) -> str:
        """Fill a template with appropriate words"""
        filled = template
        
        # Extract placeholders
        placeholders = [ph[1:-1] for ph in filled.split('{') if '}' in ph]
        
        # Map placeholder categories to word categories
        category_map = {
            "noun": "noun",
            "verb": "verb",
            "adj": "adj",
            "adv": "adv",
            "pronoun": "pronoun"
        }
        
        # Fill each placeholder
        for placeholder in placeholders:
            category = category_map.get(placeholder)
            if not category or not word_categories.get(category):
                # Fallback to function words for unknown placeholders
                category = "function"
            
            available_words = word_categories.get(category, [])
            
            # If we have context words that match the category, prioritize them
            if context_words:
                matching_context_words = []
                for word in context_words:
                    word_obj = self.lexical_memory.words.get(word)
                    if word_obj and word_obj.pos.lower() == category.upper():
                        matching_context_words.append(word)
                
                if matching_context_words and random.random() < 0.7:  # 70% chance to use context word
                    chosen_word = random.choice(matching_context_words)
                    filled = filled.replace(f"{{{placeholder}}}", chosen_word, 1)
                    continue
            
            if available_words:
                chosen_word = random.choice(available_words)
                filled = filled.replace(f"{{{placeholder}}}", chosen_word, 1)
            else:
                # Fallback for missing category
                filled = filled.replace(f"{{{placeholder}}}", "thing", 1)
        
        # Post-processing for verb tense based on grammar complexity
        if grammar_complexity < 0.5 and "verb" in word_categories and random.random() > grammar_complexity:
            # Apply errors to verbs based on grammar complexity
            words = filled.split()
            for i, word in enumerate(words):
                if word in word_categories["verb"]:
                    # 50% chance to use base form instead of correct inflection
                    if random.random() < 0.5:
                        word_obj = self.lexical_memory.words.get(word)
                        if word_obj:
                            words[i] = word_obj.lemma
            
            filled = " ".join(words)
        
        return filled
    
    def _add_speech_errors(self, utterance: str, stage: LanguageDevelopmentStage) -> str:
        """Add age-appropriate speech errors"""
        if stage == LanguageDevelopmentStage.PRE_LINGUISTIC:
            return utterance  # No modifications for babbling
        
        words = utterance.split()
        result = []
        
        for word in words:
            # Original word with punctuation preserved
            original = word
            # Clean word for processing
            clean_word = word.strip(".,;:!?\"'()[]{}").lower()
            
            if len(clean_word) <= 2:
                # Don't modify very short words
                result.append(original)
                continue
            
            # Apply phonetic simplifications
            if stage == LanguageDevelopmentStage.HOLOPHRASTIC:
                # More errors at holophrastic stage
                if len(clean_word) > 3 and random.random() < 0.7:
                    # Simplify longer words (truncate or reduce clusters)
                    if random.random() < 0.5:
                        # Truncate
                        clean_word = clean_word[:2]
                    else:
                        # Replace consonant clusters 
                        for cluster in ["str", "pl", "bl", "gr", "dr", "tr", "fr", "th"]:
                            if cluster in clean_word:
                                replacement = cluster[0]
                                clean_word = clean_word.replace(cluster, replacement)
                
                # Preserve punctuation
                for c in original:
                    if not c.isalnum():
                        clean_word += c
                        
                result.append(clean_word)
                
            elif stage == LanguageDevelopmentStage.TELEGRAPHIC:
                # Fewer errors at telegraphic stage
                if len(clean_word) > 4 and random.random() < 0.4:
                    # Simplify longer words
                    if random.random() < 0.3:
                        # Truncate
                        clean_word = clean_word[:3]
                    else:
                        # Replace difficult sounds
                        for difficult, simple in [("th", "d"), ("r", "w"), ("l", "w"), ("s", "th")]:
                            if difficult in clean_word and random.random() < 0.3:
                                clean_word = clean_word.replace(difficult, simple)
                
                # Preserve punctuation
                for c in original:
                    if not c.isalnum():
                        clean_word += c
                        
                result.append(clean_word)
                
            else:
                # No modifications for more advanced stages
                result.append(original)
        
        return " ".join(result)