# developmental_stages.py
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator
import logging

logger = logging.getLogger("LanguageDevelopment")

class LanguageFeature(str, Enum):
    """Features of language that develop over time"""
    PHONETICS = "phonetics"  # Sound production
    VOCABULARY = "vocabulary"  # Word learning
    GRAMMAR = "grammar"  # Sentence structure
    PRAGMATICS = "pragmatics"  # Language use in context
    COMPREHENSION = "comprehension"  # Understanding language
    EXPRESSION = "expression"  # Producing language

class LanguageDevelopmentStage(str, Enum):
    """
    Stages of language development, from first sounds to complex syntax
    Aligned with but more specific than HumanMindDevelopmentStage
    """
    PRE_LINGUISTIC = "pre_linguistic"  # 0-12 months (real age)
    HOLOPHRASTIC = "holophrastic"  # 12-18 months (single word)
    TELEGRAPHIC = "telegraphic"  # 18-24 months (2-3 word combinations)
    SIMPLE_SYNTAX = "simple_syntax"  # 2-3 years
    COMPLEX_SYNTAX = "complex_syntax"  # 3-5 years
    ADVANCED = "advanced"  # 5+ years

    @property
    def description(self) -> str:
        """Return description of this language development stage"""
        descriptions = {
            self.PRE_LINGUISTIC: "Babbling, crying, and other non-linguistic vocalizations.",
            self.HOLOPHRASTIC: "Single words used to express complex ideas (e.g., 'Milk' for 'I want milk').",
            self.TELEGRAPHIC: "Two to three word combinations without grammatical markers (e.g., 'Mommy go').",
            self.SIMPLE_SYNTAX: "Simple sentences with basic grammar (e.g., 'I want milk').",
            self.COMPLEX_SYNTAX: "Complex sentences with subordinate clauses (e.g., 'I want milk because I'm thirsty').",
            self.ADVANCED: "Full language competence including abstract concepts and sophisticated syntax."
        }
        return descriptions[self]
    
    @property
    def expected_utterance_length(self) -> Tuple[int, int]:
        """Return expected min and max words in utterances at this stage"""
        utterance_lengths = {
            self.PRE_LINGUISTIC: (0, 0),  # No words
            self.HOLOPHRASTIC: (0, 1),    # 0-1 words
            self.TELEGRAPHIC: (1, 3),     # 1-3 words
            self.SIMPLE_SYNTAX: (2, 5),   # 2-5 words
            self.COMPLEX_SYNTAX: (4, 10), # 4-10 words
            self.ADVANCED: (5, 20)        # 5-20 words
        }
        return utterance_lengths[self]
    
    @property
    def feature_levels(self) -> Dict[LanguageFeature, float]:
        """Return expected development level of each language feature at this stage"""
        levels = {
            self.PRE_LINGUISTIC: {
                LanguageFeature.PHONETICS: 0.2,
                LanguageFeature.VOCABULARY: 0.0,
                LanguageFeature.GRAMMAR: 0.0,
                LanguageFeature.PRAGMATICS: 0.1,
                LanguageFeature.COMPREHENSION: 0.1,
                LanguageFeature.EXPRESSION: 0.05
            },
            self.HOLOPHRASTIC: {
                LanguageFeature.PHONETICS: 0.4,
                LanguageFeature.VOCABULARY: 0.2,
                LanguageFeature.GRAMMAR: 0.0,
                LanguageFeature.PRAGMATICS: 0.2,
                LanguageFeature.COMPREHENSION: 0.3,
                LanguageFeature.EXPRESSION: 0.2
            },
            self.TELEGRAPHIC: {
                LanguageFeature.PHONETICS: 0.6,
                LanguageFeature.VOCABULARY: 0.4,
                LanguageFeature.GRAMMAR: 0.2,
                LanguageFeature.PRAGMATICS: 0.3,
                LanguageFeature.COMPREHENSION: 0.5,
                LanguageFeature.EXPRESSION: 0.4
            },
            self.SIMPLE_SYNTAX: {
                LanguageFeature.PHONETICS: 0.7,
                LanguageFeature.VOCABULARY: 0.6,
                LanguageFeature.GRAMMAR: 0.5,
                LanguageFeature.PRAGMATICS: 0.5,
                LanguageFeature.COMPREHENSION: 0.7,
                LanguageFeature.EXPRESSION: 0.6
            },
            self.COMPLEX_SYNTAX: {
                LanguageFeature.PHONETICS: 0.8,
                LanguageFeature.VOCABULARY: 0.8,
                LanguageFeature.GRAMMAR: 0.7,
                LanguageFeature.PRAGMATICS: 0.7,
                LanguageFeature.COMPREHENSION: 0.8,
                LanguageFeature.EXPRESSION: 0.8
            },
            self.ADVANCED: {
                LanguageFeature.PHONETICS: 1.0,
                LanguageFeature.VOCABULARY: 1.0,
                LanguageFeature.GRAMMAR: 1.0,
                LanguageFeature.PRAGMATICS: 1.0,
                LanguageFeature.COMPREHENSION: 1.0,
                LanguageFeature.EXPRESSION: 1.0
            }
        }
        return levels[self]

    @classmethod
    def get_stage_for_vocabulary_size(cls, vocabulary_size: int) -> 'LanguageDevelopmentStage':
        """Determine language stage based on vocabulary size"""
        if vocabulary_size < 5:
            return cls.PRE_LINGUISTIC
        elif vocabulary_size < 50:
            return cls.HOLOPHRASTIC
        elif vocabulary_size < 200:
            return cls.TELEGRAPHIC
        elif vocabulary_size < 500:
            return cls.SIMPLE_SYNTAX
        elif vocabulary_size < 1000:
            return cls.COMPLEX_SYNTAX
        else:
            return cls.ADVANCED
    
    @classmethod
    def get_stage_for_age(cls, age_days: float) -> 'LanguageDevelopmentStage':
        """Determine language stage based on age in days (accelerated)"""
        # Accelerated development for simulation
        if age_days < 3:
            return cls.PRE_LINGUISTIC
        elif age_days < 10:
            return cls.HOLOPHRASTIC
        elif age_days < 30:
            return cls.TELEGRAPHIC
        elif age_days < 90:
            return cls.SIMPLE_SYNTAX
        elif age_days < 180:
            return cls.COMPLEX_SYNTAX
        else:
            return cls.ADVANCED

class LanguageCapabilities(BaseModel):
    """Current language capabilities of the neural child"""
    stage: LanguageDevelopmentStage = Field(LanguageDevelopmentStage.PRE_LINGUISTIC)
    feature_levels: Dict[LanguageFeature, float] = Field(default_factory=lambda: {feature: 0.0 for feature in LanguageFeature})
    vocabulary_size: int = Field(0, ge=0)
    max_utterance_length: int = Field(0, ge=0)
    grammar_complexity: float = Field(0.0, ge=0.0, le=1.0)
    
    # This validator might be the issue - it's not initializing properly
    @field_validator('feature_levels')
    @classmethod
    def initialize_features(cls, v):
        """Initialize feature levels if not provided"""
        if not v:
            return {feature: 0.0 for feature in LanguageFeature}
        
        # Make sure all enum values are in the dictionary
        for feature in LanguageFeature:
            if feature not in v:
                v[feature] = 0.0
        return v
    
    def update_from_stage(self) -> None:
        """Update capabilities based on the current stage"""
        # Get expected values from the stage
        expected_levels = self.stage.feature_levels
        expected_length = self.stage.expected_utterance_length
        
        # Update values (gradual approach toward expected values)
        learning_rate = 0.1
        for feature, expected in expected_levels.items():
            current = self.feature_levels[feature]
            if expected > current:
                self.feature_levels[feature] = min(expected, current + learning_rate)
            
        self.max_utterance_length = expected_length[1]
        self.grammar_complexity = self.feature_levels[LanguageFeature.GRAMMAR]
    
    def update_from_vocabulary(self, vocabulary_size: int) -> None:
        """Update stage and capabilities based on vocabulary size"""
        self.vocabulary_size = vocabulary_size
        new_stage = LanguageDevelopmentStage.get_stage_for_vocabulary_size(vocabulary_size)
        
        if new_stage != self.stage:
            logger.info(f"Language development progressed from {self.stage} to {new_stage}")
            self.stage = new_stage
            self.update_from_stage()

class DevelopmentTracker:
    """Tracks and manages language development"""
    
    def __init__(self, initial_age_days: float = 0.0):
        """Initialize the language development tracker"""
        initial_stage = LanguageDevelopmentStage.get_stage_for_age(initial_age_days)
        self.capabilities = LanguageCapabilities(stage=initial_stage)
        self.capabilities.update_from_stage()
        self.age_days = initial_age_days
        
        logger.info(f"Language development initialized at stage: {initial_stage}")
    
    def update(self, age_days: float, vocabulary_size: int) -> None:
        """Update language development based on age and vocabulary"""
        self.age_days = age_days
        
        # Check if age alone suggests progression
        age_based_stage = LanguageDevelopmentStage.get_stage_for_age(age_days)
        vocab_based_stage = LanguageDevelopmentStage.get_stage_for_vocabulary_size(vocabulary_size)
        
        # Use the less advanced stage (development requires both age and vocabulary)
        stages = [age_based_stage, vocab_based_stage]
        ordered_stages = list(LanguageDevelopmentStage)
        current_stage_idx = min(ordered_stages.index(stage) for stage in stages)
        new_stage = ordered_stages[current_stage_idx]
        
        if new_stage != self.capabilities.stage:
            logger.info(f"Language development updated to stage: {new_stage} (age: {age_days:.1f}, vocab: {vocabulary_size})")
            self.capabilities.stage = new_stage
            
        self.capabilities.vocabulary_size = vocabulary_size
        self.capabilities.update_from_stage()
    
    def get_capabilities(self) -> LanguageCapabilities:
        """Get current language capabilities"""
        return self.capabilities