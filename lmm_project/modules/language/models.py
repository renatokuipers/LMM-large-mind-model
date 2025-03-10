from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

class PhonemeModel(BaseModel):
    """
    Model for phoneme recognition and processing
    
    Represents the sound units of language and their recognition capabilities
    """
    phoneme_inventory: Dict[str, float] = Field(
        default_factory=dict, 
        description="Phonemes and their recognition confidence (0.0-1.0)"
    )
    phonotactic_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules for phoneme combinations in the language"
    )
    phoneme_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Categories of phonemes (vowels, consonants, etc.)"
    )
    phoneme_features: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Phonological features for each phoneme"
    )
    language_specific_sounds: Set[str] = Field(
        default_factory=set,
        description="Language-specific phonemes being learned"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class WordModel(BaseModel):
    """
    Model for word learning and lexical knowledge
    
    Represents the vocabulary and word-related knowledge
    """
    vocabulary: Dict[str, float] = Field(
        default_factory=dict,
        description="Known words and their familiarity (0.0-1.0)"
    )
    word_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Categories of words (nouns, verbs, etc.)"
    )
    word_embeddings: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Semantic embeddings for known words"
    )
    word_associations: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Word associations and semantic networks"
    )
    word_frequencies: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency of word encounters"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class GrammarModel(BaseModel):
    """
    Model for grammatical knowledge and structure
    
    Represents the rules and patterns of language structure
    """
    grammatical_structures: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Known grammatical structures and patterns"
    )
    morphological_rules: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Rules for word formation and inflection"
    )
    syntactic_patterns: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Sentence construction patterns"
    )
    grammatical_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Categories of grammatical elements"
    )
    rule_confidence: Dict[str, float] = Field(
        default_factory=dict,
        description="Confidence in grammatical rules (0.0-1.0)"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class SemanticModel(BaseModel):
    """
    Model for semantic understanding and meaning
    
    Represents knowledge of word and sentence meanings
    """
    concept_network: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Network of concepts and their relationships"
    )
    semantic_features: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Semantic features for concepts"
    )
    contextual_meanings: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Context-dependent meanings"
    )
    semantic_categories: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Categories of semantic elements"
    )
    concept_embeddings: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Vector representations of concepts"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class ExpressionModel(BaseModel):
    """
    Model for language expression and generation
    
    Represents capabilities for producing language
    """
    expression_templates: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Templates for language production"
    )
    communication_intents: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Mappings of intents to expressions"
    )
    pragmatic_rules: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Rules for appropriate language use in context"
    )
    speech_acts: Dict[str, List[Dict[str, Any]]] = Field(
        default_factory=dict,
        description="Types of speech acts and their implementations"
    )
    fluency_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Metrics for expression fluency (0.0-1.0)"
    )
    last_updated: datetime = Field(default_factory=datetime.now)

class LanguageModel(BaseModel):
    """
    Comprehensive language acquisition and processing model
    
    Integrates all aspects of language knowledge and capability
    """
    phonemes: PhonemeModel = Field(default_factory=PhonemeModel)
    vocabulary: WordModel = Field(default_factory=WordModel)
    grammar: GrammarModel = Field(default_factory=GrammarModel)
    semantics: SemanticModel = Field(default_factory=SemanticModel)
    expression: ExpressionModel = Field(default_factory=ExpressionModel)
    
    developmental_level: float = Field(
        default=0.0, 
        ge=0.0, 
        le=1.0,
        description="Overall developmental level of language (0.0-1.0)"
    )
    
    component_levels: Dict[str, float] = Field(
        default_factory=lambda: {
            "phoneme_recognition": 0.0,
            "word_learning": 0.0,
            "grammar_acquisition": 0.0,
            "semantic_processing": 0.0,
            "expression_generation": 0.0
        },
        description="Development levels of individual components"
    )
    
    module_id: str = Field(default="language")
    last_updated: datetime = Field(default_factory=datetime.now)

class LanguageNeuralState(BaseModel):
    """
    Neural state information for language networks
    
    Tracks the state of neural networks for language components
    """
    phoneme_recognition_development: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Development level of phoneme recognition network"
    )
    word_learning_development: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Development level of word learning network"
    )
    grammar_acquisition_development: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Development level of grammar acquisition network"
    )
    semantic_processing_development: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Development level of semantic processing network"
    )
    expression_generation_development: float = Field(
        0.0, ge=0.0, le=1.0, 
        description="Development level of expression generation network"
    )
    
    # Track recent activations for each neural component
    recent_phoneme_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the phoneme recognition network"
    )
    recent_word_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the word learning network"
    )
    recent_grammar_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the grammar acquisition network"
    )
    recent_semantic_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the semantic processing network"
    )
    recent_expression_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the expression generation network"
    )
    
    # Network performance metrics
    phoneme_recognition_accuracy: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="Accuracy of phoneme recognition network"
    )
    word_learning_accuracy: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="Accuracy of word learning network"
    )
    grammar_acquisition_accuracy: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="Accuracy of grammar acquisition network"
    )
    semantic_processing_accuracy: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="Accuracy of semantic processing network"
    )
    expression_generation_accuracy: float = Field(
        0.5, ge=0.0, le=1.0, 
        description="Accuracy of expression generation network"
    )
    
    # Last update timestamp
    last_updated: datetime = Field(
        default_factory=datetime.now, 
        description="When neural state was last updated"
    )
    
    def update_accuracy(self, component: str, accuracy: float) -> None:
        """
        Update the accuracy for a specific component
        
        Args:
            component: The component to update
            accuracy: The new accuracy value
        """
        if component == "phoneme_recognition":
            self.phoneme_recognition_accuracy = accuracy
        elif component == "word_learning":
            self.word_learning_accuracy = accuracy
        elif component == "grammar_acquisition":
            self.grammar_acquisition_accuracy = accuracy
        elif component == "semantic_processing":
            self.semantic_processing_accuracy = accuracy
        elif component == "expression_generation":
            self.expression_generation_accuracy = accuracy
            
        self.last_updated = datetime.now()
    
    def add_activation(self, component: str, activation: Dict[str, Any]) -> None:
        """
        Add a new activation for a specific component
        
        Args:
            component: The component with the activation
            activation: The activation details
        """
        if component == "phoneme_recognition":
            self.recent_phoneme_activations.append(activation)
            # Keep only the most recent activations
            if len(self.recent_phoneme_activations) > 20:
                self.recent_phoneme_activations.pop(0)
        elif component == "word_learning":
            self.recent_word_activations.append(activation)
            if len(self.recent_word_activations) > 20:
                self.recent_word_activations.pop(0)
        elif component == "grammar_acquisition":
            self.recent_grammar_activations.append(activation)
            if len(self.recent_grammar_activations) > 20:
                self.recent_grammar_activations.pop(0)
        elif component == "semantic_processing":
            self.recent_semantic_activations.append(activation)
            if len(self.recent_semantic_activations) > 20:
                self.recent_semantic_activations.pop(0)
        elif component == "expression_generation":
            self.recent_expression_activations.append(activation)
            if len(self.recent_expression_activations) > 20:
                self.recent_expression_activations.pop(0)
                
        self.last_updated = datetime.now()
