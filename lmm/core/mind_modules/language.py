"""
Language module for the Large Mind Model (LMM).

This module handles language processing and understanding for the LMM,
including text analysis, language complexity assessment, and concept extraction.
"""
from typing import Dict, List, Optional, Union, Any
import re
import nltk
from textblob import TextBlob
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from collections import defaultdict

from lmm.utils.config import get_config
from lmm.utils.logging import get_logger
from lmm.core.mind_modules.base import MindModule
from lmm.core.development.stages import DevelopmentalStage

# Initialize NLTK resources if needed
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/wordnet')
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('punkt', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)

# Initialize spaCy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    # If spaCy model is not found, download it
    import os
    if not os.path.exists(os.path.join(os.path.expanduser("~"), ".spacy")):
        os.makedirs(os.path.join(os.path.expanduser("~"), ".spacy"))
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize Hugging Face transformers for NER (lazy loading)
ner_model = None
ner_tokenizer = None

logger = get_logger("lmm.mind_modules.language")

class LanguageModule(MindModule):
    """
    Handles language processing for the LMM.
    
    This module manages language understanding, text analysis,
    concept extraction, and language complexity assessment, adapting
    to the LMM's developmental stage.
    """
    
    def __init__(self):
        """Initialize the Language Module."""
        super().__init__("Language")
        
        # Language development parameters
        self.vocabulary_size = 500  # Starts small, increases with development
        self.grammar_complexity = 0.3  # Linguistic complexity understanding
        self.abstraction_level = 0.2  # Ability to understand abstract concepts
        
        # Language processing history
        self.processing_history = []
        
        # Concept tracker
        self.learned_concepts = {}
        
        logger.info("Initialized Language Module")
    
    def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input for language operations.
        
        Args:
            input_data: Dictionary containing input data
                - input: Input text
                - relevant_memories: Optional relevant memories
                - emotional_state: Optional emotional state
                - developmental_stage: Current developmental stage
                
        Returns:
            Dictionary with language processing results
        """
        # Extract input parameters
        text = input_data.get("input", "")
        relevant_memories = input_data.get("relevant_memories", [])
        emotional_state = input_data.get("emotional_state", {})
        stage = input_data.get("developmental_stage", DevelopmentalStage.PRENATAL.value)
        
        # Update developmental parameters
        self._update_developmental_parameters(stage)
        
        # Process the text
        complexity = self._analyze_complexity(text)
        concepts = self._extract_concepts(text)
        sentiment = self._analyze_sentiment(text)
        entities = self._extract_entities(text)
        
        # Calculate comprehension level based on developmental parameters
        comprehension_level = self._calculate_comprehension(complexity)
        
        # Record processing
        self.processing_history.append({
            "text_length": len(text),
            "complexity": complexity,
            "concepts_count": len(concepts),
            "comprehension_level": comprehension_level,
            "developmental_stage": stage
        })
        
        # Limit history size
        if len(self.processing_history) > 100:
            self.processing_history = self.processing_history[-100:]
        
        # Update learned concepts
        for concept in concepts:
            if concept in self.learned_concepts:
                self.learned_concepts[concept]["count"] += 1
                if len(self.learned_concepts[concept]["examples"]) < 3:
                    self.learned_concepts[concept]["examples"].append(text[:50])
            else:
                self.learned_concepts[concept] = {
                    "count": 1,
                    "first_seen": stage,
                    "examples": [text[:50]]
                }
        
        # Return combined results
        return {
            "complexity": complexity,
            "concepts": concepts,
            "sentiment": sentiment,
            "entities": entities,
            "comprehension_level": comprehension_level,
            "vocabulary_size": self.vocabulary_size,
            "grammar_complexity": self.grammar_complexity,
            "abstraction_level": self.abstraction_level
        }
    
    def _analyze_complexity(self, text: str) -> Dict[str, float]:
        """
        Analyze the complexity of text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with complexity metrics
        """
        # Tokenize text
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        
        # Calculate basic metrics
        sentence_count = len(sentences)
        word_count = len(words)
        avg_sentence_length = word_count / max(1, sentence_count)
        
        # Calculate vocabulary richness
        unique_words = set(word.lower() for word in words)
        vocabulary_richness = len(unique_words) / max(1, word_count)
        
        # Calculate word complexity
        complex_words = [word for word in unique_words if len(word) > 6]
        word_complexity = len(complex_words) / max(1, len(unique_words))
        
        # Check for complex grammatical structures
        complex_structures = 0
        # Check for subordinate clauses
        for sentence in sentences:
            if any(marker in sentence.lower() for marker in 
                  ["because", "although", "though", "since", "while", "if", "unless"]):
                complex_structures += 1
        
        grammatical_complexity = complex_structures / max(1, sentence_count)
        
        # Overall complexity score (normalized to 0-1)
        overall_complexity = (
            0.2 * avg_sentence_length / 20 +  # Normalize by assuming max of 20 words per sentence
            0.3 * vocabulary_richness +
            0.3 * word_complexity +
            0.2 * grammatical_complexity
        )
        
        overall_complexity = min(1.0, overall_complexity)  # Cap at 1.0
        
        return {
            "overall": overall_complexity,
            "sentence_length": avg_sentence_length,
            "vocabulary_richness": vocabulary_richness,
            "word_complexity": word_complexity,
            "grammatical_complexity": grammatical_complexity
        }
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        
        Args:
            text: Input text
            
        Returns:
            List of key concepts
        """
        # Simplified concept extraction using NLTK
        words = nltk.word_tokenize(text.lower())
        
        # Remove stopwords (a simple set for demonstration)
        stopwords = {"the", "a", "an", "in", "on", "at", "of", "to", "for", "with", 
                    "and", "or", "but", "is", "are", "was", "were", "be", "been", 
                    "being", "have", "has", "had", "do", "does", "did", "will", 
                    "would", "shall", "should", "can", "could", "may", "might", 
                    "must", "i", "you", "he", "she", "it", "we", "they"}
        
        meaningful_words = [word for word in words if word.isalpha() and word not in stopwords]
        
        # Extract nouns and noun phrases as concepts
        text_blob = TextBlob(text)
        noun_phrases = text_blob.noun_phrases
        
        # Add individual nouns not caught in phrases
        pos_tags = text_blob.tags
        nouns = [word for word, tag in pos_tags if tag.startswith('NN')]
        
        # Combine and deduplicate
        all_concepts = list(set([phrase.lower() for phrase in noun_phrases] + 
                              [noun.lower() for noun in nouns]))
        
        # Sort by length (longer phrases are often more meaningful)
        all_concepts.sort(key=len, reverse=True)
        
        # Limit to top concepts
        return all_concepts[:10]
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment in text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary with sentiment analysis
        """
        # Use TextBlob for sentiment analysis
        blob = TextBlob(text)
        
        return {
            "polarity": blob.sentiment.polarity,  # -1.0 to 1.0
            "subjectivity": blob.sentiment.subjectivity  # 0.0 to 1.0
        }
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text using advanced NLP techniques.
        
        This comprehensive entity recognition system uses multiple techniques:
        1. spaCy's statistical NER model for base entity recognition
        2. Transformer-based NER model for advanced contextual entities
        3. Pattern-based recognition for specialized entities
        4. Entity linking to consolidate duplicate entities
        5. Confidence scoring based on multiple detection methods
        
        Args:
            text: Input text to analyze
            
        Returns:
            List of entity dictionaries with text, type, confidence, and metadata
        """
        global ner_model, ner_tokenizer
        
        # Early return for empty text
        if not text.strip():
            return []
            
        # Get development-appropriate complexity level
        config = get_config()
        stage = config.development.current_stage
        
        # Store all detected entities with their sources
        entity_candidates = defaultdict(list)  # (text, type) -> [source1, source2]
        
        # 1. Basic entity extraction with spaCy
        doc = nlp(text)
        for ent in doc.ents:
            entity_key = (ent.text, self._map_spacy_entity_type(ent.label_))
            entity_candidates[entity_key].append({
                "source": "spacy",
                "confidence": 0.75,  # Base confidence for spaCy
                "span": (ent.start_char, ent.end_char),
                "metadata": {"is_root": ent.root.text}
            })
        
        # 2. Advanced transformer-based NER for developmental stages beyond infancy
        if stage not in [DevelopmentalStage.PRENATAL.value, DevelopmentalStage.INFANCY.value]:
            # Lazy-load the transformer model only when needed
            if ner_model is None or ner_tokenizer is None:
                try:
                    model_name = "dslim/bert-base-NER"  # Efficient NER model
                    ner_tokenizer = AutoTokenizer.from_pretrained(model_name)
                    ner_model = AutoModelForTokenClassification.from_pretrained(model_name)
                    
                    logger.info(f"Loaded transformer NER model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load transformer NER model: {str(e)}")
            
            if ner_model is not None and ner_tokenizer is not None:
                try:
                    # Create NER pipeline
                    ner = pipeline("ner", model=ner_model, tokenizer=ner_tokenizer, aggregation_strategy="simple")
                    
                    # Get predictions
                    entities = ner(text)
                    
                    # Add to candidates
                    for entity in entities:
                        entity_type = entity['entity_group']
                        entity_text = entity['word']
                        entity_key = (entity_text, entity_type)
                        
                        entity_candidates[entity_key].append({
                            "source": "transformer",
                            "confidence": float(entity['score']),
                            "span": (entity['start'], entity['end']),
                            "metadata": {}
                        })
                except Exception as e:
                    logger.warning(f"Error in transformer NER processing: {str(e)}")
        
        # 3. Pattern-based recognition for specific entity types
        
        # 3.1 Names (capitalized sequences)
        name_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'
        potential_names = re.finditer(name_pattern, text)
        
        for match in potential_names:
            name = match.group(0)
            if name.lower() not in ["i", "you", "he", "she", "they", "we", "january", "february", 
                                   "march", "april", "may", "june", "july", "august", "september", 
                                   "october", "november", "december", "monday", "tuesday", "wednesday", 
                                   "thursday", "friday", "saturday", "sunday"]:
                entity_key = (name, "PERSON")
                entity_candidates[entity_key].append({
                    "source": "pattern",
                    "confidence": 0.6,  # Lower confidence for pattern matching
                    "span": (match.start(), match.end()),
                    "metadata": {"pattern": "capitalized"}
                })
        
        # 3.2 Numbers
        number_pattern = r'\b\d+(?:,\d+)*(?:\.\d+)?(?:\s*%|\s*dollars|\s*USD)?\b'
        numbers = re.finditer(number_pattern, text)
        
        for match in numbers:
            number = match.group(0)
            entity_type = "NUMBER"
            
            # Check for currency or percentage
            if any(currency in number.lower() for currency in ["$", "dollar", "usd", "eur", "euro"]):
                entity_type = "MONEY"
            elif "%" in number:
                entity_type = "PERCENTAGE"
                
            entity_key = (number, entity_type)
            entity_candidates[entity_key].append({
                "source": "pattern",
                "confidence": 0.8,  # Higher confidence for number patterns
                "span": (match.start(), match.end()),
                "metadata": {"pattern": "number"}
            })
        
        # 3.3 Dates
        date_patterns = [
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',  # MM/DD/YYYY
            r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4}\b',  # Month Day, Year
            r'\b\d{1,2}(?:st|nd|rd|th)?\s+(?:of\s+)?(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?,?\s+\d{2,4}\b'  # Day Month Year
        ]
        
        for pattern in date_patterns:
            dates = re.finditer(pattern, text, re.IGNORECASE)
            for match in dates:
                date = match.group(0)
                entity_key = (date, "DATE")
                entity_candidates[entity_key].append({
                    "source": "pattern",
                    "confidence": 0.85,  # High confidence for date patterns
                    "span": (match.start(), match.end()),
                    "metadata": {"pattern": "date"}
                })
        
        # 3.4 Locations (common location indicators)
        location_patterns = [
            r'\b(?:in|at|to|from)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',  # Preposition + capitalized
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:Street|Avenue|Road|Blvd|Boulevard|Lane|Drive|Court|Plaza|Square|Highway|Freeway|Pkwy|Parkway)\b'  # Street names
        ]
        
        for pattern in location_patterns:
            locations = re.finditer(pattern, text)
            for match in locations:
                # Extract the location part (group 1 if available, else full match)
                location = match.group(1) if match.lastindex and match.lastindex >= 1 else match.group(0)
                entity_key = (location, "LOCATION")
                entity_candidates[entity_key].append({
                    "source": "pattern",
                    "confidence": 0.7,
                    "span": (match.start(), match.end()),
                    "metadata": {"pattern": "location"}
                })
        
        # 4. Entity consolidation and confidence scoring
        consolidated_entities = []
        
        for (entity_text, entity_type), sources in entity_candidates.items():
            # Skip common words mistakenly identified as entities
            if entity_text.lower() in ["the", "a", "an", "and", "or", "but", "on", "in", "at", "to"]:
                continue
                
            # Calculate confidence based on number and quality of sources
            confidence = max(source["confidence"] for source in sources)
            
            # Boost confidence if multiple detection methods found this entity
            unique_sources = set(source["source"] for source in sources)
            if len(unique_sources) > 1:
                confidence = min(confidence + 0.1 * (len(unique_sources) - 1), 1.0)
            
            # Get spans and find the most complete mention
            spans = [source["span"] for source in sources]
            span = min(spans, key=lambda x: x[0])  # Use the earliest mention
            
            # Combine metadata
            combined_metadata = {}
            for source in sources:
                combined_metadata.update(source["metadata"])
            
            consolidated_entities.append({
                "text": entity_text,
                "type": entity_type,
                "confidence": confidence,
                "span": span,
                "metadata": combined_metadata,
                "detection_methods": list(unique_sources)
            })
        
        # Sort by confidence (highest first) then by position in text
        consolidated_entities.sort(key=lambda x: (-x["confidence"], x["span"][0]))
        
        # Limit number of entities based on developmental stage
        max_entities = {
            DevelopmentalStage.PRENATAL.value: 3,
            DevelopmentalStage.INFANCY.value: 5,
            DevelopmentalStage.CHILDHOOD.value: 10,
            DevelopmentalStage.ADOLESCENCE.value: 15,
            DevelopmentalStage.ADULTHOOD.value: 20
        }.get(stage, 10)
        
        return consolidated_entities[:max_entities]
    
    def _map_spacy_entity_type(self, spacy_type: str) -> str:
        """Map spaCy entity types to our standard types."""
        mapping = {
            "PERSON": "PERSON",
            "ORG": "ORGANIZATION",
            "GPE": "LOCATION",
            "LOC": "LOCATION",
            "PRODUCT": "PRODUCT",
            "DATE": "DATE",
            "TIME": "TIME",
            "MONEY": "MONEY",
            "PERCENT": "PERCENTAGE",
            "CARDINAL": "NUMBER",
            "ORDINAL": "NUMBER",
            "QUANTITY": "QUANTITY",
            "WORK_OF_ART": "CREATIVE_WORK",
            "FAC": "FACILITY",
            "NORP": "GROUP",
            "EVENT": "EVENT",
            "LAW": "LAW"
        }
        return mapping.get(spacy_type, "MISC")
    
    def _calculate_comprehension(self, complexity: Dict[str, float]) -> float:
        """
        Calculate comprehension level based on text complexity and development.
        
        Args:
            complexity: Dictionary with complexity metrics
            
        Returns:
            Comprehension level (0.0-1.0)
        """
        # Get overall complexity
        overall_complexity = complexity["overall"]
        
        # Calculate based on developmental parameters
        vocabulary_factor = self.vocabulary_size / 5000  # Normalized to 0-1 assuming max 5000 words
        grammar_factor = self.grammar_complexity
        abstraction_factor = self.abstraction_level
        
        # Combine factors to get comprehension level
        comprehension = (
            0.4 * vocabulary_factor +
            0.4 * grammar_factor +
            0.2 * abstraction_factor
        )
        
        # Limit by text complexity (can't comprehend beyond capabilities)
        if overall_complexity > comprehension:
            # The more complex the text compared to capabilities, the lower the comprehension
            difference = overall_complexity - comprehension
            comprehension = max(0.1, comprehension - difference)
        
        return min(1.0, comprehension)  # Cap at 1.0
    
    def _update_developmental_parameters(self, stage: str) -> None:
        """
        Update language parameters based on developmental stage.
        
        Args:
            stage: Current developmental stage
        """
        # Define language development by stage
        stage_params = {
            DevelopmentalStage.PRENATAL.value: {
                "vocabulary_size": 0,
                "grammar_complexity": 0.0,
                "abstraction_level": 0.0
            },
            DevelopmentalStage.INFANCY.value: {
                "vocabulary_size": 100,
                "grammar_complexity": 0.1,
                "abstraction_level": 0.1
            },
            DevelopmentalStage.EARLY_CHILDHOOD.value: {
                "vocabulary_size": 500,
                "grammar_complexity": 0.3,
                "abstraction_level": 0.2
            },
            DevelopmentalStage.MIDDLE_CHILDHOOD.value: {
                "vocabulary_size": 2000,
                "grammar_complexity": 0.6,
                "abstraction_level": 0.5
            },
            DevelopmentalStage.ADOLESCENCE.value: {
                "vocabulary_size": 5000,
                "grammar_complexity": 0.8,
                "abstraction_level": 0.7
            },
            DevelopmentalStage.ADULTHOOD.value: {
                "vocabulary_size": 10000,
                "grammar_complexity": 0.9,
                "abstraction_level": 0.9
            }
        }
        
        # Get parameters for current stage
        params = stage_params.get(stage, stage_params[DevelopmentalStage.PRENATAL.value])
        
        # Update parameters
        self.vocabulary_size = params["vocabulary_size"]
        self.grammar_complexity = params["grammar_complexity"]
        self.abstraction_level = params["abstraction_level"]
    
    def get_module_status(self) -> Dict[str, Any]:
        """
        Get the current status of the language module.
        
        Returns:
            Dictionary with module status
        """
        # Get the base status
        status = super().get_module_status()
        
        # Get top concepts
        top_concepts = sorted(
            [(concept, data["count"]) for concept, data in self.learned_concepts.items()],
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        # Add language-specific status
        status.update({
            "vocabulary_size": self.vocabulary_size,
            "grammar_complexity": self.grammar_complexity,
            "abstraction_level": self.abstraction_level,
            "learned_concepts_count": len(self.learned_concepts),
            "top_concepts": dict(top_concepts),
            "recent_processing_avg_complexity": sum(item["complexity"]["overall"] 
                                                 for item in self.processing_history[-10:]) / max(1, min(10, len(self.processing_history)))
        })
        
        return status 