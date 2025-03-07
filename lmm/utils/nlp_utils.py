"""
NLP utilities for advanced semantic memory extraction in the LMM system.

This module provides sophisticated natural language processing functions
to extract semantic knowledge from text, including entity recognition,
concept extraction, relationship identification, and knowledge categorization.
"""
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from collections import Counter
import logging
import random

# Import Pydantic for validation
from pydantic import BaseModel, Field

# Initialize logger
logger = logging.getLogger("lmm.utils.nlp_utils")

# Define models for semantic knowledge extraction
class SemanticConcept(BaseModel):
    """Model for a semantic concept extracted from text."""
    term: str = Field(..., description="The concept term or phrase")
    type: str = Field(..., description="Type of concept (noun, verb, noun_phrase, etc.)")
    frequency: int = Field(1, description="Frequency of the concept in the analyzed text")
    importance: float = Field(0.5, description="Calculated importance score (0.0-1.0)")

class SemanticRelationship(BaseModel):
    """Model for a relationship between semantic concepts or entities."""
    source: str = Field(..., description="Source entity or concept")
    source_type: str = Field(..., description="Type of the source (entity, concept)")
    target: str = Field(..., description="Target entity or concept") 
    target_type: str = Field(..., description="Type of the target (entity, concept)")
    relation_type: str = Field(..., description="Type of relationship")
    context: str = Field(..., description="Context sentence for the relationship")
    confidence: float = Field(0.5, description="Confidence score for the relationship")

class SemanticKnowledgeItem(BaseModel):
    """Model for a semantic knowledge item extracted from text."""
    content: str = Field(..., min_length=5, description="The semantic knowledge content")
    category: str = Field(..., description="Primary knowledge category")
    subcategory: str = Field(..., description="Specific knowledge subcategory")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this knowledge")
    importance: float = Field(..., ge=0.0, le=1.0, description="Importance score for this knowledge")
    abstraction_level: str = Field(..., description="Level of abstraction") 
    context: Optional[str] = Field(None, description="Original context if available")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# Knowledge taxonomy
KNOWLEDGE_TAXONOMY = {
    "FACTUAL": {
        "subcategories": ["DEFINITION", "PROPERTY", "CATEGORY", "STATISTIC", "TEMPORAL"],
        "confidence_threshold": 0.7
    },
    "CONCEPTUAL": {
        "subcategories": ["CONCEPT", "THEORY", "PRINCIPLE", "FRAMEWORK"],
        "confidence_threshold": 0.65
    },
    "PROCEDURAL": {
        "subcategories": ["STEP", "METHOD", "TECHNIQUE", "PROTOCOL"],
        "confidence_threshold": 0.75
    },
    "RELATIONAL": {
        "subcategories": ["CAUSE_EFFECT", "CORRELATION", "DEPENDENCY", "HIERARCHY", "ASSOCIATION", "CONNECTION"],
        "confidence_threshold": 0.6
    },
    "EVALUATIVE": {
        "subcategories": ["OPINION", "ASSESSMENT", "JUDGMENT", "CRITIQUE"],
        "confidence_threshold": 0.5
    }
}

def extract_named_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using NLTK's NER.
    
    Args:
        text: Input text to analyze
        
    Returns:
        Dictionary of entity types and their instances
    """
    try:
        import nltk
        from nltk import ne_chunk, pos_tag, word_tokenize
        
        # Ensure NLTK resources are available
        required_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('chunkers/maxent_ne_chunker', 'maxent_ne_chunker'),
            ('corpora/words', 'words'),
            ('chunkers/maxent_ne_chunker_tab', 'maxent_ne_chunker_tab')
        ]
        
        for resource_path, resource_name in required_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                logger.info(f"Downloading required NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
        
        entities = {
            "PERSON": [],
            "ORGANIZATION": [],
            "LOCATION": [],
            "DATE": [],
            "TIME": [],
            "MONEY": [],
            "PERCENT": [],
            "FACILITY": [],
            "GPE": [],  # Geo-Political Entity
            "OTHER": []
        }
        
        # Extract named entities
        tokens = word_tokenize(text)
        pos_tagged = pos_tag(tokens)
        named_entities = ne_chunk(pos_tagged)
        
        # Process the named entities
        for chunk in named_entities:
            if hasattr(chunk, 'label'):
                entity_type = chunk.label()
                entity_text = " ".join([word for word, tag in chunk.leaves()])
                
                if entity_type in entities:
                    if entity_text not in entities[entity_type]:
                        entities[entity_type].append(entity_text)
                else:
                    if entity_text not in entities["OTHER"]:
                        entities["OTHER"].append(entity_text)
        
        # Identify potential dates and times using regex patterns
        date_patterns = [
            r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}\b',
            r'\b\d{1,2}/\d{1,2}/\d{2,4}\b',
            r'\b\d{4}-\d{2}-\d{2}\b'
        ]
        
        time_patterns = [
            r'\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b',
            r'\b\d{1,2}\s*(?:AM|PM|am|pm)\b'
        ]
        
        for pattern in date_patterns:
            for match in re.finditer(pattern, text):
                date = match.group(0)
                if date not in entities["DATE"]:
                    entities["DATE"].append(date)
        
        for pattern in time_patterns:
            for match in re.finditer(pattern, text):
                time = match.group(0)
                if time not in entities["TIME"]:
                    entities["TIME"].append(time)
        
        return entities
    
    except Exception as e:
        logger.error(f"Error in entity extraction: {str(e)}")
        return {"PERSON": [], "ORGANIZATION": [], "LOCATION": [], "DATE": [], "TIME": [], "OTHER": []}

def extract_key_concepts(text: str) -> List[Dict[str, Any]]:
    """
    Extract key concepts and keywords from text.
    
    Args:
        text: Input text to analyze
        
    Returns:
        List of concept dictionaries with term, type, frequency, and importance
    """
    try:
        import nltk
        from nltk import pos_tag, word_tokenize
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        from collections import Counter
        
        # Ensure NLTK resources are available
        required_resources = [
            ('tokenizers/punkt', 'punkt'),
            ('corpora/stopwords', 'stopwords'),
            ('corpora/wordnet', 'wordnet'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger')
        ]
        
        for resource_path, resource_name in required_resources:
            try:
                nltk.data.find(resource_path)
            except LookupError:
                nltk.download(resource_name, quiet=True)
        
        # Initialize NLP components
        stop_words = set(stopwords.words('english'))
        lemmatizer = WordNetLemmatizer()
        
        # Tokenize and tag parts of speech
        tokens = word_tokenize(text.lower())
        tagged = pos_tag(tokens)
        
        # Extract nouns, verbs, and adjectives (potential concepts)
        nouns = [lemmatizer.lemmatize(word) for word, tag in tagged 
                if tag.startswith('NN') and word not in stop_words and len(word) > 2]
                
        verbs = [lemmatizer.lemmatize(word, pos='v') for word, tag in tagged 
                if tag.startswith('VB') and word not in stop_words and len(word) > 2]
                
        adjectives = [word for word, tag in tagged 
                    if tag.startswith('JJ') and word not in stop_words and len(word) > 2]
        
        # Count occurrences to find most frequent terms
        noun_counter = Counter(nouns)
        verb_counter = Counter(verbs)
        adj_counter = Counter(adjectives)
        
        # Extract noun phrases (potential complex concepts)
        from nltk.chunk import RegexpParser
        
        grammar = r"""
            NP: {<DT|PP\$>?<JJ.*>*<NN.*>+}  # Noun phrase
            VP: {<VB.*><NP|PP>}             # Verb phrase
            CONCEPT: {<NP><VP>}             # Conceptual relationship
        """
        
        chunk_parser = RegexpParser(grammar)
        chunked = chunk_parser.parse(tagged)
        
        noun_phrases = []
        for subtree in chunked.subtrees():
            if subtree.label() == 'NP':
                np = " ".join([word for word, tag in subtree.leaves()])
                noun_phrases.append(np)
        
        # Create concept objects
        concepts = []
        
        # Add frequent nouns as concepts
        for noun, count in noun_counter.most_common(10):
            concepts.append({
                "term": noun,
                "type": "noun",
                "frequency": count,
                "importance": min(0.9, count / len(tokens) * 10)
            })
        
        # Add frequent verbs
        for verb, count in verb_counter.most_common(5):
            concepts.append({
                "term": verb,
                "type": "verb",
                "frequency": count,
                "importance": min(0.8, count / len(tokens) * 8)
            })
        
        # Add frequent adjectives
        for adj, count in adj_counter.most_common(5):
            concepts.append({
                "term": adj,
                "type": "adjective",
                "frequency": count,
                "importance": min(0.7, count / len(tokens) * 6)
            })
        
        # Add noun phrases
        noun_phrase_counter = Counter(noun_phrases)
        for np, count in noun_phrase_counter.most_common(7):
            if len(np.split()) > 1:  # Only multi-word phrases
                concepts.append({
                    "term": np,
                    "type": "noun_phrase",
                    "frequency": count,
                    "importance": min(0.95, count / len(tokens) * 15)
                })
        
        return concepts
        
    except Exception as e:
        logger.error(f"Error in concept extraction: {str(e)}")
        return []

def extract_relationships(sentences: List[str], entities: Dict[str, List[str]], 
                          concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract relationships between entities and concepts.
    
    Args:
        sentences: List of sentences from the text
        entities: Dictionary of named entities
        concepts: List of extracted concepts
        
    Returns:
        List of relationship dictionaries
    """
    try:
        relationships = []
        
        # Skip if no entities or concepts were found
        if not entities or not concepts or all(len(entity_list) == 0 for entity_list in entities.values()):
            logger.warning("No entities found for relationship extraction")
            return relationships
            
        if not concepts:
            logger.warning("No concepts found for relationship extraction")
            return relationships
        
        # Flatten entities for easier processing
        all_entities = []
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                all_entities.append({"text": entity, "type": entity_type})
        
        # Extract concept terms
        concept_terms = [c["term"] for c in concepts]
        
        # Find sentences with co-occurring entities/concepts
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Find entities in this sentence
            sentence_entities = []
            for entity in all_entities:
                if entity["text"].lower() in sentence_lower:
                    sentence_entities.append(entity)
            
            # Find concepts in this sentence
            sentence_concepts = []
            for concept in concept_terms:
                if concept.lower() in sentence_lower:
                    sentence_concepts.append(concept)
            
            # Create relationships for co-occurring entities
            for i, entity1 in enumerate(sentence_entities):
                for entity2 in sentence_entities[i+1:]:
                    relationships.append({
                        "source": entity1["text"],
                        "source_type": "entity",
                        "target": entity2["text"],
                        "target_type": "entity",
                        "relation_type": "co-occurrence",
                        "context": sentence,
                        "confidence": 0.7
                    })
            
            # Create relationships between entities and concepts
            for entity in sentence_entities:
                for concept in sentence_concepts:
                    relationships.append({
                        "source": entity["text"],
                        "source_type": "entity",
                        "target": concept,
                        "target_type": "concept",
                        "relation_type": "association",
                        "context": sentence,
                        "confidence": 0.6
                    })
        
        return relationships
        
    except Exception as e:
        logger.error(f"Error in relationship extraction: {str(e)}")
        return []

def categorize_knowledge(text: str, concepts: List[Dict[str, Any]], 
                        entities: Dict[str, List[str]],
                        relationships: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Categorize extracted knowledge into semantic taxonomy categories.
    
    Args:
        text: Original text
        concepts: Extracted concepts
        entities: Named entities
        relationships: Extracted relationships
        
    Returns:
        List of categorized knowledge items
    """
    try:
        from textblob import TextBlob
        from nltk.tokenize import sent_tokenize
        
        # Ensure NLTK resources are available
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
        
        categorized_items = []
        blob = TextBlob(text)
        sentences = sent_tokenize(text)
        
        # Analyze for factual statements (typically using subject-verb-object patterns)
        for sentence in blob.sentences:
            # Skip questions and commands
            if str(sentence).endswith('?') or str(sentence).endswith('!'):
                continue
            
            # Handle cases where TextBlob might not correctly process sentences
            try:
                sentence_words = sentence.words
                subjectivity = sentence.sentiment.subjectivity
                polarity = sentence.sentiment.polarity
            except:
                # If TextBlob processing fails, use simpler approach
                sentence_words = str(sentence).split()
                subjectivity = 0.5  # Neutral subjectivity
                polarity = 0.0  # Neutral polarity
                
            # Look for factual statements
            if len(sentence_words) >= 4 and subjectivity < 0.4:
                categorized_items.append({
                    "content": str(sentence),
                    "category": "FACTUAL",
                    "subcategory": "PROPERTY" if "is" in sentence_words or "are" in sentence_words else "STATEMENT",
                    "confidence": 0.7 + (0.2 * (1 - subjectivity)),
                    "abstract_level": "specific"
                })
            
            # Look for conceptual knowledge
            elif subjectivity < 0.6 and any(concept["type"] == "noun_phrase" for concept in concepts):
                matching_concepts = [c for c in concepts if c["type"] == "noun_phrase" and c["term"].lower() in str(sentence).lower()]
                if matching_concepts:
                    categorized_items.append({
                        "content": str(sentence),
                        "category": "CONCEPTUAL",
                        "subcategory": "CONCEPT",
                        "confidence": 0.65 + (0.1 * matching_concepts[0]["importance"]),
                        "abstract_level": "abstract" if subjectivity < 0.3 else "moderate",
                        "related_concepts": [c["term"] for c in matching_concepts]
                    })
            
            # Look for procedural knowledge
            elif any(action_word in str(sentence).lower() for action_word in ["how to", "steps", "process", "method", "way to"]):
                categorized_items.append({
                    "content": str(sentence),
                    "category": "PROCEDURAL",
                    "subcategory": "METHOD",
                    "confidence": 0.75,
                    "abstract_level": "specific"
                })
            
            # Look for relationship knowledge
            elif any(rel_word in str(sentence).lower() for rel_word in ["because", "therefore", "thus", "since", "as a result"]):
                categorized_items.append({
                    "content": str(sentence),
                    "category": "RELATIONAL",
                    "subcategory": "CAUSE_EFFECT",
                    "confidence": 0.7,
                    "abstract_level": "relational"
                })
            
            # Look for evaluative knowledge
            elif subjectivity > 0.6:
                categorized_items.append({
                    "content": str(sentence),
                    "category": "EVALUATIVE",
                    "subcategory": "OPINION",
                    "confidence": 0.6 + (0.3 * subjectivity),
                    "abstract_level": "subjective",
                    "sentiment_polarity": polarity
                })
        
        # Process entity-based factual knowledge
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # For people, organizations, etc.
                if entity_type in ["PERSON", "ORGANIZATION", "GPE", "FACILITY"]:
                    # Find sentences mentioning this entity
                    for sentence in sentences:
                        if entity.lower() in sentence.lower():
                            categorized_items.append({
                                "content": f"Identified {entity_type.lower()}: {entity}",
                                "context": sentence,
                                "category": "FACTUAL",
                                "subcategory": "CATEGORY",
                                "confidence": 0.85,
                                "abstract_level": "specific",
                                "entity_type": entity_type
                            })
                            break
                
                # For dates and times
                elif entity_type in ["DATE", "TIME"]:
                    # Find sentences mentioning this time reference
                    for sentence in sentences:
                        if entity.lower() in sentence.lower():
                            categorized_items.append({
                                "content": f"Temporal reference: {entity}",
                                "context": sentence,
                                "category": "FACTUAL",
                                "subcategory": "TEMPORAL",
                                "confidence": 0.9,
                                "abstract_level": "specific",
                                "entity_type": entity_type
                            })
                            break
        
        # Process relationship knowledge
        for relationship in relationships:
            if relationship["confidence"] > 0.6:
                categorized_items.append({
                    "content": f"Relationship between {relationship['source']} and {relationship['target']}",
                    "context": relationship["context"],
                    "category": "RELATIONAL",
                    "subcategory": "ASSOCIATION" if relationship["relation_type"] == "association" else "CONNECTION",
                    "confidence": relationship["confidence"],
                    "abstract_level": "relational",
                    "source": relationship["source"],
                    "target": relationship["target"]
                })
        
        return categorized_items
        
    except Exception as e:
        logger.error(f"Error in knowledge categorization: {str(e)}")
        return []

def calculate_memory_importance(
    item: Dict[str, Any], 
    sentiment_score: float,
    developmental_stage: str
) -> Tuple[float, str]:
    """
    Calculate memory importance score and corresponding importance level.
    
    Args:
        item: Knowledge item
        sentiment_score: Text sentiment score (-1.0 to 1.0)
        developmental_stage: Current developmental stage
        
    Returns:
        Tuple of (importance_score, importance_level)
    """
    from lmm.memory.persistence import MemoryImportance
    from lmm.core.development.stages import DevelopmentalStage
    
    try:
        # Calculate base developmental modifiers
        dev_modifier = 0.5  # Default modifier
        if developmental_stage == DevelopmentalStage.NEWBORN.value:
            dev_modifier = 0.2  # Limited semantic processing
        elif developmental_stage == DevelopmentalStage.INFANT.value:
            dev_modifier = 0.4  # Basic semantic processing
        elif developmental_stage == DevelopmentalStage.TODDLER.value:
            dev_modifier = 0.6  # Growing semantic capabilities
        elif developmental_stage == DevelopmentalStage.CHILD.value:
            dev_modifier = 0.8  # Developed semantic processing
        elif developmental_stage == DevelopmentalStage.ADOLESCENT.value:
            dev_modifier = 0.9  # Nearly full semantic capabilities
        elif developmental_stage == DevelopmentalStage.ADULT.value:
            dev_modifier = 1.0  # Full semantic processing
            
        # Calculate importance score based on multiple factors
        base_importance = item["confidence"] * 0.7
        
        # Adjust importance based on developmental stage and interests
        importance_score = base_importance * dev_modifier
        
        # Adjust importance based on sentiment intensity
        sentiment_intensity = abs(sentiment_score)
        importance_score = importance_score * (1 + sentiment_intensity * 0.3)
        
        # Adjust importance based on entity types present
        if "entity_type" in item and item["entity_type"] in ["PERSON", "ORGANIZATION"]:
            importance_score = importance_score * 1.2
        
        # Cap importance score
        importance_score = min(0.95, importance_score)
        
        # Determine memory importance level based on score
        memory_importance = MemoryImportance.LOW.value
        if importance_score > 0.7:
            memory_importance = MemoryImportance.HIGH.value
        elif importance_score > 0.5:
            memory_importance = MemoryImportance.MEDIUM.value
        
        return (importance_score, memory_importance)
        
    except Exception as e:
        logger.error(f"Error calculating memory importance: {str(e)}")
        return (0.5, "medium")  # Default values

# Simple self-test code
if __name__ == "__main__":
    print("Testing NLP utilities...")
    
    # Test text
    test_text = """
    John Smith, the CEO of Acme Corporation, announced a new AI product on January 15th, 2023.
    The product uses advanced machine learning algorithms to improve customer service.
    According to Smith, this technology will revolutionize how companies interact with their customers
    because it can understand complex queries and respond with high accuracy.
    The company's stock price increased by 5% after the announcement.
    """
    
    print("\nExtracting entities...")
    entities = extract_named_entities(test_text)
    for entity_type, items in entities.items():
        if items:
            print(f"{entity_type}: {items}")
    
    print("\nExtracting concepts...")
    concepts = extract_key_concepts(test_text)
    print(f"Found {len(concepts)} concepts")
    
    print("\nExtraction complete - NLP utilities are working") 