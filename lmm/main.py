"""
Main application entry point for the Large Mind Model (LMM).

This module ties together all the components of the LMM system and
provides the main application logic.
"""
import os
import sys
import time
import argparse
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
import random

from lmm.utils.config import get_config, load_config_from_dict
from lmm.utils.logging import get_logger, setup_logger
from lmm.core.mother.caregiver import MotherCaregiver
from lmm.core.development.stages import DevelopmentalStageManager
from lmm.core.development.learning import LearningManager
from lmm.memory.persistence import MemoryManager, MemoryType, MemoryImportance
from lmm.memory.advanced_memory import AdvancedMemoryManager

# Import mind modules
from lmm.core.mind_modules.emotion import EmotionModule
from lmm.core.mind_modules.language import LanguageModule
from lmm.core.mind_modules.memory import MemoryModule
from lmm.core.mind_modules.social import SocialCognitionModule
from lmm.core.mind_modules.consciousness import ConsciousnessModule
from lmm.core.mind_modules.thought import ThoughtModule

logger = get_logger("lmm.main")

class LargeMindsModel:
    """
    Large Mind Model (LMM) main class.
    
    This class coordinates all the components of the LMM system and
    provides the main application API.
    """
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize the LMM.
        
        Args:
            config_dict: Optional configuration dictionary
        """
        # Load configuration
        if config_dict:
            load_config_from_dict(config_dict)
        config = get_config()
        
        # Initialize components
        self.mother = MotherCaregiver()
        self.stage_manager = DevelopmentalStageManager()
        self.learning_manager = LearningManager()
        
        # Initialize mind modules
        self.memory_module = MemoryModule()
        self.emotional_module = EmotionModule()
        self.language_module = LanguageModule()
        self.social_module = SocialCognitionModule()
        self.consciousness_module = ConsciousnessModule()
        self.thought_module = ThoughtModule()
        
        # Track interactions
        self.interaction_count = 0
        
        logger.info("Initialized Large Mind Model")
    
    def interact(self, message: str, stream: bool = False) -> str:
        """
        Process an interaction with the LMM.
        
        Args:
            message: Message to process
            stream: Whether to stream the response
            
        Returns:
            Response from the LMM
        """
        # Increment interaction count
        self.interaction_count += 1
        current_stage = self.stage_manager.get_current_stage()
        logger.info(f"Processing interaction {self.interaction_count} in stage {current_stage}")
        
        # Get emotional state before processing
        emotional_state = self.emotional_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Store user message in memory
        message_memory = {
            "operation": "store",
            "parameters": {
                "content": f"User: {message}",
                "memory_type": MemoryType.EPISODIC.value,
                "importance": MemoryImportance.MEDIUM.value,
                "context_tags": ["user_message"],
                "metadata": {
                    "interaction_number": self.interaction_count,
                    "emotional_state": emotional_state.get("state", {})
                }
            },
            "developmental_stage": current_stage
        }
        self.memory_module.process(message_memory)
        
        # Retrieve relevant memories for context
        memory_search = {
            "operation": "search",
            "parameters": {
                "query": message,
                "limit": 5,
                "min_activation": 0.3,
                "retrieval_strategy": "combined"
            },
            "developmental_stage": current_stage
        }
        memory_result = self.memory_module.process(memory_search)
        relevant_memories = memory_result.get("memories", [])
        
        # Process language understanding
        language_result = self.language_module.process({
            "input": message,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Process social cognition
        social_result = self.social_module.process({
            "input": message,
            "language_understanding": language_result,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Process consciousness (self-awareness, reflection)
        consciousness_result = self.consciousness_module.process({
            "input": message,
            "language_understanding": language_result,
            "social_understanding": social_result,
            "relevant_memories": relevant_memories,
            "emotional_state": emotional_state.get("state", {}),
            "developmental_stage": current_stage
        })
        
        # Process thought generation and reasoning
        thought_result = self.thought_module.process({
            "operation": "generate_thought",
            "content": message,
            "context": {
                "language_understanding": language_result,
                "social_understanding": social_result,
                "consciousness_state": consciousness_result
            },
            "emotional_state": emotional_state.get("state", {}),
            "memory_activations": [m.get("id") for m in relevant_memories],
            "consciousness_state": consciousness_result,
            "developmental_stage": current_stage
        })
        
        # Generate response from mother based on developmental stage
        try:
            response = self.mother.respond(
                message=message,
                stage=current_stage,
                language_understanding=language_result,
                social_understanding=social_result,
                consciousness_state=consciousness_result,
                thought_state=thought_result,
                memories=relevant_memories,
                emotional_state=emotional_state.get("state", {}),
                stream=stream
            )
            
            # Update emotional state after processing
            self.emotional_module.process({
                "operation": "update",
                "input": message,
                "response": response,
                "developmental_stage": current_stage
            })
            
            # Store response in memory
            response_memory = {
                "operation": "store",
                "parameters": {
                    "content": f"LMM: {response}",
                    "memory_type": MemoryType.EPISODIC.value,
                    "importance": MemoryImportance.MEDIUM.value,
                    "context_tags": ["lmm_response"],
                    "metadata": {
                        "interaction_number": self.interaction_count,
                        "emotional_state": emotional_state.get("state", {})
                    }
                },
                "developmental_stage": current_stage
            }
            self.memory_module.process(response_memory)
            
            # Perform reflection on the interaction
            reflection_result = self.thought_module.process({
                "operation": "reflect",
                "content": f"User: {message}\nLMM: {response}",
                "context": {
                    "language_understanding": language_result,
                    "social_understanding": social_result,
                    "consciousness_state": consciousness_result,
                    "thought_state": thought_result
                },
                "emotional_state": emotional_state.get("state", {}),
                "memory_activations": [m.get("id") for m in relevant_memories],
                "consciousness_state": consciousness_result,
                "developmental_stage": current_stage
            })
            
            # Store insights from reflection if available
            if reflection_result.get("success") and reflection_result.get("insights"):
                insights = reflection_result.get("insights", [])
                for insight in insights[:3]:  # Store up to 3 insights
                    insight_memory = {
                        "operation": "store",
                        "parameters": {
                            "content": f"Insight: {insight}",
                            "memory_type": MemoryType.SEMANTIC.value,
                            "importance": MemoryImportance.HIGH.value,
                            "context_tags": ["reflection", "insight"],
                            "metadata": {
                                "interaction_number": self.interaction_count,
                                "source": "thought_reflection"
                            }
                        },
                        "developmental_stage": current_stage
                    }
                    self.memory_module.process(insight_memory)
                
                # Update consciousness with meta-thoughts
                meta_thoughts = reflection_result.get("meta_thoughts", [])
                if meta_thoughts:
                    self.consciousness_module.process({
                        "operation": "update_metacognition",
                        "input": meta_thoughts[0] if meta_thoughts else "",
                        "developmental_stage": current_stage
                    })
            
            # Store interaction in memory for potential semantic memory formation
            self._store_interaction_memory(message, response, current_stage)
            
            # Update learning metrics based on interaction
            self.learning_manager.update_metrics(
                interaction_count=self.interaction_count,
                message=message,
                response=response,
                language_understanding=language_result,
                social_understanding=social_result,
                consciousness_state=consciousness_result,
                emotional_state=emotional_state.get("state", {}),
                developmental_stage=current_stage
            )
            
            # Check for stage progression
            self.stage_manager.check_progression(self.learning_manager.get_metrics())
            
            return response
        except Exception as e:
            logger.error(f"Error in interaction: {str(e)}")
            return f"Error: {str(e)}"
    
    def _store_interaction_memory(self, message: str, response: str, current_stage: str) -> None:
        """
        Store an interaction in memory with advanced semantic knowledge extraction.
        
        This method processes conversation data to extract:
        1. Episodic memory (the raw conversation)
        2. Semantic memories (extracted knowledge, concepts, entities)
        3. Relationships between concepts
        
        Args:
            message: User message
            response: LMM response
            current_stage: Current developmental stage
        """
        # Store conversation as episodic memory
        episodic_memory = {
            "operation": "store",
            "parameters": {
                "content": f"Conversation:\nUser: {message}\nLMM: {response}",
                "memory_type": MemoryType.EPISODIC.value,
                "importance": MemoryImportance.MEDIUM.value,
                "context_tags": ["conversation", f"interaction_{self.interaction_count}"],
                "metadata": {
                    "interaction_number": self.interaction_count,
                    "timestamp": datetime.now().isoformat()
                }
            },
            "developmental_stage": current_stage
        }
        episodic_result = self.memory_module.process(episodic_memory)
        episodic_id = episodic_result.get("memory_id")
        
        # Skip semantic extraction for very short interactions
        if len(message) < 15 and len(response) < 15:
            return
            
        # Process the conversation for semantic memory extraction
        self._extract_semantic_memories(message, response, current_stage, episodic_id)
    
    def _extract_semantic_memories(self, message: str, response: str, current_stage: str, source_id: Optional[int] = None) -> None:
        """
        Extract semantic memories from conversation using advanced NLP techniques.
        
        This implementation:
        1. Identifies key concepts, entities, and relationships
        2. Creates a taxonomy-based categorization
        3. Scores importance based on multiple factors
        4. Establishes associative connections
        5. Performs knowledge consolidation
        
        Args:
            message: User message
            response: LMM response
            current_stage: Current developmental stage
            source_id: ID of the source episodic memory
        """
        try:
            import nltk
            from nltk.tokenize import sent_tokenize, word_tokenize
            from nltk.corpus import stopwords
            from nltk.stem import WordNetLemmatizer
            from nltk.chunk import ne_chunk
            from nltk.tag import pos_tag
            from textblob import TextBlob
            import re
            from collections import Counter
            
            # Ensure NLTK resources are available
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords', quiet=True)
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger', quiet=True)
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                nltk.download('wordnet', quiet=True)
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
            except LookupError:
                nltk.download('maxent_ne_chunker', quiet=True)
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words', quiet=True)
                
            # Create a combined text from message and response for analysis
            combined_text = f"{message}\n{response}"
            
            # Initialize NLP components
            stop_words = set(stopwords.words('english'))
            lemmatizer = WordNetLemmatizer()
            
            # Step 1: Extract key concepts using NLP techniques
            
            # Sentiment analysis
            blob = TextBlob(combined_text)
            sentiment = blob.sentiment
            
            # Tokenize and clean text
            sentences = sent_tokenize(combined_text)
            
            # Extract entities
            entities = self._extract_named_entities(combined_text)
            
            # Extract key phrases and concepts
            concepts = self._extract_key_concepts(combined_text, stop_words, lemmatizer)
            
            # Extract relationships
            relationships = self._extract_relationships(sentences, entities, concepts)
            
            # Step 2: Apply taxonomy-based categorization
            categorized_knowledge = self._categorize_knowledge(concepts, entities, relationships, combined_text)
            
            # Step 3: Calculate importance scores
            semantic_memories = self._generate_semantic_memories(
                categorized_knowledge, 
                entities, 
                sentiment, 
                current_stage,
                source_id
            )
            
            # Step 4: Store the semantic memories
            stored_memory_ids = []
            for semantic_memory in semantic_memories:
                result = self.memory_module.process(semantic_memory)
                if result.get("success"):
                    stored_memory_ids.append(result.get("memory_id"))
            
            # Step 5: Create associations between related semantic memories
            self._associate_semantic_memories(stored_memory_ids)
            
            # Step 6: Consolidate with existing knowledge if appropriate
            if random.random() < 0.3:  # Periodic consolidation
                self.memory_module.process({
                    "operation": "consolidate",
                    "parameters": {"force": False},
                    "developmental_stage": current_stage
                })
                
        except Exception as e:
            logger.error(f"Error in semantic memory extraction: {str(e)}")
            # Fall back to simple extraction if advanced processing fails
            self._fallback_semantic_extraction(message, response, current_stage)
    
    def _extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text using NLTK's NER."""
        from nltk import ne_chunk, pos_tag, word_tokenize
        
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
        current_entity = []
        current_type = None
        
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
    
    def _extract_key_concepts(self, text: str, stop_words: set, lemmatizer) -> List[Dict[str, Any]]:
        """Extract key concepts and keywords from text."""
        from nltk import pos_tag, word_tokenize
        from collections import Counter
        
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
    
    def _extract_relationships(self, sentences: List[str], entities: Dict[str, List[str]], 
                               concepts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract relationships between entities and concepts."""
        relationships = []
        
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
    
    def _categorize_knowledge(self, concepts: List[Dict[str, Any]], 
                             entities: Dict[str, List[str]],
                             relationships: List[Dict[str, Any]],
                             original_text: str) -> List[Dict[str, Any]]:
        """Categorize extracted knowledge into a semantic taxonomy."""
        from textblob import TextBlob
        
        # Define taxonomy categories for semantic knowledge
        taxonomy = {
            "FACTUAL": {
                "subcategories": ["DEFINITION", "PROPERTY", "CATEGORY", "STATISTIC"],
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
                "subcategories": ["CAUSE_EFFECT", "CORRELATION", "DEPENDENCY", "HIERARCHY"],
                "confidence_threshold": 0.6
            },
            "EVALUATIVE": {
                "subcategories": ["OPINION", "ASSESSMENT", "JUDGMENT", "CRITIQUE"],
                "confidence_threshold": 0.5
            }
        }
        
        categorized_items = []
        blob = TextBlob(original_text)
        
        # Analyze for factual statements (typically using subject-verb-object patterns)
        for sentence in blob.sentences:
            # Skip questions and commands
            if sentence.ends_with('?') or sentence.ends_with('!'):
                continue
                
            # Look for factual statements
            if len(sentence.words) >= 4 and sentence.sentiment.subjectivity < 0.4:
                categorized_items.append({
                    "content": str(sentence),
                    "category": "FACTUAL",
                    "subcategory": "PROPERTY" if "is" in sentence.words or "are" in sentence.words else "STATEMENT",
                    "confidence": 0.7 + (0.2 * (1 - sentence.sentiment.subjectivity)),
                    "abstract_level": "specific"
                })
            
            # Look for conceptual knowledge
            elif sentence.sentiment.subjectivity < 0.6 and any(concept["type"] == "noun_phrase" for concept in concepts):
                matching_concepts = [c for c in concepts if c["type"] == "noun_phrase" and c["term"].lower() in str(sentence).lower()]
                if matching_concepts:
                    categorized_items.append({
                        "content": str(sentence),
                        "category": "CONCEPTUAL",
                        "subcategory": "CONCEPT",
                        "confidence": 0.65 + (0.1 * matching_concepts[0]["importance"]),
                        "abstract_level": "abstract" if sentence.sentiment.subjectivity < 0.3 else "moderate",
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
            elif sentence.sentiment.subjectivity > 0.6:
                categorized_items.append({
                    "content": str(sentence),
                    "category": "EVALUATIVE",
                    "subcategory": "OPINION",
                    "confidence": 0.6 + (0.3 * sentence.sentiment.subjectivity),
                    "abstract_level": "subjective",
                    "sentiment_polarity": sentence.sentiment.polarity
                })
        
        # Process entity-based factual knowledge
        for entity_type, entity_list in entities.items():
            for entity in entity_list:
                # For people, organizations, etc.
                if entity_type in ["PERSON", "ORGANIZATION", "GPE", "FACILITY"]:
                    # Find sentences mentioning this entity
                    for sentence in blob.sentences:
                        if entity.lower() in str(sentence).lower():
                            categorized_items.append({
                                "content": f"Identified {entity_type.lower()}: {entity}",
                                "context": str(sentence),
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
                    for sentence in blob.sentences:
                        if entity.lower() in str(sentence).lower():
                            categorized_items.append({
                                "content": f"Temporal reference: {entity}",
                                "context": str(sentence),
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
    
    def _generate_semantic_memories(
        self, 
        categorized_knowledge: List[Dict[str, Any]], 
        entities: Dict[str, List[str]],
        sentiment,
        current_stage: str,
        source_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate structured semantic memories from categorized knowledge."""
        from pydantic import BaseModel, Field
        
        # Define models for strict typing and validation
        class SemanticKnowledgeItem(BaseModel):
            content: str = Field(..., min_length=5, description="The semantic knowledge content")
            category: str = Field(..., description="Primary knowledge category")
            subcategory: str = Field(..., description="Specific knowledge subcategory")
            confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for this knowledge")
            importance: float = Field(..., ge=0.0, le=1.0, description="Importance score for this knowledge")
            abstraction_level: str = Field(..., description="Level of abstraction")
            context: Optional[str] = Field(None, description="Original context if available")
            metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
            
        semantic_memories = []
        
        # Calculate base developmental modifiers
        dev_modifier = 0.5  # Default modifier
        if current_stage == DevelopmentalStage.NEWBORN.value:
            dev_modifier = 0.2  # Limited semantic processing
        elif current_stage == DevelopmentalStage.INFANT.value:
            dev_modifier = 0.4  # Basic semantic processing
        elif current_stage == DevelopmentalStage.TODDLER.value:
            dev_modifier = 0.6  # Growing semantic capabilities
        elif current_stage == DevelopmentalStage.CHILD.value:
            dev_modifier = 0.8  # Developed semantic processing
        elif current_stage == DevelopmentalStage.ADOLESCENT.value:
            dev_modifier = 0.9  # Nearly full semantic capabilities
        elif current_stage == DevelopmentalStage.ADULT.value:
            dev_modifier = 1.0  # Full semantic processing
            
        # Process each categorized item into a semantic memory
        for item in categorized_knowledge:
            try:
                # Skip items with too low confidence based on developmental stage
                if item["confidence"] * dev_modifier < 0.4:
                    continue
                
                # Calculate importance score based on multiple factors
                base_importance = item["confidence"] * 0.7
                
                # Adjust importance based on developmental stage and interests
                importance_score = base_importance * dev_modifier
                
                # Adjust importance based on sentiment intensity (if applicable)
                if "sentiment_polarity" in item:
                    sentiment_intensity = abs(item["sentiment_polarity"])
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
                
                # Generate context tags
                context_tags = ["semantic", item["category"].lower()]
                if "subcategory" in item:
                    context_tags.append(item["subcategory"].lower())
                
                # Add source information
                related_memories = []
                if source_id:
                    related_memories.append(source_id)
                
                # Create metadata with rich information
                metadata = {
                    "source_interaction": self.interaction_count,
                    "confidence": item["confidence"],
                    "abstraction_level": item.get("abstract_level", "general"),
                    "extraction_method": "nlp_advanced"
                }
                
                # Add other relevant metadata from the item
                for key, value in item.items():
                    if key not in ["content", "category", "subcategory", "confidence", "abstract_level"]:
                        metadata[key] = value
                
                # Validate using Pydantic model
                validated_item = SemanticKnowledgeItem(
                    content=item["content"],
                    category=item["category"],
                    subcategory=item["subcategory"],
                    confidence=item["confidence"],
                    importance=importance_score,
                    abstraction_level=item.get("abstract_level", "general"),
                    context=item.get("context"),
                    metadata=metadata
                )
                
                # Create the memory storage request
                semantic_memory = {
                    "operation": "store",
                    "parameters": {
                        "content": validated_item.content,
                        "memory_type": MemoryType.SEMANTIC.value,
                        "importance": memory_importance,
                        "context_tags": context_tags,
                        "related_memories": related_memories,
                        "metadata": validated_item.metadata
                    },
                    "developmental_stage": current_stage
                }
                
                semantic_memories.append(semantic_memory)
                
            except Exception as e:
                logger.warning(f"Error generating semantic memory: {str(e)}")
        
        return semantic_memories
    
    def _associate_semantic_memories(self, memory_ids: List[int]) -> None:
        """Create associations between related semantic memories."""
        # Skip if there aren't enough memories to associate
        if len(memory_ids) < 2:
            return
            
        # Create associations between memories
        for i, memory_id1 in enumerate(memory_ids):
            for memory_id2 in memory_ids[i+1:]:
                # Associate with a moderate strength
                self.memory_module.process({
                    "operation": "associate",
                    "parameters": {
                        "memory_id1": memory_id1,
                        "memory_id2": memory_id2,
                        "strength": 0.5  # Moderate association
                    }
                })
    
    def _fallback_semantic_extraction(self, message: str, response: str, current_stage: str) -> None:
        """Simple fallback extraction if advanced NLP processing fails."""
        if len(message) > 30:
            semantic_memory = {
                "operation": "store",
                "parameters": {
                    "content": f"Learned from conversation: {message[:100]}...",
                    "memory_type": MemoryType.SEMANTIC.value,
                    "importance": MemoryImportance.LOW.value,
                    "context_tags": ["learning", "conversation_derived"],
                    "metadata": {
                        "source_interaction": self.interaction_count,
                        "extraction_method": "fallback_simple"
                    }
                },
                "developmental_stage": current_stage
            }
            self.memory_module.process(semantic_memory)
    
    def get_development_status(self) -> Dict[str, Any]:
        """
        Get the current developmental status of the LMM.
        
        Returns:
            Dictionary with developmental status
        """
        # Get stage info
        stage_info = self.stage_manager.get_status()
        
        # Get learning analytics
        learning_metrics = self.learning_manager.get_metrics()
        
        # Combine information
        status = {
            "current_stage": stage_info["current_stage"],
            "stage_progress": stage_info["stage_progress"],
            "overall_progress": stage_info["overall_progress"],
            "interaction_count": self.interaction_count,
            "learning_metrics": learning_metrics,
            "brain_development": {
                "language_capacity": learning_metrics.get("language_complexity", 0),
                "emotional_awareness": learning_metrics.get("emotional_awareness", 0),
                "social_understanding": learning_metrics.get("social_understanding", 0),
                "cognitive_capability": learning_metrics.get("cognitive_capability", 0),
                "self_awareness": learning_metrics.get("self_awareness", 0)
            }
        }
        
        return status
    
    def get_memory_status(self) -> Dict[str, Any]:
        """
        Get the status of the memory system.
        
        Returns:
            Dictionary with memory status
        """
        memory_stats = self.memory_module.process({
            "operation": "get_stats"
        })
        
        working_memory = self.memory_module.process({
            "operation": "get_working_memory"
        })
        
        return {
            "memory_stats": memory_stats.get("stats", {}),
            "working_memory": working_memory.get("contents", [])
        }
    
    def get_mind_modules_status(self) -> Dict[str, Any]:
        """
        Get the status of all mind modules.
        
        Returns:
            Dictionary with mind modules status
        """
        modules_status = {
            "memory": self.memory_module.get_module_status(),
            "emotional": self.emotional_module.get_module_status(),
            "language": self.language_module.get_module_status(),
            "social": self.social_module.get_module_status(),
            "consciousness": self.consciousness_module.get_module_status(),
            "thought": self.thought_module.get_module_status()
        }
        
        return modules_status
    
    def recall_memories(
        self, 
        query: str, 
        memory_type: Optional[str] = None,
        limit: int = 5,
        min_activation: float = 0.0,
        context_tags: Optional[List[str]] = None,
        retrieval_strategy: str = "combined"
    ) -> List[Dict[str, Any]]:
        """
        Recall memories based on query.
        
        Args:
            query: Search query
            memory_type: Optional memory type filter
            limit: Maximum number of results
            min_activation: Minimum memory activation level
            context_tags: Optional context tags to filter by
            retrieval_strategy: Strategy for retrieval (vector, graph, context, combined)
            
        Returns:
            List of matching memories
        """
        # Get current stage
        current_stage = self.stage_manager.get_current_stage()
        
        # Search memories
        memory_search = {
            "operation": "search",
            "parameters": {
                "query": query,
                "memory_type": memory_type,
                "limit": limit,
                "min_activation": min_activation,
                "context_tags": context_tags,
                "retrieval_strategy": retrieval_strategy
            },
            "developmental_stage": current_stage
        }
        
        memory_result = self.memory_module.process(memory_search)
        memories = memory_result.get("memories", [])
        
        # Format output for external use
        formatted_memories = []
        for memory in memories:
            formatted_memory = {
                "id": memory.get("id"),
                "content": memory.get("content"),
                "type": memory.get("type"),
                "importance": memory.get("importance"),
                "created_at": memory.get("created_at"),
                "retrieval_score": memory.get("retrieval_score", 0.0)
            }
            
            # Add reconstruction information if available
            if memory.get("reconstructed"):
                formatted_memory["reconstructed"] = True
                formatted_memory["confidence"] = memory.get("confidence", 0.5)
                
            formatted_memories.append(formatted_memory)
        
        # Update retrieval statistics for visualization
        self.update_retrieval_stats(query, formatted_memories)
        
        return formatted_memories
    
    def get_introspection(self) -> str:
        """
        Get introspective information about the LMM's current state.
        
        Returns:
            Introspective response
        """
        # Get current stage and status
        current_stage = self.stage_manager.get_current_stage()
        development_status = self.get_development_status()
        memory_status = self.get_memory_status()
        
        # Get cognitive metrics from learning manager
        learning_metrics = self.learning_manager.get_metrics()
        cognitive_capacity = learning_metrics.get("cognitive_capacity", 0.0)
        attention_focus = learning_metrics.get("current_attention_focus", "unknown")
        cognitive_load = learning_metrics.get("cognitive_load", 0.0)
        
        # Get consciousness state
        consciousness_state = self.consciousness_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Get emotional state
        emotional_state = self.emotional_module.process({
            "operation": "get_state",
            "developmental_stage": current_stage
        })
        
        # Construct introspection
        introspection = f"Current developmental stage: {current_stage}\n\n"
        
        # Add cognitive metrics
        introspection += "Cognitive Status:\n"
        introspection += f"- Cognitive capacity: {cognitive_capacity:.2f}\n"
        introspection += f"- Current attention focus: {attention_focus}\n"
        introspection += f"- Cognitive load: {cognitive_load:.2f}\n"
        
        # Add consciousness insights
        introspection += "\nSelf-awareness Insights:\n"
        for insight in consciousness_state.get("recent_insights", [])[:3]:
            introspection += f"- {insight}\n"
        
        # Add emotional state
        emotions = emotional_state.get("state", {})
        introspection += "\nEmotional State:\n"
        for emotion, intensity in emotions.items():
            if intensity > 0.1:
                introspection += f"- {emotion}: {intensity:.2f}\n"
        
        # Add memory status
        working_memory = memory_status.get("working_memory", [])
        if working_memory:
            introspection += "\nCurrently in working memory:\n"
            for memory in working_memory[:3]:
                introspection += f"- {memory.get('content', '')[:50]}...\n"
        
        return introspection
    
    def simulate_development(self, iterations: int = 100) -> Dict[str, Any]:
        """
        Simulate development over time to observe learning patterns.
        
        Args:
            iterations: Number of iterations to simulate
            
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Simulating development over {iterations} iterations")
        
        # Get current stage
        current_stage = self.stage_manager.get_current_stage()
        
        # Use learning manager to simulate development
        simulation_results = self.learning_manager.simulate_development(iterations)
        
        # After simulation, consolidate memories with findings
        consolidation_results = self.memory_module.process({
            "operation": "consolidate",
            "parameters": {"force": True},
            "developmental_stage": current_stage
        })
        
        # Add memory consolidation to results
        simulation_results["memory_consolidation"] = {
            "consolidated_count": consolidation_results.get("consolidated_count", 0)
        }
        
        return simulation_results
    
    def save_state(self) -> None:
        """Save the current state of the LMM."""
        # This would save all component states to persistent storage
        logger.info("Saved LMM state")
    
    def set_developmental_stage(self, stage: str) -> None:
        """
        Set the developmental stage manually.
        
        Args:
            stage: Stage to set
        """
        self.stage_manager.set_stage(stage)
        
        # Update memory parameters for new stage
        self.memory_module.process({
            "developmental_stage": stage
        })
        
        logger.info(f"Manually set developmental stage to {stage}")

    def get_memory_graph(self, limit: int = 50) -> Dict[str, Any]:
        """
        Get the memory association graph data.
        
        Args:
            limit: Maximum number of nodes to include
            
        Returns:
            Dictionary with memory graph data
        """
        # Get memory module to access advanced memory features
        if hasattr(self, "memory_module"):
            memory_graph_result = self.memory_module.process({
                "operation": "get_memory_graph",
                "parameters": {"limit": limit}
            })
            
            if memory_graph_result.get("success"):
                return memory_graph_result.get("graph", {})
        
        # Return empty graph if memory module is not available
        return {"nodes": [], "edges": []}

    def update_retrieval_stats(self, query: str, retrieved_memories: List[Dict[str, Any]]) -> None:
        """
        Update memory retrieval statistics.
        
        Args:
            query: The search query
            retrieved_memories: List of retrieved memories
        """
        # This method would be used by any visualization tool to track retrieval patterns
        avg_score = 0.0
        if retrieved_memories:
            avg_score = sum(memory.get("retrieval_score", 0.0) for memory in retrieved_memories) / len(retrieved_memories)
        
        # Store retrieval stats for visualization
        if hasattr(self, "_retrieval_stats"):
            self._retrieval_stats["counts"].append(len(retrieved_memories))
            self._retrieval_stats["scores"].append(avg_score)
            self._retrieval_stats["timestamps"].append(datetime.now())
            self._retrieval_stats["queries"].append(query)
            
            # Limit history size
            if len(self._retrieval_stats["counts"]) > 100:
                self._retrieval_stats["counts"] = self._retrieval_stats["counts"][-100:]
                self._retrieval_stats["scores"] = self._retrieval_stats["scores"][-100:]
                self._retrieval_stats["timestamps"] = self._retrieval_stats["timestamps"][-100:]
                self._retrieval_stats["queries"] = self._retrieval_stats["queries"][-100:]
        else:
            self._retrieval_stats = {
                "counts": [len(retrieved_memories)],
                "scores": [avg_score],
                "timestamps": [datetime.now()],
                "queries": [query]
            }

    def get_retrieval_stats(self) -> Dict[str, Any]:
        """
        Get memory retrieval statistics.
        
        Returns:
            Dictionary with memory retrieval statistics
        """
        return self._retrieval_stats
        
    def process_cognitive_query(self, query: str, depth: int = 3) -> Dict[str, Any]:
        """
        Process a complex cognitive query using thought integration.
        
        This method implements advanced cognitive processing by integrating
        the thought module with memory and consciousness to produce deeper
        insights and connections.
        
        Args:
            query: The cognitive query to process
            depth: The recursive depth of cognitive processing
            
        Returns:
            Dictionary with processed results
        """
        logger.info(f"Processing cognitive query: {query[:50]}... (depth={depth})")
        current_stage = self.stage_manager.get_current_stage()
        
        # Initial thought generation
        thought_result = self.thought_module.process({
            "operation": "generate_thought",
            "content": query,
            "developmental_stage": current_stage
        })
        
        if not thought_result.get("success"):
            return {"success": False, "error": "Failed to generate initial thought"}
        
        thought = thought_result.get("thought", {})
        thought_content = thought.get("content", "")
        
        # Retrieve memories based on the thought
        memory_result = self.memory_module.process({
            "operation": "search",
            "parameters": {
                "query": thought_content,
                "limit": 5,
                "min_activation": 0.3,
                "retrieval_strategy": "semantic"
            },
            "developmental_stage": current_stage
        })
        
        relevant_memories = memory_result.get("memories", [])
        
        # Generate insights through reflection
        reflection_result = self.thought_module.process({
            "operation": "reflect",
            "content": thought_content,
            "context": {"relevant_memories": relevant_memories},
            "memory_activations": [m.get("id") for m in relevant_memories],
            "developmental_stage": current_stage
        })
        
        insights = reflection_result.get("insights", [])
        patterns = reflection_result.get("patterns", [])
        meta_thoughts = reflection_result.get("meta_thoughts", [])
        
        # Recursive cognitive exploration if depth allows
        deeper_insights = []
        if depth > 1 and insights:
            # Use the most significant insight to explore deeper
            primary_insight = insights[0] if insights else ""
            if primary_insight:
                deeper_result = self.process_cognitive_query(primary_insight, depth - 1)
                deeper_insights = deeper_result.get("insights", [])
        
        # Update consciousness with the highest-level insight
        if meta_thoughts:
            self.consciousness_module.process({
                "operation": "update_metacognition",
                "input": meta_thoughts[0] if meta_thoughts else "",
                "developmental_stage": current_stage
            })
        
        # Build result structure
        result = {
            "success": True,
            "query": query,
            "initial_thought": thought,
            "insights": insights,
            "patterns": patterns,
            "meta_thoughts": meta_thoughts,
            "deeper_insights": deeper_insights,
            "relevant_memories": relevant_memories,
            "depth_reached": depth
        }
        
        return result
        
    def launch_dashboard(self, port: int = 8050) -> Any:
        """
        Launch the development dashboard.
        
        Args:
            port: Port to run the dashboard on
            
        Returns:
            Dashboard instance
        """
        try:
            from lmm.visualization.dashboard import DevelopmentDashboard
            dashboard = DevelopmentDashboard(lmm_instance=self, port=port)
            dashboard.start_background()
            logger.info(f"Launched dashboard on port {port}")
            return dashboard
        except ImportError as e:
            logger.error(f"Failed to launch dashboard: {e}")
            print(f"Error: Failed to launch dashboard - {e}")
            return None

def main():
    """Main entry point."""
    # Setup argument parser
    parser = argparse.ArgumentParser(description="Large Mind Model")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--interactive", action="store_true", help="Run in interactive mode")
    parser.add_argument("--stage", help="Set initial developmental stage")
    parser.add_argument("--simulate", type=int, help="Run development simulation with N iterations")
    parser.add_argument("--dashboard", action="store_true", help="Launch development dashboard")
    parser.add_argument("--dashboard-port", type=int, default=8050, help="Port for the dashboard")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize LMM
    config_dict = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
    
    lmm = LargeMindsModel(config_dict)
    
    # Set stage if provided
    if args.stage:
        lmm.set_developmental_stage(args.stage)
    
    # Run simulation if requested
    if args.simulate:
        results = lmm.simulate_development(args.simulate)
        print(f"Simulation results: {results}")
        return
    
    # Launch dashboard if requested
    dashboard = None
    if args.dashboard:
        dashboard = lmm.launch_dashboard(port=args.dashboard_port)
        print(f"Dashboard launched on port {args.dashboard_port}. Access at http://localhost:{args.dashboard_port}")
        # Add a small sleep to ensure the dashboard has time to start
        time.sleep(1)
    
    # Run in interactive mode
    if args.interactive:
        print("LMM Interactive Mode. Type 'exit' to quit.")
        print("Special commands:")
        print(" - 'status': Get development status")
        print(" - 'memory': Get memory status")
        print(" - 'recall <query>': Recall memories")
        print(" - 'introspect': Get introspection")
        print(" - 'modules': Get mind modules status")
        print(" - 'consolidate': Force memory consolidation")
        print(" - 'simulate <N>': Run development simulation")
        print(" - 'stage <stage>': Set developmental stage")
        print(" - 'dashboard': Launch visualization dashboard")
        
        while True:
            try:
                message = input("\nYou: ")
                
                if message.lower() == 'exit':
                    break
                
                # Handle special commands
                if message.lower() == 'status':
                    status = lmm.get_development_status()
                    print("\nDevelopment Status:")
                    print(f"Stage: {status['current_stage']}")
                    print(f"Progress: {status['stage_progress']:.2f}")
                    print(f"Overall Progress: {status['overall_progress']:.2f}")
                    print(f"Interactions: {status['interaction_count']}")
                    print("\nBrain Development:")
                    for key, value in status['brain_development'].items():
                        print(f"- {key}: {value:.3f}")
                    continue
                
                if message.lower() == 'memory':
                    memory_status = lmm.get_memory_status()
                    stats = memory_status.get("memory_stats", {})
                    working = memory_status.get("working_memory", [])
                    
                    print("\nMemory Status:")
                    print(f"Total memories: {stats.get('total_memories', 0)}")
                    print(f"Working memory: {len(working)}/{stats.get('working_memory_capacity', 0)} items")
                    
                    strength_dist = stats.get("strength_distribution", {})
                    print("\nMemory Strength Distribution:")
                    for category, count in strength_dist.items():
                        print(f"- {category}: {count}")
                        
                    print("\nWorking Memory Contents:")
                    for memory in working:
                        print(f"- {memory.get('content', '')[:50]}...")
                    continue
                
                if message.lower().startswith('recall '):
                    query = message[7:]
                    memories = lmm.recall_memories(query, limit=3)
                    
                    print(f"\nMemories related to '{query}':")
                    for memory in memories:
                        print(f"\n- {memory['content']}")
                        print(f"  Type: {memory['type']}, Score: {memory.get('retrieval_score', 0):.2f}")
                        if memory.get("reconstructed"):
                            print(f"  Note: This memory may be partially reconstructed. Confidence: {memory.get('confidence', 0):.2f}")
                    continue
                
                if message.lower() == 'introspect':
                    print("\n" + lmm.get_introspection())
                    continue
                
                if message.lower() == 'modules':
                    modules = lmm.get_mind_modules_status()
                    
                    print("\nMind Modules Status:")
                    for name, module in modules.items():
                        print(f"\n{name.upper()} Module: {module.get('status', 'unknown')}")
                        if name == 'memory':
                            counts = module.get('memory_counts', {})
                            print(f"  Total memories: {counts.get('total', 0)}")
                            print(f"  Working memory: {module.get('working_memory', {}).get('usage', 0)}/{module.get('working_memory', {}).get('capacity', 0)}")
                    continue
                
                if message.lower() == 'consolidate':
                    result = lmm.memory_module.process({
                        "operation": "consolidate",
                        "parameters": {"force": True}
                    })
                    
                    print(f"\nConsolidated {result.get('consolidated_count', 0)} memories.")
                    continue
                
                if message.lower().startswith('simulate '):
                    try:
                        iterations = int(message.split()[1])
                        results = lmm.simulate_development(iterations)
                        
                        print(f"\nSimulation completed with {iterations} iterations.")
                        print(f"Learning progress: {results.get('learning_progress', 0):.3f}")
                        print(f"Memories consolidated: {results.get('memory_consolidation', {}).get('consolidated_count', 0)}")
                    except (ValueError, IndexError):
                        print("Invalid simulate command. Use 'simulate <number>'.")
                    continue
                
                if message.lower().startswith('stage '):
                    stage = message[6:].strip()
                    try:
                        lmm.set_developmental_stage(stage)
                        print(f"\nSet developmental stage to {stage}.")
                    except ValueError:
                        print(f"Invalid stage: {stage}")
                    continue
                
                if message.lower() == 'dashboard':
                    if dashboard:
                        print("\nDashboard is already running")
                    else:
                        dashboard = lmm.launch_dashboard()
                        print(f"\nLaunched dashboard. Access at http://localhost:8050")
                    continue
                
                # Normal interaction
                response = lmm.interact(message)
                print(f"\nLMM: {response}")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    else:
        # Non-interactive mode would define an API or other interface
        pass

if __name__ == "__main__":
    main() 