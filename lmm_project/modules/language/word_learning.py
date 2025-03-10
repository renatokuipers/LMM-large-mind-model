# TODO: Implement the WordLearning class to acquire and manage vocabulary
# This component should be able to:
# - Learn new words from context and direct instruction
# - Connect words to meanings and concepts
# - Build and maintain a lexicon of known words
# - Track word frequency and familiarity

# TODO: Implement developmental progression in word learning:
# - Simple sound-object associations in early stages
# - Vocabulary explosion in early childhood
# - Growing semantic networks in later childhood
# - Abstract and specialized vocabulary in adolescence/adulthood

# TODO: Create mechanisms for:
# - Fast mapping: Form initial word-concept connections
# - Semantic enrichment: Develop deeper word meanings over time
# - Word retrieval: Access words efficiently from memory
# - Lexical organization: Structure vocabulary by semantic relationships

# TODO: Implement different word types and learning patterns:
# - Concrete nouns: Objects, people, places
# - Action verbs: Movement, change, activities
# - Descriptive adjectives: Properties, qualities
# - Abstract concepts: Ideas, emotions, principles

# TODO: Connect to memory and perception systems
# Word learning should be tied to perceptual experiences
# and should store word knowledge in semantic memory

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.language.models import WordModel, LanguageNeuralState
from lmm_project.modules.language.neural_net import WordNetwork, get_device
from lmm_project.utils.llm_client import LLMClient

class WordLearning(BaseModule):
    """
    Learns and processes words and their meanings
    
    This module is responsible for vocabulary acquisition,
    word-meaning associations, and lexical development.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic word recognition",
        0.2: "First words acquisition",
        0.4: "Vocabulary spurt",
        0.6: "Semantic network formation",
        0.8: "Word-definition comprehension",
        1.0: "Complete lexical system"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the word learning module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize word model
        self.word_model = WordModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = WordNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.word_learning_development = self.development_level
        
        # Initialize with basic vocabulary based on development level
        self._initialize_vocabulary()
        
        # Recent inputs queue (for tracking word exposure)
        self.recent_inputs = deque(maxlen=100)
        
        # For embedding generation
        self.llm_client = LLMClient()
    
    def _initialize_vocabulary(self):
        """Initialize basic vocabulary based on development level"""
        # Basic vocabulary at earliest stages (first words)
        basic_words = {
            "mama": 0.8, "dada": 0.8, "baby": 0.7, "milk": 0.7, "hi": 0.7, "bye": 0.7
        }
        
        # Add basic words to vocabulary
        for word, familiarity in basic_words.items():
            # Scale familiarity by development level
            scaled_familiarity = familiarity * max(0.3, self.development_level)
            self.word_model.vocabulary[word] = scaled_familiarity
        
        # Set basic categories
        self.word_model.word_categories["people"] = ["mama", "dada", "baby"]
        self.word_model.word_categories["food"] = ["milk"]
        self.word_model.word_categories["greetings"] = ["hi", "bye"]
        
        # Initialize word frequencies
        for word in basic_words:
            self.word_model.word_frequencies[word] = max(1, int(10 * self.development_level))
        
        # Add more vocabulary with increased development
        if self.development_level >= 0.3:
            # Early childhood vocabulary expansion
            early_words = {
                "water": 0.7, "dog": 0.7, "cat": 0.7, "ball": 0.7, 
                "yes": 0.7, "no": 0.7, "more": 0.7, "want": 0.7
            }
            
            # Add early words
            for word, familiarity in early_words.items():
                # Scale familiarity by development beyond threshold
                scaled_familiarity = familiarity * ((self.development_level - 0.3) / 0.7)
                self.word_model.vocabulary[word] = scaled_familiarity
                self.word_model.word_frequencies[word] = max(1, int(8 * (self.development_level - 0.3) / 0.7))
            
            # Update categories
            self.word_model.word_categories["animals"] = ["dog", "cat"]
            self.word_model.word_categories["objects"] = ["ball"]
            self.word_model.word_categories["food"].append("water")
            self.word_model.word_categories["communication"] = ["yes", "no", "more", "want"]
        
        if self.development_level >= 0.5:
            # Generate word embeddings for known words
            self._initialize_word_embeddings()
    
    def _initialize_word_embeddings(self):
        """
        Initialize embeddings for words in vocabulary using LLM API
        
        Processes all words that need embeddings in batches for efficiency
        and uses a retry mechanism to handle API failures.
        """
        # Only get embeddings for words we don't already have
        words_to_embed = [w for w in self.word_model.vocabulary 
                          if w not in self.word_model.word_embeddings]
        
        if not words_to_embed:
            return
            
        # Create a logging list of failed words for retry
        if not hasattr(self, "_failed_embeddings"):
            self._failed_embeddings = set()
        
        # Process in batches of up to 20 words for efficiency
        batch_size = 20
        for batch_start in range(0, len(words_to_embed), batch_size):
            batch_end = min(batch_start + batch_size, len(words_to_embed))
            current_batch = words_to_embed[batch_start:batch_end]
            
            # Skip tiny batches
            if len(current_batch) == 0:
                continue
                
            # Try primary model first
            max_retries = 2
            for attempt in range(max_retries):
                try:
                    # Get embeddings from LLM client
                    embeddings = self.llm_client.get_embedding(
                        current_batch,
                        embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
                    )
                    
                    # Process successful embeddings
                    if isinstance(embeddings, list):
                        # Check if we got a list of lists (one per word) or a single embedding
                        if len(current_batch) == 1:
                            # Single word case
                            if isinstance(embeddings[0], list):
                                # Nested structure - use inner list
                                self.word_model.word_embeddings[current_batch[0]] = embeddings[0]
                            else:
                                # Flat list - use as is
                                self.word_model.word_embeddings[current_batch[0]] = embeddings
                                
                            # Initialize associations if developed enough
                            if self.development_level >= 0.6 and current_batch[0] not in self.word_model.word_associations:
                                self.word_model.word_associations[current_batch[0]] = []
                        else:
                            # Multiple words case
                            for i, word in enumerate(current_batch):
                                if i < len(embeddings):
                                    if isinstance(embeddings[i], list):
                                        self.word_model.word_embeddings[word] = embeddings[i]
                                    else:
                                        # If embeddings is a list of values instead of list of lists
                                        # This shouldn't happen with current API but handle it
                                        print(f"Warning: Unexpected embedding format for batch. Creating per-word embeddings.")
                                        # Try to get individual embeddings
                                        try:
                                            word_embedding = self.llm_client.get_embedding(word)
                                            if isinstance(word_embedding, list):
                                                if isinstance(word_embedding[0], list):
                                                    self.word_model.word_embeddings[word] = word_embedding[0]
                                                else:
                                                    self.word_model.word_embeddings[word] = word_embedding
                                        except Exception as e:
                                            print(f"Error getting embedding for single word '{word}': {e}")
                                            self._failed_embeddings.add(word)
                                    
                                    # Initialize associations if developed enough
                                    if self.development_level >= 0.6 and word not in self.word_model.word_associations:
                                        self.word_model.word_associations[word] = []
                    
                    # If we got here, this batch was successful
                    break
                    
                except Exception as e:
                    print(f"Warning: Embedding batch attempt {attempt+1} failed: {e}")
                    
                    if attempt == max_retries - 1:
                        # Last attempt, try fallback model
                        try:
                            # Get embeddings with fallback model
                            embeddings = self.llm_client.get_embedding(
                                current_batch,
                                embedding_model="text-embedding-ada-002"
                            )
                            
                            # Process successful embeddings (same logic as above)
                            if isinstance(embeddings, list):
                                # Handle single word vs. multiple word cases
                                if len(current_batch) == 1:
                                    if isinstance(embeddings[0], list):
                                        self.word_model.word_embeddings[current_batch[0]] = embeddings[0]
                                    else:
                                        self.word_model.word_embeddings[current_batch[0]] = embeddings
                                        
                                    # Initialize associations if developed enough
                                    if self.development_level >= 0.6 and current_batch[0] not in self.word_model.word_associations:
                                        self.word_model.word_associations[current_batch[0]] = []
                                else:
                                    for i, word in enumerate(current_batch):
                                        if i < len(embeddings):
                                            if isinstance(embeddings[i], list):
                                                self.word_model.word_embeddings[word] = embeddings[i]
                                            
                                            # Initialize associations if developed enough
                                            if self.development_level >= 0.6 and word not in self.word_model.word_associations:
                                                self.word_model.word_associations[word] = []
                            
                            print(f"Successfully used fallback embedding model for batch")
                            
                        except Exception as fallback_error:
                            print(f"ERROR: Fallback embedding also failed: {fallback_error}")
                            # Add words to failed list for later retry
                            for word in current_batch:
                                self._failed_embeddings.add(word)
        
        # Check if we have any failed words and log them
        if self._failed_embeddings:
            print(f"Words that couldn't be embedded: {len(self._failed_embeddings)}")
            # We'll try these again in future calls
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the word learning module
        
        Args:
            input_data: Input data dictionary
            
        Returns:
            Dict with processing results
        """
        # Validate input
        if not isinstance(input_data, dict):
            return {
                "status": "error",
                "message": "Input must be a dictionary"
            }
        
        # Extract process ID if provided
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Extract operation
        operation = input_data.get("operation", "recognize")
        
        # Dispatch to appropriate handler
        if operation == "recognize":
            return self._recognize_word(input_data, process_id)
        elif operation == "learn_word":
            return self._learn_word(input_data, process_id)
        elif operation == "associate_words":
            return self._associate_words(input_data, process_id)
        elif operation == "query_vocabulary":
            return self._query_vocabulary(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _recognize_word(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Recognize a word from input features
        
        Args:
            input_data: Input data dictionary including word features
            process_id: Process identifier
            
        Returns:
            Dict with recognition results
        """
        # Check for word features or phoneme sequence
        if "word_features" not in input_data and "phoneme_sequence" not in input_data and "word" not in input_data:
            return {
                "status": "error",
                "message": "Missing word_features, phoneme_sequence, or word for recognition",
                "process_id": process_id
            }
        
        # Process based on input type
        if "word" in input_data:
            # Direct word recognition (already segmented)
            word = input_data["word"]
            
            # Check if word is in vocabulary
            if word in self.word_model.vocabulary:
                familiarity = self.word_model.vocabulary[word]
                
                # Increase familiarity and frequency with exposure
                new_familiarity = min(1.0, familiarity + 0.01)
                self.word_model.vocabulary[word] = new_familiarity
                self.word_model.word_frequencies[word] = self.word_model.word_frequencies.get(word, 0) + 1
                
                # Record in recent inputs
                self.recent_inputs.append({
                    "type": "word_recognition",
                    "word": word,
                    "familiarity": new_familiarity,
                    "timestamp": datetime.now()
                })
                
                # Record activation in neural state
                self.neural_state.add_activation("word_learning", {
                    'operation': 'recognize',
                    'word': word,
                    'familiarity': new_familiarity
                })
                
                # Return recognition results
                return {
                    "status": "success",
                    "word": word,
                    "recognized": True,
                    "familiarity": new_familiarity,
                    "frequency": self.word_model.word_frequencies[word],
                    "categories": [cat for cat, words in self.word_model.word_categories.items() if word in words],
                    "process_id": process_id
                }
            else:
                # Word not in vocabulary
                return {
                    "status": "success",
                    "word": word,
                    "recognized": False,
                    "message": "Word not in vocabulary",
                    "process_id": process_id
                }
        
        elif "word_features" in input_data:
            # Features-based recognition
            word_features = input_data["word_features"]
            
            # Convert to tensor if needed
            if not isinstance(word_features, torch.Tensor):
                word_features = torch.tensor(word_features, dtype=torch.float32)
            
            # Ensure batch dimension
            if len(word_features.shape) == 1:
                word_features = word_features.unsqueeze(0)
            
            # Process through network
            word_features = word_features.to(self.device)
            with torch.no_grad():
                output = self.network(word_features, operation="recognize")
            
            # Get word probabilities and confidence
            word_probs = output["word_probs"].cpu().numpy()[0]
            confidence = output["confidence"].cpu().item()
            
            # Get top words based on vocabulary we know
            vocab_words = list(self.word_model.vocabulary.keys())
            word_probs_dict = {}
            
            # Map probabilities to words (up to vocabulary size)
            num_words = min(len(vocab_words), len(word_probs))
            for i in range(num_words):
                word_probs_dict[vocab_words[i]] = float(word_probs[i])
            
            # Get top words
            top_words = sorted(word_probs_dict.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Record activation in neural state
            self.neural_state.add_activation("word_learning", {
                'operation': 'recognize_features',
                'confidence': confidence,
                'top_word': top_words[0][0] if top_words else None
            })
            
            # Return recognition results
            return {
                "status": "success",
                "recognized_words": top_words,
                "confidence": confidence,
                "developmental_level": self.development_level,
                "process_id": process_id
            }
            
        else:
            # Phoneme sequence recognition
            phoneme_sequence = input_data["phoneme_sequence"]
            phoneme_str = "".join(phoneme_sequence)
            
            # Very simple word recognition from phoneme sequence
            # In a real implementation, would use more sophisticated matching
            recognized_word = None
            max_similarity = 0.0
            
            for word in self.word_model.vocabulary:
                # Calculate simple string similarity
                similarity = self._string_similarity(phoneme_str, word)
                if similarity > 0.7 and similarity > max_similarity:
                    max_similarity = similarity
                    recognized_word = word
            
            if recognized_word:
                # Update familiarity and frequency
                familiarity = self.word_model.vocabulary[recognized_word]
                new_familiarity = min(1.0, familiarity + 0.01)
                self.word_model.vocabulary[recognized_word] = new_familiarity
                self.word_model.word_frequencies[recognized_word] = self.word_model.word_frequencies.get(recognized_word, 0) + 1
                
                # Record activation in neural state
                self.neural_state.add_activation("word_learning", {
                    'operation': 'recognize_phonemes',
                    'word': recognized_word,
                    'similarity': max_similarity
                })
                
                # Return recognition results
                return {
                    "status": "success",
                    "phoneme_sequence": phoneme_sequence,
                    "recognized_word": recognized_word,
                    "similarity": max_similarity,
                    "familiarity": new_familiarity,
                    "process_id": process_id
                }
            else:
                # No matching word found
                return {
                    "status": "success",
                    "phoneme_sequence": phoneme_sequence,
                    "recognized": False,
                    "message": "No matching word found",
                    "process_id": process_id
                }
    
    def _learn_word(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Learn a new word or update an existing word
        
        Args:
            input_data: Input data dictionary including word and optional meaning
            process_id: Process identifier
            
        Returns:
            Dict with learning results
        """
        # Check for word
        if "word" not in input_data:
            return {
                "status": "error",
                "message": "Missing word for learning",
                "process_id": process_id
            }
        
        # Get word and category
        word = input_data["word"]
        category = input_data.get("category")
        meaning = input_data.get("meaning")
        
        # Check if word already in vocabulary
        if word in self.word_model.vocabulary:
            # Update existing word
            current_familiarity = self.word_model.vocabulary[word]
            
            # Increase familiarity with reinforcement, limited by development level
            max_familiarity = min(0.95, 0.4 + (self.development_level * 0.6))
            new_familiarity = min(max_familiarity, current_familiarity + 0.05)
            self.word_model.vocabulary[word] = new_familiarity
            
            # Update frequency
            self.word_model.word_frequencies[word] = self.word_model.word_frequencies.get(word, 0) + 1
            
            # Update category if provided and different
            if category and category in self.word_model.word_categories:
                if word not in self.word_model.word_categories[category]:
                    self.word_model.word_categories[category].append(word)
                    
            status = "updated"
        else:
            # Add new word with initial familiarity based on development
            initial_familiarity = min(0.6, 0.2 + (self.development_level * 0.4))
            self.word_model.vocabulary[word] = initial_familiarity
            self.word_model.word_frequencies[word] = 1
            
            # Add to category if provided
            if category:
                if category in self.word_model.word_categories:
                    self.word_model.word_categories[category].append(word)
                else:
                    self.word_model.word_categories[category] = [word]
                    
            status = "added"
        
        # Get word embedding if development level is sufficient and meaning is provided
        if self.development_level >= 0.5 and (meaning or word not in self.word_model.word_embeddings):
            # Track if embedding is needed
            embedding_needed = True
            
            # Check if we should use meaning for a more accurate embedding
            if meaning and len(meaning) > 0:
                embedding_text = word + ": " + meaning  # Include meaning for better embedding
            else:
                embedding_text = word
            
            # Try primary embedding model first
            try:
                embedding = self.llm_client.get_embedding(
                    embedding_text,
                    embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
                )
                
                # Process embedding based on format
                if isinstance(embedding, list):
                    if isinstance(embedding[0], list):
                        self.word_model.word_embeddings[word] = embedding[0]
                    else:
                        self.word_model.word_embeddings[word] = embedding
                
                # Mark as successful
                embedding_needed = False
            except Exception as primary_error:
                print(f"Warning: Primary embedding model failed for '{word}': {primary_error}")
            
            # Try fallback model if primary failed
            if embedding_needed:
                try:
                    embedding = self.llm_client.get_embedding(
                        embedding_text,
                        embedding_model="text-embedding-ada-002"
                    )
                    
                    # Process embedding based on format
                    if isinstance(embedding, list):
                        if isinstance(embedding[0], list):
                            self.word_model.word_embeddings[word] = embedding[0]
                        else:
                            self.word_model.word_embeddings[word] = embedding
                    
                    # Mark as successful
                    embedding_needed = False
                    print(f"Successfully used fallback embedding model for '{word}'")
                except Exception as fallback_error:
                    print(f"Warning: Fallback embedding also failed for '{word}': {fallback_error}")
            
            # If both attempts failed, mark for later retry
            if embedding_needed:
                if not hasattr(self, "_failed_embeddings"):
                    self._failed_embeddings = set()
                
                self._failed_embeddings.add(word)
                print(f"Added '{word}' to failed embeddings list for future retry")
                
                # We don't use hash-based embeddings anymore - we'll retry later
                # and use similarity search based on word form instead
        
        # Record activation in neural state
        self.neural_state.add_activation("word_learning", {
            'operation': 'learn_word',
            'word': word,
            'status': status
        })
        
        # Return learning results
        return {
            "status": "success",
            "word": word,
            "familiarity": self.word_model.vocabulary[word],
            "learning_status": status,
            "has_embedding": word in self.word_model.word_embeddings,
            "process_id": process_id
        }
    
    def _associate_words(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Create associations between words
        
        Args:
            input_data: Input data dictionary including words to associate
            process_id: Process identifier
            
        Returns:
            Dict with association results
        """
        # Check for words to associate
        if "word1" not in input_data or "word2" not in input_data:
            return {
                "status": "error",
                "message": "Missing word1 or word2 for association",
                "process_id": process_id
            }
        
        word1 = input_data["word1"]
        word2 = input_data["word2"]
        
        # Check if both words are in vocabulary
        if word1 not in self.word_model.vocabulary or word2 not in self.word_model.vocabulary:
            return {
                "status": "error",
                "message": "One or both words not in vocabulary",
                "process_id": process_id
            }
        
        # Check development level for associations
        if self.development_level < 0.4:
            return {
                "status": "undeveloped",
                "message": "Word association requires higher development level (0.4+)",
                "current_level": self.development_level,
                "process_id": process_id
            }
        
        # Create embedding tensors
        if word1 in self.word_model.word_embeddings and word2 in self.word_model.word_embeddings:
            # Use existing embeddings
            word1_embedding = self.word_model.word_embeddings[word1]
            word2_embedding = self.word_model.word_embeddings[word2]
            
            # Convert to tensors
            tensor1 = torch.tensor(word1_embedding, dtype=torch.float32).unsqueeze(0)
            tensor2 = torch.tensor(word2_embedding, dtype=torch.float32).unsqueeze(0)
            
            # Simple cosine similarity
            if tensor1.shape == tensor2.shape:
                similarity = torch.cosine_similarity(tensor1, tensor2).item()
            else:
                # Adjust dimensions if necessary
                max_dim = max(tensor1.size(1), tensor2.size(1))
                if tensor1.size(1) < max_dim:
                    tensor1 = torch.cat([tensor1, torch.zeros(1, max_dim - tensor1.size(1))], dim=1)
                if tensor2.size(1) < max_dim:
                    tensor2 = torch.cat([tensor2, torch.zeros(1, max_dim - tensor2.size(1))], dim=1)
                
                similarity = torch.cosine_similarity(tensor1, tensor2).item()
        else:
            # Fall back to simple string similarity if embeddings unavailable
            similarity = self._string_similarity(word1, word2)
        
        # Update word associations in model
        if word1 not in self.word_model.word_associations:
            self.word_model.word_associations[word1] = []
        if word2 not in self.word_model.word_associations:
            self.word_model.word_associations[word2] = []
        
        # Add bidirectional associations if they don't exist
        if word2 not in self.word_model.word_associations[word1]:
            self.word_model.word_associations[word1].append(word2)
        if word1 not in self.word_model.word_associations[word2]:
            self.word_model.word_associations[word2].append(word1)
        
        # Record activation in neural state
        self.neural_state.add_activation("word_learning", {
            'operation': 'associate_words',
            'word1': word1,
            'word2': word2,
            'similarity': similarity
        })
        
        # Return association results
        return {
            "status": "success",
            "word1": word1,
            "word2": word2,
            "similarity": similarity,
            "word1_associations": self.word_model.word_associations.get(word1, []),
            "word2_associations": self.word_model.word_associations.get(word2, []),
            "process_id": process_id
        }
    
    def _query_vocabulary(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query the vocabulary information
        
        Args:
            input_data: Input data dictionary including query parameters
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return summary of vocabulary
            return {
                "status": "success",
                "vocabulary_size": len(self.word_model.vocabulary),
                "categories": list(self.word_model.word_categories.keys()),
                "most_frequent": sorted(self.word_model.word_frequencies.items(), 
                                      key=lambda x: x[1], reverse=True)[:10],
                "developmental_level": self.development_level,
                "process_id": process_id
            }
        
        elif query_type == "word":
            # Check for word
            if "word" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing word for word query",
                    "process_id": process_id
                }
                
            word = input_data["word"]
            
            # Check if word exists in vocabulary
            if word not in self.word_model.vocabulary:
                return {
                    "status": "error",
                    "message": f"Word not found: {word}",
                    "process_id": process_id
                }
                
            # Get word information
            response = {
                "status": "success",
                "word": word,
                "familiarity": self.word_model.vocabulary[word],
                "frequency": self.word_model.word_frequencies.get(word, 0),
                "categories": [cat for cat, words in self.word_model.word_categories.items() if word in words],
                "process_id": process_id
            }
            
            # Add associations if available
            if word in self.word_model.word_associations:
                response["associations"] = self.word_model.word_associations[word]
                
            # Add embedding information if available
            if word in self.word_model.word_embeddings:
                # Don't include full embedding as it can be large
                response["has_embedding"] = True
                
            return response
        
        elif query_type == "category":
            # Check for category
            if "category" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing category for category query",
                    "process_id": process_id
                }
                
            category = input_data["category"]
            
            # Check if category exists
            if category not in self.word_model.word_categories:
                return {
                    "status": "error",
                    "message": f"Category not found: {category}",
                    "available_categories": list(self.word_model.word_categories.keys()),
                    "process_id": process_id
                }
                
            # Get words in category with their familiarity
            category_words = {}
            for word in self.word_model.word_categories[category]:
                if word in self.word_model.vocabulary:
                    category_words[word] = self.word_model.vocabulary[word]
                    
            return {
                "status": "success",
                "category": category,
                "words": category_words,
                "word_count": len(category_words),
                "process_id": process_id
            }
            
        elif query_type == "similar":
            # Check for query word
            if "word" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing word for similarity query",
                    "process_id": process_id
                }
                
            word = input_data["word"]
            
            # Check development level
            if self.development_level < 0.5:
                return {
                    "status": "undeveloped",
                    "message": "Similarity search requires higher development level (0.5+)",
                    "current_level": self.development_level,
                    "process_id": process_id
                }
                
            # Check if word exists in vocabulary and has embedding
            if word not in self.word_model.vocabulary:
                return {
                    "status": "error",
                    "message": f"Word not found in vocabulary: {word}",
                    "process_id": process_id
                }
                
            if word not in self.word_model.word_embeddings:
                return {
                    "status": "error",
                    "message": f"No embedding available for word: {word}",
                    "process_id": process_id
                }
                
            # Find similar words by embedding similarity
            similarities = {}
            target_embedding = torch.tensor(self.word_model.word_embeddings[word], dtype=torch.float32)
            
            for other_word, other_embedding in self.word_model.word_embeddings.items():
                if other_word != word:
                    other_tensor = torch.tensor(other_embedding, dtype=torch.float32)
                    
                    # Ensure tensors have same dimensions
                    if target_embedding.shape != other_tensor.shape:
                        continue
                        
                    # Calculate similarity
                    similarity = torch.cosine_similarity(
                        target_embedding.unsqueeze(0), 
                        other_tensor.unsqueeze(0)
                    ).item()
                    
                    similarities[other_word] = similarity
                    
            # Get top similar words
            top_similar = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]
            
            return {
                "status": "success",
                "word": word,
                "similar_words": top_similar,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _string_similarity(self, str1: str, str2: str) -> float:
        """
        Calculate simple string similarity
        
        Args:
            str1: First string
            str2: Second string
            
        Returns:
            Similarity score between 0 and 1
        """
        # Simplified Levenshtein distance ratio
        if not str1 or not str2:
            return 0.0
            
        len1, len2 = len(str1), len(str2)
        if len1 == 0 or len2 == 0:
            return 0.0
            
        # Simple length ratio as a baseline
        length_ratio = min(len1, len2) / max(len1, len2)
        
        # Count matching characters
        matches = sum(c1 == c2 for c1, c2 in zip(str1, str2))
        match_ratio = matches / max(len1, len2)
        
        # Combine metrics
        return (length_ratio * 0.4) + (match_ratio * 0.6)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of the module
        
        Args:
            amount: Amount to increase development by
            
        Returns:
            New development level
        """
        old_level = self.development_level
        
        # Update development level
        self.development_level = max(0.0, min(1.0, self.development_level + amount))
        
        # Update neural network
        self.network.set_development_level(self.development_level)
        
        # Update neural state
        self.neural_state.word_learning_development = self.development_level
        self.neural_state.last_updated = datetime.now()
        
        # Check if crossed a milestone
        for level in sorted(self.development_milestones.keys()):
            if old_level < level <= self.development_level:
                milestone = self.development_milestones[level]
                
                # Publish milestone event if we have an event bus
                if self.event_bus:
                    self.event_bus.publish({
                        "sender": self.module_id,
                        "message_type": "development_milestone",
                        "content": {
                            "module": "word_learning",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Update vocabulary for new development level
                self._initialize_vocabulary()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the word learning module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "vocabulary_size": len(self.word_model.vocabulary),
            "category_count": len(self.word_model.word_categories),
            "embedding_count": len(self.word_model.word_embeddings),
            "association_count": sum(len(assocs) for assocs in self.word_model.word_associations.values())
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "word_model": self.word_model.dict(),
            "developmental_level": self.development_level,
            "neural_state": {
                "development": self.neural_state.word_learning_development,
                "accuracy": self.neural_state.word_learning_accuracy
            }
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        Load a previously saved state
        
        Args:
            state: The state to load
        """
        # Load module ID
        self.module_id = state["module_id"]
        
        # Load development level
        self.development_level = state["developmental_level"]
        self.network.set_development_level(self.development_level)
        
        # Load word model
        if "word_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.word_model = parse_obj_as(WordModel, state["word_model"])
            except Exception as e:
                print(f"Error loading word model: {e}")
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            self.neural_state.word_learning_development = ns.get("development", self.development_level)
            self.neural_state.word_learning_accuracy = ns.get("accuracy", 0.5)
