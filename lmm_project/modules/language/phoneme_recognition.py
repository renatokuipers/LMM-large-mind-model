# TODO: Implement the PhonemeRecognition class to identify basic speech sounds
# This component should be able to:
# - Recognize phonemes in speech input
# - Differentiate between similar phonemes
# - Adapt to different speakers and accents
# - Develop phonological awareness

# TODO: Implement developmental progression in phoneme recognition:
# - Basic categorical perception in early stages
# - Growing phoneme differentiation in early childhood
# - Phonological rule understanding in later childhood
# - Automaticity in phoneme processing in adulthood

# TODO: Create mechanisms for:
# - Acoustic analysis: Extract relevant sound features
# - Phoneme categorization: Classify sounds as specific phonemes
# - Speaker normalization: Adjust for speaker differences
# - Phonological rule learning: Understand phoneme patterns

# TODO: Implement phonological awareness capabilities:
# - Phoneme isolation: Identify individual sounds
# - Phoneme blending: Combine sounds into words
# - Phoneme segmentation: Break words into component sounds
# - Phoneme manipulation: Add, delete, or substitute sounds

# TODO: Connect to perception and word learning systems
# Phoneme recognition should draw on auditory perception
# and feed into word learning processes

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.language.models import PhonemeModel, LanguageNeuralState
from lmm_project.modules.language.neural_net import PhonemeNetwork, get_device
from lmm_project.utils.llm_client import LLMClient

class PhonemeRecognition(BaseModule):
    """
    Recognizes and processes phonemes (speech sounds)
    
    This module is responsible for learning to recognize phonemes,
    differentiate between similar sounds, and learn phonotactic rules.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic sound discrimination",
        0.2: "Native language sound category formation",
        0.4: "Phoneme boundary detection",
        0.6: "Phonotactic rule learning",
        0.8: "Non-native phoneme attenuation",
        1.0: "Complete phonological system"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the phoneme recognition module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize phoneme model
        self.phoneme_model = PhonemeModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = PhonemeNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.phoneme_recognition_development = self.development_level
        
        # Initialize basic phoneme inventory based on development level
        self._initialize_phoneme_inventory()
        
        # Recent inputs queue (for tracking recent phoneme exposures)
        self.recent_inputs = deque(maxlen=100)
        
        # For embedding generation when needed
        self.llm_client = LLMClient()
    
    def _initialize_phoneme_inventory(self):
        """Initialize basic phoneme inventory based on development level"""
        # Basic vowels (recognized at earliest stages)
        basic_vowels = {"a": 0.7, "i": 0.7, "u": 0.7}
        
        # Add basic vowels to inventory
        for phoneme, recognition in basic_vowels.items():
            # Scale recognition by development level
            scaled_recognition = recognition * max(0.3, self.development_level)
            self.phoneme_model.phoneme_inventory[phoneme] = scaled_recognition
        
        # Add to vowel category
        self.phoneme_model.phoneme_categories["vowels"] = list(basic_vowels.keys())
        
        if self.development_level >= 0.2:
            # Basic consonants (recognized at slightly later stages)
            basic_consonants = {"m": 0.6, "b": 0.6, "p": 0.6, "t": 0.6, "d": 0.6}
            
            # Add basic consonants to inventory
            for phoneme, recognition in basic_consonants.items():
                # Scale recognition by development level
                scaled_recognition = recognition * ((self.development_level - 0.2) / 0.8)
                self.phoneme_model.phoneme_inventory[phoneme] = scaled_recognition
            
            # Add to consonant category
            self.phoneme_model.phoneme_categories["consonants"] = list(basic_consonants.keys())
        
        if self.development_level >= 0.4:
            # More complex phonemes
            complex_phonemes = {"f": 0.5, "v": 0.5, "s": 0.5, "z": 0.5, "k": 0.5, "g": 0.5}
            
            # Add complex phonemes to inventory
            for phoneme, recognition in complex_phonemes.items():
                # Scale recognition by development level
                scaled_recognition = recognition * ((self.development_level - 0.4) / 0.6)
                self.phoneme_model.phoneme_inventory[phoneme] = scaled_recognition
            
            # Add to consonant category
            self.phoneme_model.phoneme_categories["consonants"].extend(list(complex_phonemes.keys()))
            
            # Basic phonotactic rules
            self.phoneme_model.phonotactic_rules.append({
                "description": "Consonant-vowel sequence",
                "pattern": "CV",
                "confidence": 0.6 * ((self.development_level - 0.4) / 0.6)
            })
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the phoneme recognition module
        
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
            return self._recognize_phonemes(input_data, process_id)
        elif operation == "analyze_patterns":
            return self._analyze_phoneme_patterns(input_data, process_id)
        elif operation == "learn_phoneme":
            return self._learn_phoneme(input_data, process_id)
        elif operation == "query_inventory":
            return self._query_phoneme_inventory(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _recognize_phonemes(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Recognize phonemes in audio input
        
        Args:
            input_data: Input data dictionary including audio features
            process_id: Process identifier
            
        Returns:
            Dict with recognition results
        """
        # Check for audio features
        if "audio_features" not in input_data:
            return {
                "status": "error",
                "message": "Missing audio_features for phoneme recognition",
                "process_id": process_id
            }
        
        # Get audio features
        audio_features = input_data["audio_features"]
        
        # Convert to tensor if needed
        if not isinstance(audio_features, torch.Tensor):
            audio_features = torch.tensor(audio_features, dtype=torch.float32)
        
        # Ensure batch dimension
        if len(audio_features.shape) == 1:
            audio_features = audio_features.unsqueeze(0)
        
        # Process through network
        audio_features = audio_features.to(self.device)
        with torch.no_grad():
            output = self.network(audio_features, operation="recognize")
        
        # Get top phonemes
        phoneme_probs = output["phoneme_probs"].cpu().numpy()[0]
        confidence = output["confidence"].cpu().item()
        
        # Create list of phonemes with probabilities
        all_phonemes = list(self.phoneme_model.phoneme_inventory.keys())
        phoneme_probs_dict = {}
        
        # Map probabilities to phonemes (up to the size of our inventory)
        for i, prob in enumerate(phoneme_probs[:min(len(all_phonemes), len(phoneme_probs))]):
            phoneme_probs_dict[all_phonemes[i]] = float(prob)
        
        # Get top phonemes
        top_phonemes = sorted(phoneme_probs_dict.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Record in recent inputs
        self.recent_inputs.append({
            "type": "phoneme_recognition",
            "phonemes": phoneme_probs_dict,
            "confidence": confidence,
            "timestamp": datetime.now()
        })
        
        # Record activation in neural state
        self.neural_state.add_activation("phoneme_recognition", {
            'operation': 'recognize',
            'confidence': confidence,
            'top_phoneme': top_phonemes[0][0] if top_phonemes else None
        })
        
        # Return recognized phonemes
        return {
            "status": "success",
            "recognized_phonemes": top_phonemes,
            "confidence": confidence,
            "developmental_level": self.development_level,
            "process_id": process_id
        }
    
    def _analyze_phoneme_similarity(self, phoneme1: str, phoneme2: str) -> float:
        """
        Analyze phonetic similarity between two phonemes
        
        Uses the LLM API for advanced phonetic similarity when available,
        falls back to feature-based comparison.
        
        Args:
            phoneme1: First phoneme
            phoneme2: Second phoneme
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # If comparing with self, return 1.0
        if phoneme1 == phoneme2:
            return 1.0
            
        # Check if we have features for both phonemes
        if (phoneme1 in self.phoneme_model.phoneme_features and 
            phoneme2 in self.phoneme_model.phoneme_features):
            
            # Get phoneme features
            features1 = self.phoneme_model.phoneme_features[phoneme1]
            features2 = self.phoneme_model.phoneme_features[phoneme2]
            
            # Calculate similarity based on shared features
            shared_features = 0
            total_features = 0
            
            # Compare common phonetic features
            for feature in ["place", "manner", "voicing", "vowel_height", "vowel_backness", "rounded"]:
                if feature in features1 and feature in features2:
                    total_features += 1
                    if features1[feature] == features2[feature]:
                        shared_features += 1
            
            # If we have features to compare
            if total_features > 0:
                return shared_features / total_features
        
        # If phoneme features aren't available or development level is high,
        # try using the LLM API for phonetic similarity
        if self.development_level >= 0.6:
            try:
                # Create prompts for embedding comparison
                phoneme1_desc = f"Phoneme: {phoneme1}. Pronunciation characteristics and articulatory features."
                phoneme2_desc = f"Phoneme: {phoneme2}. Pronunciation characteristics and articulatory features."
                
                # Get embeddings for both phonemes
                embedding1 = self.llm_client.get_embedding(
                    phoneme1_desc,
                    embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
                )
                
                embedding2 = self.llm_client.get_embedding(
                    phoneme2_desc,
                    embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
                )
                
                # Process embeddings to calculate similarity
                if isinstance(embedding1, list) and isinstance(embedding2, list):
                    # Handle nested lists
                    if isinstance(embedding1[0], list):
                        embedding1 = embedding1[0]
                    if isinstance(embedding2[0], list):
                        embedding2 = embedding2[0]
                    
                    # Ensure same length for comparison
                    min_length = min(len(embedding1), len(embedding2))
                    
                    # Calculate cosine similarity
                    dot_product = sum(a * b for a, b in zip(embedding1[:min_length], embedding2[:min_length]))
                    magnitude1 = sum(a * a for a in embedding1[:min_length]) ** 0.5
                    magnitude2 = sum(b * b for b in embedding2[:min_length]) ** 0.5
                    
                    if magnitude1 > 0 and magnitude2 > 0:
                        similarity = dot_product / (magnitude1 * magnitude2)
                        # Scale to 0.0-1.0 range (cosine similarity is between -1 and 1)
                        return (similarity + 1) / 2
            
            except Exception as e:
                print(f"Warning: Failed to calculate phoneme similarity using LLM API: {e}")
        
        # Fallback: basic string similarity
        # Count matching characters
        min_len = min(len(phoneme1), len(phoneme2))
        max_len = max(len(phoneme1), len(phoneme2))
        
        matches = 0
        for i in range(min_len):
            if phoneme1[i] == phoneme2[i]:
                matches += 1
        
        # Return similarity ratio
        return matches / max_len
    
    def _analyze_phoneme_patterns(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Analyze patterns in phoneme sequences
        
        Args:
            input_data: Input data dictionary including phoneme sequence
            process_id: Process identifier
            
        Returns:
            Dict with analysis results
        """
        # Check for phoneme sequence
        if "phoneme_sequence" not in input_data:
            return {
                "status": "error",
                "message": "Missing phoneme_sequence for pattern analysis",
                "process_id": process_id
            }
        
        # Get phoneme sequence
        phoneme_sequence = input_data["phoneme_sequence"]
        
        # Ensure we have a list of phonemes
        if isinstance(phoneme_sequence, str):
            phoneme_sequence = phoneme_sequence.split()
        
        # If sequence is too short for pattern analysis
        if len(phoneme_sequence) < 2:
            return {
                "status": "success",
                "message": "Sequence too short for pattern analysis",
                "patterns": [],
                "process_id": process_id
            }
        
        # Development level affects pattern recognition sophistication
        dev_level = self.development_level
        
        # Find patterns in the sequence
        patterns = []
        
        # Look for repeated phonemes
        for i in range(len(phoneme_sequence) - 1):
            # For exact repetitions
            if phoneme_sequence[i] == phoneme_sequence[i + 1]:
                patterns.append({
                    "type": "repetition",
                    "phonemes": [phoneme_sequence[i]],
                    "position": i,
                    "confidence": 0.9
                })
            
            # For similar phonemes (if development level is sufficient)
            elif dev_level >= 0.4:
                similarity = self._analyze_phoneme_similarity(phoneme_sequence[i], phoneme_sequence[i + 1])
                if similarity >= 0.7:  # High similarity threshold
                    patterns.append({
                        "type": "similar_phonemes",
                        "phonemes": [phoneme_sequence[i], phoneme_sequence[i + 1]],
                        "position": i,
                        "similarity": similarity,
                        "confidence": similarity * 0.8
                    })
        
        # Look for common sequences (if development level is sufficient)
        if dev_level >= 0.3:
            for rule in self.phoneme_model.phonotactic_rules:
                rule_phonemes = rule["sequence"]
                
                # Convert to list if it's a string
                if isinstance(rule_phonemes, str):
                    rule_phonemes = rule_phonemes.split()
                
                # Check if rule sequence exists in the input sequence
                for i in range(len(phoneme_sequence) - len(rule_phonemes) + 1):
                    match = True
                    for j in range(len(rule_phonemes)):
                        if phoneme_sequence[i + j] != rule_phonemes[j]:
                            match = False
                            break
                    
                    if match:
                        patterns.append({
                            "type": "known_pattern",
                            "pattern_name": rule.get("name", "unnamed"),
                            "phonemes": rule_phonemes,
                            "position": i,
                            "confidence": rule.get("confidence", 0.7)
                        })
        
        # Look for vowel harmony (if development level is sufficient)
        if dev_level >= 0.6:
            vowels = [p for p in phoneme_sequence 
                     if p in self.phoneme_model.phoneme_categories.get("vowels", [])]
            
            if len(vowels) >= 2:
                # Check for vowel harmony (similar vowels)
                harmony_score = 0
                for i in range(len(vowels) - 1):
                    similarity = self._analyze_phoneme_similarity(vowels[i], vowels[i + 1])
                    harmony_score += similarity
                
                # Calculate average harmony
                if len(vowels) > 1:
                    avg_harmony = harmony_score / (len(vowels) - 1)
                    
                    if avg_harmony >= 0.6:  # Threshold for harmony
                        patterns.append({
                            "type": "vowel_harmony",
                            "vowels": vowels,
                            "harmony_score": avg_harmony,
                            "confidence": avg_harmony * 0.8
                        })
        
        # Return results
        return {
            "status": "success",
            "patterns": patterns,
            "sequence_length": len(phoneme_sequence),
            "development_level": dev_level,
            "process_id": process_id
        }
    
    def _learn_phoneme(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Learn a new phoneme or update recognition for existing phoneme
        
        Args:
            input_data: Input data dictionary including phoneme and examples
            process_id: Process identifier
            
        Returns:
            Dict with learning results
        """
        # Check for phoneme
        if "phoneme" not in input_data:
            return {
                "status": "error",
                "message": "Missing phoneme for learning",
                "process_id": process_id
            }
        
        # Get phoneme and examples
        phoneme = input_data["phoneme"]
        examples = input_data.get("examples", [])
        
        # Get category if provided
        category = input_data.get("category", "unknown")
        
        # Check if phoneme already exists
        if phoneme in self.phoneme_model.phoneme_inventory:
            # Update recognition confidence
            current_confidence = self.phoneme_model.phoneme_inventory[phoneme]
            # Increase recognition, but limited by development level
            max_confidence = min(0.95, 0.3 + (self.development_level * 0.7))
            new_confidence = min(max_confidence, current_confidence + 0.05)
            self.phoneme_model.phoneme_inventory[phoneme] = new_confidence
            
            status = "updated"
        else:
            # Add new phoneme with initial confidence based on development
            initial_confidence = min(0.5, 0.2 + (self.development_level * 0.3))
            self.phoneme_model.phoneme_inventory[phoneme] = initial_confidence
            
            # Add to appropriate category
            if category in self.phoneme_model.phoneme_categories:
                if phoneme not in self.phoneme_model.phoneme_categories[category]:
                    self.phoneme_model.phoneme_categories[category].append(phoneme)
            else:
                self.phoneme_model.phoneme_categories[category] = [phoneme]
            
            status = "added"
        
        # If examples provided, create phoneme features
        if examples and len(examples) > 0:
            # Simple feature creation using mean of example features
            example_features = []
            
            for example in examples[:5]:  # Limit to 5 examples
                # Convert to feature vector (simple hash-based features)
                feature_vec = np.zeros(128)
                
                # Set features based on example characters
                for i, c in enumerate(example[:10]):  # Limit to 10 chars
                    pos = (hash(c) + i) % 120
                    feature_vec[pos] = 1.0
                
                example_features.append(feature_vec)
            
            # Calculate mean features if we have examples
            if example_features:
                mean_features = np.mean(example_features, axis=0)
                
                # Store in phoneme features
                self.phoneme_model.phoneme_features[phoneme] = {
                    "vector": mean_features.tolist(),
                    "examples": examples[:5]
                }
        
        # Record activation in neural state
        self.neural_state.add_activation("phoneme_recognition", {
            'operation': 'learn_phoneme',
            'phoneme': phoneme,
            'status': status
        })
        
        # Return learning results
        return {
            "status": "success",
            "phoneme": phoneme,
            "recognition_confidence": self.phoneme_model.phoneme_inventory[phoneme],
            "learning_status": status,
            "process_id": process_id
        }
    
    def _query_phoneme_inventory(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query the phoneme inventory
        
        Args:
            input_data: Input data dictionary including query parameters
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return all phonemes
            return {
                "status": "success",
                "phoneme_inventory": dict(self.phoneme_model.phoneme_inventory),
                "phoneme_categories": dict(self.phoneme_model.phoneme_categories),
                "phonotactic_rules": self.phoneme_model.phonotactic_rules,
                "process_id": process_id
            }
        
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
            if category not in self.phoneme_model.phoneme_categories:
                return {
                    "status": "error",
                    "message": f"Category not found: {category}",
                    "available_categories": list(self.phoneme_model.phoneme_categories.keys()),
                    "process_id": process_id
                }
            
            # Get phonemes in category with their recognition confidence
            category_phonemes = {}
            for phoneme in self.phoneme_model.phoneme_categories[category]:
                if phoneme in self.phoneme_model.phoneme_inventory:
                    category_phonemes[phoneme] = self.phoneme_model.phoneme_inventory[phoneme]
            
            return {
                "status": "success",
                "category": category,
                "phonemes": category_phonemes,
                "process_id": process_id
            }
        
        elif query_type == "rules":
            # Return phonotactic rules
            return {
                "status": "success",
                "phonotactic_rules": self.phoneme_model.phonotactic_rules,
                "rule_count": len(self.phoneme_model.phonotactic_rules),
                "process_id": process_id
            }
        
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
        }
    
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
        self.neural_state.phoneme_recognition_development = self.development_level
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
                            "module": "phoneme_recognition",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Update phoneme inventory for new development level
                self._initialize_phoneme_inventory()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the phoneme recognition module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "phoneme_model": self.phoneme_model.dict(),
            "developmental_level": self.development_level,
            "inventory_size": len(self.phoneme_model.phoneme_inventory),
            "category_count": len(self.phoneme_model.phoneme_categories),
            "rule_count": len(self.phoneme_model.phonotactic_rules)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "phoneme_model": self.phoneme_model.dict(),
            "developmental_level": self.development_level,
            "neural_state": {
                "development": self.neural_state.phoneme_recognition_development,
                "accuracy": self.neural_state.phoneme_recognition_accuracy
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
        
        # Load phoneme model
        if "phoneme_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.phoneme_model = parse_obj_as(PhonemeModel, state["phoneme_model"])
            except Exception as e:
                print(f"Error loading phoneme model: {e}")
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            self.neural_state.phoneme_recognition_development = ns.get("development", self.development_level)
            self.neural_state.phoneme_recognition_accuracy = ns.get("accuracy", 0.5) 
