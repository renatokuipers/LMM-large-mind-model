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
# - Phoneme identification: Recognize distinct sound units
# - Phoneme manipulation: Add/remove/change sounds
# - Syllable awareness: Recognize syllable boundaries
# - Pattern recognition: Identify rhymes and alliteration

# TODO: Connect to perception and word learning systems
# Phoneme recognition should draw on auditory perception
# and feed into word learning processes

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
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
    
    def _analyze_phoneme_patterns(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Analyze phoneme patterns to identify phonotactic rules
        
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
        
        # Check development level
        if self.development_level < 0.4:
            return {
                "status": "undeveloped",
                "message": "Pattern analysis requires higher development level (0.4+)",
                "current_level": self.development_level,
                "process_id": process_id
            }
        
        # Convert sequence to features
        sequence_features = []
        for phoneme in phoneme_sequence:
            # Create simple one-hot-like features
            # In a real implementation, would use actual phonological features
            feature_vec = np.zeros(128)
            
            # Set simple hash-based feature
            hash_val = hash(phoneme) % 100
            feature_vec[hash_val] = 1.0
            
            sequence_features.append(feature_vec)
        
        # Convert to tensor
        sequence_tensor = torch.tensor(np.array(sequence_features), dtype=torch.float32)
        sequence_tensor = sequence_tensor.to(self.device)
        
        # Process through network
        with torch.no_grad():
            output = self.network(sequence_tensor.mean(dim=0, keepdim=True), operation="analyze")
        
        # Extract pattern features
        pattern_features = output["phoneme_features"].cpu().numpy()[0]
        
        # Build a simplified pattern description
        # Identify consonant-vowel patterns
        cv_pattern = ""
        for phoneme in phoneme_sequence:
            if phoneme in self.phoneme_model.phoneme_categories.get("vowels", []):
                cv_pattern += "V"
            elif phoneme in self.phoneme_model.phoneme_categories.get("consonants", []):
                cv_pattern += "C"
            else:
                cv_pattern += "?"
        
        # Check for existing patterns or create new ones
        pattern_found = False
        for rule in self.phoneme_model.phonotactic_rules:
            if rule["pattern"] == cv_pattern:
                # Increase confidence in existing rule
                rule["confidence"] = min(1.0, rule["confidence"] + 0.05)
                pattern_found = True
                break
        
        # Add new pattern if not found and meets minimum criteria
        if not pattern_found and len(cv_pattern) >= 2:
            # Only mature enough systems can create new rules
            if self.development_level >= 0.6:
                self.phoneme_model.phonotactic_rules.append({
                    "description": f"Observed phoneme pattern",
                    "pattern": cv_pattern,
                    "confidence": 0.3,
                    "examples": [phoneme_sequence]
                })
        
        # Record activation in neural state
        self.neural_state.add_activation("phoneme_recognition", {
            'operation': 'analyze_patterns',
            'pattern': cv_pattern
        })
        
        # Return analysis results
        return {
            "status": "success",
            "cv_pattern": cv_pattern,
            "pattern_features": pattern_features.tolist(),
            "developmental_level": self.development_level,
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
