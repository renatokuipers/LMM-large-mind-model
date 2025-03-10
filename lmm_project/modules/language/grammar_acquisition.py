# TODO: Implement the GrammarAcquisition class to learn and apply grammatical rules
# This component should be able to:
# - Identify grammatical patterns from language input
# - Extract and formalize grammatical rules
# - Apply learned rules in language comprehension and production
# - Handle syntactic processing and sentence structure

# TODO: Implement developmental progression in grammar acquisition:
# - Simple two-word combinations in early stages
# - Basic sentence structures in early childhood
# - Complex grammar and exceptions in later childhood
# - Advanced syntax and pragmatics in adolescence/adulthood

# TODO: Create mechanisms for:
# - Pattern detection: Identify recurring grammatical structures
# - Rule extraction: Formalize explicit and implicit rules
# - Syntactic parsing: Analyze sentence structure
# - Grammatical error detection: Identify violations of learned rules

# TODO: Implement different grammatical concepts:
# - Word order rules (syntax)
# - Morphological rules (word formation)
# - Agreement rules (subject-verb, etc.)
# - Dependency relationships between sentence elements

# TODO: Connect to word learning and semantic processing
# Grammar acquisition should work with lexical knowledge
# and contribute to meaning extraction

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.base.module import BaseModule
from lmm_project.event_bus import EventBus
from lmm_project.modules.language.models import GrammarModel, LanguageNeuralState
from lmm_project.modules.language.neural_net import GrammarNetwork, get_device
from lmm_project.utils.llm_client import LLMClient

class GrammarAcquisition(BaseModule):
    """
    Learns and processes grammatical structures
    
    This module is responsible for acquiring grammatical rules,
    syntactic patterns, and morphological regularities.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic word ordering",
        0.2: "Two-word combinations",
        0.4: "Early grammatical markers",
        0.6: "Complex sentence structures",
        0.8: "Rule generalization and exceptions",
        1.0: "Complete grammatical system"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the grammar acquisition module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize grammar model
        self.grammar_model = GrammarModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = GrammarNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.grammar_acquisition_development = self.development_level
        
        # Initialize with basic grammar structures based on development level
        self._initialize_grammar_structures()
        
        # Recent inputs queue (for tracking grammar exposure)
        self.recent_inputs = deque(maxlen=100)
        
        # For embedding generation when needed
        self.llm_client = LLMClient()
    
    def _initialize_grammar_structures(self):
        """Initialize basic grammatical structures based on development level"""
        # Basic structures at earliest stages
        if self.development_level >= 0.0:
            # Simple word ordering patterns (SVO - Subject-Verb-Object)
            self.grammar_model.syntactic_patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "name": "Basic word order",
                "pattern": "SV",  # Subject-Verb
                "examples": ["Baby sleep", "Dog run"],
                "confidence": 0.6 * max(0.2, self.development_level)
            })
            
            # Initialize grammatical categories
            self.grammar_model.grammatical_categories["subject"] = ["baby", "dog", "mama", "dada"]
            self.grammar_model.grammatical_categories["verb"] = ["sleep", "run", "eat", "go"]
        
        if self.development_level >= 0.2:
            # Two-word combinations and early object usage
            self.grammar_model.syntactic_patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "name": "Subject-Verb-Object",
                "pattern": "SVO",  # Subject-Verb-Object
                "examples": ["Baby want milk", "Mama see dog"],
                "confidence": 0.6 * ((self.development_level - 0.2) / 0.8)
            })
            
            # Update grammatical categories
            self.grammar_model.grammatical_categories["object"] = ["milk", "ball", "dog", "book"]
            
            # Simple morphological rules
            self.grammar_model.morphological_rules["plural"] = {
                "rule_id": str(uuid.uuid4()),
                "description": "Add 's' to make plural",
                "pattern": "{word} + s",
                "examples": {"dog": "dogs", "cat": "cats"},
                "confidence": 0.5 * ((self.development_level - 0.2) / 0.8)
            }
        
        if self.development_level >= 0.4:
            # More complex structures
            self.grammar_model.syntactic_patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "name": "Subject-Verb-Object with modifier",
                "pattern": "S V O M",  # Subject-Verb-Object-Modifier
                "examples": ["Baby drink milk now", "Dog play ball outside"],
                "confidence": 0.6 * ((self.development_level - 0.4) / 0.6)
            })
            
            # Add modifiers category
            self.grammar_model.grammatical_categories["modifier"] = ["now", "here", "there", "outside"]
            
            # More morphological rules
            self.grammar_model.morphological_rules["past_tense"] = {
                "rule_id": str(uuid.uuid4()),
                "description": "Add 'ed' for past tense",
                "pattern": "{word} + ed",
                "examples": {"play": "played", "jump": "jumped"},
                "confidence": 0.5 * ((self.development_level - 0.4) / 0.6)
            }
            
            # Update rule confidence for existing rules
            if "plural" in self.grammar_model.morphological_rules:
                self.grammar_model.morphological_rules["plural"]["confidence"] = 0.7
        
        if self.development_level >= 0.6:
            # Complex sentence structures
            self.grammar_model.syntactic_patterns.append({
                "pattern_id": str(uuid.uuid4()),
                "name": "Compound sentence",
                "pattern": "S1 V1 and S2 V2",
                "examples": ["Baby eat and mama smile", "Dog bark and cat run"],
                "confidence": 0.6 * ((self.development_level - 0.6) / 0.4)
            })
            
            # Add conjunction category
            self.grammar_model.grammatical_categories["conjunction"] = ["and", "but", "or"]
            
            # More complex morphological rules
            self.grammar_model.morphological_rules["present_progressive"] = {
                "rule_id": str(uuid.uuid4()),
                "description": "Add 'ing' for present progressive",
                "pattern": "{be} + {word} + ing",
                "examples": {"run": "is running", "eat": "is eating"},
                "confidence": 0.5 * ((self.development_level - 0.6) / 0.4)
            }
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the grammar acquisition module
        
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
        operation = input_data.get("operation", "analyze")
        
        # Dispatch to appropriate handler
        if operation == "analyze":
            return self._analyze_grammar(input_data, process_id)
        elif operation == "learn_rule":
            return self._learn_grammar_rule(input_data, process_id)
        elif operation == "check_grammar":
            return self._check_grammaticality(input_data, process_id)
        elif operation == "query_grammar":
            return self._query_grammar(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _analyze_grammar(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Analyze a sentence or utterance for grammatical structure
        
        Args:
            input_data: Input data dictionary including sentence
            process_id: Process identifier
            
        Returns:
            Dict with analysis results
        """
        # Check for sentence
        if "sentence" not in input_data:
            return {
                "status": "error",
                "message": "Missing sentence for grammar analysis",
                "process_id": process_id
            }
        
        sentence = input_data["sentence"]
        
        # Tokenize sentence (simple split for this implementation)
        tokens = sentence.split()
        
        # Convert tokens to feature vectors
        token_features = []
        for token in tokens:
            # Simple feature creation (in a real implementation, these would be richer)
            token_vector = np.zeros(128)
            
            # Set hash-based features
            hash_val = hash(token) % 100
            token_vector[hash_val] = 1.0
            
            # Add positional information
            pos_index = tokens.index(token) % 10
            token_vector[100 + pos_index] = 1.0
            
            token_features.append(token_vector)
        
        # Convert to tensor
        if token_features:
            token_tensor = torch.tensor(np.array(token_features), dtype=torch.float32)
            token_tensor = token_tensor.to(self.device)
            
            # Process through network
            with torch.no_grad():
                # First pass just using the first token as a representative
                initial_output = self.network(token_tensor[0:1], operation="recognize")
                
                # Second pass with the sequence if available
                if len(token_tensor) > 1:
                    sequence_output = self.network(token_tensor[0:1], operation="predict", sequence=token_tensor)
                    prediction_quality = sequence_output.get("quality", torch.tensor([0.5])).item()
                else:
                    prediction_quality = 0.5
        else:
            # Empty sentence
            return {
                "status": "error",
                "message": "Empty sentence provided",
                "process_id": process_id
            }
        
        # Identify grammatical structure
        structure_found = False
        matched_pattern = None
        
        # Simple grammatical categorization of tokens
        token_categories = []
        for token in tokens:
            category = "unknown"
            for cat, words in self.grammar_model.grammatical_categories.items():
                if token.lower() in words:
                    category = cat
                    break
            token_categories.append(category)
        
        # Convert to simplified pattern
        pattern_str = " ".join([cat[0].upper() if cat != "unknown" else "X" for cat in token_categories])
        
        # Match against known patterns
        for pattern in self.grammar_model.syntactic_patterns:
            pattern_tokens = pattern["pattern"].split()
            # Simple pattern matching (in a real implementation, would be more sophisticated)
            if len(pattern_tokens) == len(token_categories):
                matches = True
                for i, pat in enumerate(pattern_tokens):
                    if pat != token_categories[i][0].upper() and token_categories[i] != "unknown":
                        matches = False
                        break
                
                if matches:
                    structure_found = True
                    matched_pattern = pattern
                    break
        
        # Record activation in neural state
        self.neural_state.add_activation("grammar_acquisition", {
            'operation': 'analyze',
            'pattern_found': structure_found,
            'pattern': pattern_str if structure_found else None
        })
        
        # Record in recent inputs
        self.recent_inputs.append({
            "type": "grammar_analysis",
            "sentence": sentence,
            "pattern": pattern_str,
            "timestamp": datetime.now()
        })
        
        # Return analysis results
        result = {
            "status": "success",
            "sentence": sentence,
            "token_count": len(tokens),
            "detected_pattern": pattern_str,
            "token_categories": list(zip(tokens, token_categories)),
            "grammatical": structure_found,
            "prediction_quality": prediction_quality,
            "development_level": self.development_level,
            "process_id": process_id
        }
        
        if matched_pattern:
            result["matched_structure"] = {
                "name": matched_pattern["name"],
                "pattern": matched_pattern["pattern"],
                "confidence": matched_pattern["confidence"]
            }
        
        return result
    
    def _learn_grammar_rule(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Learn a new grammatical rule or update an existing one
        
        Args:
            input_data: Input data dictionary including rule info
            process_id: Process identifier
            
        Returns:
            Dict with learning results
        """
        # Check for required fields
        if "rule_type" not in input_data:
            return {
                "status": "error",
                "message": "Missing rule_type for grammar rule learning",
                "process_id": process_id
            }
        
        rule_type = input_data["rule_type"]
        
        # Different handling for different rule types
        if rule_type == "syntactic":
            # Syntactic pattern rule
            if "pattern" not in input_data or "name" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing pattern or name for syntactic rule",
                    "process_id": process_id
                }
            
            pattern = input_data["pattern"]
            name = input_data["name"]
            examples = input_data.get("examples", [])
            
            # Check if this pattern already exists
            for existing in self.grammar_model.syntactic_patterns:
                if existing["pattern"] == pattern:
                    # Update existing pattern
                    existing["name"] = name
                    if examples:
                        existing["examples"] = examples
                    
                    # Increase confidence
                    existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
                    
                    # Record activation in neural state
                    self.neural_state.add_activation("grammar_acquisition", {
                        'operation': 'update_syntactic_rule',
                        'pattern': pattern,
                        'confidence': existing["confidence"]
                    })
                    
                    return {
                        "status": "success",
                        "message": "Updated existing syntactic pattern",
                        "pattern": pattern,
                        "confidence": existing["confidence"],
                        "process_id": process_id
                    }
            
            # Create new pattern
            pattern_obj = {
                "pattern_id": str(uuid.uuid4()),
                "name": name,
                "pattern": pattern,
                "examples": examples,
                "confidence": 0.5  # Initial confidence
            }
            
            # Development level affects initial confidence
            pattern_obj["confidence"] *= max(0.5, self.development_level)
            
            # Add to patterns
            self.grammar_model.syntactic_patterns.append(pattern_obj)
            
            # Record activation in neural state
            self.neural_state.add_activation("grammar_acquisition", {
                'operation': 'learn_syntactic_rule',
                'pattern': pattern,
                'confidence': pattern_obj["confidence"]
            })
            
            return {
                "status": "success",
                "message": "Learned new syntactic pattern",
                "pattern": pattern,
                "confidence": pattern_obj["confidence"],
                "process_id": process_id
            }
            
        elif rule_type == "morphological":
            # Morphological rule
            if "rule_name" not in input_data or "description" not in input_data or "pattern" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing rule_name, description, or pattern for morphological rule",
                    "process_id": process_id
                }
            
            rule_name = input_data["rule_name"]
            description = input_data["description"]
            pattern = input_data["pattern"]
            examples = input_data.get("examples", {})
            
            # Check if this rule already exists
            if rule_name in self.grammar_model.morphological_rules:
                # Update existing rule
                existing = self.grammar_model.morphological_rules[rule_name]
                existing["description"] = description
                existing["pattern"] = pattern
                
                if examples:
                    existing["examples"] = examples
                
                # Increase confidence
                existing["confidence"] = min(1.0, existing.get("confidence", 0.5) + 0.1)
                
                # Record activation in neural state
                self.neural_state.add_activation("grammar_acquisition", {
                    'operation': 'update_morphological_rule',
                    'rule_name': rule_name,
                    'confidence': existing["confidence"]
                })
                
                return {
                    "status": "success",
                    "message": "Updated existing morphological rule",
                    "rule_name": rule_name,
                    "confidence": existing["confidence"],
                    "process_id": process_id
                }
            
            # Create new rule
            rule_obj = {
                "rule_id": str(uuid.uuid4()),
                "description": description,
                "pattern": pattern,
                "examples": examples,
                "confidence": 0.5  # Initial confidence
            }
            
            # Development level affects initial confidence
            rule_obj["confidence"] *= max(0.5, self.development_level)
            
            # Add to rules
            self.grammar_model.morphological_rules[rule_name] = rule_obj
            
            # Record activation in neural state
            self.neural_state.add_activation("grammar_acquisition", {
                'operation': 'learn_morphological_rule',
                'rule_name': rule_name,
                'confidence': rule_obj["confidence"]
            })
            
            return {
                "status": "success",
                "message": "Learned new morphological rule",
                "rule_name": rule_name,
                "confidence": rule_obj["confidence"],
                "process_id": process_id
            }
            
        elif rule_type == "category":
            # Grammatical category
            if "category" not in input_data or "words" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing category or words for category rule",
                    "process_id": process_id
                }
            
            category = input_data["category"]
            words = input_data["words"]
            
            if not isinstance(words, list):
                words = [words]
            
            # Check if this category already exists
            if category in self.grammar_model.grammatical_categories:
                # Update existing category
                existing_words = self.grammar_model.grammatical_categories[category]
                
                # Add new words
                for word in words:
                    if word not in existing_words:
                        existing_words.append(word)
                
                # Record activation in neural state
                self.neural_state.add_activation("grammar_acquisition", {
                    'operation': 'update_category',
                    'category': category,
                    'word_count': len(existing_words)
                })
                
                return {
                    "status": "success",
                    "message": "Updated existing grammatical category",
                    "category": category,
                    "word_count": len(existing_words),
                    "process_id": process_id
                }
            
            # Create new category
            self.grammar_model.grammatical_categories[category] = words
            
            # Record activation in neural state
            self.neural_state.add_activation("grammar_acquisition", {
                'operation': 'learn_category',
                'category': category,
                'word_count': len(words)
            })
            
            return {
                "status": "success",
                "message": "Learned new grammatical category",
                "category": category,
                "word_count": len(words),
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown rule_type: {rule_type}",
                "process_id": process_id
            }
    
    def _check_grammaticality(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Check if a sentence is grammatically correct
        
        Args:
            input_data: Input data dictionary including sentence
            process_id: Process identifier
            
        Returns:
            Dict with grammaticality assessment
        """
        # Check for sentence
        if "sentence" not in input_data:
            return {
                "status": "error",
                "message": "Missing sentence for grammaticality check",
                "process_id": process_id
            }
        
        sentence = input_data["sentence"]
        
        # Check development level
        if self.development_level < 0.3:
            return {
                "status": "undeveloped",
                "message": "Grammaticality judgment requires higher development level (0.3+)",
                "current_level": self.development_level,
                "process_id": process_id
            }
        
        # Tokenize sentence
        tokens = sentence.split()
        
        # Convert tokens to features
        if tokens:
            # Create a simple feature vector for the whole sentence
            sentence_vector = np.zeros(128)
            
            for i, token in enumerate(tokens):
                pos = (hash(token) + i) % 120
                sentence_vector[pos] = 1.0
            
            # Convert to tensor
            sentence_tensor = torch.tensor(sentence_vector, dtype=torch.float32).unsqueeze(0)
            sentence_tensor = sentence_tensor.to(self.device)
            
            # Process through network
            with torch.no_grad():
                output = self.network(sentence_tensor, operation="judge")
            
            grammaticality = float(output["grammaticality"].cpu().item())
            certainty = float(output["certainty"].cpu().item())
        else:
            # Empty sentence
            grammaticality = 0.0
            certainty = 1.0
        
        # Record activation in neural state
        self.neural_state.add_activation("grammar_acquisition", {
            'operation': 'check_grammaticality',
            'grammaticality': grammaticality,
            'certainty': certainty
        })
        
        # Record in recent inputs
        self.recent_inputs.append({
            "type": "grammaticality_check",
            "sentence": sentence,
            "grammaticality": grammaticality,
            "timestamp": datetime.now()
        })
        
        # Return grammaticality assessment
        return {
            "status": "success",
            "sentence": sentence,
            "grammatical": grammaticality > 0.5,
            "grammaticality_score": grammaticality,
            "certainty": certainty,
            "development_level": self.development_level,
            "process_id": process_id
        }
    
    def _query_grammar(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query the grammar knowledge
        
        Args:
            input_data: Input data dictionary including query parameters
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return summary of grammar knowledge
            return {
                "status": "success",
                "syntactic_patterns": len(self.grammar_model.syntactic_patterns),
                "morphological_rules": len(self.grammar_model.morphological_rules),
                "grammatical_categories": list(self.grammar_model.grammatical_categories.keys()),
                "development_level": self.development_level,
                "process_id": process_id
            }
        
        elif query_type == "syntactic":
            # Return syntactic patterns
            patterns = []
            for pattern in self.grammar_model.syntactic_patterns:
                patterns.append({
                    "name": pattern["name"],
                    "pattern": pattern["pattern"],
                    "confidence": pattern["confidence"]
                })
            
            return {
                "status": "success",
                "syntactic_patterns": patterns,
                "count": len(patterns),
                "process_id": process_id
            }
        
        elif query_type == "morphological":
            # Return morphological rules
            rules = {}
            for rule_name, rule in self.grammar_model.morphological_rules.items():
                rules[rule_name] = {
                    "description": rule["description"],
                    "pattern": rule["pattern"],
                    "confidence": rule["confidence"]
                }
            
            return {
                "status": "success",
                "morphological_rules": rules,
                "count": len(rules),
                "process_id": process_id
            }
        
        elif query_type == "category":
            # Check for category
            category = input_data.get("category")
            
            if category and category in self.grammar_model.grammatical_categories:
                # Return specific category
                return {
                    "status": "success",
                    "category": category,
                    "words": self.grammar_model.grammatical_categories[category],
                    "word_count": len(self.grammar_model.grammatical_categories[category]),
                    "process_id": process_id
                }
            else:
                # Return all categories
                categories = {}
                for cat, words in self.grammar_model.grammatical_categories.items():
                    categories[cat] = len(words)
                
                return {
                    "status": "success",
                    "categories": categories,
                    "count": len(categories),
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
        self.neural_state.grammar_acquisition_development = self.development_level
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
                            "module": "grammar_acquisition",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Update grammar structures for new development level
                self._initialize_grammar_structures()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the grammar acquisition module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "syntactic_patterns": len(self.grammar_model.syntactic_patterns),
            "morphological_rules": len(self.grammar_model.morphological_rules),
            "grammatical_categories": len(self.grammar_model.grammatical_categories)
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "grammar_model": self.grammar_model.dict(),
            "developmental_level": self.development_level,
            "neural_state": {
                "development": self.neural_state.grammar_acquisition_development,
                "accuracy": self.neural_state.grammar_acquisition_accuracy
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
        
        # Load grammar model
        if "grammar_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.grammar_model = parse_obj_as(GrammarModel, state["grammar_model"])
            except Exception as e:
                print(f"Error loading grammar model: {e}")
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            self.neural_state.grammar_acquisition_development = ns.get("development", self.development_level)
            self.neural_state.grammar_acquisition_accuracy = ns.get("accuracy", 0.5) 
