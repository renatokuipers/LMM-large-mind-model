# TODO: Implement the ExpressionGenerator class to produce language output
# This component should be able to:
# - Generate coherent linguistic expressions from concepts
# - Apply grammatical rules to structure output
# - Select appropriate vocabulary for the intended meaning
# - Adapt expression style to different contexts and purposes

# TODO: Implement developmental progression in language production:
# - Simple sounds and single words in early stages
# - Basic grammatical combinations in early childhood
# - Complex sentences in later childhood
# - Sophisticated and context-appropriate expression in adulthood

# TODO: Create mechanisms for:
# - Conceptual encoding: Translate concepts to linguistic form
# - Grammatical structuring: Apply syntactic rules to output
# - Lexical selection: Choose appropriate words for meanings
# - Pragmatic adjustment: Adapt expression to social context

# TODO: Implement different expression types:
# - Declarative statements: Convey information
# - Questions: Request information
# - Imperatives: Direct or request actions
# - Expressives: Convey emotions and attitudes

# TODO: Connect to semantic processing and social understanding
# Expression should build on semantic representations
# and be shaped by social context understanding

from typing import Dict, List, Any, Optional, Set, Tuple
import torch
import uuid
import numpy as np
from datetime import datetime
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.modules.language.models import ExpressionModel, LanguageNeuralState
from lmm_project.modules.language.neural_net import ExpressionNetwork, get_device
from lmm_project.utils.llm_client import LLMClient

class ExpressionGenerator(BaseModule):
    """
    Generates language expressions from meaning
    
    This module is responsible for producing language output,
    translating intentions and meanings into words and sentences.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic vocalization",
        0.2: "Single word utterances",
        0.4: "Two-word combinations",
        0.6: "Simple sentences",
        0.8: "Complex sentences",
        1.0: "Full expressive language"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the expression generator module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level (0.0 to 1.0)
        """
        super().__init__(module_id, event_bus)
        
        # Initialize expression model
        self.expression_model = ExpressionModel()
        
        # Set initial development level
        self.development_level = max(0.0, min(1.0, development_level))
        
        # Initialize neural network
        self.device = get_device()
        self.network = ExpressionNetwork().to(self.device)
        self.network.set_development_level(self.development_level)
        
        # Initialize neural state
        self.neural_state = LanguageNeuralState()
        self.neural_state.expression_generation_development = self.development_level
        
        # Initialize with expression templates based on development level
        self._initialize_expression_templates()
        
        # Recent outputs queue (for tracking recent expressions)
        self.recent_outputs = deque(maxlen=100)
        
        # For embedding generation when needed
        self.llm_client = LLMClient()
    
    def _initialize_expression_templates(self):
        """Initialize expression templates based on development level"""
        # Reset existing templates
        self.expression_model.expression_templates = []
        
        # Basic expression templates at earliest stages
        if self.development_level >= 0.0:
            # Single word expressions for basic needs
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Basic request",
                "template": "{object}",
                "examples": ["milk", "mama", "up"],
                "complexity": 0.1,
                "confidence": 0.8 * max(0.2, self.development_level)
            })
            
            # Initialize communication intents
            self.expression_model.communication_intents["request"] = {
                "core_words": ["want", "give", "more"],
                "templates": ["Basic request"]
            }
            
            # Initialize fluency metrics
            self.expression_model.fluency_metrics["word_clarity"] = 0.3 * max(0.3, self.development_level)
            self.expression_model.fluency_metrics["response_time"] = 0.2 * max(0.3, self.development_level)
        
        if self.development_level >= 0.2:
            # Two-word combinations
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Agent-action",
                "template": "{agent} {action}",
                "examples": ["mama come", "baby eat", "dog run"],
                "complexity": 0.3,
                "confidence": 0.7 * ((self.development_level - 0.2) / 0.8)
            })
            
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Action-object",
                "template": "{action} {object}",
                "examples": ["want milk", "see dog", "throw ball"],
                "complexity": 0.3,
                "confidence": 0.7 * ((self.development_level - 0.2) / 0.8)
            })
            
            # Update communication intents
            self.expression_model.communication_intents["describe"] = {
                "core_words": ["see", "look", "big", "small"],
                "templates": ["Agent-action"]
            }
            
            # Update fluency metrics
            self.expression_model.fluency_metrics["word_connections"] = 0.3 * ((self.development_level - 0.2) / 0.8)
        
        if self.development_level >= 0.4:
            # Simple sentences
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Simple sentence",
                "template": "{agent} {action} {object}",
                "examples": ["I want milk", "Dog chase ball", "Baby eat food"],
                "complexity": 0.5,
                "confidence": 0.7 * ((self.development_level - 0.4) / 0.6)
            })
            
            # Add more communication intents
            self.expression_model.communication_intents["inform"] = {
                "core_words": ["is", "has", "can", "will"],
                "templates": ["Simple sentence"]
            }
            
            # Add simple pragmatic rules
            self.expression_model.pragmatic_rules.append({
                "rule_id": str(uuid.uuid4()),
                "name": "Polite request",
                "condition": "request + formal",
                "adjustment": "Add 'please'",
                "examples": ["Want milk -> Want milk please"],
                "confidence": 0.6 * ((self.development_level - 0.4) / 0.6)
            })
            
            # Update fluency metrics
            self.expression_model.fluency_metrics["sentence_formation"] = 0.4 * ((self.development_level - 0.4) / 0.6)
        
        if self.development_level >= 0.6:
            # Complex sentences
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Complex sentence",
                "template": "{agent} {action} {object} {modifier}",
                "examples": ["I want milk now", "Dog runs fast outside", "Baby sleeps quietly in bed"],
                "complexity": 0.7,
                "confidence": 0.7 * ((self.development_level - 0.6) / 0.4)
            })
            
            # Add compound sentences
            self.expression_model.expression_templates.append({
                "template_id": str(uuid.uuid4()),
                "name": "Compound sentence",
                "template": "{clause1} and {clause2}",
                "examples": ["I am hungry and I want food", "Dog runs and cat jumps"],
                "complexity": 0.8,
                "confidence": 0.6 * ((self.development_level - 0.6) / 0.4)
            })
            
            # Add more communication intents
            self.expression_model.communication_intents["explain"] = {
                "core_words": ["because", "so", "when", "if"],
                "templates": ["Complex sentence", "Compound sentence"]
            }
            
            # Add more speech acts
            self.expression_model.speech_acts["request"] = [
                {
                    "form": "Can you {action} {object}?",
                    "examples": ["Can you get milk?", "Can you help me?"],
                    "politeness": 0.7
                },
                {
                    "form": "I would like {object}, please.",
                    "examples": ["I would like water, please."],
                    "politeness": 0.9
                }
            ]
            
            # Update fluency metrics
            self.expression_model.fluency_metrics["grammatical_accuracy"] = 0.5 * ((self.development_level - 0.6) / 0.4)
            self.expression_model.fluency_metrics["vocabulary_diversity"] = 0.5 * ((self.development_level - 0.6) / 0.4)
        
        # Generate template embeddings if development level is sufficient
        if self.development_level >= 0.5:
            self._generate_template_embeddings()
    
    def _generate_template_embeddings(self):
        """
        Generate embeddings for expression templates using the LLM API
        
        This allows more sophisticated template selection based on semantic similarity
        to the intended expression rather than just template names.
        """
        # Only process templates that don't already have embeddings
        templates_to_embed = [
            template for template in self.expression_model.expression_templates
            if "embedding" not in template
        ]
        
        if not templates_to_embed:
            return
            
        for template in templates_to_embed:
            # Create a representative text for this template
            # Use examples and template structure
            
            # Start with the template name
            embedding_text = f"Template: {template['name']}. "
            
            # Add template structure
            embedding_text += f"Structure: {template['template']}. "
            
            # Add examples if available
            if "examples" in template and template["examples"]:
                embedding_text += f"Examples: {', '.join(template['examples'])}. "
            
            # Try to get embedding
            try:
                raw_embedding = self.llm_client.get_embedding(
                    embedding_text,
                    embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
                )
                
                # Process the embedding
                if isinstance(raw_embedding, list):
                    if isinstance(raw_embedding[0], list):
                        template["embedding"] = raw_embedding[0]
                    else:
                        template["embedding"] = raw_embedding
                    
                    print(f"Generated embedding for template '{template['name']}'")
                
            except Exception as e:
                print(f"Warning: Failed to generate embedding for template '{template['name']}': {e}")
                
                # Try fallback model
                try:
                    raw_embedding = self.llm_client.get_embedding(
                        embedding_text,
                        embedding_model="text-embedding-ada-002"
                    )
                    
                    # Process the embedding
                    if isinstance(raw_embedding, list):
                        if isinstance(raw_embedding[0], list):
                            template["embedding"] = raw_embedding[0]
                        else:
                            template["embedding"] = raw_embedding
                        
                        print(f"Generated embedding for template '{template['name']}' using fallback model")
                    
                except Exception as fallback_error:
                    print(f"ERROR: Fallback embedding also failed for template '{template['name']}': {fallback_error}")
    
    def _find_template_by_similarity(self, intent_text, max_complexity=1.0):
        """
        Find the most suitable template based on semantic similarity to intent
        
        Args:
            intent_text: Text describing the intention
            max_complexity: Maximum template complexity to consider
            
        Returns:
            The most semantically similar template within complexity constraints
        """
        # Check if we have embeddings for templates
        templates_with_embeddings = [
            template for template in self.expression_model.expression_templates
            if "embedding" in template and template["complexity"] <= max_complexity
        ]
        
        if not templates_with_embeddings:
            # Fall back to traditional selection if no embeddings available
            return None
            
        try:
            # Get embedding for intent text
            intent_embedding = self.llm_client.get_embedding(
                intent_text,
                embedding_model="text-embedding-nomic-embed-text-v1.5@q4_k_m"
            )
            
            if not isinstance(intent_embedding, list):
                return None
                
            # Flatten if nested
            if isinstance(intent_embedding[0], list):
                intent_embedding = intent_embedding[0]
                
            # Calculate similarity to each template
            best_similarity = -1
            best_template = None
            
            for template in templates_with_embeddings:
                template_embedding = template["embedding"]
                
                # Ensure both embeddings have the same length for comparison
                min_length = min(len(intent_embedding), len(template_embedding))
                
                # Calculate cosine similarity
                dot_product = sum(a * b for a, b in zip(intent_embedding[:min_length], template_embedding[:min_length]))
                intent_magnitude = sum(a * a for a in intent_embedding[:min_length]) ** 0.5
                template_magnitude = sum(b * b for b in template_embedding[:min_length]) ** 0.5
                
                if intent_magnitude > 0 and template_magnitude > 0:
                    similarity = dot_product / (intent_magnitude * template_magnitude)
                    
                    # Apply a complexity bonus (prefer more complex templates when similar)
                    complexity_bonus = template["complexity"] * 0.1
                    adjusted_similarity = similarity + complexity_bonus
                    
                    if adjusted_similarity > best_similarity:
                        best_similarity = adjusted_similarity
                        best_template = template
            
            return best_template
            
        except Exception as e:
            print(f"Warning: Error finding template by similarity: {e}")
            return None
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to the expression generator module
        
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
        operation = input_data.get("operation", "generate")
        
        # Dispatch to appropriate handler
        if operation == "generate":
            return self._generate_expression(input_data, process_id)
        elif operation == "evaluate":
            return self._evaluate_expression(input_data, process_id)
        elif operation == "learn_template":
            return self._learn_expression_template(input_data, process_id)
        elif operation == "query_expressions":
            return self._query_expressions(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _generate_expression(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Generate a language expression based on intent
        
        Args:
            input_data: Input data dictionary including intent information
            process_id: Process identifier
            
        Returns:
            Dict with generated expression
        """
        # Check for intent
        if "intent" not in input_data:
            return {
                "status": "error",
                "message": "Missing intent for expression generation",
                "process_id": process_id
            }
        
        intent = input_data["intent"]
        context = input_data.get("context", {})
        
        # Development level constrains complexity
        max_complexity = min(1.0, 0.2 + (self.development_level * 0.8))
        chosen_template = None
        
        # Use semantic template selection if development level is high enough
        if self.development_level >= 0.5:
            # Create a more descriptive intent text for semantic matching
            intent_description = f"Intent: {intent}. "
            
            # Add context information if available
            if context:
                context_desc = ", ".join([f"{k}: {v}" for k, v in context.items() 
                                        if isinstance(v, str) and len(v) < 100])
                if context_desc:
                    intent_description += f"Context: {context_desc}."
            
            # Try to find template by semantic similarity
            semantic_template = self._find_template_by_similarity(intent_description, max_complexity)
            if semantic_template:
                chosen_template = semantic_template
        
        # If semantic selection failed or development level is too low, use traditional selection
        if not chosen_template:
            suitable_templates = []
            
            # Check if we have specific templates for this intent
            if intent in self.expression_model.communication_intents:
                intent_info = self.expression_model.communication_intents[intent]
                template_names = intent_info.get("templates", [])
                
                # Find the template objects
                for template in self.expression_model.expression_templates:
                    if template["name"] in template_names and template["complexity"] <= max_complexity:
                        suitable_templates.append(template)
            
            # If no suitable templates found, use templates within complexity constraints
            if not suitable_templates:
                for template in self.expression_model.expression_templates:
                    if template["complexity"] <= max_complexity:
                        suitable_templates.append(template)
            
            # If still no templates, return error
            if not suitable_templates:
                return {
                    "status": "undeveloped",
                    "message": "No suitable expression templates available at current development level",
                    "development_level": self.development_level,
                    "process_id": process_id
                }
            
            # Sort by complexity (descending) to use the most complex templates possible
            suitable_templates.sort(key=lambda t: t["complexity"], reverse=True)
            
            # Choose the best template (most complex that we can handle)
            chosen_template = suitable_templates[0]
        
        # Generate expression by filling template
        expression = chosen_template["template"]
        
        # Find placeholders in the template
        placeholders = []
        current = ""
        in_placeholder = False
        
        for char in expression:
            if char == '{':
                in_placeholder = True
                current = ""
            elif char == '}' and in_placeholder:
                in_placeholder = False
                placeholders.append(current)
            elif in_placeholder:
                current += char
        
        # Fill placeholders from context
        filled_expression = expression
        missing_placeholders = []
        
        for placeholder in placeholders:
            if placeholder in context:
                filled_expression = filled_expression.replace(f"{{{placeholder}}}", context[placeholder])
            else:
                missing_placeholders.append(placeholder)
        
        # Check for missing placeholders
        if missing_placeholders:
            return {
                "status": "error",
                "message": f"Missing context for placeholders: {', '.join(missing_placeholders)}",
                "template": chosen_template["name"],
                "required_context": missing_placeholders,
                "process_id": process_id
            }
        
        # Apply appropriate speech act modifications if available
        if self.development_level >= 0.6 and intent in self.expression_model.speech_acts:
            speech_acts = self.expression_model.speech_acts[intent]
            
            # Use a simple speech act modification if available
            if speech_acts and "politeness" in context:
                politeness = float(context.get("politeness", 0.5))
                
                # Find the speech act with closest politeness level
                closest_act = min(speech_acts, key=lambda act: abs(act.get("politeness", 0.5) - politeness))
                
                # Apply the speech act form if it doesn't reduce complexity too much
                if closest_act["politeness"] <= max_complexity + 0.1:
                    # This is a simple approximation - in a real system, would properly
                    # restructure the sentence according to the speech act
                    if "{action}" in closest_act["form"] and "action" in context:
                        if "{object}" in closest_act["form"] and "object" in context:
                            filled_expression = closest_act["form"].replace("{action}", context["action"]).replace("{object}", context["object"])
        
        # Generate expression features for neural processing
        intent_features = np.zeros(128)
        
        # Simple feature creation based on intent and context
        intent_hash = hash(intent) % 50
        intent_features[intent_hash] = 1.0
        
        # Add context features
        for i, (key, value) in enumerate(context.items()[:10]):  # Limit to first 10 context items
            key_hash = (hash(key) + i) % 20
            intent_features[50 + key_hash] = 1.0
            
            value_hash = (hash(str(value)) + i) % 20
            intent_features[70 + value_hash] = 1.0
        
        # Convert to tensor
        intent_tensor = torch.tensor(intent_features, dtype=torch.float32).unsqueeze(0)
        intent_tensor = intent_tensor.to(self.device)
        
        # Process through network
        with torch.no_grad():
            # Generate expression plan
            output = self.network(
                intent_tensor, 
                operation="plan",
                sequence_length=min(10, len(filled_expression.split()))
            )
            
            # Get plan quality
            plan_quality = output["plan_quality"].cpu().item()
            
            # Apply development factor to quality
            fluency = plan_quality * max(0.3, self.development_level)
        
        # Apply developmental constraints to expression
        if self.development_level < 0.2:
            # Single word stage - limit to first word
            words = filled_expression.split()
            if words:
                filled_expression = words[0]
                
        elif self.development_level < 0.4:
            # Two-word stage - limit to first two words
            words = filled_expression.split()
            if len(words) > 2:
                filled_expression = " ".join(words[:2])
        
        # Record in recent outputs
        self.recent_outputs.append({
            "type": "expression_generation",
            "intent": intent,
            "expression": filled_expression,
            "template": chosen_template["name"],
            "fluency": fluency,
            "timestamp": datetime.now()
        })
        
        # Record activation in neural state
        self.neural_state.add_activation("expression_generation", {
            'operation': 'generate',
            'intent': intent,
            'template': chosen_template["name"],
            'fluency': fluency
        })
        
        # Return generated expression
        return {
            "status": "success",
            "expression": filled_expression,
            "intent": intent,
            "template_used": chosen_template["name"],
            "fluency": fluency,
            "development_level": self.development_level,
            "process_id": process_id
        }
    
    def _evaluate_expression(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Evaluate the quality of a language expression
        
        Args:
            input_data: Input data dictionary including expression to evaluate
            process_id: Process identifier
            
        Returns:
            Dict with evaluation results
        """
        # Check for expression
        if "expression" not in input_data:
            return {
                "status": "error",
                "message": "Missing expression for evaluation",
                "process_id": process_id
            }
        
        expression = input_data["expression"]
        intent = input_data.get("intent")
        
        # Create expression features
        expression_words = expression.split()
        
        if not expression_words:
            return {
                "status": "error",
                "message": "Empty expression provided",
                "process_id": process_id
            }
        
        # Create a context tensor from the expression
        expression_tensor = torch.zeros((len(expression_words), 128), dtype=torch.float32)
        
        for i, word in enumerate(expression_words):
            # Set word features
            word_hash = hash(word) % 100
            expression_tensor[i, word_hash] = 1.0
            
            # Add positional information
            expression_tensor[i, 100 + min(10, i)] = 1.0
            
            # Add intent information if provided
            if intent:
                intent_hash = hash(intent) % 10
                expression_tensor[i, 110 + intent_hash] = 1.0
        
        expression_tensor = expression_tensor.to(self.device)
        
        # Create a simple intent tensor if intent provided
        if intent:
            intent_features = np.zeros(128)
            intent_hash = hash(intent) % 50
            intent_features[intent_hash] = 1.0
            
            intent_tensor = torch.tensor(intent_features, dtype=torch.float32).unsqueeze(0)
            intent_tensor = intent_tensor.to(self.device)
        else:
            # Use simple placeholder
            intent_tensor = torch.zeros((1, 128), dtype=torch.float32).to(self.device)
        
        # Process through network
        with torch.no_grad():
            output = self.network(
                intent_tensor,
                operation="evaluate",
                context=expression_tensor
            )
            
            # Get fluency and confidence
            fluency = output["fluency"].cpu().item()
            confidence = output["evaluation_confidence"].cpu().item()
        
        # Determine appropriate fluency metrics based on expression complexity
        metrics = {}
        
        # Word-level fluency
        metrics["word_clarity"] = self.expression_model.fluency_metrics.get("word_clarity", 0.3) * fluency
        
        # Add higher-level metrics based on development
        if len(expression_words) > 1 and self.development_level >= 0.2:
            metrics["word_connections"] = self.expression_model.fluency_metrics.get("word_connections", 0.3) * fluency
            
        if len(expression_words) > 2 and self.development_level >= 0.4:
            metrics["sentence_formation"] = self.expression_model.fluency_metrics.get("sentence_formation", 0.3) * fluency
            
        if len(expression_words) > 3 and self.development_level >= 0.6:
            metrics["grammatical_accuracy"] = self.expression_model.fluency_metrics.get("grammatical_accuracy", 0.3) * fluency
            metrics["vocabulary_diversity"] = self.expression_model.fluency_metrics.get("vocabulary_diversity", 0.3) * fluency
        
        # Overall fluency (weighted average of metrics)
        overall_fluency = sum(metrics.values()) / max(1, len(metrics))
        
        # Record activation in neural state
        self.neural_state.add_activation("expression_generation", {
            'operation': 'evaluate',
            'expression_length': len(expression_words),
            'fluency': overall_fluency
        })
        
        # Return evaluation results
        return {
            "status": "success",
            "expression": expression,
            "fluency": overall_fluency,
            "evaluation_confidence": confidence,
            "metrics": metrics,
            "development_level": self.development_level,
            "process_id": process_id
        }
    
    def _learn_expression_template(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Learn a new expression template
        
        Args:
            input_data: Input data dictionary including template information
            process_id: Process identifier
            
        Returns:
            Dict with learning results
        """
        # Check for required fields
        if "name" not in input_data or "template" not in input_data:
            return {
                "status": "error",
                "message": "Missing name or template for learning expression template",
                "process_id": process_id
            }
        
        name = input_data["name"]
        template = input_data["template"]
        examples = input_data.get("examples", [])
        complexity = input_data.get("complexity", 0.5)
        intent = input_data.get("intent")
        
        # Check for existing template with same name
        for existing in self.expression_model.expression_templates:
            if existing["name"] == name:
                # Update existing template
                existing["template"] = template
                if examples:
                    existing["examples"] = examples
                existing["complexity"] = complexity
                existing["confidence"] = min(1.0, existing["confidence"] + 0.1)
                
                # Record activation in neural state
                self.neural_state.add_activation("expression_generation", {
                    'operation': 'update_template',
                    'template_name': name,
                    'complexity': complexity
                })
                
                # Update intent mapping if provided
                if intent and intent in self.expression_model.communication_intents:
                    if name not in self.expression_model.communication_intents[intent]["templates"]:
                        self.expression_model.communication_intents[intent]["templates"].append(name)
                
                return {
                    "status": "success",
                    "message": "Updated existing expression template",
                    "template_name": name,
                    "process_id": process_id
                }
        
        # Create new template
        new_template = {
            "template_id": str(uuid.uuid4()),
            "name": name,
            "template": template,
            "examples": examples,
            "complexity": complexity,
            "confidence": 0.5  # Initial confidence
        }
        
        # Development level affects initial confidence
        new_template["confidence"] *= max(0.5, self.development_level)
        
        # Add to templates
        self.expression_model.expression_templates.append(new_template)
        
        # Add to intent mapping if provided
        if intent:
            if intent in self.expression_model.communication_intents:
                if name not in self.expression_model.communication_intents[intent]["templates"]:
                    self.expression_model.communication_intents[intent]["templates"].append(name)
            else:
                self.expression_model.communication_intents[intent] = {
                    "core_words": [],
                    "templates": [name]
                }
        
        # Record activation in neural state
        self.neural_state.add_activation("expression_generation", {
            'operation': 'learn_template',
            'template_name': name,
            'complexity': complexity
        })
        
        return {
            "status": "success",
            "message": "Learned new expression template",
            "template_name": name,
            "process_id": process_id
        }
    
    def _query_expressions(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """
        Query expression information
        
        Args:
            input_data: Input data dictionary including query parameters
            process_id: Process identifier
            
        Returns:
            Dict with query results
        """
        # Get query type
        query_type = input_data.get("query_type", "all")
        
        if query_type == "all":
            # Return summary of expression capabilities
            return {
                "status": "success",
                "templates_count": len(self.expression_model.expression_templates),
                "intents": list(self.expression_model.communication_intents.keys()),
                "fluency_metrics": self.expression_model.fluency_metrics,
                "development_level": self.development_level,
                "process_id": process_id
            }
        
        elif query_type == "templates":
            # Return expression templates
            # Sort by complexity for easier review
            sorted_templates = sorted(self.expression_model.expression_templates, key=lambda t: t["complexity"])
            
            templates = []
            for template in sorted_templates:
                templates.append({
                    "name": template["name"],
                    "template": template["template"],
                    "complexity": template["complexity"],
                    "confidence": template["confidence"]
                })
                
            return {
                "status": "success",
                "templates": templates,
                "count": len(templates),
                "process_id": process_id
            }
        
        elif query_type == "intent":
            # Check for intent
            if "intent" not in input_data:
                return {
                    "status": "error",
                    "message": "Missing intent for intent query",
                    "process_id": process_id
                }
                
            intent = input_data["intent"]
            
            # Check if intent exists
            if intent not in self.expression_model.communication_intents:
                return {
                    "status": "error",
                    "message": f"Intent not found: {intent}",
                    "available_intents": list(self.expression_model.communication_intents.keys()),
                    "process_id": process_id
                }
                
            # Get intent information
            intent_info = self.expression_model.communication_intents[intent]
            
            # Get related templates
            template_details = []
            for template_name in intent_info["templates"]:
                for template in self.expression_model.expression_templates:
                    if template["name"] == template_name:
                        template_details.append({
                            "name": template["name"],
                            "template": template["template"],
                            "complexity": template["complexity"]
                        })
                        break
                        
            return {
                "status": "success",
                "intent": intent,
                "core_words": intent_info["core_words"],
                "templates": template_details,
                "process_id": process_id
            }
            
        elif query_type == "speech_acts":
            # Return speech acts
            speech_acts = {}
            for intent, acts in self.expression_model.speech_acts.items():
                speech_acts[intent] = []
                for act in acts:
                    speech_acts[intent].append({
                        "form": act["form"],
                        "politeness": act.get("politeness", 0.5)
                    })
                    
            return {
                "status": "success",
                "speech_acts": speech_acts,
                "count": sum(len(acts) for acts in self.expression_model.speech_acts.values()),
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
        self.neural_state.expression_generation_development = self.development_level
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
                            "module": "expression_generation",
                            "milestone": milestone,
                            "level": level
                        }
                    })
                
                # Update expression templates for new development level
                self._initialize_expression_templates()
        
        return self.development_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the expression generator module
        
        Returns:
            Dict representing the current state
        """
        return {
            "module_id": self.module_id,
            "developmental_level": self.development_level,
            "template_count": len(self.expression_model.expression_templates),
            "intent_count": len(self.expression_model.communication_intents),
            "speech_act_count": sum(len(acts) for acts in self.expression_model.speech_acts.values())
        }
    
    def save_state(self) -> Dict[str, Any]:
        """
        Save the current state for persistence
        
        Returns:
            Dict with serializable state
        """
        return {
            "module_id": self.module_id,
            "expression_model": self.expression_model.dict(),
            "developmental_level": self.development_level,
            "neural_state": {
                "development": self.neural_state.expression_generation_development,
                "accuracy": self.neural_state.expression_generation_accuracy
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
        
        # Load expression model
        if "expression_model" in state:
            try:
                # Create new model from dict
                from pydantic import parse_obj_as
                self.expression_model = parse_obj_as(ExpressionModel, state["expression_model"])
            except Exception as e:
                print(f"Error loading expression model: {e}")
        
        # Load neural state
        if "neural_state" in state:
            ns = state["neural_state"]
            self.neural_state.expression_generation_development = ns.get("development", self.development_level)
            self.neural_state.expression_generation_accuracy = ns.get("accuracy", 0.5) 
