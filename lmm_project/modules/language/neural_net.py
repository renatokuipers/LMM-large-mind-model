import torch 
import torch.nn as nn 
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple, List, Union

def get_device() -> torch.device:
    """
    Get the appropriate device for tensor operations.
    
    Returns:
        torch.device: CUDA device if available, otherwise CPU
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class PhonemeNetwork(nn.Module):
    """
    Neural network for phoneme recognition and processing
    
    This network processes audio input to recognize phonemes and
    learns phonological patterns in the language.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_phonemes: int = 50):
        """
        Initialize the phoneme recognition network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_phonemes: Number of phonemes to recognize
        """
        super().__init__()
        
        # Main layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Phoneme recognition head
        self.phoneme_classifier = nn.Linear(hidden_dim, num_phonemes)
        
        # Phoneme embedding
        self.phoneme_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU()
        )
        
        # Developmental factor (grows with learning)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_data: torch.Tensor, operation: str = "recognize") -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            input_data: Input tensor [batch_size, input_dim]
            operation: Type of operation to perform
                "recognize": Recognize phonemes in input
                "analyze": Extract phonological features
                "embed": Generate phoneme embeddings
            
        Returns:
            Dictionary of output tensors
        """
        # Get base encoding
        x = self.encoder(input_data)
        
        # Apply developmental scaling to more complex operations
        dev_factor = torch.sigmoid(self.developmental_factor * 10)
        
        # Perform operation
        if operation == "recognize":
            # Phoneme recognition
            logits = self.phoneme_classifier(x)
            probabilities = F.softmax(logits, dim=-1)
            
            # Confidence increases with development
            confidence = torch.sigmoid(torch.max(logits, dim=1)[0]) * dev_factor
            
            return {
                "phoneme_logits": logits,
                "phoneme_probs": probabilities,
                "confidence": confidence
            }
            
        elif operation == "analyze":
            # Extract phonological features
            features = self.feature_extractor(x)
            
            # Feature clarity improves with development
            feature_clarity = features * dev_factor
            
            return {
                "phoneme_features": feature_clarity,
                "encoding": x
            }
            
        elif operation == "embed":
            # Generate phoneme embeddings
            embeddings = self.phoneme_embedding(x)
            
            return {
                "phoneme_embedding": embeddings,
                "encoding": x
            }
            
        else:
            # Default to basic encoding
            return {
                "encoding": x
            }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class WordNetwork(nn.Module):
    """
    Neural network for word learning and lexical processing
    
    This network learns words and their associations.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, vocab_size: int = 1000):
        """
        Initialize the word learning network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            vocab_size: Maximum vocabulary size
        """
        super().__init__()
        
        # Main layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Word recognition head
        self.word_classifier = nn.Linear(hidden_dim, vocab_size)
        
        # Word embedding
        self.word_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Word generation (from meaning to word)
        self.word_generator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size)
        )
        
        # Association network
        self.association_network = nn.Bilinear(output_dim, output_dim, 1)
        
        # Developmental factor (grows with learning)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_data: torch.Tensor, operation: str = "recognize", 
               context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            input_data: Input tensor [batch_size, input_dim]
            operation: Type of operation to perform
                "recognize": Recognize words in input
                "embed": Generate word embeddings
                "associate": Find associations between words
                "generate": Generate word from meaning
            context: Optional context tensor for operations that need it
            
        Returns:
            Dictionary of output tensors
        """
        # Get base encoding
        x = self.encoder(input_data)
        
        # Apply developmental scaling
        dev_factor = torch.sigmoid(self.developmental_factor * 10)
        
        # Perform operation
        if operation == "recognize":
            # Word recognition
            logits = self.word_classifier(x)
            probabilities = F.softmax(logits, dim=-1)
            
            # Confidence increases with development
            confidence = torch.sigmoid(torch.max(logits, dim=1)[0]) * dev_factor
            
            return {
                "word_logits": logits,
                "word_probs": probabilities,
                "confidence": confidence
            }
            
        elif operation == "embed":
            # Generate word embeddings
            embeddings = self.word_embedding(x)
            
            # Semantic richness increases with development
            embeddings = embeddings * (0.5 + 0.5 * dev_factor)
            
            return {
                "word_embedding": embeddings,
                "encoding": x
            }
            
        elif operation == "associate" and context is not None:
            # Generate embeddings for input and context
            input_embedding = self.word_embedding(x)
            context_encoded = self.encoder(context)
            context_embedding = self.word_embedding(context_encoded)
            
            # Compute association strength
            association = self.association_network(input_embedding, context_embedding)
            
            # Association strength influenced by development
            association = association * dev_factor
            
            return {
                "association_strength": association,
                "input_embedding": input_embedding,
                "context_embedding": context_embedding
            }
            
        elif operation == "generate":
            # Generate word from meaning embedding
            embedding = self.word_embedding(x)
            word_logits = self.word_generator(embedding)
            word_probs = F.softmax(word_logits, dim=-1)
            
            # Generation quality improves with development
            quality = torch.sigmoid(torch.max(word_logits, dim=1)[0]) * dev_factor
            
            return {
                "word_logits": word_logits,
                "word_probs": word_probs,
                "generation_quality": quality
            }
            
        else:
            # Default to basic encoding
            return {
                "encoding": x
            }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class GrammarNetwork(nn.Module):
    """
    Neural network for grammar acquisition and processing
    
    This network learns grammatical structures and rules.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_structures: int = 50):
        """
        Initialize the grammar acquisition network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_structures: Number of grammatical structures to recognize
        """
        super().__init__()
        
        # Main layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Structure recognition head
        self.structure_classifier = nn.Linear(hidden_dim, num_structures)
        
        # Structure embedding
        self.structure_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Grammaticality judgment
        self.grammaticality_judge = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Sequence prediction (for next-word prediction based on grammar)
        self.sequence_predictor = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        self.prediction_head = nn.Linear(hidden_dim, input_dim)
        
        # Developmental factor (grows with learning)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_data: torch.Tensor, operation: str = "recognize", 
               sequence: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            input_data: Input tensor [batch_size, input_dim]
            operation: Type of operation to perform
                "recognize": Recognize grammatical structures
                "judge": Judge grammaticality
                "embed": Generate structure embeddings
                "predict": Predict next element in sequence
            sequence: Optional sequence tensor for sequence operations
            
        Returns:
            Dictionary of output tensors
        """
        # Get base encoding
        x = self.encoder(input_data)
        
        # Apply developmental scaling
        dev_factor = torch.sigmoid(self.developmental_factor * 10)
        
        # Perform operation
        if operation == "recognize":
            # Structure recognition
            logits = self.structure_classifier(x)
            probabilities = F.softmax(logits, dim=-1)
            
            # Confidence increases with development
            confidence = torch.sigmoid(torch.max(logits, dim=1)[0]) * dev_factor
            
            return {
                "structure_logits": logits,
                "structure_probs": probabilities,
                "confidence": confidence
            }
            
        elif operation == "judge":
            # Judge grammaticality
            grammaticality = self.grammaticality_judge(x)
            
            # Judgment accuracy increases with development
            certainty = grammaticality * (0.5 + 0.5 * dev_factor)
            
            return {
                "grammaticality": grammaticality,
                "certainty": certainty
            }
            
        elif operation == "embed":
            # Generate structure embeddings
            embeddings = self.structure_embedding(x)
            
            return {
                "structure_embedding": embeddings,
                "encoding": x
            }
            
        elif operation == "predict" and sequence is not None:
            # Process sequence through GRU
            outputs, hidden = self.sequence_predictor(sequence)
            
            # Predict next element - handle both 2D and 3D outputs
            if outputs.dim() == 3:
                # Batch x Sequence x Features
                prediction = self.prediction_head(outputs[:, -1, :])
            else:
                # Batch x Features (single timestep output)
                prediction = self.prediction_head(outputs)
            
            # Prediction quality improves with development
            quality = torch.norm(prediction, dim=1, keepdim=True) * dev_factor
            
            return {
                "prediction": prediction,
                "quality": quality,
                "hidden": hidden
            }
            
        else:
            # Default to basic encoding
            return {
                "encoding": x
            }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class SemanticNetwork(nn.Module):
    """
    Neural network for semantic processing
    
    This network learns meanings, concepts, and semantic relations.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_concepts: int = 200):
        """
        Initialize the semantic processing network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_concepts: Number of basic concepts to recognize
        """
        super().__init__()
        
        # Main layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Concept recognition
        self.concept_classifier = nn.Linear(hidden_dim, num_concepts)
        
        # Semantic embedding
        self.semantic_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Relation recognition
        self.relation_network = nn.Bilinear(output_dim, output_dim, hidden_dim // 4)
        self.relation_classifier = nn.Linear(hidden_dim // 4, 10)  # 10 basic relation types
        
        # Contextual understanding
        self.context_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Developmental factor (grows with learning)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_data: torch.Tensor, operation: str = "understand", 
               context: Optional[torch.Tensor] = None, 
               second_concept: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            input_data: Input tensor [batch_size, input_dim]
            operation: Type of operation to perform
                "understand": Extract meaning from input
                "embed": Generate semantic embeddings
                "relate": Find relations between concepts
                "contextualize": Understand meaning in context
            context: Optional context tensor for context operations
            second_concept: Optional tensor for relation operations
            
        Returns:
            Dictionary of output tensors
        """
        # Get base encoding
        x = self.encoder(input_data)
        
        # Apply developmental scaling
        dev_factor = torch.sigmoid(self.developmental_factor * 10)
        
        # Perform operation
        if operation == "understand":
            # Concept recognition
            logits = self.concept_classifier(x)
            probabilities = F.softmax(logits, dim=-1)
            
            # Understanding depth increases with development
            depth = torch.max(probabilities, dim=1)[0] * dev_factor
            
            return {
                "concept_logits": logits,
                "concept_probs": probabilities,
                "understanding_depth": depth
            }
            
        elif operation == "embed":
            # Generate semantic embeddings
            embeddings = self.semantic_embedding(x)
            
            # Semantic richness increases with development
            richness = torch.norm(embeddings, dim=1, keepdim=True) * dev_factor
            
            return {
                "semantic_embedding": embeddings,
                "semantic_richness": richness,
                "encoding": x
            }
            
        elif operation == "relate" and second_concept is not None:
            # Generate embeddings for both concepts
            input_embedding = self.semantic_embedding(x)
            
            # Either use encoded second concept or encode it
            if second_concept.size(-1) == output_dim:
                concept2_embedding = second_concept
            else:
                concept2_encoded = self.encoder(second_concept)
                concept2_embedding = self.semantic_embedding(concept2_encoded)
            
            # Compute relation features
            relation_features = self.relation_network(input_embedding, concept2_embedding)
            relation_logits = self.relation_classifier(relation_features)
            relation_probs = F.softmax(relation_logits, dim=-1)
            
            # Relation understanding improves with development
            clarity = torch.max(relation_probs, dim=1)[0] * dev_factor
            
            return {
                "relation_logits": relation_logits,
                "relation_probs": relation_probs,
                "relation_clarity": clarity
            }
            
        elif operation == "contextualize" and context is not None:
            # Process context
            context_encoded = self.encoder(context)
            
            # Combine input and context
            combined = torch.cat([x, context_encoded], dim=1)
            contextualized = self.context_network(combined)
            
            # Contextual understanding improves with development
            context_effect = torch.norm(contextualized - self.semantic_embedding(x), dim=1, keepdim=True) * dev_factor
            
            return {
                "contextualized_embedding": contextualized,
                "context_effect": context_effect
            }
            
        else:
            # Default to basic encoding
            return {
                "encoding": x
            }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class ExpressionNetwork(nn.Module):
    """
    Neural network for language expression generation
    
    This network generates language expressions based on intent.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, vocab_size: int = 1000):
        """
        Initialize the expression generation network
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            vocab_size: Vocabulary size for word generation
        """
        super().__init__()
        
        # Main intent encoder
        self.intent_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Expression planning
        self.expression_planner = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True
        )
        
        # Word generation
        self.word_generator = nn.Linear(hidden_dim, vocab_size)
        
        # Fluency evaluation
        self.fluency_evaluator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Expression embedding
        self.expression_embedding = nn.Linear(hidden_dim, output_dim)
        
        # Developmental factor (grows with learning)
        self.developmental_factor = nn.Parameter(torch.tensor(0.1), requires_grad=False)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, input_data: torch.Tensor, operation: str = "generate", 
               sequence_length: int = 5, 
               context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the network
        
        Args:
            input_data: Input tensor [batch_size, input_dim] (intent representation)
            operation: Type of operation to perform
                "generate": Generate language expression
                "plan": Plan expression structure
                "evaluate": Evaluate expression fluency
            sequence_length: Length of sequence to generate
            context: Optional context for expression generation
            
        Returns:
            Dictionary of output tensors
        """
        # Get intent encoding
        x = self.intent_encoder(input_data)
        
        # Apply developmental scaling
        dev_factor = torch.sigmoid(self.developmental_factor * 10)
        
        # Batch size
        batch_size = x.size(0)
        
        # Perform operation
        if operation == "generate":
            # Initialize generation
            hidden = x.unsqueeze(0).repeat(2, 1, 1)  # Initial hidden state from intent
            current_input = x.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            
            # Generate sequence
            word_logits_sequence = []
            
            for i in range(sequence_length):
                # Generate next step
                output, hidden = self.expression_planner(current_input, hidden)
                
                # Predict word
                word_logits = self.word_generator(output.squeeze(1))
                word_logits_sequence.append(word_logits)
                
                # Set up next input (teacher forcing would use actual words here)
                current_input = output
            
            # Stack logits sequences
            word_logits_sequence = torch.stack(word_logits_sequence, dim=1)
            
            # Generate word probabilities
            word_probs_sequence = F.softmax(word_logits_sequence, dim=-1)
            
            # Fluency increases with development
            fluency = self.fluency_evaluator(hidden[-1]) * dev_factor
            
            return {
                "word_logits_sequence": word_logits_sequence,
                "word_probs_sequence": word_probs_sequence,
                "fluency": fluency,
                "hidden_state": hidden
            }
            
        elif operation == "plan":
            # Plan expression structure without generating specific words
            hidden = x.unsqueeze(0).repeat(2, 1, 1)
            
            # Create sequence input (repeat intent)
            sequence_input = x.unsqueeze(1).repeat(1, sequence_length, 1)
            
            # Generate plan
            plan_sequence, hidden = self.expression_planner(sequence_input, hidden)
            
            # Plan quality improves with development
            plan_quality = self.fluency_evaluator(hidden[-1]) * dev_factor
            
            return {
                "expression_plan": plan_sequence,
                "plan_quality": plan_quality
            }
            
        elif operation == "evaluate" and context is not None:
            # Encode context (existing expression)
            _, hidden = self.expression_planner(context, None)
            
            # Evaluate fluency
            fluency = self.fluency_evaluator(hidden[-1])
            
            # Evaluation accuracy improves with development
            confidence = fluency * dev_factor
            
            return {
                "fluency": fluency,
                "evaluation_confidence": confidence
            }
            
        else:
            # Default to basic encoding
            embedding = self.expression_embedding(x)
            
            return {
                "expression_embedding": embedding,
                "intent_encoding": x
            }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))
