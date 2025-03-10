import torch
import torch.nn as nn
import torch.nn.functional as F
import psutil
import gc
from typing import Dict, List, Any, Optional, Tuple, Union

from lmm_project.utils.llm_client import LLMClient

def get_device() -> torch.device:
    """
    Get the appropriate device (GPU if available, otherwise CPU) with memory management
    
    Returns:
        torch.device: The device to use for tensor operations
    """
    if torch.cuda.is_available():
        # Check available GPU memory before deciding
        try:
            free_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
            # Require at least 1GB of free memory
            if free_memory > 1024 * 1024 * 1024:
                return torch.device("cuda")
            else:
                print(f"Warning: CUDA available but low memory ({free_memory / (1024**3):.2f} GB free). Using CPU.")
                return torch.device("cpu")
        except Exception as e:
            print(f"Error checking CUDA memory: {e}. Falling back to CPU.")
            return torch.device("cpu")
    return torch.device("cpu")

def get_semantic_embedding(text: str, cache: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
    """
    Generate semantic embedding for text instead of random vectors
    
    Args:
        text: Text to generate embedding for
        cache: Optional cache dictionary to avoid repeated embedding generation
        
    Returns:
        torch.Tensor: Embedding tensor
    """
    # Check cache first if provided
    if cache is not None and text in cache:
        return cache[text]
        
    try:
        # Generate embedding
        client = LLMClient()
        embedding = client.get_embedding(text)
        
        # Convert to tensor
        embedding_tensor = torch.tensor(embedding, dtype=torch.float32)
        
        # Add to cache if provided
        if cache is not None:
            cache[text] = embedding_tensor
            
        return embedding_tensor
    except Exception as e:
        print(f"Error generating embedding: {e}. Falling back to random vector.")
        # Fallback to random vector of appropriate size (768 for most embedding models)
        return torch.randn(768, dtype=torch.float32) * 0.1

def clean_cuda_memory():
    """
    Clean up CUDA memory to prevent OOM errors
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

class DevelopmentalLayer(nn.Module):
    """
    Layer that gradually increases in complexity with development level
    
    Provides smooth transition between developmental stages rather than
    abrupt changes.
    """
    
    def __init__(self, 
                in_features: int, 
                out_features: int, 
                min_dev_level: float = 0.0,
                max_dev_level: float = 1.0):
        """
        Initialize developmental layer
        
        Args:
            in_features: Input feature dimension
            out_features: Output feature dimension
            min_dev_level: Minimum development level where layer begins to function
            max_dev_level: Development level where layer reaches full functionality
        """
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.min_dev_level = min_dev_level
        self.max_dev_level = max_dev_level
    
    def forward(self, x: torch.Tensor, dev_level: float) -> torch.Tensor:
        """
        Forward pass with developmental modulation
        
        Args:
            x: Input tensor
            dev_level: Current development level (0.0 to 1.0)
            
        Returns:
            Output tensor modulated by development level
        """
        # Calculate modulation factor (smooth transition)
        if dev_level <= self.min_dev_level:
            mod_factor = 0.0
        elif dev_level >= self.max_dev_level:
            mod_factor = 1.0
        else:
            # Smooth interpolation between min and max
            mod_factor = (dev_level - self.min_dev_level) / (self.max_dev_level - self.min_dev_level)
        
        # Ensure input has proper shape
        if len(x.shape) == 1:
            x = x.unsqueeze(0)  # Add batch dimension
        
        # Apply layer with modulation
        output = self.linear(x)
        
        # Apply modulation with proper broadcasting
        if len(output.shape) == 1:
            output = output.unsqueeze(0)
        
        # Reshape mod_factor for proper broadcasting
        mod_factor = torch.tensor(mod_factor, device=output.device).view(1, 1)
        
        return output * mod_factor

class SelfConceptNetwork(nn.Module):
    """
    Neural network for processing and developing self-concept
    
    This network processes information about the self and integrates it
    into a coherent self-representation.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the self-concept network
        
        Args:
            input_dim: Dimension of input features (set to embedding dim)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
        """
        super().__init__()
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers
        self.hidden1 = nn.Linear(hidden_dim, hidden_dim)
        self.hidden2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Domain-specific processing paths
        self.domain_physical = DevelopmentalLayer(hidden_dim, hidden_dim//2, min_dev_level=0.3, max_dev_level=0.6)
        self.domain_social = DevelopmentalLayer(hidden_dim, hidden_dim//2, min_dev_level=0.3, max_dev_level=0.6)
        self.domain_academic = DevelopmentalLayer(hidden_dim, hidden_dim//2, min_dev_level=0.4, max_dev_level=0.7)
        self.domain_emotional = DevelopmentalLayer(hidden_dim, hidden_dim//2, min_dev_level=0.5, max_dev_level=0.8)
        
        # Integration layer
        self.integration = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layers
        self.attribute_content = nn.Linear(hidden_dim, output_dim)
        self.attribute_confidence = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.1, max_dev_level=0.4)
        self.attribute_importance = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.2, max_dev_level=0.5)
        self.attribute_valence = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.3, max_dev_level=0.6)
        
        # Global self-evaluation
        self.global_self_esteem = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.4, max_dev_level=0.7)
        
        # Developmental factor (modulates network with development level)
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # Embedding cache to prevent repeated embedding generation
        self.embedding_cache = {}
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, input_data: torch.Tensor, domain: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Process input data to update self-concept
        
        Args:
            input_data: Input data tensor [batch_size, input_dim]
            domain: Optional domain specifier for domain-specific processing
            
        Returns:
            Dictionary containing processed outputs
        """
        # Get development level (direct value rather than sigmoid)
        dev_level = self.developmental_factor.item()
        
        # Check input dimensionality and reshape if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
            
        # Initial layers 
        x = F.relu(self.input_layer(input_data))
        x = self.dropout(x)
        
        # Hidden layers with gradual developmental progression
        h1 = F.relu(self.hidden1(x))
        h1 = self.dropout(h1)
        
        # Second hidden layer gradually activates with development
        if dev_level > 0.2:
            # Smooth transition factor
            trans_factor = min(1.0, (dev_level - 0.2) / 0.3)
            h2 = F.relu(self.hidden2(h1))
            h2 = self.dropout(h2)
            # Weighted combination
            x = h1 * (1 - trans_factor) + h2 * trans_factor
        else:
            x = h1
        
        # Domain-specific processing with gradual development
        domain_x = None
        if domain is not None:
            if domain == "physical":
                domain_x = F.relu(self.domain_physical(x, dev_level))
            elif domain == "social":
                domain_x = F.relu(self.domain_social(x, dev_level))
            elif domain == "academic":
                domain_x = F.relu(self.domain_academic(x, dev_level))
            elif domain == "emotional":
                domain_x = F.relu(self.domain_emotional(x, dev_level))
        
        # Integrate domain-specific processing if available
        if domain_x is not None and domain_x.abs().sum() > 0:
            x = torch.cat([x, domain_x], dim=1)
            x = F.relu(self.integration(x))
        
        # Generate outputs
        attribute_content = self.attribute_content(x)
        
        # Meta-cognitive aspects develop gradually
        confidence = torch.sigmoid(self.attribute_confidence(x, dev_level))
        importance = torch.sigmoid(self.attribute_importance(x, dev_level))
        valence = torch.tanh(self.attribute_valence(x, dev_level))
        self_esteem = torch.sigmoid(self.global_self_esteem(x, dev_level))
        
        # Clean up memory if needed
        if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
            clean_cuda_memory()
        
        return {
            "attribute_content": attribute_content,
            "confidence": confidence,
            "importance": importance,
            "valence": valence,
            "self_esteem": self_esteem
        }
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the self-concept network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class NarrativeNetwork(nn.Module):
    """
    Neural network for processing and integrating personal narrative
    
    This network processes experiences and integrates them into
    a coherent personal narrative with themes and meaning.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the narrative network
        
        Args:
            input_dim: Dimension of input features (set to embedding dim)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
        """
        super().__init__()
        
        # Input processing
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Experience processing
        self.event_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Theme extraction with developmental adaptation
        self.theme_extraction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, hidden_dim//2)
        )
        
        # Interpretation generation
        self.interpretation = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim//2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Event importance evaluation
        self.importance_evaluation = nn.Linear(hidden_dim, 1)
        
        # Emotional processing
        self.emotional_impact = nn.Linear(hidden_dim, output_dim)
        
        # Coherence evaluation
        self.coherence_evaluation = nn.Linear(hidden_dim, 1)
        
        # Developmental factor
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # LSTM for temporal sequence processing (more advanced narrative abilities)
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Embedding cache to prevent repeated embedding generation
        self.embedding_cache = {}
    
    def forward(self, 
               input_data: torch.Tensor, 
               operation: str = "process_event",
               past_events: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input data for narrative operations
        
        Args:
            input_data: Input data tensor [batch_size, input_dim]
            operation: Operation to perform
                "process_event": Process a new event
                "extract_theme": Extract themes from events
                "evaluate_coherence": Evaluate narrative coherence
            past_events: Optional tensor of past events for contextual processing
                [batch_size, num_events, hidden_dim]
                
        Returns:
            Dictionary containing processed outputs
        """
        # Get development level (direct value rather than sigmoid)
        dev_level = self.developmental_factor.item()
        
        # Check input dimensionality and reshape if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "process_event":
            # Event encoding
            event_encoding = self.event_encoder(x)
            
            # Event importance - gradually develops
            importance_factor = min(1.0, max(0.2, dev_level))
            importance = torch.sigmoid(self.importance_evaluation(event_encoding) * importance_factor)
            
            # Emotional processing - even early development has this
            emotional_impact = torch.tanh(self.emotional_impact(event_encoding))
            
            # Context-sensitive processing with gradual development
            # Calculate threshold and transition smoothness
            context_threshold = 0.4  # When context sensitivity begins
            transition_width = 0.2  # How quickly it develops
            
            # Initialize outputs
            theme_vector = None
            interpretation = None
            temporal_context = event_encoding
            
            # Process with past events if provided
            if past_events is not None:
                batch_size = input_data.size(0)
                
                # Gradually activate context sensitivity
                if dev_level > context_threshold:
                    # Calculate smooth transition factor
                    context_factor = min(1.0, (dev_level - context_threshold) / transition_width)
                    
                    # Add current event to past events
                    event_expanded = event_encoding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                    
                    # Process temporal sequence with LSTM if we have past events
                    if past_events.size(1) > 0:
                        all_events = torch.cat([past_events, event_expanded], dim=1)
                        seq_output, _ = self.temporal_lstm(all_events)
                        # Gradually transition to using temporal context
                        lstm_context = seq_output[:, -1, :]  # Get last output
                        temporal_context = event_encoding * (1 - context_factor) + lstm_context * context_factor
                    
                    # Theme extraction with smooth developmental progression
                    theme_threshold = 0.5  # When theme extraction begins
                    theme_width = 0.2  # How quickly it develops
                    
                    if dev_level > theme_threshold:
                        theme_factor = min(1.0, (dev_level - theme_threshold) / theme_width)
                        
                        # Extract theme
                        raw_theme = self.theme_extraction(temporal_context)
                        
                        # Apply developmental factor
                        theme_vector = raw_theme * theme_factor
                        
                        # Generate interpretation by combining event and theme
                        context_augmented = torch.cat([event_encoding, theme_vector], dim=1)
                        interpretation = self.interpretation(context_augmented) * theme_factor
                    else:
                        # Partial theme development
                        raw_theme = self.theme_extraction(temporal_context)
                        theme_vector = torch.zeros_like(raw_theme)
                        interpretation = torch.zeros((batch_size, self.interpretation[-1].out_features),
                                                   device=input_data.device)
                else:
                    # No context sensitivity yet
                    theme_vector = torch.zeros((batch_size, self.theme_extraction[0].out_features), 
                                             device=input_data.device)
                    interpretation = torch.zeros((batch_size, self.interpretation[-1].out_features),
                                               device=input_data.device)
            else:
                # No past events provided
                theme_vector = torch.zeros((input_data.size(0), self.theme_extraction[0].out_features), 
                                         device=input_data.device)
                interpretation = torch.zeros((input_data.size(0), self.interpretation[-1].out_features),
                                           device=input_data.device)
            
            # Clean up memory if needed
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                clean_cuda_memory()
            
            return {
                "event_encoding": event_encoding,
                "importance": importance,
                "emotional_impact": emotional_impact,
                "theme_vector": theme_vector,
                "interpretation": interpretation,
                "temporal_context": temporal_context
            }
            
        elif operation == "extract_theme":
            # Theme extraction from events
            # Gradual development of theme extraction
            theme_threshold = 0.3  # When theme extraction begins
            theme_width = 0.3  # How quickly it develops
            
            if dev_level < theme_threshold:
                # Early development - no real theme extraction yet
                null_theme = torch.zeros_like(self.theme_extraction(x))
                return {"theme_vector": null_theme}
            else:
                # Calculate smooth transition factor
                theme_factor = min(1.0, (dev_level - theme_threshold) / theme_width)
                
                # Extract theme with developmental scaling
                raw_theme = self.theme_extraction(x)
                theme_vector = raw_theme * theme_factor
                
                return {"theme_vector": theme_vector}
            
        elif operation == "evaluate_coherence":
            # Evaluate narrative coherence with gradual development
            coherence_threshold = 0.4  # When coherence evaluation begins
            coherence_width = 0.3  # How quickly it develops
            
            # Initialize with developmentally appropriate defaults
            if dev_level < coherence_threshold:
                base_coherence = 0.3 * (dev_level / coherence_threshold)  # Very basic coherence
                coherence = torch.tensor([base_coherence], device=input_data.device)
            else:
                # Calculate smooth transition factor
                coherence_factor = min(1.0, (dev_level - coherence_threshold) / coherence_width)
                
                # Process with temporal LSTM if past events available
                if past_events is not None and past_events.size(1) > 0:
                    # Use temporal sequence for coherence evaluation
                    seq_output, _ = self.temporal_lstm(past_events)
                    temporal_features = seq_output[:, -1, :]
                    raw_coherence = self.coherence_evaluation(temporal_features)
                else:
                    # Fallback to simpler coherence evaluation
                    raw_coherence = self.coherence_evaluation(x)
                
                # Apply developmental factor
                coherence = torch.sigmoid(raw_coherence * coherence_factor)
            
            return {"coherence": coherence}
        
        # Default return
        return {"features": x}
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the narrative network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class PreferenceNetwork(nn.Module):
    """
    Neural network for processing preferences and values
    
    This network processes experiences to form preferences and
    extracts values from preference patterns.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the preference network
        
        Args:
            input_dim: Dimension of input features (set to embedding dim)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
        """
        super().__init__()
        
        # Input processing
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Preference formation
        self.preference_formation = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Preference evaluation with developmental layers
        self.valence_evaluation = nn.Linear(hidden_dim, 1)  # Basic approach/avoid develops early
        self.strength_evaluation = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.2, max_dev_level=0.5)
        self.certainty_evaluation = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.3, max_dev_level=0.6)
        
        # Value extraction (higher-level preferences)
        self.value_extraction = DevelopmentalLayer(hidden_dim, output_dim, min_dev_level=0.5, max_dev_level=0.8)
        
        # Value importance evaluation
        self.value_importance = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.5, max_dev_level=0.8)
        
        # Preference application (using preferences to make decisions)
        self.preference_application = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Developmental factor
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # Embedding cache to prevent repeated embedding generation
        self.embedding_cache = {}
    
    def forward(self, 
               input_data: torch.Tensor, 
               operation: str = "form_preference",
               context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input data for preference operations
        
        Args:
            input_data: Input data tensor [batch_size, input_dim]
            operation: Operation to perform
                "form_preference": Form a preference from experience
                "extract_value": Extract values from preferences
                "apply_preference": Apply preferences to a choice
            context: Optional context tensor for preference application
                
        Returns:
            Dictionary containing processed outputs
        """
        # Get development level (direct value rather than sigmoid)
        dev_level = self.developmental_factor.item()
        
        # Check input dimensionality and reshape if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "form_preference":
            # Preference formation
            preference_encoding = self.preference_formation(x)
            
            # Basic preference evaluation (valence develops early)
            valence = torch.tanh(self.valence_evaluation(preference_encoding))
            
            # More nuanced preferences develop gradually
            strength = torch.sigmoid(self.strength_evaluation(preference_encoding, dev_level))
            certainty = torch.sigmoid(self.certainty_evaluation(preference_encoding, dev_level))
            
            # Value extraction develops later
            value_vector = self.value_extraction(preference_encoding, dev_level)
            value_importance = torch.sigmoid(self.value_importance(preference_encoding, dev_level))
            
            # Clean up memory if needed
            if torch.cuda.is_available() and torch.cuda.memory_allocated() > 0.8 * torch.cuda.get_device_properties(0).total_memory:
                clean_cuda_memory()
            
            return {
                "preference_encoding": preference_encoding,
                "valence": valence,
                "strength": strength,
                "certainty": certainty,
                "value_vector": value_vector,
                "value_importance": value_importance
            }
            
        elif operation == "extract_value":
            # Value extraction from preference patterns
            # Gradual development of value extraction
            value_threshold = 0.5  # When value extraction begins
            value_width = 0.3  # How quickly it develops
            
            # Preference encoding is available at all development levels
            preference_encoding = self.preference_formation(x)
            
            # Value extraction with developmental scaling
            value_vector = self.value_extraction(preference_encoding, dev_level)
            value_importance = torch.sigmoid(self.value_importance(preference_encoding, dev_level))
            
            return {
                "value_vector": value_vector,
                "value_importance": value_importance
            }
            
        elif operation == "apply_preference" and context is not None:
            # Apply preferences to a choice context
            preference_encoding = self.preference_formation(x)
            
            # Calculate application threshold and factor
            application_threshold = 0.3  # When preference application begins
            application_width = 0.3  # How quickly it develops
            
            if dev_level < application_threshold:
                # Very basic preference application
                decision_factor = dev_level / application_threshold
                combined = torch.cat([preference_encoding, context], dim=1)
                basic_influence = F.tanh(combined.mean(dim=1, keepdim=True)) * decision_factor
                
                return {
                    "decision_influence": basic_influence,
                    "confidence": torch.tensor([0.3 * decision_factor], device=input_data.device)
                }
            else:
                # More sophisticated preference application
                application_factor = min(1.0, (dev_level - application_threshold) / application_width)
                
                # Combine preference with context
                combined = torch.cat([preference_encoding, context], dim=1)
                
                # Generate decision influence
                decision_influence = self.preference_application(combined) * application_factor
                
                # Decision confidence based on preference strength and certainty
                strength = torch.sigmoid(self.strength_evaluation(preference_encoding, dev_level))
                certainty = torch.sigmoid(self.certainty_evaluation(preference_encoding, dev_level))
                confidence = strength * certainty * application_factor
                
                return {
                    "decision_influence": decision_influence,
                    "confidence": confidence
                }
        
        # Default return
        return {"features": x}
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the preference network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))

class PersonalityNetwork(nn.Module):
    """
    Neural network for processing and developing personality traits
    
    This network identifies consistent patterns across behaviors
    and organizes them into stable traits.
    """
    
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 64, num_traits: int = 10):
        """
        Initialize the personality network
        
        Args:
            input_dim: Dimension of input features (set to embedding dim)
            hidden_dim: Dimension of hidden layers
            output_dim: Dimension of output features
            num_traits: Number of personality traits to track
        """
        super().__init__()
        
        # Input processing
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Behavioral pattern recognition
        self.pattern_recognition = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Trait extraction with developmental adaptation
        self.trait_extraction = DevelopmentalLayer(hidden_dim, num_traits, min_dev_level=0.2, max_dev_level=0.6)
        
        # Dimension organization (e.g., Big Five)
        self.dimension_organization = DevelopmentalLayer(num_traits, 5, min_dev_level=0.5, max_dev_level=0.8)
        
        # Trait application (predicting behavior from traits)
        self.trait_application = nn.Sequential(
            nn.Linear(num_traits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trait stability tracking
        self.stability_evaluation = DevelopmentalLayer(hidden_dim, 1, min_dev_level=0.3, max_dev_level=0.7)
        
        # Developmental factor
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # Running average of trait scores for stability
        self.register_buffer('trait_averages', torch.zeros(num_traits))
        self.register_buffer('trait_update_count', torch.zeros(1))
        
        # Embedding cache to prevent repeated embedding generation
        self.embedding_cache = {}
    
    def forward(self, 
               input_data: torch.Tensor, 
               operation: str = "extract_traits",
               context: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Process input data for personality operations
        
        Args:
            input_data: Input data tensor [batch_size, input_dim]
            operation: Operation to perform
                "extract_traits": Extract traits from behavioral patterns
                "predict_behavior": Predict behavior based on traits
                "evaluate_stability": Evaluate trait stability
            context: Optional context tensor for contextual processing
                
        Returns:
            Dictionary containing processed outputs
        """
        # Get development level (direct value rather than sigmoid)
        dev_level = self.developmental_factor.item()
        
        # Check input dimensionality and reshape if needed
        if len(input_data.shape) == 1:
            input_data = input_data.unsqueeze(0)  # Add batch dimension
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "extract_traits":
            # Extract traits from behavioral patterns
            patterns = self.pattern_recognition(x)
            trait_scores = torch.sigmoid(self.trait_extraction(patterns, dev_level))
            
            # Update running averages if we have a batch
            if trait_scores.size(0) > 0:
                batch_avg = trait_scores.mean(0)  # Average across batch dimension
                self.trait_averages = (self.trait_averages * self.trait_update_count + batch_avg) / (self.trait_update_count + 1)
                self.trait_update_count += 1
            
            # Organize into dimensions if development level allows
            if dev_level >= 0.5:
                dimension_scores = torch.sigmoid(self.dimension_organization(trait_scores, dev_level))
            else:
                dimension_scores = torch.zeros(trait_scores.size(0), 5, device=trait_scores.device)
            
            return {
                "patterns": trait_scores,
                "dimensions": dimension_scores
            }
            
        elif operation == "predict_behavior":
            # Predict behavior based on traits
            if context is None:
                context = torch.zeros(x.size(0), self.trait_application[0].in_features, device=x.device)
            
            # Apply traits to predict behavior
            behavior = self.trait_application(context)
            
            return {
                "behavior": behavior
            }
            
        elif operation == "evaluate_stability":
            # Evaluate trait stability with gradual development
            stability_threshold = 0.3  # When stability evaluation begins
            stability_width = 0.4  # How quickly it develops
            
            # Extract traits
            patterns = self.pattern_recognition(x)
            trait_scores = torch.sigmoid(self.trait_extraction(patterns, dev_level))
            
            # Calculate stability based on developmental level
            if dev_level < stability_threshold or self.trait_update_count == 0:
                # Basic stability assessment
                base_stability = 0.2 + (dev_level / stability_threshold) * 0.3
                stability = torch.tensor([base_stability], device=input_data.device)
            else:
                # Calculate smooth transition factor
                stability_factor = min(1.0, (dev_level - stability_threshold) / stability_width)
                
                # Calculate deviation from running averages
                # Ensure trait_scores and trait_averages have compatible shapes
                if len(trait_scores.shape) == 1:
                    trait_scores = trait_scores.unsqueeze(0)
                if len(self.trait_averages.shape) == 1:
                    self.trait_averages = self.trait_averages.unsqueeze(0)
                
                # Calculate mean deviation across batch
                deviation = torch.abs(trait_scores - self.trait_averages)
                raw_stability = 1.0 - torch.clamp(deviation.mean(), 0.0, 1.0)
                
                # Apply developmental factor
                stability = raw_stability * stability_factor + 0.5 * (1.0 - stability_factor)
                
            return {
                "stability": stability
            }
        
        # Default return
        return {"features": x}
    
    def set_development_level(self, level: float) -> None:
        """
        Set the developmental level of the personality network
        
        Args:
            level: Development level (0.0 to 1.0)
        """
        with torch.no_grad():
            self.developmental_factor.copy_(torch.tensor(max(0.0, min(1.0, level))))
