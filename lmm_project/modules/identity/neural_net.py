import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple

def get_device() -> torch.device:
    """
    Get the appropriate device (GPU if available, otherwise CPU)
    
    Returns:
        torch.device: The device to use for tensor operations
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

class SelfConceptNetwork(nn.Module):
    """
    Neural network for processing and developing self-concept
    
    This network processes information about the self and integrates it
    into a coherent self-representation.
    """
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the self-concept network
        
        Args:
            input_dim: Dimension of input features
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
        self.domain_physical = nn.Linear(hidden_dim, hidden_dim//2)
        self.domain_social = nn.Linear(hidden_dim, hidden_dim//2)
        self.domain_academic = nn.Linear(hidden_dim, hidden_dim//2)
        self.domain_emotional = nn.Linear(hidden_dim, hidden_dim//2)
        
        # Integration layer
        self.integration = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Output layers
        self.attribute_content = nn.Linear(hidden_dim, output_dim)
        self.attribute_confidence = nn.Linear(hidden_dim, 1)
        self.attribute_importance = nn.Linear(hidden_dim, 1)
        self.attribute_valence = nn.Linear(hidden_dim, 1)
        
        # Global self-evaluation
        self.global_self_esteem = nn.Linear(hidden_dim, 1)
        
        # Developmental factor (modulates network with development level)
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
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
        # Get developmental modulation factor
        dev_factor = torch.sigmoid(self.developmental_factor * 5)
        
        # Initial layers with developmental modulation
        x = F.relu(self.input_layer(input_data))
        x = self.dropout(x)
        
        # Hidden layers
        h1 = F.relu(self.hidden1(x))
        h1 = self.dropout(h1)
        
        # Development affects depth of processing
        if dev_factor > 0.3:
            h2 = F.relu(self.hidden2(h1))
            h2 = self.dropout(h2)
            x = h2
        else:
            x = h1
        
        # Domain-specific processing based on development level
        if dev_factor > 0.5 and domain is not None:
            if domain == "physical":
                domain_x = F.relu(self.domain_physical(x))
            elif domain == "social":
                domain_x = F.relu(self.domain_social(x))
            elif domain == "academic":
                domain_x = F.relu(self.domain_academic(x))
            elif domain == "emotional":
                domain_x = F.relu(self.domain_emotional(x))
            else:
                domain_x = x
                
            # Concatenate domain-specific and general processing
            x = torch.cat([x, domain_x], dim=1)
            x = F.relu(self.integration(x))
        
        # Generate outputs
        attribute_content = self.attribute_content(x)
        
        # Meta-cognitive aspects develop with maturity
        if dev_factor > 0.2:
            confidence = torch.sigmoid(self.attribute_confidence(x))
            importance = torch.sigmoid(self.attribute_importance(x))
            valence = torch.tanh(self.attribute_valence(x))
            self_esteem = torch.sigmoid(self.global_self_esteem(x))
        else:
            # Limited metacognitive abilities at early development
            confidence = torch.sigmoid(torch.randn_like(self.attribute_confidence(x)) * 0.1 + 0.5)
            importance = torch.sigmoid(torch.randn_like(self.attribute_importance(x)) * 0.1 + 0.5)
            valence = torch.tanh(torch.randn_like(self.attribute_valence(x)) * 0.1)
            self_esteem = torch.sigmoid(torch.randn_like(self.global_self_esteem(x)) * 0.1 + 0.5)
        
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
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the narrative network
        
        Args:
            input_dim: Dimension of input features
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
        
        # Theme extraction
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
        # Get developmental modulation factor
        dev_factor = torch.sigmoid(self.developmental_factor * 5)
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "process_event":
            # Event encoding
            event_encoding = self.event_encoder(x)
            
            # Event importance (develops with maturity)
            importance = torch.sigmoid(self.importance_evaluation(event_encoding))
            
            # Emotional processing
            emotional_impact = torch.tanh(self.emotional_impact(event_encoding))
            
            # Context-sensitive processing with development
            if dev_factor > 0.4 and past_events is not None:
                # Add current event to past events
                batch_size = input_data.size(0)
                event_expanded = event_encoding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
                
                # Process temporal sequence with LSTM
                if past_events.size(1) > 0:
                    all_events = torch.cat([past_events, event_expanded], dim=1)
                    seq_output, _ = self.temporal_lstm(all_events)
                    temporal_context = seq_output[:, -1, :]  # Get last output
                else:
                    temporal_context = event_encoding
                
                # Theme extraction with context
                if dev_factor > 0.6:
                    theme_vector = self.theme_extraction(temporal_context)
                    # Generate interpretation with thematic understanding
                    context_augmented = torch.cat([event_encoding, theme_vector], dim=1)
                    interpretation = self.interpretation(context_augmented)
                else:
                    theme_vector = torch.zeros((batch_size, self.theme_extraction[0].out_features), 
                                             device=input_data.device)
                    interpretation = torch.zeros((batch_size, self.interpretation[-1].out_features),
                                               device=input_data.device)
            else:
                # Simple processing without context at early development
                temporal_context = event_encoding
                theme_vector = torch.zeros((input_data.size(0), self.theme_extraction[0].out_features), 
                                         device=input_data.device)
                interpretation = torch.zeros((input_data.size(0), self.interpretation[-1].out_features),
                                           device=input_data.device)
            
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
            # At early development, themes are simple and concrete
            if dev_factor < 0.4:
                theme_vector = torch.randn_like(self.theme_extraction(x)) * 0.1
            else:
                theme_vector = self.theme_extraction(x)
                
            return {
                "theme_vector": theme_vector
            }
            
        elif operation == "evaluate_coherence":
            # Evaluate narrative coherence
            # At early development, coherence evaluation is limited
            if dev_factor < 0.6:
                coherence = torch.sigmoid(torch.tensor([0.3], device=input_data.device))
            else:
                # Process with temporal LSTM if past events available
                if past_events is not None and past_events.size(1) > 0:
                    seq_output, _ = self.temporal_lstm(past_events)
                    temporal_features = seq_output[:, -1, :]
                    coherence = torch.sigmoid(self.coherence_evaluation(temporal_features))
                else:
                    coherence = torch.sigmoid(self.coherence_evaluation(x))
            
            return {
                "coherence": coherence
            }
        
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
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64):
        """
        Initialize the preference network
        
        Args:
            input_dim: Dimension of input features
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
        
        # Preference evaluation
        self.valence_evaluation = nn.Linear(hidden_dim, 1)
        self.strength_evaluation = nn.Linear(hidden_dim, 1)
        self.certainty_evaluation = nn.Linear(hidden_dim, 1)
        
        # Value extraction (higher-level preferences)
        self.value_extraction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim//2, output_dim)
        )
        
        # Value importance evaluation
        self.value_importance = nn.Linear(output_dim, 1)
        
        # Preference application (using preferences to make decisions)
        self.preference_application = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Developmental factor
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
    
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
        # Get developmental modulation factor
        dev_factor = torch.sigmoid(self.developmental_factor * 5)
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "form_preference":
            # Preference formation
            preference_encoding = self.preference_formation(x)
            
            # Basic preference evaluation (develops with maturity)
            if dev_factor < 0.2:
                # Simple approach/avoid at early stages
                valence = torch.tanh(self.valence_evaluation(preference_encoding)) 
                strength = torch.ones_like(self.strength_evaluation(preference_encoding)) * 0.5
                certainty = torch.ones_like(self.certainty_evaluation(preference_encoding)) * 0.5
            else:
                # More nuanced preferences with development
                valence = torch.tanh(self.valence_evaluation(preference_encoding))
                strength = torch.sigmoid(self.strength_evaluation(preference_encoding))
                certainty = torch.sigmoid(self.certainty_evaluation(preference_encoding))
            
            # Value extraction develops later
            if dev_factor > 0.6:
                value_vector = self.value_extraction(preference_encoding)
                value_importance = torch.sigmoid(self.value_importance(value_vector))
            else:
                value_vector = torch.zeros((input_data.size(0), self.value_extraction[-1].out_features), 
                                         device=input_data.device)
                value_importance = torch.zeros((input_data.size(0), 1), device=input_data.device)
            
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
            # Only available at higher development levels
            if dev_factor < 0.6:
                value_vector = torch.zeros((input_data.size(0), self.value_extraction[-1].out_features), 
                                         device=input_data.device)
                value_importance = torch.zeros((input_data.size(0), 1), device=input_data.device)
            else:
                preference_encoding = self.preference_formation(x)
                value_vector = self.value_extraction(preference_encoding)
                value_importance = torch.sigmoid(self.value_importance(value_vector))
                
            return {
                "value_vector": value_vector,
                "value_importance": value_importance
            }
            
        elif operation == "apply_preference" and context is not None:
            # Apply preferences to a choice context
            preference_encoding = self.preference_formation(x)
            
            # Combine preference with context
            combined = torch.cat([preference_encoding, context], dim=1)
            
            # Generate decision influence
            decision_influence = self.preference_application(combined)
            
            # Decision confidence based on preference strength and certainty
            strength = torch.sigmoid(self.strength_evaluation(preference_encoding))
            certainty = torch.sigmoid(self.certainty_evaluation(preference_encoding))
            confidence = strength * certainty
            
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
    
    def __init__(self, input_dim: int = 128, hidden_dim: int = 256, output_dim: int = 64, num_traits: int = 10):
        """
        Initialize the personality network
        
        Args:
            input_dim: Dimension of input features
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
        
        # Trait extraction
        self.trait_extraction = nn.Sequential(
            nn.Linear(hidden_dim, num_traits),
            nn.Sigmoid()  # Traits from 0.0 to 1.0
        )
        
        # Dimension organization (e.g., Big Five)
        self.dimension_organization = nn.Linear(num_traits, 5)
        
        # Trait application (predicting behavior from traits)
        self.trait_application = nn.Sequential(
            nn.Linear(num_traits, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Trait stability tracking
        self.stability_evaluation = nn.Linear(hidden_dim, 1)
        
        # Developmental factor
        self.developmental_factor = nn.Parameter(torch.tensor(0.0), requires_grad=False)
        
        # Running average of trait scores for stability
        register_buffer = lambda name, tensor: self.register_buffer(name, tensor)
        register_buffer('trait_averages', torch.zeros(num_traits))
        register_buffer('trait_update_count', torch.zeros(1))
    
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
        # Get developmental modulation factor
        dev_factor = torch.sigmoid(self.developmental_factor * 5)
        
        # Initial processing
        x = F.relu(self.input_layer(input_data))
        
        if operation == "extract_traits":
            # Behavioral pattern recognition
            patterns = self.pattern_recognition(x)
            
            # Trait extraction (develops with maturity)
            if dev_factor < 0.2:
                # Very simple temperamental tendencies at early stages
                trait_scores = torch.sigmoid(torch.randn_like(self.trait_extraction(patterns)) * 0.1 + 0.5)
                # No dimension organization at early stages
                dimension_scores = torch.zeros((input_data.size(0), 5), device=input_data.device)
                stability = torch.tensor([0.1], device=input_data.device)
            else:
                # More structured trait extraction with development
                trait_scores = self.trait_extraction(patterns)
                
                # Update running averages for stability tracking
                if self.training:
                    batch_avg = trait_scores.mean(0)
                    old_count = self.trait_update_count
                    new_count = old_count + 1
                    self.trait_averages.copy_((self.trait_averages * old_count + batch_avg) / new_count)
                    self.trait_update_count.copy_(new_count)
                
                # Stability calculation
                stability = torch.sigmoid(self.stability_evaluation(patterns))
                
                # Dimension organization develops later
                if dev_factor > 0.6:
                    dimension_scores = torch.tanh(self.dimension_organization(trait_scores))
                else:
                    dimension_scores = torch.zeros((input_data.size(0), 5), device=input_data.device)
            
            return {
                "trait_scores": trait_scores,
                "dimension_scores": dimension_scores,
                "stability": stability,
                "patterns": patterns
            }
            
        elif operation == "predict_behavior":
            # Predict behavior from traits
            # Only meaningful at higher development levels
            if dev_factor < 0.4:
                behavior_prediction = torch.randn((input_data.size(0), self.trait_application[-1].out_features), 
                                                device=input_data.device) * 0.1
            else:
                # Extract traits first
                patterns = self.pattern_recognition(x)
                trait_scores = self.trait_extraction(patterns)
                
                # Context-sensitive behavior prediction
                if context is not None and dev_factor > 0.6:
                    # Modulate traits based on context (trait x context interaction)
                    context_processed = F.relu(self.input_layer(context))
                    # Simple context influence through multiplication
                    trait_scores = trait_scores * torch.sigmoid(context_processed.mean(dim=1, keepdim=True))
                
                # Predict behavior from traits
                behavior_prediction = self.trait_application(trait_scores)
                
            return {
                "behavior_prediction": behavior_prediction
            }
            
        elif operation == "evaluate_stability":
            # Evaluate trait stability
            patterns = self.pattern_recognition(x)
            trait_scores = self.trait_extraction(patterns)
            
            # Calculate deviation from running averages
            if self.trait_update_count > 0:
                deviation = torch.abs(trait_scores.mean(0) - self.trait_averages)
                stability = 1.0 - torch.clamp(deviation.mean(), 0.0, 1.0)
            else:
                stability = torch.tensor([0.5], device=input_data.device)
                
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
