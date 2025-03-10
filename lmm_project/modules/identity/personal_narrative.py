# TODO: Implement the PersonalNarrative class to create autobiographical continuity
# This component should be able to:
# - Construct a coherent story of personal experiences
# - Integrate new experiences into the ongoing narrative
# - Identify themes and patterns across experiences
# - Maintain temporal continuity of identity

# TODO: Implement developmental progression in personal narrative:
# - Simple episodic sequences in early stages
# - Chronological life stories in childhood
# - Theme-based integration in adolescence
# - Complex, meaning-focused narratives in adulthood

# TODO: Create mechanisms for:
# - Narrative construction: Form coherent stories from experiences
# - Causal connection: Link events with causal relationships
# - Thematic integration: Identify recurring themes and patterns
# - Meaning-making: Extract personal significance from events

# TODO: Implement narrative characteristics:
# - Coherence: Logical and temporal consistency
# - Complexity: Multilayered interpretation of events
# - Agency: Sense of control in one's life story
# - Emotional tone: Overall valence of the narrative

# TODO: Connect to episodic memory and belief systems
# Personal narrative should draw on episodic memories
# and influence/be influenced by the belief system

import logging
import uuid
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime

import numpy as np
import torch
from collections import deque

from lmm_project.modules.base_module import BaseModule
from lmm_project.core.event_bus import EventBus
from lmm_project.core.message import Message
from lmm_project.modules.identity.models import NarrativeEvent, NarrativeTheme, PersonalNarrative as NarrativeModel, IdentityNeuralState
from lmm_project.modules.identity.neural_net import NarrativeNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class PersonalNarrative(BaseModule):
    """
    Creates and maintains autobiographical continuity
    
    This module constructs a coherent story from experiences,
    providing a sense of continuity and meaning to identity.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Simple episodic memory",
        0.2: "Temporal sequences",
        0.4: "Causal connections",
        0.6: "Thematic integration",
        0.8: "Coherent life story",
        1.0: "Meaning-focused narrative"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the personal narrative module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="personal_narrative", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device for neural processing
        self.device = get_device()
        
        # Initialize neural network
        self.network = NarrativeNetwork().to(self.device)
        self.network.set_development_level(development_level)
        
        # Initialize personal narrative state
        self.narrative = NarrativeModel()
        
        # Initialize neural state for tracking
        self.neural_state = IdentityNeuralState()
        self.neural_state.narrative_development = development_level
        
        # Event memory cache (for temporal sequence processing)
        self.event_embeddings = {}  # event_id -> embedding tensor
        
        # Available life periods (expands with development)
        self.available_life_periods = ["immediate"]
        self._adjust_life_periods_for_development()
        
        # Recent processing queue
        self.recent_inputs = deque(maxlen=100)
        
        logger.info(f"Personal narrative module initialized at development level {development_level:.2f}")
    
    def _adjust_life_periods_for_development(self):
        """Adjust available life periods based on developmental level"""
        if self.development_level < 0.2:
            # Very limited temporal span at early stages
            self.available_life_periods = ["immediate"]
            
        elif self.development_level < 0.4:
            # Basic temporal categories
            self.available_life_periods = ["immediate", "recent_past"]
            
        elif self.development_level < 0.6:
            # More differentiated temporal categories
            self.available_life_periods = ["immediate", "recent_past", "early_memories"]
            
        elif self.development_level < 0.8:
            # Life stage temporal categories
            self.available_life_periods = ["immediate", "recent_past", "early_memories", "childhood", "current_period"]
            
        else:
            # Full life narrative categories
            self.available_life_periods = [
                "immediate", "recent_past", "early_memories", "early_childhood", 
                "middle_childhood", "adolescence", "young_adulthood", "current_period", "anticipated_future"
            ]
            
        logger.info(f"Personal narrative life periods adjusted to: {self.available_life_periods}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update personal narrative
        
        Args:
            input_data: Dictionary containing narrative operations
                Required keys:
                - 'operation': The operation to perform
                  Options: 'add_event', 'update_event', 'extract_theme', 'query_narrative'
                
                For 'add_event' operation:
                - 'title': Title of the event
                - 'description': Description of what happened
                - 'interpretation': (Optional) Personal meaning of the event
                - 'emotional_impact': (Optional) Emotional reactions to the event
                - 'age_period': Life period when this event occurred
                - 'importance': (Optional) Subjective importance (0.0 to 1.0)
                
                For 'update_event' operation:
                - 'event_id': ID of the event to update
                - 'interpretation': (Optional) Updated interpretation
                - 'importance': (Optional) Updated importance
                - 'emotional_impact': (Optional) Updated emotional impact
                
                For 'extract_theme' operation:
                - 'event_ids': List of event IDs to analyze for themes
                - 'name': (Optional) Suggested theme name
                
                For 'query_narrative' operation:
                - 'query_type': Type of query ('events', 'themes', 'coherence', 'all')
                - 'life_period': (Optional) Life period to filter by
            
        Returns:
            Dictionary with the results of narrative processing
        """
        operation = input_data.get("operation", "query_narrative")
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Process based on operation
        if operation == "add_event":
            return self._add_event(input_data, process_id)
        elif operation == "update_event":
            return self._update_event(input_data, process_id)
        elif operation == "extract_theme":
            return self._extract_theme(input_data, process_id)
        elif operation == "query_narrative":
            return self._query_narrative(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _add_event(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Add a new event to the personal narrative"""
        # Extract required data
        title = input_data.get("title")
        description = input_data.get("description")
        age_period = input_data.get("age_period")
        
        if not title:
            return {"status": "error", "message": "No title provided", "process_id": process_id}
        if not description:
            return {"status": "error", "message": "No description provided", "process_id": process_id}
        if not age_period:
            return {"status": "error", "message": "No age_period provided", "process_id": process_id}
            
        # Check if age period is available at current development level
        if age_period not in self.available_life_periods:
            return {
                "status": "undeveloped",
                "message": f"Life period '{age_period}' not available at current development level",
                "available_life_periods": self.available_life_periods,
                "process_id": process_id
            }
            
        # Extract optional data
        interpretation = input_data.get("interpretation", "")
        emotional_impact = input_data.get("emotional_impact", {})
        importance = input_data.get("importance", 0.5)
        
        # Get past events for context (if development permits)
        past_events_tensor = None
        if self.development_level >= 0.4 and self.event_embeddings:
            # Create tensor of past event embeddings for contextual processing
            past_embeddings = list(self.event_embeddings.values())
            if past_embeddings:
                past_events_tensor = torch.stack(past_embeddings, dim=0).unsqueeze(0)  # [1, num_events, hidden_dim]
        
        # Process event through neural network
        input_features = self._extract_features(description)
        
        with torch.no_grad():
            network_output = self.network(
                input_features.to(self.device),
                operation="process_event",
                past_events=past_events_tensor.to(self.device) if past_events_tensor is not None else None
            )
        
        # Create new event
        event_id = str(uuid.uuid4())
        
        # Generate interpretation based on development level
        if not interpretation and self.development_level >= 0.6:
            # At higher development levels, generate interpretation from neural network
            interpretation_embedding = network_output["interpretation"]
            # For demo, convert embedding to simple text
            if torch.sum(interpretation_embedding) > 0:
                interpretation = f"This event relates to my sense of {self._get_narrative_theme_from_embedding(interpretation_embedding)}"
        
        # Adjust importance based on neural network and development level
        if self.development_level >= 0.2:
            importance = max(0.0, min(1.0, importance * 0.5 + network_output["importance"].item() * 0.5))
        
        # Create the event
        new_event = NarrativeEvent(
            event_id=event_id,
            title=title,
            description=description,
            interpretation=interpretation,
            emotional_impact=emotional_impact,
            importance=importance,
            age_period=age_period,
            themes=[],  # No themes initially
            related_events=[]  # No related events initially
        )
        
        # Add to narrative
        self.narrative.add_event(new_event)
        
        # Store event embedding for future reference
        self.event_embeddings[event_id] = network_output["event_encoding"].cpu().squeeze(0)
        
        # Record activation in neural state
        self.neural_state.add_activation('narrative', {
            'operation': 'add_event',
            'age_period': age_period,
            'importance': importance
        })
        
        # Check for automatic theme extraction at higher development levels
        if self.development_level >= 0.6 and len(self.narrative.events) >= 3:
            # Try to extract themes automatically after adding events
            self._auto_extract_themes()
        
        # Add to recent inputs
        self.recent_inputs.append({
            "type": "add_event",
            "data": input_data,
            "timestamp": datetime.now()
        })
        
        # Check and update narrative coherence
        self._update_narrative_coherence()
        
        return {
            "status": "success",
            "event_id": event_id,
            "operation": "add_event",
            "event": new_event.dict(),
            "process_id": process_id
        }
    
    def _update_event(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Update an existing event in the personal narrative"""
        # Extract data
        event_id = input_data.get("event_id")
        
        if not event_id:
            return {"status": "error", "message": "No event_id provided", "process_id": process_id}
            
        # Check if event exists
        if event_id not in self.narrative.events:
            return {
                "status": "error", 
                "message": f"Event with ID {event_id} not found", 
                "process_id": process_id
            }
            
        # Get the existing event
        event = self.narrative.events[event_id]
        
        # Update fields if provided
        updated = False
        
        if "interpretation" in input_data:
            event.interpretation = input_data["interpretation"]
            updated = True
            
        if "importance" in input_data:
            event.importance = max(0.0, min(1.0, float(input_data["importance"])))
            updated = True
            
        if "emotional_impact" in input_data:
            # Update emotional impact (can be partial update)
            new_impact = input_data["emotional_impact"]
            if isinstance(new_impact, dict):
                for emotion, intensity in new_impact.items():
                    event.emotional_impact[emotion] = intensity
                updated = True
                
        if "title" in input_data:
            event.title = input_data["title"]
            updated = True
            
        if "description" in input_data:
            event.description = input_data["description"]
            updated = True
            
            # Re-process through neural network to update embeddings
            input_features = self._extract_features(event.description)
            
            with torch.no_grad():
                network_output = self.network(
                    input_features.to(self.device),
                    operation="process_event"
                )
                
            # Update stored embedding
            self.event_embeddings[event_id] = network_output["event_encoding"].cpu().squeeze(0)
        
        if updated:
            # Update timestamp
            event.updated_at = datetime.now()
            
            # Update in narrative
            self.narrative.events[event_id] = event
            self.narrative.last_updated = datetime.now()
            
            # Record activation in neural state
            self.neural_state.add_activation('narrative', {
                'operation': 'update_event',
                'event_id': event_id,
                'age_period': event.age_period
            })
            
            # Check and update narrative coherence
            self._update_narrative_coherence()
            
            # Add to recent inputs
            self.recent_inputs.append({
                "type": "update_event",
                "data": input_data,
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "event_id": event_id,
                "operation": "update_event",
                "event": event.dict(),
                "process_id": process_id
            }
        else:
            return {
                "status": "not_modified",
                "message": "No changes were made to the event",
                "process_id": process_id
            }
    
    def _extract_theme(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Extract a theme from a set of events"""
        # Theme extraction requires higher development
        if self.development_level < 0.4:
            return {
                "status": "error",
                "message": "Theme extraction not available at current development level",
                "process_id": process_id
            }
            
        # Extract data
        event_ids = input_data.get("event_ids", [])
        suggested_name = input_data.get("name")
        suggested_description = input_data.get("description")
        
        # Validate input
        if not event_ids:
            return {
                "status": "error",
                "message": "No event IDs provided for theme extraction",
                "process_id": process_id
            }
            
        # Get valid events
        valid_events = []
        for event_id in event_ids:
            if event_id in self.narrative.events:
                valid_events.append(self.narrative.events[event_id])
                
        if not valid_events:
            return {
                "status": "error",
                "message": "No valid events found with provided IDs",
                "process_id": process_id
            }
            
        # Get embeddings for events
        event_embeddings = []
        for event in valid_events:
            if event.event_id in self.event_embeddings:
                event_embeddings.append(self.event_embeddings[event.event_id])
                
        if not event_embeddings:
            # Re-process events to get embeddings
            for event in valid_events:
                input_features = self._extract_features(event.description)
                
                with torch.no_grad():
                    network_output = self.network(
                        input_features.to(self.device),
                        operation="process_event"
                    )
                    
                self.event_embeddings[event.event_id] = network_output["event_encoding"].cpu().squeeze(0)
                event_embeddings.append(self.event_embeddings[event.event_id])
                
        # Average embeddings to get theme representation
        if event_embeddings:
            # Stack embeddings and average them
            stacked_embeddings = torch.stack(event_embeddings)
            avg_embedding = stacked_embeddings.mean(dim=0)
            
            # The neural network expects input of shape [batch_size, input_dim] where input_dim is 128
            # But our avg_embedding has shape [hidden_dim] which is 256
            # We need to project it down to the expected input dimension
            
            # Check the shape of avg_embedding
            if avg_embedding.dim() == 1:
                # If it's a 1D tensor, add batch dimension
                avg_embedding = avg_embedding.unsqueeze(0)  # Shape becomes [1, hidden_dim]
                
            # Move avg_embedding to the device before processing
            avg_embedding = avg_embedding.to(self.device)
                
            # If the embedding dimension doesn't match the expected input dimension (128),
            # we need to project it to the correct size
            if avg_embedding.shape[1] != 128:
                # Create a simple projection if needed (first time)
                if not hasattr(self, 'projection_layer'):
                    input_dim = avg_embedding.shape[1]  # Current dimension (likely 256)
                    target_dim = 128  # Target dimension for the network input
                    self.projection_layer = torch.nn.Linear(input_dim, target_dim)
                    # Ensure the projection layer is on the same device as the input tensor
                    self.projection_layer.to(self.device)
                
                # Apply the projection
                avg_embedding = self.projection_layer(avg_embedding)
                
            # Process through neural network for theme extraction
            with torch.no_grad():
                theme_output = self.network(
                    avg_embedding,  # Already on the correct device
                    operation="extract_theme"
                )
                
            theme_vector = theme_output["theme_vector"].cpu().squeeze(0)
        else:
            return {
                "status": "error",
                "message": "Could not generate embeddings for events",
                "process_id": process_id
            }
            
        # Generate theme name and description if not provided
        if not suggested_name:
            # Generate theme name based on common elements
            if self.development_level < 0.6:
                suggested_name = f"Theme from {valid_events[0].age_period}"
            else:
                suggested_name = self._get_narrative_theme_from_embedding(theme_vector)
                
        # Generate theme description
        if self.development_level < 0.6:
            description = f"Common elements in events from {valid_events[0].age_period}"
        else:
            # More sophisticated theme description at higher development levels
            common_elements = []
            if len(valid_events) >= 2:
                # Look for common emotional impacts
                emotions = {}
                for event in valid_events:
                    for emotion, intensity in event.emotional_impact.items():
                        if emotion not in emotions:
                            emotions[emotion] = []
                        emotions[emotion].append(intensity)
                
                # Find emotions present in multiple events
                for emotion, intensities in emotions.items():
                    if len(intensities) >= len(valid_events) / 2:
                        avg_intensity = sum(intensities) / len(intensities)
                        if avg_intensity > 0.5:
                            common_elements.append(f"strong {emotion}")
                        else:
                            common_elements.append(emotion)
                            
            if not common_elements:
                description = f"A recurring pattern across events in my life"
            else:
                description = f"A recurring pattern involving {', '.join(common_elements)}"
                
        # Calculate emotional tone of the theme
        emotional_tone = 0.0
        emotion_count = 0
        for event in valid_events:
            for emotion, intensity in event.emotional_impact.items():
                # Simple mapping of common emotions to valence
                if emotion in ["joy", "happiness", "excitement", "satisfaction", "pride"]:
                    emotional_tone += intensity
                    emotion_count += 1
                elif emotion in ["sadness", "fear", "anger", "disappointment", "shame"]:
                    emotional_tone -= intensity
                    emotion_count += 1
        
        if emotion_count > 0:
            emotional_tone = emotional_tone / emotion_count
        emotional_tone = max(-1.0, min(1.0, emotional_tone))
        
        # Calculate importance of the theme
        importance = sum(event.importance for event in valid_events) / len(valid_events)
        
        # Create the theme
        theme_id = str(uuid.uuid4())
        new_theme = NarrativeTheme(
            theme_id=theme_id,
            name=suggested_name,
            description=description,
            events=[event.event_id for event in valid_events],
            emotional_tone=emotional_tone,
            importance=importance
        )
        
        # Add to narrative
        self.narrative.add_theme(new_theme)
        
        # Record activation in neural state
        self.neural_state.add_activation('narrative', {
            'operation': 'extract_theme',
            'event_count': len(valid_events),
            'importance': importance
        })
        
        # Add to recent inputs
        self.recent_inputs.append({
            "type": "extract_theme",
            "data": input_data,
            "timestamp": datetime.now()
        })
        
        # Check and update narrative coherence
        self._update_narrative_coherence()
        
        return {
            "status": "success",
            "theme_id": theme_id,
            "operation": "extract_theme",
            "theme": new_theme.dict(),
            "process_id": process_id
        }
    
    def _query_narrative(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Query personal narrative information"""
        query_type = input_data.get("query_type", "all")
        life_period = input_data.get("life_period")
        
        if query_type == "events":
            # Return events, filtered by life period if specified
            if life_period:
                if life_period not in self.narrative.life_periods:
                    return {
                        "status": "not_found",
                        "message": f"No events found in life period '{life_period}'",
                        "process_id": process_id
                    }
                    
                # Get events in the specified life period
                period_events = {}
                for event_id in self.narrative.life_periods[life_period]:
                    if event_id in self.narrative.events:
                        period_events[event_id] = self.narrative.events[event_id].dict()
                        
                return {
                    "status": "success",
                    "operation": "query_narrative",
                    "query_type": "events",
                    "life_period": life_period,
                    "events": period_events,
                    "count": len(period_events),
                    "process_id": process_id
                }
            else:
                # Return all events
                return {
                    "status": "success",
                    "operation": "query_narrative",
                    "query_type": "events",
                    "events": {id: event.dict() for id, event in self.narrative.events.items()},
                    "count": len(self.narrative.events),
                    "process_id": process_id
                }
                
        elif query_type == "themes":
            # Return all themes
            return {
                "status": "success",
                "operation": "query_narrative",
                "query_type": "themes",
                "themes": {id: theme.dict() for id, theme in self.narrative.themes.items()},
                "count": len(self.narrative.themes),
                "process_id": process_id
            }
            
        elif query_type == "coherence":
            # Return narrative coherence information
            return {
                "status": "success",
                "operation": "query_narrative",
                "query_type": "coherence",
                "coherence": self.narrative.coherence,
                "emotional_tone": self.narrative.emotional_tone,
                "agency": self.narrative.agency,
                "process_id": process_id
            }
            
        elif query_type == "all":
            # Return complete narrative state
            return {
                "status": "success",
                "operation": "query_narrative",
                "query_type": "all",
                "narrative": self.narrative.dict(),
                "available_life_periods": self.available_life_periods,
                "development_level": self.development_level,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _auto_extract_themes(self):
        """Automatically extract themes from events at higher development levels"""
        # Require sufficient development and events
        if self.development_level < 0.6 or len(self.narrative.events) < 3:
            return
            
        # Group events by life period
        period_events = {}
        for event_id, event in self.narrative.events.items():
            if event.age_period not in period_events:
                period_events[event.age_period] = []
            period_events[event.age_period].append(event_id)
            
        # Extract themes for each period with enough events
        for period, event_ids in period_events.items():
            if len(event_ids) >= 3:
                # Check if we already have a theme for this period
                period_themes = []
                for theme_id, theme in self.narrative.themes.items():
                    if all(event_id in theme.events for event_id in event_ids):
                        period_themes.append(theme_id)
                        
                # Skip if we already have a theme for this period
                if period_themes:
                    continue
                    
                # Extract a theme
                self._extract_theme({
                    "event_ids": event_ids,
                    "name": f"Theme from {period}"
                }, str(uuid.uuid4()))
    
    def _update_narrative_coherence(self):
        """Update narrative coherence metrics"""
        # Skip if not enough events
        if len(self.narrative.events) < 2:
            self.narrative.coherence = 0.5
            self.narrative.emotional_tone = 0.0
            self.narrative.agency = 0.5
            return
            
        # Use neural network to evaluate coherence if development permits
        if self.development_level >= 0.6 and self.event_embeddings:
            # Create tensor of all event embeddings
            event_embeddings = list(self.event_embeddings.values())
            if event_embeddings:
                all_events_tensor = torch.stack(event_embeddings, dim=0).unsqueeze(0)  # [1, num_events, hidden_dim]
                
                with torch.no_grad():
                    coherence_output = self.network(
                        torch.zeros((1, 128), device=self.device),  # Dummy input
                        operation="evaluate_coherence",
                        past_events=all_events_tensor.to(self.device)
                    )
                    
                # Update coherence value
                self.narrative.coherence = coherence_output["coherence"].item()
        else:
            # Basic coherence calculation based on number of themes
            themes_ratio = min(1.0, len(self.narrative.themes) / max(1, len(self.narrative.events) / 3))
            self.narrative.coherence = 0.3 + 0.4 * themes_ratio
            
        # Calculate emotional tone across all events
        if self.narrative.events:
            total_tone = 0.0
            emotion_count = 0
            for event in self.narrative.events.values():
                for emotion, intensity in event.emotional_impact.items():
                    # Simple mapping of common emotions to valence
                    if emotion in ["joy", "happiness", "excitement", "satisfaction", "pride"]:
                        total_tone += intensity
                        emotion_count += 1
                    elif emotion in ["sadness", "fear", "anger", "disappointment", "shame"]:
                        total_tone -= intensity
                        emotion_count += 1
            
            if emotion_count > 0:
                self.narrative.emotional_tone = max(-1.0, min(1.0, total_tone / emotion_count))
                
        # Calculate agency based on content (very simplistic)
        if self.development_level >= 0.4:
            self.narrative.agency = min(0.8, 0.4 + self.development_level * 0.4)
        else:
            self.narrative.agency = 0.4
            
        # Update timestamp
        self.narrative.last_updated = datetime.now()
    
    def _get_narrative_theme_from_embedding(self, embedding: torch.Tensor) -> str:
        """Generate a theme name from an embedding"""
        # Basic theme possibilities
        basic_themes = [
            "personal growth", "achievement", "relationships", "challenge", 
            "change", "learning", "responsibility", "identity", "connection"
        ]
        
        # For demonstration, use the embedding to select a theme
        # In a real implementation, this would use more sophisticated methods
        if isinstance(embedding, torch.Tensor):
            embedding_sum = torch.sum(embedding).item()
            theme_index = abs(hash(str(embedding_sum))) % len(basic_themes)
            return basic_themes[theme_index]
        else:
            # Random fallback
            return basic_themes[abs(hash(str(time.time()))) % len(basic_themes)]
    
    def _extract_features(self, data) -> torch.Tensor:
        """
        Extract features from input data for neural processing
        
        Args:
            data: Text or other data to extract features from
            
        Returns:
            Tensor of features [1, feature_dim]
        """
        # For demonstration, create simple random features
        # In a real implementation, this would use proper feature extraction
        feature_dim = 128  # This should match the input_dim of the NarrativeNetwork
        
        if isinstance(data, str):
            # Seed random generator with hash of string to ensure consistent features
            seed = hash(data) % 10000
            np.random.seed(seed)
            
            # Generate "features" based on the text
            features = np.random.randn(1, feature_dim)  # Add batch dimension
            features = features / np.linalg.norm(features)  # Normalize
            
        elif isinstance(data, dict):
            # For dictionary data, use keys and values to generate features
            seed = hash(str(sorted(data.items()))) % 10000
            np.random.seed(seed)
            
            # Generate "features" based on the dictionary
            features = np.random.randn(1, feature_dim)  # Add batch dimension
            features = features / np.linalg.norm(features)  # Normalize
            
        else:
            # Default random features
            features = np.random.randn(1, feature_dim)  # Add batch dimension
            features = features / np.linalg.norm(features)  # Normalize
        
        return torch.FloatTensor(features)
    
    def update_development(self, amount: float) -> float:
        """
        Update the developmental level of this module
        
        Args:
            amount: Amount to increase development
            
        Returns:
            New developmental level
        """
        old_level = self.development_level
        
        # Update base development level
        new_level = super().update_development(amount)
        
        # Update network development level
        self.network.set_development_level(new_level)
        
        # Update neural state
        self.neural_state.narrative_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # If crossing a developmental threshold, adjust available life periods
        if int(old_level * 5) != int(new_level * 5):
            self._adjust_life_periods_for_development()
            
            # Re-evaluate narrative coherence with new capabilities
            self._update_narrative_coherence()
        
        logger.info(f"Personal narrative development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Get base state from parent
        base_state = super().get_state()
        
        # Add narrative-specific state
        narrative_dict = self.narrative.dict()
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.narrative_development,
            "accuracy": self.neural_state.narrative_accuracy,
            "recent_activations_count": len(self.neural_state.recent_narrative_activations)
        }
        
        # Combine states
        combined_state = {
            **base_state, 
            "narrative": narrative_dict,
            "available_life_periods": self.available_life_periods,
            "neural_state": neural_state
        }
        
        return combined_state
