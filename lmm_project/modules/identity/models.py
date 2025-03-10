from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime
import uuid
from pydantic import BaseModel, Field

class SelfAttribute(BaseModel):
    """
    Represents a single attribute of the self-concept
    
    Attributes include beliefs about the self in various domains
    """
    attribute_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this attribute")
    domain: str = Field(..., description="Domain of this attribute (e.g., 'physical', 'academic', 'social')")
    content: str = Field(..., description="Content of the self-attribute")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="Confidence in this self-belief")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Personal importance of this attribute")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Emotional valence of this attribute")
    evidence: List[str] = Field(default_factory=list, description="Supporting evidence for this attribute")
    sources: List[str] = Field(default_factory=list, description="Sources of this belief (e.g., 'personal experience', 'social feedback')")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class SelfConcept(BaseModel):
    """
    Represents the overall self-concept
    
    The self-concept is the collection of beliefs about oneself
    """
    attributes: Dict[str, SelfAttribute] = Field(default_factory=dict, description="Self-attributes by ID")
    domains: Dict[str, List[str]] = Field(default_factory=dict, description="Attribute IDs organized by domain")
    global_self_esteem: float = Field(0.5, ge=0.0, le=1.0, description="Overall evaluation of self-worth")
    clarity: float = Field(0.5, ge=0.0, le=1.0, description="Clarity and coherence of self-concept")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Stability of self-concept over time")
    complexity: float = Field(0.5, ge=0.0, le=1.0, description="Complexity and differentiation of self-concept")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result
    
    def add_attribute(self, attribute: SelfAttribute) -> None:
        """
        Add an attribute to the self-concept
        
        Args:
            attribute: The attribute to add
        """
        self.attributes[attribute.attribute_id] = attribute
        
        # Add to domain mapping
        if attribute.domain not in self.domains:
            self.domains[attribute.domain] = []
        
        if attribute.attribute_id not in self.domains[attribute.domain]:
            self.domains[attribute.domain].append(attribute.attribute_id)
            
        self.last_updated = datetime.now()

class NarrativeEvent(BaseModel):
    """
    Represents a significant event in the personal narrative
    
    Events include important personal experiences and their interpretation
    """
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this event")
    title: str = Field(..., description="Brief title for this event")
    description: str = Field(..., description="Description of what happened")
    interpretation: str = Field("", description="Personal meaning or interpretation of the event")
    emotional_impact: Dict[str, float] = Field(default_factory=dict, description="Emotional impact (emotion: intensity)")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Subjective importance of this event")
    age_period: str = Field(..., description="Life period when this event occurred (e.g., 'early childhood')")
    themes: List[str] = Field(default_factory=list, description="Themes associated with this event")
    related_events: List[str] = Field(default_factory=list, description="IDs of related events")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class NarrativeTheme(BaseModel):
    """
    Represents a recurring theme in the personal narrative
    
    Themes connect events through common patterns or meanings
    """
    theme_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this theme")
    name: str = Field(..., description="Name of this theme")
    description: str = Field(..., description="Description of what this theme represents")
    events: List[str] = Field(default_factory=list, description="IDs of events associated with this theme")
    emotional_tone: float = Field(0.0, ge=-1.0, le=1.0, description="Overall emotional tone of this theme")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Subjective importance of this theme")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class PersonalNarrative(BaseModel):
    """
    Represents the autobiographical narrative
    
    The personal narrative is the story one constructs about oneself
    """
    events: Dict[str, NarrativeEvent] = Field(default_factory=dict, description="Narrative events by ID")
    themes: Dict[str, NarrativeTheme] = Field(default_factory=dict, description="Narrative themes by ID")
    life_periods: Dict[str, List[str]] = Field(default_factory=dict, description="Event IDs organized by life period")
    coherence: float = Field(0.5, ge=0.0, le=1.0, description="Narrative coherence and integration")
    emotional_tone: float = Field(0.0, ge=-1.0, le=1.0, description="Overall emotional tone of the narrative")
    agency: float = Field(0.5, ge=0.0, le=1.0, description="Sense of personal agency in the narrative")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result
    
    def add_event(self, event: NarrativeEvent) -> None:
        """
        Add an event to the personal narrative
        
        Args:
            event: The event to add
        """
        self.events[event.event_id] = event
        
        # Add to life period mapping
        if event.age_period not in self.life_periods:
            self.life_periods[event.age_period] = []
        
        if event.event_id not in self.life_periods[event.age_period]:
            self.life_periods[event.age_period].append(event.event_id)
        
        # Update related themes
        for theme_id in event.themes:
            if theme_id in self.themes:
                if event.event_id not in self.themes[theme_id].events:
                    theme = self.themes[theme_id]
                    theme.events.append(event.event_id)
                    theme.updated_at = datetime.now()
                    self.themes[theme_id] = theme
            
        self.last_updated = datetime.now()
    
    def add_theme(self, theme: NarrativeTheme) -> None:
        """
        Add a theme to the personal narrative
        
        Args:
            theme: The theme to add
        """
        self.themes[theme.theme_id] = theme
        
        # Update related events
        for event_id in theme.events:
            if event_id in self.events:
                if theme.theme_id not in self.events[event_id].themes:
                    event = self.events[event_id]
                    event.themes.append(theme.theme_id)
                    event.updated_at = datetime.now()
                    self.events[event_id] = event
        
        self.last_updated = datetime.now()

class Preference(BaseModel):
    """
    Represents a preference for or against something
    
    Preferences include likes, dislikes, and value judgments
    """
    preference_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this preference")
    domain: str = Field(..., description="Domain of this preference (e.g., 'food', 'music', 'activities')")
    target: str = Field(..., description="Target of the preference")
    valence: float = Field(0.0, ge=-1.0, le=1.0, description="Degree of liking/disliking")
    strength: float = Field(0.5, ge=0.0, le=1.0, description="Strength of the preference")
    certainty: float = Field(0.5, ge=0.0, le=1.0, description="Certainty of the preference")
    reasons: List[str] = Field(default_factory=list, description="Reasons for this preference")
    related_experiences: List[str] = Field(default_factory=list, description="Related experiences that shaped this preference")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class Value(BaseModel):
    """
    Represents a personal value
    
    Values are abstract principles or qualities that are important to the individual
    """
    value_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this value")
    name: str = Field(..., description="Name of this value")
    description: str = Field(..., description="Description of what this value represents")
    importance: float = Field(0.5, ge=0.0, le=1.0, description="Importance of this value")
    related_preferences: List[str] = Field(default_factory=list, description="IDs of preferences related to this value")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class PreferenceSystem(BaseModel):
    """
    Represents the system of preferences and values
    
    The preference system organizes likes, dislikes, and values
    """
    preferences: Dict[str, Preference] = Field(default_factory=dict, description="Preferences by ID")
    values: Dict[str, Value] = Field(default_factory=dict, description="Values by ID")
    domains: Dict[str, List[str]] = Field(default_factory=dict, description="Preference IDs organized by domain")
    consistency: float = Field(0.5, ge=0.0, le=1.0, description="Consistency of preferences across domains")
    value_hierarchy: List[str] = Field(default_factory=list, description="Value IDs in order of importance")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result
    
    def add_preference(self, preference: Preference) -> None:
        """
        Add a preference to the system
        
        Args:
            preference: The preference to add
        """
        self.preferences[preference.preference_id] = preference
        
        # Add to domain mapping
        if preference.domain not in self.domains:
            self.domains[preference.domain] = []
        
        if preference.preference_id not in self.domains[preference.domain]:
            self.domains[preference.domain].append(preference.preference_id)
            
        self.last_updated = datetime.now()
    
    def add_value(self, value: Value) -> None:
        """
        Add a value to the system
        
        Args:
            value: The value to add
        """
        self.values[value.value_id] = value
        
        # Update value hierarchy
        if value.value_id not in self.value_hierarchy:
            # Insert based on importance
            for i, v_id in enumerate(self.value_hierarchy):
                if self.values[v_id].importance < value.importance:
                    self.value_hierarchy.insert(i, value.value_id)
                    break
            else:
                self.value_hierarchy.append(value.value_id)
                
        self.last_updated = datetime.now()

class PersonalityTrait(BaseModel):
    """
    Represents a personality trait
    
    Traits are stable patterns of thinking, feeling, and behaving
    """
    trait_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this trait")
    name: str = Field(..., description="Name of this trait")
    description: str = Field(..., description="Description of what this trait represents")
    score: float = Field(..., ge=0.0, le=1.0, description="Strength of this trait")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Stability of this trait over time")
    behavioral_instances: List[str] = Field(default_factory=list, description="Examples of behaviors reflecting this trait")
    opposing_trait: Optional[str] = Field(None, description="ID of the opposing trait, if any")
    dimension: Optional[str] = Field(None, description="Dimension this trait belongs to (e.g., 'extraversion')")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class TraitDimension(BaseModel):
    """
    Represents a dimension along which traits vary
    
    Dimensions organize traits into bipolar continuums
    """
    dimension_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique identifier for this dimension")
    name: str = Field(..., description="Name of this dimension")
    description: str = Field(..., description="Description of what this dimension represents")
    positive_pole: str = Field(..., description="Description of the high end of this dimension")
    negative_pole: str = Field(..., description="Description of the low end of this dimension")
    traits: List[str] = Field(default_factory=list, description="IDs of traits along this dimension")
    score: float = Field(0.5, ge=0.0, le=1.0, description="Overall score on this dimension")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["created_at"] = self.created_at
        result["updated_at"] = self.updated_at
        return result

class PersonalityProfile(BaseModel):
    """
    Represents the overall personality profile
    
    The personality profile organizes traits into a coherent whole
    """
    traits: Dict[str, PersonalityTrait] = Field(default_factory=dict, description="Traits by ID")
    dimensions: Dict[str, TraitDimension] = Field(default_factory=dict, description="Trait dimensions by ID")
    stability: float = Field(0.5, ge=0.0, le=1.0, description="Overall stability of personality")
    differentiation: float = Field(0.5, ge=0.0, le=1.0, description="Degree of differentiation among traits")
    integration: float = Field(0.5, ge=0.0, le=1.0, description="Degree of integration among traits")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result
    
    def add_trait(self, trait: PersonalityTrait) -> None:
        """
        Add a trait to the profile
        
        Args:
            trait: The trait to add
        """
        self.traits[trait.trait_id] = trait
        
        # Update dimension if applicable
        if trait.dimension and trait.dimension in self.dimensions:
            dimension = self.dimensions[trait.dimension]
            if trait.trait_id not in dimension.traits:
                dimension.traits.append(trait.trait_id)
                dimension.updated_at = datetime.now()
                
                # Recalculate dimension score
                dimension_traits = [self.traits[t] for t in dimension.traits if t in self.traits]
                if dimension_traits:
                    dimension.score = sum(t.score for t in dimension_traits) / len(dimension_traits)
                    
                self.dimensions[trait.dimension] = dimension
                
        self.last_updated = datetime.now()
    
    def add_dimension(self, dimension: TraitDimension) -> None:
        """
        Add a dimension to the profile
        
        Args:
            dimension: The dimension to add
        """
        self.dimensions[dimension.dimension_id] = dimension
        
        # Update traits
        for trait_id in dimension.traits:
            if trait_id in self.traits:
                trait = self.traits[trait_id]
                trait.dimension = dimension.dimension_id
                trait.updated_at = datetime.now()
                self.traits[trait_id] = trait
                
        self.last_updated = datetime.now()

class IdentityState(BaseModel):
    """
    Represents the overall state of identity
    
    Identity integrates self-concept, narrative, preferences, and personality
    """
    self_concept: SelfConcept = Field(default_factory=SelfConcept, description="Current self-concept")
    personal_narrative: PersonalNarrative = Field(default_factory=PersonalNarrative, description="Personal narrative")
    preference_system: PreferenceSystem = Field(default_factory=PreferenceSystem, description="Preferences and values")
    personality_profile: PersonalityProfile = Field(default_factory=PersonalityProfile, description="Personality traits")
    
    identity_integration: float = Field(0.5, ge=0.0, le=1.0, description="Integration across identity components")
    identity_clarity: float = Field(0.5, ge=0.0, le=1.0, description="Clarity and definition of identity")
    identity_stability: float = Field(0.5, ge=0.0, le=1.0, description="Stability of identity over time")
    
    module_id: str = Field(..., description="Module identifier")
    developmental_level: float = Field(0.0, ge=0.0, le=1.0, description="Overall developmental level")
    last_updated: datetime = Field(default_factory=datetime.now)
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result

class IdentityNeuralState(BaseModel):
    """
    Neural state information for identity networks
    
    Tracks the state of neural networks for identity components
    """
    self_concept_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of self-concept network")
    narrative_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of narrative network")
    preference_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of preference network")
    personality_development: float = Field(0.0, ge=0.0, le=1.0, description="Development level of personality network")
    
    # Track recent activations for each neural component
    recent_self_concept_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the self-concept network"
    )
    recent_narrative_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the narrative network"
    )
    recent_preference_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the preference network"
    )
    recent_personality_activations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Recent activations of the personality network"
    )
    
    # Network performance metrics
    self_concept_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of self-concept network")
    narrative_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of narrative network")
    preference_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of preference network")
    personality_accuracy: float = Field(0.5, ge=0.0, le=1.0, description="Accuracy of personality network")
    
    # Last update timestamp
    last_updated: datetime = Field(default_factory=datetime.now, description="When neural state was last updated")
    
    def dict(self, *args, **kwargs):
        """Convert to dictionary, preserving datetime objects"""
        result = super().dict(*args, **kwargs)
        result["last_updated"] = self.last_updated
        return result
    
    def update_accuracy(self, component: str, accuracy: float) -> None:
        """
        Update the accuracy of a component network
        
        Args:
            component: Component to update ('self_concept', 'narrative', 'preference', or 'personality')
            accuracy: New accuracy value (0.0 to 1.0)
        """
        accuracy = max(0.0, min(1.0, accuracy))
        
        if component == "self_concept":
            self.self_concept_accuracy = accuracy
        elif component == "narrative":
            self.narrative_accuracy = accuracy
        elif component == "preference":
            self.preference_accuracy = accuracy
        elif component == "personality":
            self.personality_accuracy = accuracy
            
        self.last_updated = datetime.now()
    
    def add_activation(self, component: str, activation: Dict[str, Any]) -> None:
        """
        Add a network activation record
        
        Args:
            component: Component that was activated ('self_concept', 'narrative', 'preference', or 'personality')
            activation: Activation data
        """
        # Add timestamp if not present
        if "timestamp" not in activation:
            activation["timestamp"] = datetime.now()
            
        if component == "self_concept":
            self.recent_self_concept_activations.append(activation)
            # Keep only the most recent activations (max 100)
            if len(self.recent_self_concept_activations) > 100:
                self.recent_self_concept_activations = self.recent_self_concept_activations[-100:]
                
        elif component == "narrative":
            self.recent_narrative_activations.append(activation)
            # Keep only the most recent activations (max 100)
            if len(self.recent_narrative_activations) > 100:
                self.recent_narrative_activations = self.recent_narrative_activations[-100:]
                
        elif component == "preference":
            self.recent_preference_activations.append(activation)
            # Keep only the most recent activations (max 100)
            if len(self.recent_preference_activations) > 100:
                self.recent_preference_activations = self.recent_preference_activations[-100:]
                
        elif component == "personality":
            self.recent_personality_activations.append(activation)
            # Keep only the most recent activations (max 100)
            if len(self.recent_personality_activations) > 100:
                self.recent_personality_activations = self.recent_personality_activations[-100:]
                
        self.last_updated = datetime.now()
