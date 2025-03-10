# TODO: Implement the Preferences class to track likes, dislikes, and values
# This component should be able to:
# - Represent preferences across different domains
# - Update preferences based on experiences
# - Form preference hierarchies and priorities
# - Generate preference-based choices

# TODO: Implement developmental progression in preferences:
# - Simple approach/avoid preferences in early stages
# - Concrete likes and dislikes in childhood
# - Value-based preferences in adolescence
# - Stable yet flexible preference systems in adulthood

# TODO: Create mechanisms for:
# - Preference formation: Develop likes/dislikes from experiences
# - Preference integration: Organize preferences into coherent systems
# - Value extraction: Derive abstract values from concrete preferences
# - Preference application: Use preferences to guide decisions

# TODO: Implement different preference types:
# - Sensory preferences: Likes/dislikes for physical sensations
# - Activity preferences: Preferred activities and pastimes
# - Social preferences: Preferred interaction styles and partners
# - Abstract preferences: Values and principles

# TODO: Connect to emotion and memory systems
# Preferences should be influenced by emotional responses
# and should draw on memories of past experiences

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
from lmm_project.modules.identity.models import Preference, Value, PreferenceSystem, IdentityNeuralState
from lmm_project.modules.identity.neural_net import PreferenceNetwork, get_device

# Initialize logger
logger = logging.getLogger(__name__)

class Preferences(BaseModule):
    """
    Manages preferences, likes, dislikes, and values
    
    This module tracks preferences across different domains and
    extracts higher-level values from preference patterns.
    """
    
    # Development milestones
    development_milestones = {
        0.0: "Basic approach/avoid preferences",
        0.2: "Domain-specific likes and dislikes",
        0.4: "Preference hierarchies",
        0.6: "Abstract value formation",
        0.8: "Integrated value system",
        1.0: "Sophisticated preference/value system"
    }
    
    def __init__(self, module_id: str, event_bus: Optional[EventBus] = None, development_level: float = 0.0):
        """
        Initialize the preferences module
        
        Args:
            module_id: Unique identifier for this module
            event_bus: Event bus for communication with other modules
            development_level: Initial developmental level of this module
        """
        super().__init__(
            module_id=module_id, 
            module_type="preferences", 
            event_bus=event_bus,
            development_level=development_level
        )
        
        # Initialize device for neural processing
        self.device = get_device()
        
        # Initialize neural network
        self.network = PreferenceNetwork().to(self.device)
        self.network.set_development_level(development_level)
        
        # Initialize preference system
        self.preference_system = PreferenceSystem()
        
        # Initialize neural state for tracking
        self.neural_state = IdentityNeuralState()
        self.neural_state.preference_development = development_level
        
        # Available preference domains (expands with development)
        self.available_domains = ["sensory"]
        self._adjust_domains_for_development()
        
        # Recent processing queue
        self.recent_inputs = deque(maxlen=100)
        
        logger.info(f"Preferences module initialized at development level {development_level:.2f}")
    
    def _adjust_domains_for_development(self):
        """Adjust available preference domains based on developmental level"""
        if self.development_level < 0.2:
            # Very basic sensory preferences at early stages
            self.available_domains = ["sensory"]
            
        elif self.development_level < 0.4:
            # Basic concrete preferences
            self.available_domains = ["sensory", "food", "activities", "people"]
            
        elif self.development_level < 0.6:
            # More differentiated preference domains
            self.available_domains = ["sensory", "food", "activities", "people", "entertainment", "aesthetics"]
            
        elif self.development_level < 0.8:
            # Higher-level preference domains
            self.available_domains = [
                "sensory", "food", "activities", "people", "entertainment", 
                "aesthetics", "social", "intellectual", "achievement"
            ]
            
        else:
            # Abstract and value-based preference domains
            self.available_domains = [
                "sensory", "food", "activities", "people", "entertainment", 
                "aesthetics", "social", "intellectual", "achievement", 
                "moral", "political", "spiritual", "personal_growth"
            ]
            
        logger.info(f"Preference domains adjusted to: {self.available_domains}")
    
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input to update preferences
        
        Args:
            input_data: Dictionary containing preference operations
                Required keys:
                - 'operation': The operation to perform
                  Options: 'add_preference', 'update_preference', 'extract_value', 'query_preferences'
                
                For 'add_preference' operation:
                - 'domain': Domain of the preference (e.g., 'food', 'activities')
                - 'target': Target of the preference
                - 'valence': (Optional) Degree of liking/disliking (-1.0 to 1.0)
                - 'reasons': (Optional) Reasons for this preference
                - 'experiences': (Optional) Experiences related to this preference
                
                For 'update_preference' operation:
                - 'preference_id': ID of the preference to update
                - 'valence': (Optional) Updated degree of liking/disliking
                - 'strength': (Optional) Updated strength of preference
                - 'reasons': (Optional) Additional reasons
                
                For 'extract_value' operation:
                - 'preference_ids': List of preference IDs to analyze for underlying values
                - 'name': (Required) Name of the value
                - 'description': (Required) Description of the value
                
                For 'query_preferences' operation:
                - 'query_type': Type of query ('preferences', 'values', 'domains', 'all')
                - 'domain': (Optional) Domain to filter by
            
        Returns:
            Dictionary with the results of preference processing
        """
        operation = input_data.get("operation", "query_preferences")
        process_id = input_data.get("process_id", str(uuid.uuid4()))
        
        # Process based on operation
        if operation == "add_preference":
            return self._add_preference(input_data, process_id)
        elif operation == "update_preference":
            return self._update_preference(input_data, process_id)
        elif operation == "extract_value":
            return self._extract_value(input_data, process_id)
        elif operation == "query_preferences":
            return self._query_preferences(input_data, process_id)
        else:
            return {
                "status": "error",
                "message": f"Unknown operation: {operation}",
                "process_id": process_id
            }
    
    def _add_preference(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Add a new preference"""
        # Extract required data
        domain = input_data.get("domain")
        target = input_data.get("target")
        
        if not domain:
            return {"status": "error", "message": "No domain provided", "process_id": process_id}
        if not target:
            return {"status": "error", "message": "No target provided", "process_id": process_id}
            
        # Check if domain is available at current development level
        if domain not in self.available_domains:
            return {
                "status": "undeveloped",
                "message": f"Domain '{domain}' not available at current development level",
                "available_domains": self.available_domains,
                "process_id": process_id
            }
            
        # Extract optional data
        valence = input_data.get("valence", 0.0)
        reasons = input_data.get("reasons", [])
        related_experiences = input_data.get("experiences", [])
        
        # Process preference through neural network
        input_features = self._extract_features(target)
        
        with torch.no_grad():
            network_output = self.network(
                input_features.to(self.device),
                operation="form_preference"
            )
        
        # Create new preference
        preference_id = str(uuid.uuid4())
        
        # Apply development-specific processing
        if self.development_level < 0.2:
            # At early development, preferences are simple and binary
            valence = 1.0 if valence > 0 else -1.0
            strength = 1.0
            certainty = 1.0
        else:
            # More nuanced preferences with development
            # Blend provided valence with network output
            if "valence" in input_data:
                valence = valence * 0.7 + network_output["valence"].item() * 0.3
            else:
                valence = network_output["valence"].item()
                
            # Get strength and certainty from network
            strength = network_output["strength"].item()
            certainty = network_output["certainty"].item()
            
        # Ensure values are in appropriate ranges
        valence = max(-1.0, min(1.0, valence))
        strength = max(0.0, min(1.0, strength))
        certainty = max(0.0, min(1.0, certainty))
        
        # Create preference
        new_preference = Preference(
            preference_id=preference_id,
            domain=domain,
            target=target,
            valence=valence,
            strength=strength,
            certainty=certainty,
            reasons=reasons if isinstance(reasons, list) else [reasons],
            related_experiences=related_experiences if isinstance(related_experiences, list) else [related_experiences]
        )
        
        # Add to preference system
        self.preference_system.add_preference(new_preference)
        
        # Record activation in neural state
        self.neural_state.add_activation('preference', {
            'operation': 'add_preference',
            'domain': domain,
            'valence': valence,
            'strength': strength
        })
        
        # Check for value extraction at higher development levels
        if self.development_level >= 0.6 and len(self.preference_system.preferences) >= 5:
            # Attempt automatic value extraction
            self._auto_extract_values()
        
        # Add to recent inputs
        self.recent_inputs.append({
            "type": "add_preference",
            "data": input_data,
            "timestamp": datetime.now()
        })
        
        # Update consistency score
        self._update_preference_consistency()
        
        return {
            "status": "success",
            "preference_id": preference_id,
            "operation": "add_preference",
            "preference": new_preference.dict(),
            "process_id": process_id
        }
    
    def _update_preference(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Update an existing preference"""
        # Extract data
        preference_id = input_data.get("preference_id")
        
        if not preference_id:
            return {"status": "error", "message": "No preference_id provided", "process_id": process_id}
            
        # Check if preference exists
        if preference_id not in self.preference_system.preferences:
            return {
                "status": "error", 
                "message": f"Preference with ID {preference_id} not found", 
                "process_id": process_id
            }
            
        # Get the existing preference
        preference = self.preference_system.preferences[preference_id]
        
        # Update fields if provided
        updated = False
        
        if "valence" in input_data:
            valence = max(-1.0, min(1.0, float(input_data["valence"])))
            if self.development_level < 0.2:
                # At early development, preferences are binary
                valence = 1.0 if valence > 0 else -1.0
            preference.valence = valence
            updated = True
            
        if "strength" in input_data and self.development_level >= 0.2:
            preference.strength = max(0.0, min(1.0, float(input_data["strength"])))
            updated = True
            
        if "certainty" in input_data and self.development_level >= 0.4:
            preference.certainty = max(0.0, min(1.0, float(input_data["certainty"])))
            updated = True
            
        if "reasons" in input_data:
            new_reasons = input_data["reasons"]
            if isinstance(new_reasons, list):
                preference.reasons.extend(new_reasons)
            else:
                preference.reasons.append(new_reasons)
            updated = True
            
        if "experiences" in input_data:
            new_experiences = input_data["experiences"]
            if isinstance(new_experiences, list):
                preference.related_experiences.extend(new_experiences)
            else:
                preference.related_experiences.append(new_experiences)
            updated = True
            
        if "target" in input_data:
            preference.target = input_data["target"]
            updated = True
        
        if updated:
            # Update timestamp
            preference.updated_at = datetime.now()
            
            # Update in preference system
            self.preference_system.preferences[preference_id] = preference
            self.preference_system.last_updated = datetime.now()
            
            # Record activation in neural state
            self.neural_state.add_activation('preference', {
                'operation': 'update_preference',
                'domain': preference.domain,
                'preference_id': preference_id
            })
            
            # Update consistency score
            self._update_preference_consistency()
            
            # Add to recent inputs
            self.recent_inputs.append({
                "type": "update_preference",
                "data": input_data,
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "preference_id": preference_id,
                "operation": "update_preference",
                "preference": preference.dict(),
                "process_id": process_id
            }
        else:
            return {
                "status": "not_modified",
                "message": "No changes were made to the preference",
                "process_id": process_id
            }
    
    def _extract_value(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Extract a value from a set of preferences"""
        # Value extraction requires higher development
        if self.development_level < 0.6:
            return {
                "status": "undeveloped",
                "message": "Value extraction requires higher developmental level",
                "development_level": self.development_level,
                "required_level": 0.6,
                "process_id": process_id
            }
            
        # Extract data
        preference_ids = input_data.get("preference_ids", [])
        name = input_data.get("name")
        description = input_data.get("description")
        
        if not preference_ids:
            return {"status": "error", "message": "No preference_ids provided", "process_id": process_id}
        if not name:
            return {"status": "error", "message": "No name provided", "process_id": process_id}
        if not description:
            return {"status": "error", "message": "No description provided", "process_id": process_id}
            
        # Check if preferences exist
        valid_preferences = []
        for pref_id in preference_ids:
            if pref_id in self.preference_system.preferences:
                valid_preferences.append(pref_id)
            
        if not valid_preferences:
            return {
                "status": "error", 
                "message": "None of the provided preference IDs were found", 
                "process_id": process_id
            }
            
        # Process through neural network
        # Combine embeddings from all preferences
        preference_embeddings = []
        for pref_id in valid_preferences:
            pref = self.preference_system.preferences[pref_id]
            input_features = self._extract_features(pref.target)
            
            with torch.no_grad():
                preference_output = self.network(
                    input_features.to(self.device),
                    operation="extract_value"
                )
                
            preference_embeddings.append(preference_output["value_vector"].cpu().squeeze(0))
        
        # Average the embeddings
        if preference_embeddings:
            avg_embedding = torch.stack(preference_embeddings).mean(dim=0)
            
            # Calculate importance based on network output and preference strengths
            importance_sum = 0.0
            for pref_id in valid_preferences:
                pref = self.preference_system.preferences[pref_id]
                importance_sum += pref.strength * pref.certainty
            
            if len(valid_preferences) > 0:
                importance_base = importance_sum / len(valid_preferences)
            else:
                importance_base = 0.5
                
            # Create the value
            value_id = str(uuid.uuid4())
            new_value = Value(
                value_id=value_id,
                name=name,
                description=description,
                importance=importance_base,
                related_preferences=valid_preferences
            )
            
            # Add to preference system
            self.preference_system.add_value(new_value)
            
            # Record activation in neural state
            self.neural_state.add_activation('preference', {
                'operation': 'extract_value',
                'value_name': name,
                'preference_count': len(valid_preferences),
                'importance': importance_base
            })
            
            # Update consistency score
            self._update_preference_consistency()
            
            # Add to recent inputs
            self.recent_inputs.append({
                "type": "extract_value",
                "data": input_data,
                "timestamp": datetime.now()
            })
            
            return {
                "status": "success",
                "value_id": value_id,
                "operation": "extract_value",
                "value": new_value.dict(),
                "process_id": process_id
            }
        else:
            return {
                "status": "error",
                "message": "Could not process preferences to extract value",
                "process_id": process_id
            }
    
    def _query_preferences(self, input_data: Dict[str, Any], process_id: str) -> Dict[str, Any]:
        """Query preference system information"""
        query_type = input_data.get("query_type", "all")
        domain = input_data.get("domain")
        
        if query_type == "preferences":
            # Return preferences, filtered by domain if specified
            if domain:
                if domain not in self.preference_system.domains:
                    return {
                        "status": "not_found",
                        "message": f"No preferences found in domain '{domain}'",
                        "process_id": process_id
                    }
                    
                # Get preferences in the specified domain
                domain_preferences = {}
                for pref_id in self.preference_system.domains[domain]:
                    if pref_id in self.preference_system.preferences:
                        domain_preferences[pref_id] = self.preference_system.preferences[pref_id].dict()
                        
                return {
                    "status": "success",
                    "operation": "query_preferences",
                    "query_type": "preferences",
                    "domain": domain,
                    "preferences": domain_preferences,
                    "count": len(domain_preferences),
                    "process_id": process_id
                }
            else:
                # Return all preferences
                return {
                    "status": "success",
                    "operation": "query_preferences",
                    "query_type": "preferences",
                    "preferences": {id: pref.dict() for id, pref in self.preference_system.preferences.items()},
                    "count": len(self.preference_system.preferences),
                    "process_id": process_id
                }
                
        elif query_type == "values":
            # Value queries require higher development
            if self.development_level < 0.6:
                return {
                    "status": "undeveloped",
                    "message": "Value queries require higher developmental level",
                    "development_level": self.development_level,
                    "required_level": 0.6,
                    "process_id": process_id
                }
                
            # Return all values
            return {
                "status": "success",
                "operation": "query_preferences",
                "query_type": "values",
                "values": {id: value.dict() for id, value in self.preference_system.values.items()},
                "count": len(self.preference_system.values),
                "value_hierarchy": [self.preference_system.values[v_id].dict() if v_id in self.preference_system.values else None 
                                   for v_id in self.preference_system.value_hierarchy],
                "process_id": process_id
            }
            
        elif query_type == "domains":
            # Return available domains and preferences per domain
            result = {
                "status": "success",
                "operation": "query_preferences",
                "query_type": "domains",
                "available_domains": self.available_domains,
                "domain_counts": {d: len(ids) for d, ids in self.preference_system.domains.items()},
                "process_id": process_id
            }
            
            return result
            
        elif query_type == "all":
            # Return complete preference system state
            return {
                "status": "success",
                "operation": "query_preferences",
                "query_type": "all",
                "preference_system": self.preference_system.dict(),
                "available_domains": self.available_domains,
                "development_level": self.development_level,
                "process_id": process_id
            }
            
        else:
            return {
                "status": "error",
                "message": f"Unknown query_type: {query_type}",
                "process_id": process_id
            }
    
    def _auto_extract_values(self):
        """
        Automatically attempt to extract values from preference patterns
        
        This is used at higher development levels to find emerging values
        """
        # Require higher development level
        if self.development_level < 0.6:
            return
            
        # Group preferences by domain
        domain_preferences = {}
        for pref_id, pref in self.preference_system.preferences.items():
            if pref.domain not in domain_preferences:
                domain_preferences[pref.domain] = []
            domain_preferences[pref.domain].append(pref_id)
            
        # Look for domains with enough preferences for value extraction
        for domain, pref_ids in domain_preferences.items():
            if len(pref_ids) >= 3:
                # Check valence patterns
                positive_prefs = []
                negative_prefs = []
                
                for pref_id in pref_ids:
                    pref = self.preference_system.preferences[pref_id]
                    if pref.valence > 0.3:
                        positive_prefs.append(pref_id)
                    elif pref.valence < -0.3:
                        negative_prefs.append(pref_id)
                
                # Extract values from strong preference patterns
                if len(positive_prefs) >= 3:
                    # Check if these preferences already have a common value
                    has_common_value = False
                    for value in self.preference_system.values.values():
                        common_prefs = set(value.related_preferences).intersection(set(positive_prefs))
                        if len(common_prefs) >= 3:
                            has_common_value = True
                            break
                            
                    if not has_common_value:
                        # Generate a value
                        if domain == "food":
                            value_name = "Culinary enjoyment"
                            description = "Appreciating good food and diverse tastes"
                        elif domain == "activities":
                            value_name = "Active engagement"
                            description = "Enjoying participatory and engaging activities"
                        elif domain == "social":
                            value_name = "Social connection"
                            description = "Valuing meaningful relationships with others"
                        elif domain == "intellectual":
                            value_name = "Intellectual growth"
                            description = "Valuing learning and mental stimulation"
                        elif domain == "aesthetics":
                            value_name = "Aesthetic appreciation"
                            description = "Valuing beauty and artistic expression"
                        else:
                            value_name = f"{domain.capitalize()} appreciation"
                            description = f"Valuing positive experiences in {domain}"
                            
                        # Extract the value
                        self._extract_value({
                            "preference_ids": positive_prefs,
                            "name": value_name,
                            "description": description
                        }, str(uuid.uuid4()))
    
    def _update_preference_consistency(self):
        """Update preference consistency score"""
        # Skip if not enough preferences
        if len(self.preference_system.preferences) < 3:
            self.preference_system.consistency = 0.5
            return
            
        # Basic consistency calculation based on preference patterns
        # Group by domain
        domain_preferences = {}
        for pref_id, pref in self.preference_system.preferences.items():
            if pref.domain not in domain_preferences:
                domain_preferences[pref.domain] = []
            domain_preferences[pref.domain].append(pref)
        
        # Calculate consistency within domains
        domain_consistency = {}
        for domain, prefs in domain_preferences.items():
            if len(prefs) < 2:
                domain_consistency[domain] = 1.0  # No inconsistency with single preference
                continue
                
            # Calculate variance in valence for similar targets
            targets = {}
            for pref in prefs:
                # Simple grouping by target similarity (in a real system, would use semantic similarity)
                target_key = pref.target.lower()[:5]  # Simple prefix grouping
                if target_key not in targets:
                    targets[target_key] = []
                targets[target_key].append(pref)
            
            # Calculate consistency based on valence agreement within target groups
            if not targets:
                domain_consistency[domain] = 1.0
                continue
                
            inconsistency_sum = 0
            group_count = 0
            
            for target_group in targets.values():
                if len(target_group) > 1:
                    # Calculate variance in valence
                    valences = [pref.valence for pref in target_group]
                    mean_valence = sum(valences) / len(valences)
                    variance = sum((v - mean_valence) ** 2 for v in valences) / len(valences)
                    
                    inconsistency_sum += variance
                    group_count += 1
                    
            # Convert to consistency score
            if group_count > 0:
                domain_consistency[domain] = max(0, 1.0 - inconsistency_sum / group_count)
            else:
                domain_consistency[domain] = 1.0
                
        # Average domain consistencies
        if domain_consistency:
            consistency = sum(domain_consistency.values()) / len(domain_consistency)
        else:
            consistency = 0.5
            
        # Adjust based on value integration at higher development levels
        if self.development_level >= 0.8 and self.preference_system.values:
            # Calculate the proportion of preferences that are connected to values
            preferences_with_values = set()
            for value in self.preference_system.values.values():
                preferences_with_values.update(value.related_preferences)
                
            value_coverage = len(preferences_with_values) / max(1, len(self.preference_system.preferences))
            
            # Blend consistency with value coverage
            consistency = consistency * 0.7 + value_coverage * 0.3
            
        # Update consistency score
        self.preference_system.consistency = max(0.0, min(1.0, consistency))
        self.preference_system.last_updated = datetime.now()
    
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
        feature_dim = 128
        
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
        self.neural_state.preference_development = new_level
        self.neural_state.last_updated = datetime.now()
        
        # If crossing a developmental threshold, adjust available domains
        if int(old_level * 5) != int(new_level * 5):
            self._adjust_domains_for_development()
            
            # Re-evaluate preference consistency with new capabilities
            self._update_preference_consistency()
            
            # Try to extract values if development is sufficient
            if new_level >= 0.6 and old_level < 0.6:
                self._auto_extract_values()
        
        logger.info(f"Preferences development updated to {new_level:.2f}")
        return new_level
    
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the module
        
        Returns:
            Dictionary containing current module state
        """
        # Get base state from parent
        base_state = super().get_state()
        
        # Add preference-specific state
        preference_system_dict = self.preference_system.dict()
        
        # Add neural state
        neural_state = {
            "development_level": self.neural_state.preference_development,
            "accuracy": self.neural_state.preference_accuracy,
            "recent_activations_count": len(self.neural_state.recent_preference_activations)
        }
        
        # Combine states
        combined_state = {
            **base_state, 
            "preference_system": preference_system_dict,
            "available_domains": self.available_domains,
            "neural_state": neural_state
        }
        
        return combined_state
