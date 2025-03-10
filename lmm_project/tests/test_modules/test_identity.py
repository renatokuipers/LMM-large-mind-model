"""
Test script for the Identity module.

This script demonstrates how the identity module processes different 
inputs at various developmental stages, showing how its capabilities
evolve from basic self-recognition to complex, integrated identity.
"""

import logging
import sys
from typing import Dict, Any, List, Optional
import time
import json
import os
from datetime import datetime
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

# Import identity module
from lmm_project.modules.identity import get_module as get_identity_module
from lmm_project.core.event_bus import EventBus

# Helper functions for pretty printing
def print_section(title):
    """Print a section header with formatting"""
    border = "=" * (len(title) + 4)
    print(f"\n{border}")
    print(f"| {title} |")
    print(f"{border}\n")

def print_dict(data: Dict[str, Any], indent=0, max_depth=3, current_depth=0):
    """Recursively print a dictionary with proper indentation"""
    if current_depth >= max_depth:
        print(" " * indent + "...")
        return
    
    for key, value in data.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            print_dict(value, indent + 4, max_depth, current_depth + 1)
        elif isinstance(value, list) and len(value) > 0:
            if isinstance(value[0], dict) and len(value) > 3:
                print(" " * indent + f"{key}: [{len(value)} items]")
                # Print first 3 items
                for i, item in enumerate(value[:3]):
                    print(" " * (indent + 4) + f"Item {i}:")
                    print_dict(item, indent + 8, max_depth, current_depth + 1)
                if len(value) > 3:
                    print(" " * (indent + 4) + f"... and {len(value) - 3} more items")
            else:
                print(" " * indent + f"{key}: {value}")
        else:
            print(" " * indent + f"{key}: {value}")

class IdentityTester:
    """A class to test the identity module at different developmental levels"""
    
    def __init__(self, development_level: float = 0.0):
        """Initialize the identity tester with a specific development level"""
        self.event_bus = EventBus()
        self.identity = get_identity_module(
            module_id="identity_test",
            event_bus=self.event_bus,
            development_level=development_level
        )
        
        # Keep track of test results
        self.results = []
        
        # Log initialization
        logging.info(f"Initialized IdentityTester at development level {development_level:.1f}")
    
    def test_self_concept(self, attributes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the self-concept component by adding and updating self-attributes
        
        Args:
            attributes: List of attribute data to add to the self-concept
            
        Returns:
            Dict with results of self-concept operations
        """
        print_section("Testing Self-Concept")
        
        results = {
            "added_attributes": [],
            "updated_attributes": [],
            "query_results": {}
        }
        
        # Add attributes to self-concept
        for i, attr_data in enumerate(attributes):
            process_id = f"self_concept_test_{int(time.time())}_{i}"
            
            # Create input for adding an attribute
            input_data = {
                "process_id": process_id,
                "component": "self_concept",
                "operation": "add_attribute",
                "domain": attr_data.get("domain", "general"),
                "content": attr_data.get("content", ""),
                "confidence": attr_data.get("confidence", 0.5),
                "importance": attr_data.get("importance", 0.5),
                "valence": attr_data.get("valence", 0.0)
            }
            
            # Process the input
            logging.info(f"Adding self-attribute: {attr_data.get('content', '')}")
            start_time = time.time()
            result = self.identity.process_input(input_data)
            processing_time = time.time() - start_time
            
            # Store result
            result["processing_time_ms"] = int(processing_time * 1000)
            results["added_attributes"].append(result)
            
            # Log completion
            logging.info(f"Self-attribute added in {processing_time:.3f} seconds")
            
            # Also update some attributes
            if i < len(attributes) // 2:
                # Update confidence and importance
                update_input = {
                    "process_id": f"{process_id}_update",
                    "component": "self_concept",
                    "operation": "update_attribute",
                    "attribute_id": result.get("attribute_id", ""),
                    "confidence": min(1.0, attr_data.get("confidence", 0.5) + 0.2),
                    "importance": min(1.0, attr_data.get("importance", 0.5) + 0.1)
                }
                
                logging.info(f"Updating self-attribute: {result.get('attribute_id', '')}")
                update_result = self.identity.process_input(update_input)
                results["updated_attributes"].append(update_result)
        
        # Query the self-concept
        domains = list(set([attr.get("domain", "general") for attr in attributes]))
        for domain in domains:
            query_input = {
                "process_id": f"self_query_{int(time.time())}",
                "component": "self_concept",
                "operation": "query_self",
                "query_domain": domain
            }
            
            logging.info(f"Querying self-concept for domain: {domain}")
            query_result = self.identity.process_input(query_input)
            results["query_results"][domain] = query_result
        
        # Store in test results
        self.results.append({
            "test_type": "self_concept",
            "development_level": self.identity.development_level,
            "result": results
        })
        
        return results
    
    def test_personal_narrative(self, events: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the personal narrative component by adding events and extracting themes
        
        Args:
            events: List of event data to add to the personal narrative
            
        Returns:
            Dict with results of narrative operations
        """
        print_section("Testing Personal Narrative")
        
        results = {
            "added_events": [],
            "updated_events": [],
            "extracted_themes": [],
            "query_results": {}
        }
        
        # Add events to personal narrative
        for i, event_data in enumerate(events):
            process_id = f"narrative_test_{int(time.time())}_{i}"
            
            # Create input for adding an event
            input_data = {
                "process_id": process_id,
                "component": "personal_narrative",
                "operation": "add_event",
                "title": event_data.get("title", ""),
                "description": event_data.get("description", ""),
                "interpretation": event_data.get("interpretation", ""),
                "age_period": event_data.get("age_period", "childhood"),
                "importance": event_data.get("importance", 0.5),
                "emotional_impact": event_data.get("emotional_impact", {"neutral": 0.5})
            }
            
            # Process the input
            logging.info(f"Adding narrative event: {event_data.get('title', '')}")
            start_time = time.time()
            result = self.identity.process_input(input_data)
            processing_time = time.time() - start_time
            
            # Store result
            result["processing_time_ms"] = int(processing_time * 1000)
            results["added_events"].append(result)
            
            # Log completion
            logging.info(f"Narrative event added in {processing_time:.3f} seconds")
            
            # Also update some events
            if i < len(events) // 2:
                # Update interpretation and importance
                update_input = {
                    "process_id": f"{process_id}_update",
                    "component": "personal_narrative",
                    "operation": "update_event",
                    "event_id": result.get("event_id", ""),
                    "interpretation": f"Updated: {event_data.get('interpretation', '')}",
                    "importance": min(1.0, event_data.get("importance", 0.5) + 0.2)
                }
                
                logging.info(f"Updating narrative event: {result.get('event_id', '')}")
                update_result = self.identity.process_input(update_input)
                results["updated_events"].append(update_result)
        
        # Extract themes from events
        if len(events) >= 3:
            extract_input = {
                "process_id": f"theme_extract_{int(time.time())}",
                "component": "personal_narrative",
                "operation": "extract_theme",
                "name": "Test Theme",
                "description": "A theme extracted during testing",
                "event_ids": [result.get("event_id", "") for result in results["added_events"][:3]]
            }
            
            logging.info("Extracting theme from events")
            theme_result = self.identity.process_input(extract_input)
            results["extracted_themes"].append(theme_result)
        
        # Query the narrative
        age_periods = list(set([event.get("age_period", "childhood") for event in events]))
        for period in age_periods:
            query_input = {
                "process_id": f"narrative_query_{int(time.time())}",
                "component": "personal_narrative",
                "operation": "query_narrative",
                "query_type": "age_period",
                "age_period": period
            }
            
            logging.info(f"Querying narrative for age period: {period}")
            query_result = self.identity.process_input(query_input)
            results["query_results"][period] = query_result
        
        # Store in test results
        self.results.append({
            "test_type": "personal_narrative",
            "development_level": self.identity.development_level,
            "result": results
        })
        
        return results
    
    def test_preferences(self, preferences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the preferences component by adding and updating preferences
        
        Args:
            preferences: List of preference data to add
            
        Returns:
            Dict with results of preference operations
        """
        print_section("Testing Preferences")
        
        results = {
            "added_preferences": [],
            "updated_preferences": [],
            "extracted_values": [],
            "query_results": {}
        }
        
        # Add preferences
        for i, pref_data in enumerate(preferences):
            process_id = f"preference_test_{int(time.time())}_{i}"
            
            # Create input for adding a preference
            input_data = {
                "process_id": process_id,
                "component": "preferences",
                "operation": "add_preference",
                "domain": pref_data.get("domain", "general"),
                "target": pref_data.get("target", ""),
                "valence": pref_data.get("valence", 0.0),
                "strength": pref_data.get("strength", 0.5),
                "certainty": pref_data.get("certainty", 0.5),
                "reasons": pref_data.get("reasons", [])
            }
            
            # Process the input
            logging.info(f"Adding preference: {pref_data.get('target', '')}")
            start_time = time.time()
            result = self.identity.process_input(input_data)
            processing_time = time.time() - start_time
            
            # Store result
            result["processing_time_ms"] = int(processing_time * 1000)
            results["added_preferences"].append(result)
            
            # Log completion
            logging.info(f"Preference added in {processing_time:.3f} seconds")
            
            # Also update some preferences
            if i < len(preferences) // 2:
                # Update valence and strength
                update_input = {
                    "process_id": f"{process_id}_update",
                    "component": "preferences",
                    "operation": "update_preference",
                    "preference_id": result.get("preference_id", ""),
                    "valence": max(-1.0, min(1.0, pref_data.get("valence", 0.0) + 0.3)),
                    "strength": min(1.0, pref_data.get("strength", 0.5) + 0.2)
                }
                
                logging.info(f"Updating preference: {result.get('preference_id', '')}")
                update_result = self.identity.process_input(update_input)
                results["updated_preferences"].append(update_result)
        
        # Extract values from preferences (if development level allows)
        if self.identity.development_level >= 0.4 and len(preferences) >= 3:
            extract_input = {
                "process_id": f"value_extract_{int(time.time())}",
                "component": "preferences",
                "operation": "extract_value",
                "name": "Test Value",
                "description": "A value extracted during testing",
                "preference_ids": [result.get("preference_id", "") for result in results["added_preferences"][:3]]
            }
            
            logging.info("Extracting value from preferences")
            value_result = self.identity.process_input(extract_input)
            results["extracted_values"].append(value_result)
        
        # Query the preferences
        domains = list(set([pref.get("domain", "general") for pref in preferences]))
        for domain in domains:
            query_input = {
                "process_id": f"preference_query_{int(time.time())}",
                "component": "preferences",
                "operation": "query_preferences",
                "query_type": "domain",
                "domain": domain
            }
            
            logging.info(f"Querying preferences for domain: {domain}")
            query_result = self.identity.process_input(query_input)
            results["query_results"][domain] = query_result
        
        # Store in test results
        self.results.append({
            "test_type": "preferences",
            "development_level": self.identity.development_level,
            "result": results
        })
        
        return results
    
    def test_personality_traits(self, traits: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test the personality traits component by adding and updating traits
        
        Args:
            traits: List of trait data to add
            
        Returns:
            Dict with results of trait operations
        """
        print_section("Testing Personality Traits")
        
        results = {
            "added_traits": [],
            "updated_traits": [],
            "added_dimensions": [],
            "extracted_traits": [],
            "query_results": {}
        }
        
        # First add dimensions
        dimensions = {}
        for dimension_name in ["extraversion", "agreeableness", "conscientiousness", "emotional_stability", "openness"]:
            dimension_input = {
                "process_id": f"dimension_add_{int(time.time())}",
                "component": "personality_traits",
                "operation": "add_dimension",
                "name": dimension_name.capitalize(),
                "description": f"The {dimension_name} dimension of personality",
                "positive_pole": f"High {dimension_name}",
                "negative_pole": f"Low {dimension_name}"
            }
            
            logging.info(f"Adding trait dimension: {dimension_name}")
            dimension_result = self.identity.process_input(dimension_input)
            results["added_dimensions"].append(dimension_result)
            dimensions[dimension_name] = dimension_result.get("dimension_id", "")
        
        # Add traits
        for i, trait_data in enumerate(traits):
            process_id = f"trait_test_{int(time.time())}_{i}"
            
            # Assign a dimension if not specified
            if "dimension" not in trait_data and dimensions:
                dimension_keys = list(dimensions.keys())
                trait_data["dimension"] = dimensions[dimension_keys[i % len(dimension_keys)]]
            
            # Create input for adding a trait
            input_data = {
                "process_id": process_id,
                "component": "personality_traits",
                "operation": "add_trait",
                "name": trait_data.get("name", ""),
                "description": trait_data.get("description", ""),
                "score": trait_data.get("score", 0.5),
                "stability": trait_data.get("stability", 0.5),
                "dimension": trait_data.get("dimension", "")
            }
            
            # Process the input
            logging.info(f"Adding personality trait: {trait_data.get('name', '')}")
            start_time = time.time()
            result = self.identity.process_input(input_data)
            processing_time = time.time() - start_time
            
            # Store result
            result["processing_time_ms"] = int(processing_time * 1000)
            results["added_traits"].append(result)
            
            # Log completion
            logging.info(f"Personality trait added in {processing_time:.3f} seconds")
            
            # Also update some traits
            if i < len(traits) // 2:
                # Update score and stability
                update_input = {
                    "process_id": f"{process_id}_update",
                    "component": "personality_traits",
                    "operation": "update_trait",
                    "trait_id": result.get("trait_id", ""),
                    "score": min(1.0, trait_data.get("score", 0.5) + 0.15),
                    "stability": min(1.0, trait_data.get("stability", 0.5) + 0.1)
                }
                
                logging.info(f"Updating personality trait: {result.get('trait_id', '')}")
                update_result = self.identity.process_input(update_input)
                results["updated_traits"].append(update_result)
        
        # Extract traits from behavior description
        extract_input = {
            "process_id": f"trait_extract_{int(time.time())}",
            "component": "personality_traits",
            "operation": "extract_traits",
            "text": "The person is very social and enjoys meeting new people. They are organized and plan ahead carefully. They tend to be calm under pressure and adapt well to changes."
        }
        
        logging.info("Extracting traits from behavior description")
        extract_result = self.identity.process_input(extract_input)
        results["extracted_traits"].append(extract_result)
        
        # Query the traits
        for dimension_name, dimension_id in dimensions.items():
            query_input = {
                "process_id": f"trait_query_{int(time.time())}",
                "component": "personality_traits",
                "operation": "query_traits",
                "query_type": "dimension",
                "dimension_id": dimension_id
            }
            
            logging.info(f"Querying traits for dimension: {dimension_name}")
            query_result = self.identity.process_input(query_input)
            results["query_results"][dimension_name] = query_result
        
        # Store in test results
        self.results.append({
            "test_type": "personality_traits",
            "development_level": self.identity.development_level,
            "result": results
        })
        
        return results
    
    def test_identity_integration(self) -> Dict[str, Any]:
        """
        Test the integrated identity system
        
        Returns:
            Dict with results of identity operations
        """
        print_section("Testing Identity Integration")
        
        results = {
            "identity_state": {},
            "identity_integration": {},
            "identity_query": {}
        }
        
        # Get identity state
        state_input = {
            "process_id": f"identity_state_{int(time.time())}",
            "component": "identity",
            "operation": "get_state"
        }
        
        logging.info("Getting identity state")
        state_result = self.identity.process_input(state_input)
        results["identity_state"] = state_result
        
        # Update identity integration
        integration_input = {
            "process_id": f"identity_integration_{int(time.time())}",
            "component": "identity",
            "operation": "update_integration"
        }
        
        logging.info("Updating identity integration")
        integration_result = self.identity.process_input(integration_input)
        results["identity_integration"] = integration_result
        
        # Query identity
        query_input = {
            "process_id": f"identity_query_{int(time.time())}",
            "component": "identity",
            "operation": "query_identity",
            "query_type": "summary"
        }
        
        logging.info("Querying identity summary")
        query_result = self.identity.process_input(query_input)
        results["identity_query"] = query_result
        
        # Get milestones
        milestones_input = {
            "process_id": f"identity_milestones_{int(time.time())}",
            "component": "identity",
            "operation": "query_identity",
            "query_type": "milestones"
        }
        
        logging.info("Querying identity milestones")
        milestones_result = self.identity.process_input(milestones_input)
        results["identity_milestones"] = milestones_result
        
        # Store in test results
        self.results.append({
            "test_type": "identity_integration",
            "development_level": self.identity.development_level,
            "result": results
        })
        
        return results
    
    def print_result_summary(self, result: Dict[str, Any], result_type: str = "test"):
        """Print a summary of the test result"""
        print_section("Result Summary")
        
        # Basic information
        print(f"Development Level: {self.identity.development_level:.2f}")
        print(f"Result Type: {result_type}")
        
        if result_type == "self_concept":
            # Self-concept summary
            added = result.get("added_attributes", [])
            updated = result.get("updated_attributes", [])
            queries = result.get("query_results", {})
            
            print(f"\nAdded Self-Attributes: {len(added)}")
            print(f"Updated Self-Attributes: {len(updated)}")
            print(f"Query Results: {len(queries)} domains")
            
            if added:
                print("\nSample Added Attributes:")
                for i, attr in enumerate(added[:3]):
                    print(f"  {i+1}. {attr.get('content', '')} (Domain: {attr.get('domain', '')})")
                    print(f"     Confidence: {attr.get('confidence', 0.0):.2f}, Importance: {attr.get('importance', 0.0):.2f}")
            
            if queries:
                print("\nSample Query Results:")
                for domain, query in list(queries.items())[:2]:
                    print(f"  Domain: {domain}")
                    attrs = query.get("attributes", [])
                    print(f"  Found {len(attrs)} attributes")
        
        elif result_type == "personal_narrative":
            # Narrative summary
            added = result.get("added_events", [])
            themes = result.get("extracted_themes", [])
            queries = result.get("query_results", {})
            
            print(f"\nAdded Events: {len(added)}")
            print(f"Extracted Themes: {len(themes)}")
            print(f"Query Results: {len(queries)} age periods")
            
            if added:
                print("\nSample Added Events:")
                for i, event in enumerate(added[:3]):
                    print(f"  {i+1}. {event.get('title', '')} (Age Period: {event.get('age_period', '')})")
                    print(f"     Importance: {event.get('importance', 0.0):.2f}")
            
            if themes:
                print("\nExtracted Themes:")
                for i, theme in enumerate(themes):
                    print(f"  {i+1}. {theme.get('name', '')} (Events: {len(theme.get('events', []))})")
        
        elif result_type == "preferences":
            # Preferences summary
            added = result.get("added_preferences", [])
            values = result.get("extracted_values", [])
            queries = result.get("query_results", {})
            
            print(f"\nAdded Preferences: {len(added)}")
            print(f"Extracted Values: {len(values)}")
            print(f"Query Results: {len(queries)} domains")
            
            if added:
                print("\nSample Added Preferences:")
                for i, pref in enumerate(added[:3]):
                    print(f"  {i+1}. {pref.get('target', '')} (Domain: {pref.get('domain', '')})")
                    print(f"     Valence: {pref.get('valence', 0.0):.2f}, Strength: {pref.get('strength', 0.0):.2f}")
            
            if values:
                print("\nExtracted Values:")
                for i, value in enumerate(values):
                    print(f"  {i+1}. {value.get('name', '')} (Importance: {value.get('importance', 0.0):.2f})")
        
        elif result_type == "personality_traits":
            # Personality traits summary
            added_traits = result.get("added_traits", [])
            dimensions = result.get("added_dimensions", [])
            extracted = result.get("extracted_traits", [])
            
            print(f"\nAdded Traits: {len(added_traits)}")
            print(f"Added Dimensions: {len(dimensions)}")
            print(f"Extracted Traits: {len(extracted)}")
            
            if added_traits:
                print("\nSample Added Traits:")
                for i, trait in enumerate(added_traits[:3]):
                    print(f"  {i+1}. {trait.get('name', '')} (Score: {trait.get('score', 0.0):.2f})")
                    print(f"     Stability: {trait.get('stability', 0.0):.2f}")
            
            if extracted:
                for i, extract in enumerate(extracted):
                    traits = extract.get("traits", [])
                    print(f"\nExtracted {len(traits)} traits from text")
                    for j, trait in enumerate(traits[:3]):
                        print(f"  {j+1}. {trait.get('name', '')} (Score: {trait.get('score', 0.0):.2f})")
        
        elif result_type == "identity_integration":
            # Identity integration summary
            state = result.get("identity_state", {})
            integration = result.get("identity_integration", {})
            
            if "identity_integration" in integration:
                print(f"\nIdentity Integration: {integration.get('identity_integration', 0.0):.2f}")
                print(f"Identity Stability: {integration.get('identity_stability', 0.0):.2f}")
                print(f"Identity Clarity: {integration.get('identity_clarity', 0.0):.2f}")
            
            if "components" in state:
                components = state.get("components", {})
                print("\nComponent Development Levels:")
                for component, comp_state in components.items():
                    if "developmental_level" in comp_state:
                        print(f"  {component}: {comp_state.get('developmental_level', 0.0):.2f}")
    
    def print_detailed_result(self, result: Dict[str, Any]):
        """Print a detailed view of the test result"""
        print_section("Detailed Result")
        print_dict(result, max_depth=4)
    
    def print_module_state(self):
        """Print the current state of the identity module and its submodules"""
        print_section("Identity Module State")
        
        # Get identity state
        state_input = {
            "process_id": f"state_query_{int(time.time())}",
            "component": "identity",
            "operation": "get_state"
        }
        
        state = self.identity.process_input(state_input)
        
        # Print summary
        print(f"Identity System:")
        print(f"  - Development Level: {state.get('developmental_level', 0.0):.2f}")
        print(f"  - Module ID: {state.get('module_id', 'unknown')}")
        print(f"  - Identity Integration: {state.get('identity_integration', 0.0):.2f}")
        print(f"  - Identity Stability: {state.get('identity_stability', 0.0):.2f}")
        print(f"  - Identity Clarity: {state.get('identity_clarity', 0.0):.2f}")
        
        # Print component states
        if "components" in state:
            components = state.get("components", {})
            
            for component_name, component_state in components.items():
                print(f"\n{component_name.replace('_', ' ').title()}:")
                print(f"  - Development Level: {component_state.get('developmental_level', 0.0):.2f}")
                
                if component_name == "self_concept":
                    print(f"  - Self-Attributes: {component_state.get('attribute_count', 0)}")
                    print(f"  - Self-Esteem: {component_state.get('global_self_esteem', 0.0):.2f}")
                    print(f"  - Self-Concept Clarity: {component_state.get('clarity', 0.0):.2f}")
                    
                elif component_name == "personal_narrative":
                    print(f"  - Events: {component_state.get('event_count', 0)}")
                    print(f"  - Themes: {component_state.get('theme_count', 0)}")
                    print(f"  - Narrative Coherence: {component_state.get('coherence', 0.0):.2f}")
                    
                elif component_name == "preferences":
                    print(f"  - Preferences: {component_state.get('preference_count', 0)}")
                    print(f"  - Values: {component_state.get('value_count', 0)}")
                    print(f"  - Preference Consistency: {component_state.get('consistency', 0.0):.2f}")
                    
                elif component_name == "personality_traits":
                    print(f"  - Traits: {component_state.get('trait_count', 0)}")
                    print(f"  - Dimensions: {component_state.get('dimension_count', 0)}")
                    print(f"  - Trait Stability: {component_state.get('stability', 0.0):.2f}")
    
    def set_development_level(self, level: float):
        """Set the development level of the identity module"""
        prev_level = self.identity.development_level
        self.identity.update_development(level - prev_level)
        logging.info(f"Updated development level from {prev_level:.2f} to {self.identity.development_level:.2f}")
    
    def save_results(self, filename: str = None):
        """Save test results to a JSON file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"identity_test_results_{timestamp}.json"
            
        # Create directory if it doesn't exist
        os.makedirs("test_results", exist_ok=True)
        filepath = os.path.join("test_results", filename)
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            # Create a simplified version of the result for storage
            serializable_results.append({
                "test_type": result["test_type"],
                "development_level": result["development_level"],
                "timestamp": datetime.now().isoformat(),
            })
            
        with open(filepath, "w") as f:
            json.dump(serializable_results, f, indent=2)
            
        logging.info(f"Results saved to {filepath}")

def generate_test_self_attributes(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test data for self-attributes"""
    domains = ["physical", "academic", "social", "emotional", "skills"]
    
    attributes = []
    for i in range(count):
        domain = domains[i % len(domains)]
        
        if domain == "physical":
            content = f"I am {['tall', 'athletic', 'strong', 'coordinated', 'flexible'][i % 5]}"
            valence = 0.6
        elif domain == "academic":
            content = f"I am {['intelligent', 'studious', 'analytical', 'curious', 'creative'][i % 5]}"
            valence = 0.7
        elif domain == "social":
            content = f"I am {['friendly', 'outgoing', 'helpful', 'compassionate', 'loyal'][i % 5]}"
            valence = 0.8
        elif domain == "emotional":
            content = f"I am {['stable', 'resilient', 'optimistic', 'passionate', 'calm'][i % 5]}"
            valence = 0.5
        else:  # skills
            content = f"I am good at {['writing', 'problem-solving', 'sports', 'music', 'art'][i % 5]}"
            valence = 0.6
            
        attributes.append({
            "domain": domain,
            "content": content,
            "confidence": 0.5 + (i * 0.1) % 0.5,
            "importance": 0.4 + (i * 0.15) % 0.6,
            "valence": valence
        })
    
    return attributes

def generate_test_narrative_events(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test data for narrative events"""
    age_periods = ["early_childhood", "middle_childhood", "adolescence", "early_adulthood", "adulthood"]
    
    events = []
    for i in range(count):
        age_period = age_periods[i % len(age_periods)]
        
        if age_period == "early_childhood":
            title = f"First day of kindergarten"
            description = "I remember being nervous but excited on my first day of school."
            interpretation = "This was my first step toward independence."
            emotional_impact = {"anxiety": 0.6, "excitement": 0.7}
        elif age_period == "middle_childhood":
            title = f"Learning to ride a bike"
            description = "After many tries, I finally learned to ride without training wheels."
            interpretation = "This taught me that persistence leads to success."
            emotional_impact = {"pride": 0.8, "joy": 0.7}
        elif age_period == "adolescence":
            title = f"Making the basketball team"
            description = "I practiced all summer and made the team in the fall."
            interpretation = "I learned that hard work pays off."
            emotional_impact = {"pride": 0.9, "happiness": 0.8}
        elif age_period == "early_adulthood":
            title = f"Graduating college"
            description = "After four years of hard work, I finally got my degree."
            interpretation = "This milestone opened doors for my future career."
            emotional_impact = {"pride": 0.9, "relief": 0.7, "nostalgia": 0.5}
        else:  # adulthood
            title = f"Career promotion"
            description = "After years at the company, I was promoted to a leadership role."
            interpretation = "This validated my professional skills and dedication."
            emotional_impact = {"pride": 0.8, "satisfaction": 0.7}
            
        events.append({
            "age_period": age_period,
            "title": title,
            "description": description,
            "interpretation": interpretation,
            "importance": 0.5 + (i * 0.1) % 0.5,
            "emotional_impact": emotional_impact
        })
    
    return events

def generate_test_preferences(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test data for preferences"""
    domains = ["food", "music", "activities", "environment", "people"]
    
    preferences = []
    for i in range(count):
        domain = domains[i % len(domains)]
        
        if domain == "food":
            target = ["Italian cuisine", "Sushi", "Chocolate", "Spicy food", "Vegetarian dishes"][i % 5]
            valence = 0.7
            reasons = ["It's delicious", "It reminds me of childhood", "It's healthy"]
        elif domain == "music":
            target = ["Classical music", "Jazz", "Rock", "Pop", "Electronic"][i % 5]
            valence = 0.6
            reasons = ["It's energizing", "It helps me relax", "I appreciate the complexity"]
        elif domain == "activities":
            target = ["Reading", "Hiking", "Playing video games", "Cooking", "Photography"][i % 5]
            valence = 0.8
            reasons = ["It's enjoyable", "It challenges me", "It helps me grow"]
        elif domain == "environment":
            target = ["Mountains", "Beaches", "Cities", "Forests", "Home"][i % 5]
            valence = 0.7
            reasons = ["It's peaceful", "It energizes me", "It makes me feel connected"]
        else:  # people
            target = ["Outgoing people", "Intellectuals", "Creative types", "Reliable friends", "Mentors"][i % 5]
            valence = 0.75
            reasons = ["They inspire me", "I feel comfortable with them", "We share interests"]
            
        preferences.append({
            "domain": domain,
            "target": target,
            "valence": valence if i % 3 != 0 else -valence,  # Mix in some negative preferences
            "strength": 0.5 + (i * 0.12) % 0.5,
            "certainty": 0.4 + (i * 0.14) % 0.6,
            "reasons": reasons
        })
    
    return preferences

def generate_test_personality_traits(count: int = 5) -> List[Dict[str, Any]]:
    """Generate test data for personality traits"""
    trait_info = [
        # Extraversion traits
        {
            "name": "Sociable",
            "description": "Enjoys interacting with others and seeks social situations",
            "score": 0.75,
            "dimension": "extraversion"
        },
        {
            "name": "Reserved",
            "description": "Prefers quiet environments and limited social interaction",
            "score": 0.25,
            "dimension": "extraversion"
        },
        # Agreeableness traits
        {
            "name": "Compassionate",
            "description": "Shows empathy and concern for others' well-being",
            "score": 0.85,
            "dimension": "agreeableness"
        },
        {
            "name": "Competitive",
            "description": "Focuses on personal success and achievement",
            "score": 0.35,
            "dimension": "agreeableness"
        },
        # Conscientiousness traits
        {
            "name": "Organized",
            "description": "Plans carefully and maintains order in activities",
            "score": 0.80,
            "dimension": "conscientiousness"
        },
        {
            "name": "Flexible",
            "description": "Adapts easily to changes and prefers spontaneity",
            "score": 0.30,
            "dimension": "conscientiousness"
        },
        # Emotional stability traits
        {
            "name": "Calm",
            "description": "Maintains composure even in stressful situations",
            "score": 0.70,
            "dimension": "emotional_stability"
        },
        {
            "name": "Sensitive",
            "description": "Experiences emotions intensely and reacts strongly",
            "score": 0.40,
            "dimension": "emotional_stability"
        },
        # Openness traits
        {
            "name": "Creative",
            "description": "Generates novel ideas and appreciates art and beauty",
            "score": 0.90,
            "dimension": "openness"
        },
        {
            "name": "Practical",
            "description": "Focuses on tangible results and conventional approaches",
            "score": 0.20,
            "dimension": "openness"
        }
    ]
    
    # Select a subset of traits
    selected_traits = []
    for i in range(count):
        trait = trait_info[i % len(trait_info)].copy()
        trait["stability"] = 0.3 + (i * 0.1) % 0.7  # Vary stability
        selected_traits.append(trait)
    
    return selected_traits

def test_identity_at_level(level: float) -> IdentityTester:
    """Test the identity module at a specific development level"""
    print_section(f"Testing Identity at Level {level:.1f}")
    
    tester = IdentityTester(development_level=level)
    tester.print_module_state()
    
    # Generate test data
    self_attributes = generate_test_self_attributes(5)
    narrative_events = generate_test_narrative_events(5)
    preferences = generate_test_preferences(5)
    personality_traits = generate_test_personality_traits(5)
    
    # Run tests
    print_section("Testing Self-Concept Component")
    result = tester.test_self_concept(self_attributes)
    tester.print_result_summary(result, "self_concept")
    
    print_section("Testing Personal Narrative Component")
    result = tester.test_personal_narrative(narrative_events)
    tester.print_result_summary(result, "personal_narrative")
    
    print_section("Testing Preferences Component")
    result = tester.test_preferences(preferences)
    tester.print_result_summary(result, "preferences")
    
    print_section("Testing Personality Traits Component")
    result = tester.test_personality_traits(personality_traits)
    tester.print_result_summary(result, "personality_traits")
    
    # Test integration
    print_section("Testing Identity Integration")
    result = tester.test_identity_integration()
    tester.print_result_summary(result, "identity_integration")
    
    # Save results
    tester.save_results(f"identity_level_{level:.1f}.json")
    
    return tester

def test_development_progression() -> IdentityTester:
    """Test how identity capabilities evolve across development levels"""
    print_section("Testing Development Progression")
    
    # Initialize at the lowest level
    tester = IdentityTester(development_level=0.0)
    
    # Define development stages to test
    stages = [0.0, 0.3, 0.6, 0.9]
    
    # Generate test data
    self_attributes = generate_test_self_attributes(6)
    narrative_events = generate_test_narrative_events(6)
    preferences = generate_test_preferences(6)
    personality_traits = generate_test_personality_traits(6)
    
    # Test each stage with the same inputs
    for stage in stages:
        # Set the development level
        tester.set_development_level(stage)
        
        print_section(f"Development Level: {stage:.1f}")
        tester.print_module_state()
        
        # Run tests with a subset of the data appropriate for the stage
        # Earlier stages use fewer items to reflect simpler capabilities
        count = max(2, int(2 + (stage * 4)))
        
        # Test components
        result = tester.test_self_concept(self_attributes[:count])
        tester.print_result_summary(result, "self_concept")
        
        result = tester.test_personal_narrative(narrative_events[:count])
        tester.print_result_summary(result, "personal_narrative")
        
        result = tester.test_preferences(preferences[:count])
        tester.print_result_summary(result, "preferences")
        
        result = tester.test_personality_traits(personality_traits[:count])
        tester.print_result_summary(result, "personality_traits")
        
        # Test integration (at all levels)
        result = tester.test_identity_integration()
        tester.print_result_summary(result, "identity_integration")
    
    # Save results
    tester.save_results("identity_development_progression.json")
    
    return tester

def main():
    """Main test function"""
    print_section("Identity Module Test")
    
    # Test specific development levels
    print_section("Testing Specific Development Levels")
    
    # Test basic identity (0.1)
    test_identity_at_level(0.1)
    
    # Test intermediate identity (0.5)
    test_identity_at_level(0.5)
    
    # Test advanced identity (0.9)
    test_identity_at_level(0.9)
    
    # Test progression across development stages
    test_development_progression()
    
    print_section("Test Complete")

if __name__ == "__main__":
    main()