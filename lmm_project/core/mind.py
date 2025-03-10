"""
Mind - Core cognitive architecture controller

This module implements the central Mind class that coordinates all cognitive modules,
manages developmental progression, and orchestrates the flow of information between
components of the Large Mind Model system.
"""

import logging
import time
import traceback
from typing import Dict, List, Any, Optional, Set, Tuple, TYPE_CHECKING, Union, Type
import os
import json
import threading
from datetime import datetime
from pathlib import Path

from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message
from lmm_project.core.types import DevelopmentalStage, HomeostaticSignalType, ModuleType
from lmm_project.core.exceptions import ModuleInitializationError, ModuleProcessingError

# Type annotations with strings to avoid circular imports
if TYPE_CHECKING:
    from lmm_project.modules.base_module import BaseModule
    from lmm_project.homeostasis.energy_regulation import EnergyRegulator
    from lmm_project.homeostasis.arousal_control import ArousalController
    from lmm_project.homeostasis.cognitive_load_balancer import CognitiveLoadBalancer
    from lmm_project.homeostasis.social_need_manager import SocialNeedManager

logger = logging.getLogger(__name__)

class Mind:
    """
    Central coordinator for all cognitive modules
    
    The Mind integrates all cognitive modules, manages developmental progression,
    and coordinates information flow between components.
    
    Key responsibilities:
    - Initialize and manage cognitive modules
    - Track developmental progression
    - Coordinate homeostatic regulation
    - Process inputs and generate responses
    - Maintain system state
    """
    
    def __init__(
        self, 
        event_bus: EventBus,
        state_manager: StateManager,
        initial_age: float = 0.0,
        developmental_stage: str = "prenatal",
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the Mind
        
        Args:
            event_bus: Event bus for inter-module communication
            state_manager: State manager for tracking system state
            initial_age: Initial age of the mind (0.0-1.0)
            developmental_stage: Initial developmental stage
            config: Optional configuration parameters
        
        Raises:
            ModuleInitializationError: If there's an error initializing the mind
        """
        self.event_bus = event_bus
        self.state_manager = state_manager
        self.age = initial_age
        self.developmental_stage = developmental_stage
        self.modules: Dict[str, Any] = {}  # Use Any instead of BaseModule to avoid circular imports
        self.homeostasis_systems: Dict[str, Any] = {}  # Homeostasis regulatory systems
        self.creation_time = datetime.now()
        self.last_cycle_time = datetime.now()
        self.cycle_count = 0
        self.config = config or {}
        self._lock = threading.RLock()
        self._performance_metrics = {
            "cycle_times": [],
            "module_processing_times": {},
            "error_counts": {},
            "last_error_time": None,
            "total_messages_processed": 0
        }
        self._active = False
        self._initialized = False
        
        # Register event handlers
        self.event_bus.subscribe("system_cycle_complete", self._handle_system_cycle)
        self.event_bus.subscribe("module_error", self._handle_module_error)
        self.event_bus.subscribe("performance_metric", self._handle_performance_metric)
        
        logger.info(f"Mind initialized at age {initial_age} in {developmental_stage} stage")
        
    def initialize_modules(self):
        """
        Initialize all cognitive modules
        
        This method creates instances of all required cognitive modules and 
        establishes connections between them.
        
        Returns:
            Dict[str, Any]: Initialized modules
            
        Raises:
            ModuleInitializationError: If there's an error initializing modules
        """
        if self._initialized:
            logger.warning("Modules already initialized, skipping initialization")
            return self.modules
            
        logger.info("Initializing cognitive modules...")
        
        # Import modules here to avoid circular imports
        from lmm_project.modules import get_module_classes
        
        try:
            module_classes = get_module_classes()
            
            # Create instances of all modules
            initialization_errors = []
            for module_type, module_class in module_classes.items():
                module_id = f"{module_type}_{int(time.time())}"
                try:
                    # Apply module-specific configuration if available
                    module_config = self.config.get("modules", {}).get(module_type, {})
                    
                    module = module_class(
                        module_id=module_id,
                        event_bus=self.event_bus,
                        development_level=self.age,
                        **module_config
                    )
                    self.modules[module_type] = module
                    
                    # Initialize performance tracking for this module
                    self._performance_metrics["module_processing_times"][module_type] = []
                    self._performance_metrics["error_counts"][module_type] = 0
                    
                    logger.info(f"Initialized {module_type} module")
                except Exception as e:
                    error_msg = f"Failed to initialize {module_type} module: {str(e)}"
                    logger.error(error_msg)
                    logger.debug(traceback.format_exc())
                    initialization_errors.append(error_msg)
            
            # If any modules failed to initialize, raise an error with details
            if initialization_errors:
                if len(initialization_errors) == len(module_classes):
                    raise ModuleInitializationError(f"All modules failed to initialize: {initialization_errors}")
                else:
                    logger.warning(f"Some modules failed to initialize: {initialization_errors}")
                    
            logger.info(f"Initialized {len(self.modules)} cognitive modules")
            
            # Initialize homeostasis systems
            self._initialize_homeostasis()
            
            # Mark as initialized
            self._initialized = True
            
            # Publish initialization complete event
            self.event_bus.publish(Message(
                sender="mind",
                message_type="initialization_complete",
                content={
                    "modules": list(self.modules.keys()),
                    "homeostasis_systems": list(self.homeostasis_systems.keys()),
                    "development_level": self.age,
                    "developmental_stage": self.developmental_stage
                }
            ))
            
            return self.modules
            
        except Exception as e:
            error_msg = f"Failed to initialize modules: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            raise ModuleInitializationError(error_msg)
    
    def _initialize_homeostasis(self):
        """
        Initialize homeostasis regulatory systems
        
        Returns:
            Dict[str, Any]: Initialized homeostasis systems
            
        Raises:
            ModuleInitializationError: If there's an error initializing homeostasis systems
        """
        logger.info("Initializing homeostasis systems...")
        
        # Import homeostasis components
        from lmm_project.homeostasis.energy_regulation import EnergyRegulator
        from lmm_project.homeostasis.arousal_control import ArousalController
        from lmm_project.homeostasis.cognitive_load_balancer import CognitiveLoadBalancer
        from lmm_project.homeostasis.social_need_manager import SocialNeedManager
        
        initialization_errors = []
        
        # Initialize energy regulation
        try:
            energy_config = self.config.get("homeostasis", {}).get("energy", {})
            initial_energy = energy_config.get("initial_energy", 0.8 if self.age > 0.1 else 0.5)
            
            energy_regulator = EnergyRegulator(
                event_bus=self.event_bus,
                initial_energy=initial_energy,
                **energy_config
            )
            self.homeostasis_systems["energy"] = energy_regulator
            logger.info("Initialized energy regulation system")
        except Exception as e:
            error_msg = f"Failed to initialize energy regulation: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            initialization_errors.append(error_msg)
        
        # Initialize arousal control
        try:
            arousal_config = self.config.get("homeostasis", {}).get("arousal", {})
            initial_arousal = arousal_config.get("initial_arousal", 0.4 if self.age > 0.1 else 0.2)
            
            arousal_controller = ArousalController(
                event_bus=self.event_bus,
                initial_arousal=initial_arousal,
                **arousal_config
            )
            self.homeostasis_systems["arousal"] = arousal_controller
            logger.info("Initialized arousal control system")
        except Exception as e:
            error_msg = f"Failed to initialize arousal control: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            initialization_errors.append(error_msg)
        
        # Initialize cognitive load balancer
        try:
            cognitive_config = self.config.get("homeostasis", {}).get("cognitive_load", {})
            initial_capacity = cognitive_config.get("initial_capacity", 0.3)
            working_memory_slots = cognitive_config.get("working_memory_slots", 2 + int(self.age * 5))
            
            cognitive_load_balancer = CognitiveLoadBalancer(
                event_bus=self.event_bus,
                initial_capacity=initial_capacity,
                working_memory_slots=working_memory_slots,
                **cognitive_config
            )
            self.homeostasis_systems["cognitive_load"] = cognitive_load_balancer
            logger.info("Initialized cognitive load balancer")
        except Exception as e:
            error_msg = f"Failed to initialize cognitive load balancer: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            initialization_errors.append(error_msg)
        
        # Initialize social need manager
        try:
            social_config = self.config.get("homeostasis", {}).get("social_need", {})
            initial_social_need = social_config.get("initial_social_need", 0.3 if self.age > 0.1 else 0.1)
            
            social_need_manager = SocialNeedManager(
                event_bus=self.event_bus,
                initial_social_need=initial_social_need,
                **social_config
            )
            self.homeostasis_systems["social_need"] = social_need_manager
            logger.info("Initialized social need manager")
        except Exception as e:
            error_msg = f"Failed to initialize social need manager: {str(e)}"
            logger.error(error_msg)
            logger.debug(traceback.format_exc())
            initialization_errors.append(error_msg)
        
        # If any homeostasis systems failed to initialize, raise an error with details
        if initialization_errors:
            if len(initialization_errors) == 4:  # All systems failed
                raise ModuleInitializationError(f"All homeostasis systems failed to initialize: {initialization_errors}")
            else:
                logger.warning(f"Some homeostasis systems failed to initialize: {initialization_errors}")
        
        # Notify all systems of current developmental stage
        self._update_homeostasis_development()
        
        logger.info(f"Initialized {len(self.homeostasis_systems)} homeostasis systems")
        return self.homeostasis_systems
    
    def _update_homeostasis_development(self):
        """Update all homeostasis systems with current development level"""
        # Create development update message
        dev_stage = DevelopmentalStage.from_level(self.age)
        dev_message = Message(
            sender="mind",
            message_type="development_update",
            content={
                "development_level": self.age,
                "stage": dev_stage,
                "previous_stage": self.developmental_stage if self.developmental_stage != dev_stage else None
            }
        )
        
        # Publish development update
        self.event_bus.publish(dev_message)
        self.developmental_stage = dev_stage
        
        logger.info(f"Published development update: level={self.age:.2f}, stage={dev_stage}")
        
    def _handle_system_cycle(self, message: Message):
        """Handle system cycle completion events"""
        now = datetime.now()
        delta_time = (now - self.last_cycle_time).total_seconds()
        self.last_cycle_time = now
        self.cycle_count += 1
        
        # Publish system cycle event (regularity helps homeostasis systems)
        cycle_message = Message(
            sender="mind",
            message_type="system_cycle",
            content={
                "cycle_number": self.cycle_count,
                "delta_time": delta_time,
                "current_age": self.age
            }
        )
        self.event_bus.publish(cycle_message)
        
        # Every 10 cycles, check if there are any homeostatic imbalances
        if self.cycle_count % 10 == 0:
            self._check_homeostatic_balance()
            
    def _check_homeostatic_balance(self):
        """Check if any homeostatic systems are significantly out of balance"""
        # Get most urgent homeostatic need from each system
        urgent_needs = []
        
        for system_name, system in self.homeostasis_systems.items():
            if hasattr(system, "homeostatic_system") and hasattr(system.homeostatic_system, "get_most_urgent_need"):
                urgent_need = system.homeostatic_system.get_most_urgent_need()
                if urgent_need and urgent_need[1].urgency > 0.6:  # Significant urgency
                    urgent_needs.append({
                        "system": system_name,
                        "need_type": urgent_need[0],
                        "urgency": urgent_need[1].urgency,
                        "current_value": urgent_need[1].current_value,
                        "setpoint": urgent_need[1].setpoint
                    })
        
        # If there are urgent needs, publish message
        if urgent_needs:
            # Sort by urgency
            urgent_needs.sort(key=lambda x: x["urgency"], reverse=True)
            
            # Create message
            balance_message = Message(
                sender="mind",
                message_type="homeostatic_imbalance",
                content={
                    "urgent_needs": urgent_needs,
                    "most_urgent": urgent_needs[0]
                },
                priority=int(urgent_needs[0]["urgency"] * 10)
            )
            self.event_bus.publish(balance_message)
            
            logger.info(f"Detected homeostatic imbalance: {urgent_needs[0]['system']}.{urgent_needs[0]['need_type']} (urgency: {urgent_needs[0]['urgency']:.2f})")
        
    def update_development(self, delta_time: float):
        """
        Update the mind's developmental progression
        
        Args:
            delta_time: Amount of developmental time to add
        """
        # Update mind age
        prev_age = self.age
        self.age += delta_time
        
        # Update all modules with appropriate fraction of development
        for module_type, module in self.modules.items():
            # Different modules may develop at different rates
            # Here we use a simple approach where all modules develop equally
            module.update_development(delta_time)
            
        # Update homeostasis systems with new development level
        self._update_homeostasis_development()
            
        logger.debug(f"Mind development updated: age {prev_age:.2f} -> {self.age:.2f}")
        
    def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through the complete cognitive pipeline
        
        Args:
            input_data: Dictionary containing input data
            
        Returns:
            Dictionary containing processing results
        """
        # Check homeostatic state to see if processing is possible
        energy_state = self.get_homeostatic_state("energy")
        if energy_state and energy_state.get("energy_level", 1.0) < 0.2:
            logger.warning("Energy level too low for input processing")
            return {"error": "Energy level too low for input processing"}
        
        # Check if cognitive load allows for processing
        cognitive_load = self.get_homeostatic_state("cognitive_load")
        if cognitive_load and cognitive_load.get("cognitive_load", 0.0) > 0.9:
            logger.warning("Cognitive load too high for input processing")
            return {"error": "Cognitive load too high for input processing"}
        
        results = {}
        module_outputs = {}
        
        # Start a new processing cycle
        cycle_message = Message(
            sender="mind",
            message_type="processing_cycle_start",
            content={
                "input_data": input_data,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.event_bus.publish(cycle_message)
        
        # 1. PERCEPTION - Process through perception module
        if "perception" in self.modules:
            logger.debug("Processing input through perception module")
            perception_results = self.modules["perception"].process_input(input_data)
            module_outputs["perception"] = perception_results
            results["perception"] = perception_results
        else:
            logger.warning("No perception module available")
            return {"error": "Perception module not available"}
            
        # 2. ATTENTION - Focus on relevant aspects of perceived input
        if "attention" in self.modules:
            logger.debug("Applying attention to perceived input")
            attention_results = self.modules["attention"].process_input({
                "perception_data": perception_results,
                "current_focus": self.state_manager.get_state("current_focus")
            })
            module_outputs["attention"] = attention_results
            results["attention"] = attention_results
            
            # Update salience for further processing
            salient_data = attention_results.get("salient_data", perception_results)
        else:
            salient_data = perception_results
        
        # 3. WORKING MEMORY - Place attended information in working memory
        if "memory" in self.modules and hasattr(self.modules["memory"], "update_working_memory"):
            logger.debug("Updating working memory with attended information")
            memory_results = self.modules["memory"].update_working_memory(salient_data)
            module_outputs["working_memory"] = memory_results
            results["working_memory"] = memory_results
            
            # Retrieve relevant long-term memories based on current context
            ltm_retrieval = self.modules["memory"].retrieve_relevant_memories(salient_data)
            module_outputs["memory_retrieval"] = ltm_retrieval
            results["memory_retrieval"] = ltm_retrieval
            
            # Create an integrated context combining perception, attention, and memory
            context = {
                "current_input": salient_data,
                "working_memory": memory_results,
                "relevant_memories": ltm_retrieval
            }
        else:
            context = {"current_input": salient_data}
        
        # 4. LANGUAGE PROCESSING - Process linguistic aspects of the input
        if "language" in self.modules:
            logger.debug("Processing linguistic content")
            language_results = self.modules["language"].process_input(context)
            module_outputs["language"] = language_results
            results["language"] = language_results
            
            # Add linguistic understanding to context
            context["linguistic_understanding"] = language_results
        
        # 5. EMOTIONAL PROCESSING - Evaluate emotional aspects of the input
        if "emotion" in self.modules:
            logger.debug("Processing emotional content")
            emotion_results = self.modules["emotion"].process_input(context)
            module_outputs["emotion"] = emotion_results
            results["emotion"] = emotion_results
            
            # Add emotional response to context
            context["emotional_response"] = emotion_results
        
        # 6. SOCIAL PROCESSING - Evaluate social implications
        if "social" in self.modules:
            logger.debug("Processing social implications")
            social_results = self.modules["social"].process_input(context)
            module_outputs["social"] = social_results
            results["social"] = social_results
            
            # Add social understanding to context
            context["social_understanding"] = social_results
        
        # 7. BELIEF PROCESSING - Update beliefs based on new information
        if "belief" in self.modules:
            logger.debug("Updating beliefs based on new information")
            belief_results = self.modules["belief"].process_input(context)
            module_outputs["belief"] = belief_results
            results["belief"] = belief_results
            
            # Add updated beliefs to context
            context["updated_beliefs"] = belief_results
        
        # 8. EXECUTIVE PROCESSING - Planning, decision making, inhibition
        if "executive" in self.modules:
            logger.debug("Performing executive processing")
            executive_results = self.modules["executive"].process_input(context)
            module_outputs["executive"] = executive_results
            results["executive"] = executive_results
            
            # Add executive decisions to context
            context["executive_decisions"] = executive_results
        
        # 9. CONSCIOUSNESS - Global workspace integration
        if "consciousness" in self.modules:
            logger.debug("Integrating information in consciousness")
            consciousness_results = self.modules["consciousness"].process_input({
                "module_outputs": module_outputs,
                "integrated_context": context
            })
            results["consciousness"] = consciousness_results
            
            # The conscious state represents the mind's current integrated understanding
            conscious_state = consciousness_results.get("conscious_state", {})
            self.state_manager.update_state({"conscious_state": conscious_state})
        
        # 10. CREATIVITY - Generate creative responses or novel associations
        if "creativity" in self.modules:
            logger.debug("Generating creative associations")
            creativity_results = self.modules["creativity"].process_input(context)
            results["creativity"] = creativity_results
        
        # 11. LEARNING - Update neural connections based on experience
        if "learning" in self.modules:
            logger.debug("Applying learning from current experience")
            learning_input = {
                "experience": context,
                "module_outputs": module_outputs,
                "developmental_stage": self.developmental_stage
            }
            learning_results = self.modules["learning"].process_input(learning_input)
            results["learning"] = learning_results
        
        # 12. IDENTITY - Update self-model based on experience
        if "identity" in self.modules:
            logger.debug("Updating self-model")
            identity_results = self.modules["identity"].process_input(context)
            results["identity"] = identity_results
            
        # Complete the processing cycle
        cycle_complete_message = Message(
            sender="mind",
            message_type="processing_cycle_complete",
            content={
                "processing_results": results,
                "timestamp": datetime.now().isoformat()
            }
        )
        self.event_bus.publish(cycle_complete_message)
        
        # Store this experience in memory if available
        if "memory" in self.modules and hasattr(self.modules["memory"], "store_experience"):
            self.modules["memory"].store_experience({
                "input": input_data,
                "processing_results": results,
                "timestamp": datetime.now().isoformat()
            })
        
        return results
        
    def get_homeostatic_state(self, system_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current state of homeostasis systems
        
        Args:
            system_name: Name of specific system to query, or None for all
            
        Returns:
            Dictionary containing homeostatic state information
        """
        if system_name and system_name in self.homeostasis_systems:
            # Return state of specific system
            system = self.homeostasis_systems[system_name]
            if hasattr(system, "get_state"):
                return system.get_state()
            return {}
            
        # Return state of all systems
        states = {}
        for name, system in self.homeostasis_systems.items():
            if hasattr(system, "get_state"):
                states[name] = system.get_state()
                
        return states
        
    def get_state(self) -> Dict[str, Any]:
        """
        Get the current state of the mind
        
        Returns:
            Dictionary containing mind state
        """
        modules_state = {}
        for module_type, module in self.modules.items():
            modules_state[module_type] = module.get_state()
            
        # Add homeostasis states
        homeostasis_state = self.get_homeostatic_state()
            
        return {
            "age": self.age,
            "development_level": self.age,  # Same as age, for compatibility
            "developmental_stage": self.developmental_stage,
            "modules": modules_state,
            "homeostasis": homeostasis_state,
            "creation_time": self.creation_time.isoformat(),
            "cycle_count": self.cycle_count
        }
        
    def save_state(self, state_dir: str) -> str:
        """
        Save the mind state to disk
        
        Args:
            state_dir: Directory to save state in
            
        Returns:
            Path to saved state file
        """
        # Ensure directory exists
        os.makedirs(state_dir, exist_ok=True)
        
        # Create timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Get complete state
        state = self.get_state()
        
        # Save to file
        file_path = os.path.join(state_dir, f"mind_state_{timestamp}.json")
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)
            
        logger.info(f"Mind state saved to {file_path}")
        return file_path
        
    def load_state(self, state_path: str) -> bool:
        """
        Load the mind state from disk
        
        Args:
            state_path: Path to state file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load state file
            with open(state_path, "r") as f:
                state = json.load(f)
                
            # Update mind properties
            self.age = state.get("age", self.age)
            self.developmental_stage = state.get("developmental_stage", self.developmental_stage)
            self.cycle_count = state.get("cycle_count", 0)
            
            # Update module states
            modules_state = state.get("modules", {})
            for module_type, module_state in modules_state.items():
                if module_type in self.modules and hasattr(self.modules[module_type], "load_state"):
                    self.modules[module_type].load_state(module_state)
            
            # Update homeostasis states
            homeostasis_state = state.get("homeostasis", {})
            for system_name, system_state in homeostasis_state.items():
                if system_name in self.homeostasis_systems and hasattr(self.homeostasis_systems[system_name], "load_state"):
                    self.homeostasis_systems[system_name].load_state(system_state)
            
            # Notify all systems of current development level
            self._update_homeostasis_development()
            
            logger.info(f"Mind state loaded from {state_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to load mind state: {str(e)}")
            return False
    
    def get_most_urgent_homeostatic_need(self) -> Optional[Dict[str, Any]]:
        """
        Get the most urgent homeostatic need across all systems
        
        Returns:
            Dictionary with information about the most urgent need, or None if all needs are balanced
        """
        most_urgent = None
        max_urgency = 0.0
        
        for system_name, system in self.homeostasis_systems.items():
            if hasattr(system, "homeostatic_system") and hasattr(system.homeostatic_system, "get_most_urgent_need"):
                urgent_need = system.homeostatic_system.get_most_urgent_need()
                if urgent_need and urgent_need[1].urgency > max_urgency:
                    max_urgency = urgent_need[1].urgency
                    most_urgent = {
                        "system": system_name,
                        "need_type": urgent_need[0],
                        "urgency": urgent_need[1].urgency,
                        "current_value": urgent_need[1].current_value,
                        "setpoint": urgent_need[1].setpoint,
                        "is_deficient": urgent_need[1].is_deficient,
                        "is_excessive": urgent_need[1].is_excessive
                    }
        
        return most_urgent if max_urgency > 0.1 else None

    @property
    def development_level(self) -> float:
        """
        Get the current development level
        
        Returns:
            Current development level (same as age)
        """
        return self.age

    def _handle_module_error(self, message: Message):
        """Handle module error events"""
        error_info = message.content
        module_type = error_info.get("module_type")
        error_message = error_info.get("error_message")
        error_time = datetime.now()
        
        if module_type:
            self._performance_metrics["error_counts"][module_type] += 1
            self._performance_metrics["last_error_time"] = error_time.isoformat()
            logger.error(f"Module error: {module_type}, Error: {error_message}")
        else:
            logger.error(f"Unspecified module error: {error_message}")
        
        # Publish error event
        self.event_bus.publish(Message(
            sender="mind",
            message_type="error",
            content={
                "error_type": "module_error",
                "module_type": module_type,
                "error_message": error_message,
                "error_time": error_time.isoformat()
            }
        ))

    def _handle_performance_metric(self, message: Message):
        """Handle performance metric events"""
        metric_info = message.content
        metric_type = metric_info.get("metric_type")
        metric_value = metric_info.get("metric_value")
        metric_time = datetime.now()
        
        if metric_type:
            if metric_type not in self._performance_metrics:
                self._performance_metrics[metric_type] = []
            self._performance_metrics[metric_type].append({
                "value": metric_value,
                "timestamp": metric_time.isoformat()
            })
            logger.info(f"Performance metric: {metric_type}, Value: {metric_value}")
        else:
            logger.info("Unspecified performance metric")
        
        # Publish metric event
        self.event_bus.publish(Message(
            sender="mind",
            message_type="performance_metric",
            content={
                "metric_type": metric_type,
                "metric_value": metric_value,
                "metric_time": metric_time.isoformat()
            }
        ))
