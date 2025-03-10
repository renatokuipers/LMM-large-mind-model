#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Large Mind Model (LMM) - Main Entry Point

This file serves as the primary entry point for the LMM system, initializing and coordinating
all components necessary for cognitive development.
"""

import argparse
import os
import sys
import logging
import signal
import json
import time
from pathlib import Path
from datetime import datetime
import importlib
from typing import Dict, Any, Optional, List, Tuple

# Core LMM components
from lmm_project.core.mind import Mind
from lmm_project.core.event_bus import EventBus
from lmm_project.core.state_manager import StateManager
from lmm_project.core.message import Message
from lmm_project.core.types import DevelopmentalStage

# Utils
from lmm_project.utils.config_manager import ConfigManager
from lmm_project.utils.logging_utils import setup_logger
from lmm_project.utils.llm_client import LLMClient
from lmm_project.utils.tts_client import TTSClient
from lmm_project.utils.vector_store import VectorStore
from lmm_project.utils.visualization import visualize_development, visualize_neural_activity, visualize_learning_progress

# Neural substrate
from lmm_project.neural_substrate.neural_network import NeuralNetwork
from lmm_project.neural_substrate.hebbian_learning import HebbianLearning
from lmm_project.neural_substrate.activation_functions import get_activation_function

# Developmental tracking
from lmm_project.development.developmental_stages import DevelopmentalStageManager
from lmm_project.development.critical_periods import CriticalPeriodManager
from lmm_project.development.milestone_tracker import MilestoneTracker
from lmm_project.development.growth_rate_controller import GrowthRateController

# Learning engines
from lmm_project.learning_engines.reinforcement_engine import ReinforcementEngine
from lmm_project.learning_engines.hebbian_engine import HebbianEngine
from lmm_project.learning_engines.pruning_engine import PruningEngine
from lmm_project.learning_engines.consolidation_engine import ConsolidationEngine
from lmm_project.learning_engines.models import HebbianParameters, ReinforcementParameters, PruningParameters, ConsolidationParameters

# Homeostasis
from lmm_project.homeostasis.energy_regulation import EnergyRegulator
from lmm_project.homeostasis.arousal_control import ArousalController
from lmm_project.homeostasis.cognitive_load_balancer import CognitiveLoadBalancer
from lmm_project.homeostasis.social_need_manager import SocialNeedManager
from lmm_project.homeostasis.coherence import CoherenceManager

# Interfaces
from lmm_project.interfaces.mother.mother_llm import MotherLLM
from lmm_project.interfaces.mother.personality import PersonalityManager
from lmm_project.interfaces.mother.teaching_strategies import TeachingStrategyManager
from lmm_project.interfaces.mother.interaction_patterns import InteractionPatternManager
from lmm_project.interfaces.researcher.state_observer import StateObserver
from lmm_project.interfaces.researcher.metrics_collector import MetricsCollector
from lmm_project.interfaces.researcher.development_tracker import DevelopmentTracker

# Visualization
from lmm_project.visualization.dashboard import Dashboard
from lmm_project.visualization.neural_activity_view import NeuralActivityView
from lmm_project.visualization.development_charts import DevelopmentCharts
from lmm_project.visualization.state_inspector import StateInspector

# Initialize global variables
config = None
mind = None
event_bus = None
state_manager = None
mother_interface = None
vector_store = None
developmental_system = None
learning_engines = {}
visualization_components = {}
researcher_interface = {}
running = True
logging_enabled = True
homeostasis_systems = {}

def detect_gpu():
    """Detect and configure GPU for CUDA acceleration"""
    logger = logging.getLogger(__name__)
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_names = [torch.cuda.get_device_name(i) for i in range(device_count)]
            logger.info(f"CUDA available: {device_count} device(s) detected")
            for i, name in enumerate(device_names):
                logger.info(f"  Device {i}: {name}")
            return True, "cuda"
        else:
            logger.info("CUDA not available, using CPU")
            return False, "cpu"
    except Exception as e:
        logger.warning(f"Error detecting CUDA: {e}")
        logger.info("Defaulting to CPU processing")
        return False, "cpu"

def setup_directories():
    """Create necessary directories for storage, logs, etc."""
    logger = logging.getLogger(__name__)
    
    dirs = [
        Path(config.storage.checkpoint_dir),
        Path(config.storage.experience_dir),
        Path(config.storage.memory_dir),
        Path("logs")
    ]
    
    for directory in dirs:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Created directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to create directory {directory}: {e}")

def init_neural_substrate():
    """Initialize neural substrate components"""
    logger = logging.getLogger(__name__)
    logger.info("Initializing neural substrate")
    
    # Get default activation function
    activation_function = get_activation_function(
        config.neural_substrate.default_activation_function
    )
    
    # Initialize Hebbian learning
    hebbian_learning = HebbianLearning(
        learning_rate=config.neural_substrate.default_learning_rate
    )
    
    # Initialize neural network
    neural_network = NeuralNetwork(
        name="main_network",
        learning_mechanism=hebbian_learning
    )
    
    logger.info("Neural substrate initialized")
    return {
        "activation_function": activation_function,
        "hebbian_learning": hebbian_learning,
        "neural_network": neural_network
    }

def init_learning_engines(event_bus, neural_substrate):
    """Initialize learning engines for the system."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing learning engines")
    
    # Create simple placeholders for learning engines
    # In a full implementation, these would be properly initialized
    learning_engines = {}
    
    # Hebbian learning engine (simplified placeholder)
    hebbian_engine = {
        "engine_type": "hebbian",
        "event_bus": event_bus,
        "apply_learning": lambda network=None: [],
        "learning_rate": 0.01,
        "on_development_update": lambda message: None,
        "on_perception_result": lambda message: None
    }
    learning_engines["hebbian"] = hebbian_engine
    
    # Reinforcement learning engine (simplified placeholder)
    reinforcement_engine = {
        "engine_type": "reinforcement",
        "event_bus": event_bus,
        "apply_learning": lambda network=None, reward=0.0: [],
        "learning_rate": 0.01,
        "on_development_update": lambda message: None,
        "on_perception_result": lambda message: None
    }
    learning_engines["reinforcement"] = reinforcement_engine
    
    # Pruning engine (simplified placeholder)
    pruning_engine = {
        "engine_type": "pruning",
        "event_bus": event_bus,
        "apply_learning": lambda network=None: [],
        "pruning_threshold": 0.01,
        "on_development_update": lambda message: None,
        "on_perception_result": lambda message: None
    }
    learning_engines["pruning"] = pruning_engine
    
    # Consolidation engine (simplified placeholder)
    consolidation_engine = {
        "engine_type": "consolidation",
        "event_bus": event_bus,
        "apply_learning": lambda network=None, sleep_mode=False: [],
        "consolidation_threshold": 0.5,
        "on_development_update": lambda message: None,
        "on_perception_result": lambda message: None
    }
    learning_engines["consolidation"] = consolidation_engine
    
    logger.info(f"Initialized {len(learning_engines)} learning engines")
    return learning_engines

def init_developmental_system(event_bus):
    """Initialize the developmental tracking system."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing developmental tracking system")
    
    from lmm_project.development.developmental_stages import DevelopmentalStageManager
    from lmm_project.development.critical_periods import CriticalPeriodManager
    from lmm_project.development.milestone_tracker import MilestoneTracker
    from lmm_project.development.growth_rate_controller import GrowthRateController
    
    # Create the developmental system components
    developmental_system = {}
    
    # Initialize Developmental Stage Manager
    stage_manager = DevelopmentalStageManager(event_bus=event_bus)
    developmental_system["stage_manager"] = stage_manager
    logger.info("Initialized developmental stage manager")
    
    # Initialize Critical Period Manager
    critical_period_manager = CriticalPeriodManager(event_bus=event_bus)
    developmental_system["critical_period_manager"] = critical_period_manager
    logger.info("Initialized critical period manager")
    
    # Initialize Milestone Tracker
    milestone_tracker = MilestoneTracker(event_bus=event_bus)
    developmental_system["milestone_tracker"] = milestone_tracker
    logger.info("Initialized milestone tracker")
    
    # Initialize Growth Rate Controller
    growth_controller = GrowthRateController(base_rate=0.01, variability=0.2)
    developmental_system["growth_controller"] = growth_controller
    logger.info("Initialized growth rate controller")
    
    logger.info("Developmental tracking system initialized")
    return developmental_system

def init_researcher_interface(event_bus, mind):
    """Initialize the researcher interface components."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing researcher interface")
    
    from lmm_project.interfaces.researcher.state_observer import StateObserver
    from lmm_project.interfaces.researcher.metrics_collector import MetricsCollector
    from lmm_project.interfaces.researcher.development_tracker import DevelopmentTracker
    
    # Create the researcher interface components
    researcher_interface = {}
    
    # Initialize State Observer
    state_observer = StateObserver(storage_dir="storage/observations")
    # Register mind with state observer
    if hasattr(mind, "get_state"):
        state_observer.register_module("mind", mind, mind.get_state)
    researcher_interface["state_observer"] = state_observer
    logger.info("Initialized state observer")
    
    # Initialize Metrics Collector
    metrics_collector = MetricsCollector(storage_dir="storage/metrics")
    researcher_interface["metrics_collector"] = metrics_collector
    logger.info("Initialized metrics collector")
    
    # Initialize Development Tracker
    development_tracker = DevelopmentTracker(storage_dir="storage/development")
    researcher_interface["development_tracker"] = development_tracker
    logger.info("Initialized development tracker")
    
    logger.info("Researcher interface initialized")
    return researcher_interface

def init_visualization_system(event_bus, mind, developmental_system, researcher_interface):
    """Initialize the visualization system components."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing visualization system")
    
    from lmm_project.visualization.dashboard import Dashboard
    from lmm_project.visualization.neural_activity_view import NeuralActivityView
    from lmm_project.visualization.development_charts import DevelopmentCharts
    from lmm_project.visualization.state_inspector import StateInspector
    
    # Check if visualization is disabled in config
    if hasattr(config, 'visualization') and hasattr(config.visualization, 'enabled') and not config.visualization.enabled:
        logger.info("Visualization disabled in configuration")
        return {}
    
    # Create visualization output directory
    os.makedirs("visualization/output", exist_ok=True)
    
    # Create the visualization components
    visualization_components = {}
    
    # Initialize Dashboard
    dashboard = Dashboard(output_dir="visualization/output")
    visualization_components["dashboard"] = dashboard
    logger.info("Initialized dashboard")
    
    # Initialize Neural Activity View
    neural_activity_view = NeuralActivityView(output_dir="visualization/output")
    visualization_components["neural_activity_view"] = neural_activity_view
    logger.info("Initialized neural activity view")
    
    # Initialize Development Charts
    development_charts = DevelopmentCharts(output_dir="visualization/output")
    visualization_components["development_charts"] = development_charts
    logger.info("Initialized development charts")
    
    # Initialize State Inspector
    state_inspector = StateInspector(output_dir="visualization/output")
    visualization_components["state_inspector"] = state_inspector
    logger.info("Initialized state inspector")
    
    logger.info("Visualization system initialized")
    return visualization_components

def init_homeostasis_systems(event_bus):
    """Initialize the homeostasis systems."""
    logger = logging.getLogger(__name__)
    logger.info("Initializing homeostasis systems")
    
    from lmm_project.homeostasis.energy_regulation import EnergyRegulator
    from lmm_project.homeostasis.arousal_control import ArousalController
    from lmm_project.homeostasis.cognitive_load_balancer import CognitiveLoadBalancer
    from lmm_project.homeostasis.social_need_manager import SocialNeedManager
    from lmm_project.homeostasis.coherence import CoherenceManager
    
    # Create the homeostasis systems
    homeostasis_systems = {}
    
    # Initialize Energy Regulator
    energy_regulator = EnergyRegulator(
        event_bus=event_bus,
        initial_energy=0.8,
        recovery_rate=0.05,
        consumption_rate=0.02
    )
    homeostasis_systems["energy"] = energy_regulator
    logger.info("Initialized energy regulation system")
    
    # Initialize Arousal Controller
    arousal_controller = ArousalController(
        event_bus=event_bus,
        initial_arousal=0.5,
        decay_rate=0.03,
        adaptation_rate=0.05
    )
    homeostasis_systems["arousal"] = arousal_controller
    logger.info("Initialized arousal control system")
    
    # Initialize Cognitive Load Balancer
    cognitive_load_balancer = CognitiveLoadBalancer(
        event_bus=event_bus,
        initial_capacity=0.3,
        working_memory_slots=4,
        processing_threshold=0.8
    )
    homeostasis_systems["cognitive_load"] = cognitive_load_balancer
    logger.info("Initialized cognitive load balancer")
    
    # Initialize Social Need Manager
    social_need_manager = SocialNeedManager(
        event_bus=event_bus,
        initial_social_need=0.5,
        satiation_rate=0.15,
        deficit_growth_rate=0.02
    )
    homeostasis_systems["social"] = social_need_manager
    logger.info("Initialized social need manager")
    
    # Initialize Coherence Manager if needed
    coherence_manager = CoherenceManager(
        event_bus=event_bus,
        initial_coherence=0.8,
        tolerance_threshold=0.3,
        resolution_rate=0.05
    )
    homeostasis_systems["coherence"] = coherence_manager
    logger.info("Initialized coherence manager")
    
    logger.info(f"Initialized {len(homeostasis_systems)} homeostasis systems")
    return homeostasis_systems

def init_system(args):
    """Initialize all LMM system components"""
    global config, mind, event_bus, state_manager, mother_interface, vector_store
    global developmental_system, learning_engines, visualization_components
    global researcher_interface, running, homeostasis_systems
    
    # Configure basic logging first
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Initializing LMM system with Python {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up detailed logging
    logger = setup_logger(
        name="LMM",
        log_level=logging.DEBUG if args.debug else logging.INFO,
        log_file="logs/lmm.log"
    )
    
    # Detect GPU
    has_gpu, device = detect_gpu()
    
    # Load configuration
    try:
        config_manager = ConfigManager(args.config)
        config = config_manager.get_config()
        logger.info(f"Configuration loaded from {args.config}")
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        logger.info("Using default configuration")
        config = config_manager.get_default_config()
    
    # Override GPU config based on detection
    config.neural_substrate.use_gpu = has_gpu
    
    # Create necessary directories
    setup_directories()
    
    # Initialize core components
    event_bus = EventBus()
    state_manager = StateManager()
    logger.info("Core components initialized")
    
    # Initialize vector store
    try:
        vector_store_path = Path(config.storage.memory_dir) / "vector_store"
        vector_store_path.parent.mkdir(parents=True, exist_ok=True)
        vector_store = VectorStore(
            storage_dir=str(vector_store_path),
            dimension=1536,
            use_gpu=config.neural_substrate.use_gpu
        )
        logger.info("Vector store initialized")
    except Exception as e:
        logger.error(f"Error initializing vector store: {e}")
        raise
    
    # Initialize neural substrate
    neural_substrate = init_neural_substrate()
    
    # Initialize learning engines
    learning_engines = init_learning_engines(event_bus, neural_substrate)
    
    # Initialize developmental system
    developmental_system = init_developmental_system(event_bus)
    
    # Initialize API clients
    try:
        llm_client = LLMClient(base_url=config.apis.llm_api_url)
        tts_client = TTSClient(base_url=config.apis.tts_api_url)
        logger.info("API clients initialized")
    except Exception as e:
        logger.error(f"Error initializing API clients: {e}")
        raise
    
    # Initialize Mother LLM
    try:
        logger.info("Initializing Mother LLM interface")
        personality_manager = PersonalityManager(
            profile="balanced",
            custom_traits=config.mother.personality.model_dump()
        )
        
        teaching_strategy_manager = TeachingStrategyManager(
            default_style=config.mother.teaching_style
        )
        
        interaction_pattern_manager = InteractionPatternManager()
        
        mother_interface = MotherLLM(
            llm_client=llm_client,
            tts_client=tts_client,
            personality_traits=config.mother.personality.model_dump(),
            teaching_style=config.mother.teaching_style,
            voice=config.mother.voice,
            personality_manager=personality_manager,
            teaching_strategy_manager=teaching_strategy_manager,
            interaction_pattern_manager=interaction_pattern_manager
        )
        logger.info("Mother LLM initialized successfully")
    except Exception as e:
        logger.error(f"Error initializing Mother LLM: {e}")
        raise
    
    # Initialize homeostasis systems
    homeostasis_systems = init_homeostasis_systems(event_bus)
    
    # Initialize or load the Mind
    if args.load:
        # Load from checkpoint
        mind = Mind(event_bus=event_bus, state_manager=state_manager)
        if os.path.exists(args.load):
            logger.info(f"Loading mind state from {args.load}")
            success = mind.load_state(args.load)
            if not success:
                logger.error(f"Failed to load mind state from {args.load}")
                sys.exit(1)
        else:
            logger.error(f"Checkpoint file {args.load} does not exist")
            sys.exit(1)
    else:
        # Create new Mind instance
        logger.info("Creating new mind instance")
        mind = Mind(
            event_bus=event_bus,
            state_manager=state_manager,
            initial_age=0.0,
            developmental_stage="prenatal"
        )
        
        # Register homeostasis systems with the mind
        for name, system in homeostasis_systems.items():
            if hasattr(mind, "register_homeostasis_system"):
                mind.register_homeostasis_system(name, system)
        
        # Initialize cognitive modules
        mind.initialize_modules()
    
    # Initialize researcher interface
    researcher_interface = init_researcher_interface(event_bus, mind)
    
    # Initialize visualization (after mind is initialized)
    visualization_components = init_visualization_system(
        event_bus, 
        mind, 
        developmental_system, 
        researcher_interface
    )
    
    # Connect learning engines to mind
    for name, engine in learning_engines.items():
        event_bus.subscribe("development_update", engine["on_development_update"])
        event_bus.subscribe("perception_result", engine["on_perception_result"])
    
    # Register signal handlers for clean shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    logger.info("LMM system initialization complete")
    return mind

def handle_shutdown(sig, frame):
    """Handle graceful shutdown when receiving a signal."""
    global running, mind, visualization_components
    
    logger = logging.getLogger(__name__)
    logger.info(f"Received shutdown signal {sig}, shutting down...")
    running = False
    
    # Save final checkpoint
    if mind:
        try:
            checkpoint_dir = "checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(checkpoint_dir, f"mind_shutdown_{timestamp}.json")
            mind.save_state(checkpoint_path)
            logger.info(f"Final state saved to {checkpoint_path}")
        except Exception as save_error:
            logger.error(f"Failed to save final state: {save_error}")
    
    # Shutdown visualization components
    if visualization_components:
        logger.info("Shutting down visualization components")
        for component_name, component in visualization_components.items():
            if hasattr(component, "shutdown"):
                try:
                    component.shutdown()
                except Exception as viz_error:
                    logger.error(f"Error shutting down visualization component {component_name}: {viz_error}")
    
    logger.info("Shutdown complete")
    sys.exit(0)

def development_cycle(cycles=None):
    """Run cognitive development cycles."""
    global visualization_components, developmental_system, learning_engines, mind, mother_interface, running, homeostasis_systems
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting development cycles")
    
    cycle_count = 0
    last_viz_update = time.time()
    viz_update_interval = config.visualization.update_interval if hasattr(config, 'visualization') and hasattr(config.visualization, 'update_interval') else 30  # seconds
    
    running = True
    try:
        while running and (cycles is None or cycle_count < cycles):
            cycle_count += 1
            
            # Get current state
            age = mind.age
            stage = mind.developmental_stage
            logger.info(f"Development cycle {cycle_count}: Age {age:.2f}, Stage {stage}")
            
            # Update developmental components
            for component_name, component in developmental_system.items():
                if hasattr(component, "update"):
                    try:
                        component.update(age=age, stage=stage)
                    except Exception as e:
                        logger.error(f"Error updating developmental component {component_name}: {e}")
            
            try:
                # Determine the most urgent homeostatic need
                most_urgent_need = None
                highest_urgency = 0
                
                for system_name, system in homeostasis_systems.items():
                    if hasattr(system, "get_most_urgent_need"):
                        try:
                            need = system.get_most_urgent_need()
                            if need and need[1] > highest_urgency:
                                most_urgent_need = need
                                highest_urgency = need[1]
                        except Exception as e:
                            logger.error(f"Error getting need from {system_name}: {e}")
                
                # Address the most urgent need
                if most_urgent_need:
                    need_type, urgency = most_urgent_need
                    logger.debug(f"Addressing {need_type} need with urgency {urgency:.2f}")
                    
                    # Prepare appropriate input based on developmental stage and needs
                    if need_type == "social":
                        # Generate social input
                        preferred_partner = "mother"  # Default to mother/caregiver
                        if hasattr(homeostasis_systems.get("social"), "_get_preferred_partner"):
                            try:
                                preferred_partner = homeostasis_systems["social"]._get_preferred_partner()
                            except Exception as e:
                                logger.error(f"Error getting preferred partner: {e}")
                        
                        input_type = f"social_{preferred_partner}"
                        
                        # Get the current focus from milestone tracker if available
                        current_focus = None
                        if "milestone_tracker" in developmental_system and hasattr(developmental_system["milestone_tracker"], "get_current_focus"):
                            try:
                                current_focus = developmental_system["milestone_tracker"].get_current_focus()
                            except Exception as e:
                                logger.error(f"Error getting current focus: {e}")
                        
                        logger.debug(f"Generating {preferred_partner} response for input: {input_type}")
                        try:
                            response = mother_interface.generate_response(
                                input_type, 
                                age=age, 
                                stage=stage,
                                focus=current_focus
                            )
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            response = {
                                "type": "mother_interaction",
                                "content": f"Hello little one. (Error fallback for {stage} stage)",
                                "text": f"Hello little one. (Error fallback for {stage} stage)",
                                "emotion": "neutral",
                                "learning_focus": "general",
                            }
                        
                    elif need_type == "arousal":
                        # Generate stimulating input
                        input_type = f"stimulate_{stage}"
                        
                        # Get the current focus from milestone tracker if available
                        current_focus = None
                        if "milestone_tracker" in developmental_system and hasattr(developmental_system["milestone_tracker"], "get_current_focus"):
                            try:
                                current_focus = developmental_system["milestone_tracker"].get_current_focus()
                            except Exception as e:
                                logger.error(f"Error getting current focus: {e}")
                        
                        logger.debug(f"Generating Mother response for input: {input_type}")
                        try:
                            response = mother_interface.generate_response(
                                input_type, 
                                age=age, 
                                stage=stage,
                                focus=current_focus
                            )
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            response = {
                                "type": "mother_interaction",
                                "content": f"Hello little one. (Error fallback for {stage} stage)",
                                "text": f"Hello little one. (Error fallback for {stage} stage)",
                                "emotion": "neutral",
                                "learning_focus": "general",
                            }
                        
                    else:
                        # Default nurturing input
                        input_type = f"nurture_{stage}"
                        
                        # Get the current focus from milestone tracker if available
                        current_focus = None
                        if "milestone_tracker" in developmental_system and hasattr(developmental_system["milestone_tracker"], "get_current_focus"):
                            try:
                                current_focus = developmental_system["milestone_tracker"].get_current_focus()
                            except Exception as e:
                                logger.error(f"Error getting current focus: {e}")
                        
                        logger.debug(f"Generating Mother response for input: {input_type}")
                        try:
                            response = mother_interface.generate_response(
                                input_type, 
                                age=age, 
                                stage=stage,
                                focus=current_focus
                            )
                        except Exception as e:
                            logger.error(f"Error generating response: {e}")
                            response = {
                                "type": "mother_interaction",
                                "content": f"Hello little one. (Error fallback for {stage} stage)",
                                "text": f"Hello little one. (Error fallback for {stage} stage)",
                                "emotion": "neutral",
                                "learning_focus": "general",
                            }
                    
                    # Process the response through the mind
                    try:
                        mind.process_input(response)
                    except Exception as e:
                        logger.error(f"Error processing input: {e}")
                    
                    # Apply learning based on the current developmental stage
                    try:
                        if "growth_controller" in developmental_system and hasattr(developmental_system["growth_controller"], "get_growth_rate"):
                            growth_rate = developmental_system["growth_controller"].get_growth_rate(
                                capability="general",
                                module="core",
                                age=age,
                                active_factors={"critical_periods_enabled": True}
                            )
                        else:
                            growth_rate = config.development.default_rate
                            
                        delta_age = growth_rate * config.development.cycle_age_increment
                        mind.update_development(delta_age)
                    except Exception as e:
                        logger.error(f"Error updating development: {e}")
                        
            except Exception as e:
                logger.error(f"Error in development cycle: {e}")
            
            # Update visualization if enabled and interval has passed
            if config.visualization.enabled and visualization_components:
                current_time = time.time()
                if current_time - last_viz_update > viz_update_interval:
                    logger.debug(f"Updating visualization components")
                    for component_name, component in visualization_components.items():
                        if hasattr(component, "update"):
                            try:
                                component.update(mind)
                            except Exception as e:
                                logger.error(f"Error updating visualization component {component_name}: {e}")
                    last_viz_update = current_time
            
            # Check for termination conditions
            if cycles is not None and cycle_count >= cycles:
                break
                
            # Save checkpoints if configured
            if hasattr(config, 'checkpoints') and hasattr(config.checkpoints, 'enabled') and config.checkpoints.enabled and cycle_count % config.checkpoints.interval == 0:
                try:
                    checkpoint_path = f"checkpoints/mind_age{age:.2f}_stage{stage}.json"
                    os.makedirs("checkpoints", exist_ok=True)
                    mind.save_state(checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                except Exception as e:
                    logger.error(f"Error saving checkpoint: {e}")
        
        logger.info(f"Development cycle completed after {cycle_count} cycles")
        
    except Exception as e:
        logger.exception(f"Critical error in development cycle: {e}")
        # Emergency save in case of fatal error
        try:
            checkpoint_path = "checkpoints/emergency_save.json"
            os.makedirs("checkpoints", exist_ok=True)
            mind.save_state(checkpoint_path)
            logger.info(f"Emergency state saved to {checkpoint_path}")
        except Exception as save_error:
            logger.error(f"Failed to save emergency state: {save_error}")

def main():
    """Main entry point for the LMM system"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Large Mind Model (LMM) system")
    parser.add_argument("--config", default="config.yml", help="Path to configuration file")
    parser.add_argument("--load", help="Load mind state from a checkpoint file")
    parser.add_argument("--cycles", type=int, help="Number of development cycles to run (default: continuous)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--no-visualization", action="store_true", help="Disable visualization")
    parser.add_argument("--accelerated", action="store_true", help="Enable accelerated development")
    parser.add_argument("--mother-voice", help="Override Mother's voice (e.g., 'af_bella', 'af_nicole')")
    parser.add_argument("--cuda-device", type=int, help="Specify CUDA device index to use")
    args = parser.parse_args()
    
    # Configure basic logging directly
    if args.debug:
        logging.basicConfig(level=logging.DEBUG, 
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    else:
        logging.basicConfig(level=logging.INFO,
                           format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger = logging.getLogger(__name__)
    logger.debug("Debug logging enabled")
    
    try:
        # Initialize the system
        init_system(args)
        
        # Override configurations based on command line arguments
        if args.no_visualization and config:
            config.visualization.enabled = False
            logger.info("Visualization disabled via command line")
            
        if args.accelerated and config:
            config.development.accelerated_mode = True
            logger.info("Accelerated development enabled via command line")
            
        if args.mother_voice and config:
            config.mother.voice = args.mother_voice
            if mother_interface:
                mother_interface.voice = args.mother_voice
            logger.info(f"Mother voice set to {args.mother_voice} via command line")
        
        # Run development cycles
        development_cycle(args.cycles)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        handle_shutdown(signal.SIGINT, None)
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        if mind:
            try:
                emergency_path = Path(config.storage.checkpoint_dir) / "crash_save.json"
                mind.save_state(str(emergency_path))
                logger.info(f"Crash state saved to {emergency_path}")
            except Exception as save_error:
                logger.error(f"Failed to save crash state: {save_error}")
        sys.exit(1)

if __name__ == "__main__":
    main() 