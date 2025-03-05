import os
import sys
import time
import json
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Literal, TypeVar, Generic
from datetime import datetime, timedelta
import threading
import traceback
import uuid
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
import dash.dependencies as dd
from pydantic import (
    BaseModel, Field, validator, root_validator, field_validator,
    create_model, ValidationError, conlist
)
import torch

# Add parent directory to path to import from sibling packages
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the Mind class and other components
from neural_child.mind.mind import Mind
from neural_child.mind.base import MindState, InteractionState
from neural_child.cognition.cognitive_component import CognitiveComponent
from neural_child.emotion.emotional_component import EmotionalComponent
from neural_child.language.language_component import LanguageComponent
from neural_child.memory.memory_component import MemoryComponent
from neural_child.social.social_component import SocialComponent
from utils.config import (
    DevelopmentStageConfig, MotherPersonalityConfig, 
    NeuralChildConfig, LLMConfig, DashboardConfig,
    GlobalConfig, DEFAULT_CONFIG
)

# Set default template for all plotly figures
px.defaults.template = "plotly_dark"

# Define Pydantic models for dashboard state management
class ComponentActivation(BaseModel):
    """Model representing a component's activation during processing."""
    name: str
    activation_levels: List[float] = Field(default_factory=list)
    timestamps: List[float] = Field(default_factory=list)

class InferenceSession(BaseModel):
    """Model representing an inference session with the LMM."""
    id: str
    user_input: str
    lmm_response: str
    timestamp: float
    component_activations: Dict[str, ComponentActivation] = Field(default_factory=dict)
    processing_time_ms: float

class ModelCheckpoint(BaseModel):
    """Model representing a saved checkpoint of the Mind."""
    id: str = Field(..., description="Unique identifier for the checkpoint")
    timestamp: float = Field(..., description="When the checkpoint was created")
    age_months: float = Field(..., description="Age of the Mind in months")
    developmental_stage: str = Field(..., description="Current developmental stage")
    vocabulary_size: int = Field(..., description="Size of vocabulary")
    file_path: Path = Field(..., description="Path to the saved model files")
    
    # Development metrics for key capabilities
    language_development: float = Field(..., ge=0.0, le=1.0)
    cognitive_development: float = Field(..., ge=0.0, le=1.0)
    emotional_development: float = Field(..., ge=0.0, le=1.0)
    social_development: float = Field(..., ge=0.0, le=1.0)
    memory_development: float = Field(..., ge=0.0, le=1.0)
    
    # Whether this model meets inference requirements
    inference_capable: bool = Field(False)
    
    @field_validator('inference_capable', mode='after')
    def validate_inference_capability(cls, v, info):
        """Determine if model meets minimum requirements for inference."""
        values = info.data
        if not all(k in values for k in ['language_development', 'cognitive_development']):
            return False
            
        # Define minimum thresholds for inference capability
        language_threshold = 0.7
        cognitive_threshold = 0.6
        vocabulary_threshold = 1000
        age_threshold = 36.0  # 3 years
        
        # Check if model meets thresholds
        if (values.get('language_development', 0) >= language_threshold and
            values.get('cognitive_development', 0) >= cognitive_threshold and
            values.get('vocabulary_size', 0) >= vocabulary_threshold and
            values.get('age_months', 0) >= age_threshold):
            return True
        return False

# Dashboard Mode and Training Status literals
DashboardMode = Literal["training", "management", "inference"]
TrainingStatus = Literal["idle", "running", "paused", "stopped"]

class DevelopmentalData(BaseModel):
    """Time series data for developmental metrics."""
    timestamps: List[float] = Field(default_factory=list)
    age_months: List[float] = Field(default_factory=list)
    developmental_stages: List[str] = Field(default_factory=list)
    
    # Language metrics
    receptive_language: List[float] = Field(default_factory=list)
    expressive_language: List[float] = Field(default_factory=list)
    vocabulary_size: List[int] = Field(default_factory=list)
    
    # Emotional metrics
    basic_emotions: List[float] = Field(default_factory=list)
    emotional_regulation: List[float] = Field(default_factory=list)
    emotional_complexity: List[float] = Field(default_factory=list)
    
    # Cognitive metrics
    attention: List[float] = Field(default_factory=list)
    memory: List[float] = Field(default_factory=list)
    problem_solving: List[float] = Field(default_factory=list)
    abstract_thinking: List[float] = Field(default_factory=list)
    
    # Social metrics
    attachment: List[float] = Field(default_factory=list)
    social_awareness: List[float] = Field(default_factory=list)
    empathy: List[float] = Field(default_factory=list)
    theory_of_mind: List[float] = Field(default_factory=list)
    
    def add_data_point(self, mind_state: MindState, timestamp: float):
        """Add a new data point from the current mind state."""
        self.timestamps.append(timestamp)
        self.age_months.append(mind_state.age_months)
        self.developmental_stages.append(mind_state.developmental_stage)
        
        # Extract and add metrics from mind_state
        dev_metrics = mind_state.developmental_metrics
        
        # Language metrics
        self.receptive_language.append(dev_metrics["language"].get("receptive_language", 0.0))
        self.expressive_language.append(dev_metrics["language"].get("expressive_language", 0.0))
        self.vocabulary_size.append(mind_state.vocabulary_size)
        
        # Emotional metrics
        self.basic_emotions.append(dev_metrics["emotional"].get("basic_emotions", 0.0))
        self.emotional_regulation.append(dev_metrics["emotional"].get("emotional_regulation", 0.0))
        self.emotional_complexity.append(dev_metrics["emotional"].get("emotional_complexity", 0.0))
        
        # Cognitive metrics
        self.attention.append(dev_metrics["cognitive"].get("attention", 0.0))
        self.memory.append(dev_metrics["cognitive"].get("memory", 0.0))
        self.problem_solving.append(dev_metrics["cognitive"].get("problem_solving", 0.0))
        self.abstract_thinking.append(dev_metrics["cognitive"].get("abstract_thinking", 0.0))
        
        # Social metrics
        self.attachment.append(dev_metrics["social"].get("attachment", 0.0))
        self.social_awareness.append(dev_metrics["social"].get("social_awareness", 0.0))
        self.empathy.append(dev_metrics["social"].get("empathy", 0.0))
        self.theory_of_mind.append(dev_metrics["social"].get("theory_of_mind", 0.0))
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert the developmental data to a pandas DataFrame."""
        df = pd.DataFrame({
            'timestamp': self.timestamps,
            'age_months': self.age_months,
            'developmental_stage': self.developmental_stages,
            'receptive_language': self.receptive_language,
            'expressive_language': self.expressive_language,
            'vocabulary_size': self.vocabulary_size,
            'basic_emotions': self.basic_emotions,
            'emotional_regulation': self.emotional_regulation,
            'emotional_complexity': self.emotional_complexity,
            'attention': self.attention,
            'memory': self.memory,
            'problem_solving': self.problem_solving,
            'abstract_thinking': self.abstract_thinking,
            'attachment': self.attachment,
            'social_awareness': self.social_awareness,
            'empathy': self.empathy,
            'theory_of_mind': self.theory_of_mind
        })
        return df

class InteractionHistory(BaseModel):
    """History of Mother-Child interactions."""
    interactions: List[Dict[str, Any]] = Field(default_factory=list, 
                                             description="List of interaction records")
    max_history: int = Field(100, ge=10, description="Maximum number of interactions to store")
    
    def add_interaction(self, interaction: InteractionState):
        """Add a new interaction to the history."""
        interaction_dict = interaction.model_dump()
        self.interactions.append(interaction_dict)
        
        # Limit history size
        if len(self.interactions) > self.max_history:
            self.interactions = self.interactions[-self.max_history:]
    
    def get_recent_interactions(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get the most recent interactions."""
        return self.interactions[-min(count, len(self.interactions)):]

class DashboardState(BaseModel):
    """Global state for the dashboard application."""
    # Basic configuration
    mode: DashboardMode = Field("training", description="Current dashboard mode")
    training_status: TrainingStatus = Field("idle", description="Current training status")
    last_update_time: float = Field(default_factory=time.time, description="Last state update time")
    
    # Mind instance configuration
    neural_child_config: NeuralChildConfig = Field(DEFAULT_CONFIG.neural_child, 
                                                 description="Neural Child configuration")
    mother_personality: MotherPersonalityConfig = Field(DEFAULT_CONFIG.mother_personality,
                                                      description="Mother personality configuration")
    llm_config: LLMConfig = Field(DEFAULT_CONFIG.llm, description="LLM configuration")
    
    # Data storage
    data_dir: Path = Field(Path("data"), description="Directory for data storage")
    models_dir: Path = Field(Path("models"), description="Directory for model storage")
    logs_dir: Path = Field(Path("logs"), description="Directory for log storage")
    
    # Runtime data
    developmental_data: DevelopmentalData = Field(default_factory=DevelopmentalData,
                                               description="Time series developmental data")
    interaction_history: InteractionHistory = Field(default_factory=InteractionHistory,
                                                 description="History of interactions")
    available_checkpoints: List[ModelCheckpoint] = Field(default_factory=list,
                                                      description="Available model checkpoints")
    active_checkpoint: Optional[ModelCheckpoint] = Field(None,
                                                      description="Currently loaded checkpoint")
    
    # Auto-save configuration
    auto_save_interval: int = Field(300, ge=60, description="Auto-save interval in seconds")
    last_auto_save_time: float = Field(default_factory=time.time, 
                                     description="Last auto-save time")
    
    # Training settings
    target_age_months: Optional[float] = Field(None, description="Target age to train to")
    speed_multiplier: float = Field(1.0, ge=0.1, le=100.0, 
                                  description="Training speed multiplier")
    
    # Metrics display settings
    metrics_history_limit: int = Field(1000, ge=100, le=10000,
                                     description="Maximum metrics history points")
    update_interval_ms: int = Field(1000, ge=100, le=10000, 
                                  description="UI update interval in milliseconds")
    
    # Inference settings
    inference_session_id: Optional[str] = Field(None, description="Current inference session ID")
    inference_history: List[InferenceSession] = Field(default_factory=list,
                                                   description="History of inference interactions")
    
    # Method to update from mind state
    def update_from_mind(self, mind: Mind):
        """Update dashboard state from the current Mind state."""
        # Get current timestamp
        current_time = time.time()
        
        # Update timestamp
        self.last_update_time = current_time
        
        # Add data point to developmental metrics
        self.developmental_data.add_data_point(mind.mind_state, current_time)
        
        # Check if it's time for auto-save
        if current_time - self.last_auto_save_time > self.auto_save_interval:
            # Trigger auto-save
            self._auto_save_model(mind)
            self.last_auto_save_time = current_time
    
    def _auto_save_model(self, mind: Mind):
        """Automatically save the current model state."""
        try:
            # Generate a unique identifier for this checkpoint
            checkpoint_id = f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Create a directory for this checkpoint
            checkpoint_dir = self.models_dir / checkpoint_id
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save the mind state
            mind.save()
            
            # Get current development metrics
            status = mind.get_status()
            dev_metrics = status["developmental_metrics"]
            
            # Create a checkpoint record
            checkpoint = ModelCheckpoint(
                id=checkpoint_id,
                timestamp=time.time(),
                age_months=mind.mind_state.age_months,
                developmental_stage=mind.mind_state.developmental_stage,
                vocabulary_size=mind.mind_state.vocabulary_size,
                file_path=checkpoint_dir,
                language_development=(
                    dev_metrics["language"]["receptive_language"] + 
                    dev_metrics["language"]["expressive_language"]
                ) / 2,
                cognitive_development=(
                    dev_metrics["cognitive"]["attention"] + 
                    dev_metrics["cognitive"]["memory"] + 
                    dev_metrics["cognitive"]["problem_solving"] + 
                    dev_metrics["cognitive"]["abstract_thinking"]
                ) / 4,
                emotional_development=(
                    dev_metrics["emotional"]["basic_emotions"] + 
                    dev_metrics["emotional"]["emotional_regulation"] + 
                    dev_metrics["emotional"]["emotional_complexity"]
                ) / 3,
                social_development=(
                    dev_metrics["social"]["attachment"] + 
                    dev_metrics["social"]["social_awareness"] + 
                    dev_metrics["social"]["empathy"] + 
                    dev_metrics["social"]["theory_of_mind"]
                ) / 4,
                memory_development=0.5  # Approximation based on memory component metrics
            )
            
            # Add to available checkpoints
            self.available_checkpoints.append(checkpoint)
            
            print(f"Auto-saved checkpoint: {checkpoint_id}")
            
        except Exception as e:
            print(f"Error during auto-save: {e}")

# Initialize the Dash application with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "Neural Child Dashboard"

# Create shared global state
dashboard_state = DashboardState()

# Initialize Mind instance
mind = None
training_thread = None
stop_training_event = threading.Event()

# Add a global variable to track training errors
training_error = None

# Add a helper function to ensure all PyTorch tensors are on the correct device
def ensure_device_consistency():
    """Helper function to ensure all PyTorch tensors are on the same device."""
    global mind
    
    if mind is None:
        return
    
    try:
        # Get the current device
        device = mind.device
        print(f"Ensuring all tensors are on device: {device}")
        
        # Use modern PyTorch API instead of deprecated functions
        # This also helps prevent device inconsistency issues
        if device == "cuda" and torch.cuda.is_available():
            # Set default device and dtype separately (modern approach)
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
        else:
            # Fall back to CPU if CUDA not available or device is CPU
            device = "cpu"
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
            # Update mind device if needed
            mind.device = device
        
        # Force device consistency for all neural components
        if hasattr(mind, "cognitive_component") and mind.cognitive_component:
            mind.cognitive_component.to(device)
            # Also move optimizer if it exists
            if hasattr(mind.cognitive_component, "optimizer"):
                for param_group in mind.cognitive_component.optimizer.param_groups:
                    for param in param_group['params']:
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(device)
        
        if hasattr(mind, "language_component") and mind.language_component:
            mind.language_component.to(device)
            if hasattr(mind.language_component, "optimizer"):
                for param_group in mind.language_component.optimizer.param_groups:
                    for param in param_group['params']:
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(device)
            
        if hasattr(mind, "emotional_component") and mind.emotional_component:
            mind.emotional_component.to(device)
            if hasattr(mind.emotional_component, "optimizer"):
                for param_group in mind.emotional_component.optimizer.param_groups:
                    for param in param_group['params']:
                        param.data = param.data.to(device)
                        if param.grad is not None:
                            param.grad.data = param.grad.data.to(device)
            
        if hasattr(mind, "social_component") and mind.social_component:
            mind.social_component.to(device)
            
        if hasattr(mind, "memory_component") and mind.memory_component:
            mind.memory_component.to(device)
            
        print("Device consistency enforced")
    except Exception as e:
        print(f"Error enforcing device consistency: {e}")
        traceback.print_exc()

# Call the function after initializing mind
def initialize_mind(config=None, load_existing=False):
    """Initialize or reinitialize the Mind instance."""
    global mind, training_error
    
    if config is None:
        config = dashboard_state.neural_child_config
    
    # Update config for development speed
    config.development_speed = dashboard_state.speed_multiplier
    
    # Create data directories
    base_path = dashboard_state.data_dir / "neural_child"
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Use modern PyTorch API instead of deprecated function
    torch.set_default_device(device)
    torch.set_default_dtype(torch.float32)
    
    try:
        # Initialize mind
        mind = Mind(
            config=config,
            mother_personality=dashboard_state.mother_personality,
            llm_config=dashboard_state.llm_config,
            device=device,
            base_path=base_path,
            load_existing=load_existing
        )
        
        # Enforce device consistency
        ensure_device_consistency()
        
        # Update dashboard state from mind
        dashboard_state.update_from_mind(mind)
        
        # Reset error state since initialization was successful
        training_error = None
        
        return True
    except Exception as e:
        print(f"Error initializing Mind: {e}")
        training_error = f"Error initializing Mind: {str(e)}"
        traceback.print_exc()
        return False

def training_loop():
    """Background thread function for continuous training."""
    global mind, dashboard_state, stop_training_event, training_error
    
    print("Starting training loop...")
    training_error = None  # Reset error state
    
    # Track consecutive errors to prevent infinite error loops
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    try:
        while not stop_training_event.is_set():
            # Check if we've reached target age (if set)
            if dashboard_state.target_age_months is not None:
                if mind.mind_state.age_months >= dashboard_state.target_age_months:
                    print(f"Reached target age: {dashboard_state.target_age_months} months")
                    dashboard_state.training_status = "stopped"
                    break
            
            try:
                # Process one interaction
                interaction = mind.interact_with_mother()
                
                # Add to interaction history
                dashboard_state.interaction_history.add_interaction(interaction)
                
                # Reset consecutive error counter on success
                consecutive_errors = 0
                
            except RuntimeError as e:
                error_message = str(e)
                # Handle device mismatch errors
                if "Expected all tensors to be on the same device" in error_message:
                    print("Device mismatch detected. Attempting to fix...")
                    training_error = "Device mismatch detected. Attempting to fix by switching to CPU..."
                    # Set all tensors to CPU to recover
                    device = "cpu"
                    torch.set_default_device(device)
                    torch.set_default_dtype(torch.float32)
                    
                    # Reinitialize the mind with CPU only
                    success = initialize_mind(load_existing=True)
                    if not success:
                        print("Failed to recover from device error")
                        training_error = "Failed to recover from device error. Training stopped."
                        dashboard_state.training_status = "stopped"
                        break
                    print("Recovery successful, continuing with CPU")
                    training_error = "Recovery successful, continuing with CPU."
                
                # Handle matrix dimension mismatch errors
                elif "mat1 and mat2 shapes cannot be multiplied" in error_message or "size mismatch" in error_message:
                    consecutive_errors += 1
                    print(f"Matrix dimension mismatch detected ({consecutive_errors}/{max_consecutive_errors}). Attempting to fix...")
                    training_error = f"Matrix dimension mismatch detected. Attempting to reset network architecture..."
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors. Stopping training to prevent infinite loop.")
                        training_error = "Too many consecutive matrix dimension errors. Training stopped. Try reloading a checkpoint."
                        dashboard_state.training_status = "stopped"
                        break
                    
                    # Use our specialized reset function to fix dimension issues
                    success = reset_model_dimensions()
                    
                    if not success:
                        print("Failed to recover from dimension mismatch")
                        training_error = "Failed to recover from dimension mismatch. Training stopped."
                        dashboard_state.training_status = "stopped"
                        break
                    print("Model reset successful, continuing with new architecture")
                    training_error = "Model reset successful, continuing with new architecture."
                
                else:
                    # For other runtime errors, log and continue
                    consecutive_errors += 1
                    print(f"Error during interaction: {error_message}")
                    training_error = f"Error during interaction: {error_message}"
                    traceback.print_exc()
                    
                    if consecutive_errors >= max_consecutive_errors:
                        print("Too many consecutive errors. Stopping training.")
                        training_error = "Too many consecutive errors. Training stopped."
                        dashboard_state.training_status = "stopped"
                        break
            
            # Sleep briefly to prevent maxing out CPU
            time.sleep(0.01)
        
        print("Training loop stopped")
        
    except Exception as e:
        print(f"Error in training loop: {e}")
        training_error = f"Error in training loop: {str(e)}"
        traceback.print_exc()
        dashboard_state.training_status = "stopped"

def start_training():
    """Start the training process in a background thread."""
    global training_thread, stop_training_event, dashboard_state, training_error, mind
    
    if dashboard_state.training_status == "running":
        print("Training already running")
        return
    
    # Reset error state
    training_error = None
    
    try:
        # Initialize Mind if needed
        if mind is None:
            print("Mind not initialized. Initializing...")
            success = initialize_mind()
            if not success:
                training_error = "Failed to initialize Mind. Cannot start training."
                return
        
        # Ensure device consistency before starting
        ensure_device_consistency()
        
        # Check if CUDA is available but we're using CPU (provide a warning)
        if torch.cuda.is_available() and mind.device == "cpu":
            print("WARNING: CUDA is available but using CPU. This might be due to a previous error recovery.")
            training_error = "WARNING: CUDA is available but using CPU. Training will be slower."
        
        # Reset stop event
        stop_training_event.clear()
        
        # Create and start training thread
        training_thread = threading.Thread(target=training_loop)
        training_thread.daemon = True
        training_thread.start()
        
        # Update status
        dashboard_state.training_status = "running"
        print("Training started")
        
    except Exception as e:
        print(f"Error starting training: {e}")
        training_error = f"Error starting training: {str(e)}"
        traceback.print_exc()
        dashboard_state.training_status = "stopped"

def stop_training():
    """Stop the training process."""
    global stop_training_event, dashboard_state
    
    if dashboard_state.training_status != "running":
        print("Training not running")
        return
    
    # Set stop event
    stop_training_event.set()
    
    # Update status
    dashboard_state.training_status = "stopped"
    print("Training stopped")

def create_checkpoint(name=None):
    """Create a manual checkpoint of the current model state."""
    global mind, dashboard_state
    
    if mind is None:
        print("Mind not initialized")
        return None
    
    try:
        # Generate a checkpoint ID
        if name:
            checkpoint_id = f"manual_{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            checkpoint_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create a directory for this checkpoint
        checkpoint_dir = dashboard_state.models_dir / checkpoint_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save the mind state
        mind.save()
        
        # Get current development metrics
        status = mind.get_status()
        dev_metrics = status["developmental_metrics"]
        
        # Create a checkpoint record
        checkpoint = ModelCheckpoint(
            id=checkpoint_id,
            timestamp=time.time(),
            age_months=mind.mind_state.age_months,
            developmental_stage=mind.mind_state.developmental_stage,
            vocabulary_size=mind.mind_state.vocabulary_size,
            file_path=checkpoint_dir,
            language_development=(
                dev_metrics["language"]["receptive_language"] + 
                dev_metrics["language"]["expressive_language"]
            ) / 2,
            cognitive_development=(
                dev_metrics["cognitive"]["attention"] + 
                dev_metrics["cognitive"]["memory"] + 
                dev_metrics["cognitive"]["problem_solving"] + 
                dev_metrics["cognitive"]["abstract_thinking"]
            ) / 4,
            emotional_development=(
                dev_metrics["emotional"]["basic_emotions"] + 
                dev_metrics["emotional"]["emotional_regulation"] + 
                dev_metrics["emotional"]["emotional_complexity"]
            ) / 3,
            social_development=(
                dev_metrics["social"]["attachment"] + 
                dev_metrics["social"]["social_awareness"] + 
                dev_metrics["social"]["empathy"] + 
                dev_metrics["social"]["theory_of_mind"]
            ) / 4,
            memory_development=(
                dev_metrics["cognitive"].get("memory_working_memory_capacity", 0.0) + 
                dev_metrics["cognitive"].get("memory_long_term_memory_development", 0.0) + 
                dev_metrics["cognitive"].get("memory_memory_consolidation_rate", 0.0) + 
                dev_metrics["cognitive"].get("memory_memory_retrieval_accuracy", 0.0)
            ) / 4 if any(k.startswith("memory_") for k in dev_metrics["cognitive"]) else 0.5
        )
        
        # Add to available checkpoints
        dashboard_state.available_checkpoints.append(checkpoint)
        
        print(f"Created checkpoint: {checkpoint_id}")
        return checkpoint
        
    except Exception as e:
        print(f"Error creating checkpoint: {e}")
        traceback.print_exc()
        return None

def load_checkpoint(checkpoint_id):
    """Load a specific checkpoint."""
    global mind, dashboard_state, training_error
    
    # Find the checkpoint
    checkpoint = next((cp for cp in dashboard_state.available_checkpoints if cp.id == checkpoint_id), None)
    
    if checkpoint is None:
        print(f"Checkpoint not found: {checkpoint_id}")
        training_error = f"Checkpoint not found: {checkpoint_id}"
        return False
    
    try:
        # Make sure training is stopped
        if dashboard_state.training_status == "running":
            stop_training()
            # Wait briefly for training to stop
            time.sleep(0.5)
        
        # Reset any previous errors
        training_error = None
        
        # Reinitialize mind with the checkpoint
        config = dashboard_state.neural_child_config.copy()
        config.initial_age_months = checkpoint.age_months
        
        # Determine if we should use CUDA or CPU
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            # Try with CUDA first
            device = "cuda"
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
        else:
            # Fall back to CPU
            device = "cpu"
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
            
        # First attempt load with current device
        success = initialize_mind(config=config, load_existing=True)
        
        # If initial load fails and we were using CUDA, try falling back to CPU
        if not success and use_cuda:
            print("Failed to load checkpoint with CUDA. Trying with CPU...")
            training_error = "Failed to load with CUDA. Attempting with CPU..."
            
            # Fall back to CPU
            device = "cpu"
            torch.set_default_device(device)
            torch.set_default_dtype(torch.float32)
            
            # Try again with CPU
            success = initialize_mind(config=config, load_existing=True)
        
        if success:
            # Ensure device consistency after loading
            ensure_device_consistency()
            
            # Set the active checkpoint
            dashboard_state.active_checkpoint = checkpoint
            print(f"Loaded checkpoint: {checkpoint_id}")
            
            # Display success message with device info
            if mind.device == "cuda":
                training_error = f"Successfully loaded checkpoint: {checkpoint_id} using CUDA"
            else:
                training_error = f"Successfully loaded checkpoint: {checkpoint_id} using CPU"
                
            return True
        else:
            error_msg = f"Failed to load checkpoint: {checkpoint_id}"
            print(error_msg)
            if training_error is None:
                training_error = error_msg
            return False
        
    except Exception as e:
        error_msg = f"Error loading checkpoint: {str(e)}"
        print(error_msg)
        training_error = error_msg
        traceback.print_exc()
        return False

def inference_with_mind(input_text):
    """
    Perform direct inference with the current mind state.
    This uses the neural components directly without LLM involvement.
    
    Args:
        input_text: User input text for the LMM to process
        
    Returns:
        InferenceSession object containing the response and component activations
    """
    global mind, dashboard_state
    
    if mind is None:
        raise ValueError("Mind not initialized")
    
    # Check if current model is inference capable
    if dashboard_state.active_checkpoint is None or not dashboard_state.active_checkpoint.inference_capable:
        raise ValueError("Current model is not mature enough for inference")
    
    # Start time tracking
    start_time = time.time()
    component_activations = {}
    
    try:
        # 1. Process through language component to understand the input
        language_input = {
            "mother_utterance": input_text,  # We're treating user input as if from "mother"
            "developmental_stage": mind.mind_state.developmental_stage,
            "age_months": mind.mind_state.age_months,
            "emotional_state": mind.mind_state.emotional_state
        }
        
        # Track component activation before processing
        component_activations["language"] = ComponentActivation(
            name="language",
            activation_levels=[mind.language_component.activation_level],
            timestamps=[time.time()]
        )
        
        language_output = mind.language_component.process(language_input)
        
        # Update activation tracking after processing
        component_activations["language"].activation_levels.append(mind.language_component.activation_level)
        component_activations["language"].timestamps.append(time.time())
        
        # 2. Process through cognitive component for understanding
        cognitive_input = {
            "mother_utterance": input_text,
            "complexity": 0.6,  # Moderate complexity
            "context": {"inference_mode": True},
            "emotional_state": mind.mind_state.emotional_state
        }
        
        # Track component activation before processing
        component_activations["cognitive"] = ComponentActivation(
            name="cognitive",
            activation_levels=[mind.cognitive_component.activation_level],
            timestamps=[time.time()]
        )
        
        cognitive_output = mind.cognitive_component.process(cognitive_input)
        
        # Update activation tracking after processing
        component_activations["cognitive"].activation_levels.append(mind.cognitive_component.activation_level)
        component_activations["cognitive"].timestamps.append(time.time())
        
        # 3. Process through memory component to retrieve relevant memories
        memory_input = {
            "query": {
                "type": "semantic",
                "content": {"concepts": input_text.lower().split()},
                "limit": 5
            },
            "developmental_stage": mind.mind_state.developmental_stage,
            "age_months": mind.mind_state.age_months,
            "emotional_state": mind.mind_state.emotional_state
        }
        
        # Track component activation before processing
        component_activations["memory"] = ComponentActivation(
            name="memory",
            activation_levels=[mind.memory_component.activation_level],
            timestamps=[time.time()]
        )
        
        memory_output = mind.memory_component.process(memory_input)
        
        # Update activation tracking after processing
        component_activations["memory"].activation_levels.append(mind.memory_component.activation_level)
        component_activations["memory"].timestamps.append(time.time())
        
        # 4. Process through emotional component for emotional response
        emotional_input = {
            "mother_utterance": input_text,
            "interaction_context": {
                "type": "inference",
                "valence": 0.6,  # Slightly positive
                "arousal": 0.4   # Moderate arousal
            },
            "developmental_stage": mind.mind_state.developmental_stage,
            "age_months": mind.mind_state.age_months
        }
        
        # Track component activation before processing
        component_activations["emotional"] = ComponentActivation(
            name="emotional",
            activation_levels=[mind.emotional_component.activation_level],
            timestamps=[time.time()]
        )
        
        emotional_output = mind.emotional_component.process(emotional_input)
        
        # Update activation tracking after processing
        component_activations["emotional"].activation_levels.append(mind.emotional_component.activation_level)
        component_activations["emotional"].timestamps.append(time.time())
        
        # 5. Process through social component for social understanding
        social_input = {
            "agent": "user",
            "content": input_text,
            "emotional_tone": emotional_output.get("emotional_state", {}),
            "context": {"inference_mode": True},
            "age_months": mind.mind_state.age_months
        }
        
        # Track component activation before processing
        component_activations["social"] = ComponentActivation(
            name="social",
            activation_levels=[mind.social_component.activation_level],
            timestamps=[time.time()]
        )
        
        social_output = mind.social_component.process_interaction(social_input)
        
        # Update activation tracking after processing
        component_activations["social"].activation_levels.append(mind.social_component.activation_level)
        component_activations["social"].timestamps.append(time.time())
        
        # 6. Generate a response based on all component outputs
        # Here we use the language component's vocabulary and grammar capabilities
        response_input = {
            "developmental_stage": mind.mind_state.developmental_stage,
            "age_months": mind.mind_state.age_months,
            "emotional_state": emotional_output.get("emotional_state", {}),
            "comprehension_level": cognitive_output.get("understanding_level", 0.5),
            "retrieved_memories": memory_output.get("retrieved_memories", []),
            "mother_utterance": input_text
        }
        
        # Generate response using language component with influencing factors
        language_response = mind.language_component.process(response_input)
        
        # Extract the generated utterance
        child_utterance = language_response.get("child_utterance", "")
        
        # If we got a blank response (common in early development), use cognitive output
        if not child_utterance or child_utterance == "[non-verbal]":
            cognitive_understanding = cognitive_output.get("understanding_level", 0.0)
            emotional_response = list(emotional_output.get("emotional_state", {}).items())
            
            # Sort emotions by intensity
            emotional_response.sort(key=lambda x: x[1], reverse=True)
            
            # Create a basic response based on cognitive understanding
            if cognitive_understanding < 0.3:
                child_utterance = "I don't understand."
            elif cognitive_understanding < 0.6:
                child_utterance = "I partially understand."
            else:
                # Use vocabulary from language component to construct a response
                words = []
                for _ in range(min(10, len(mind.language_component.vocabulary))):
                    if mind.language_component.vocabulary:
                        words.append(random.choice(list(mind.language_component.vocabulary)))
                child_utterance = " ".join(words)
        
        # 7. Record final component activations
        for component_name in component_activations:
            component_activations[component_name].activation_levels.append(
                mind.components[component_name].activation_level
            )
            component_activations[component_name].timestamps.append(time.time())
        
        # Calculate processing time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000
        
        # Create inference session record
        session = InferenceSession(
            id=str(uuid.uuid4()),
            user_input=input_text,
            lmm_response=child_utterance,
            timestamp=time.time(),
            component_activations=component_activations,
            processing_time_ms=processing_time_ms
        )
        
        # Save in history
        dashboard_state.inference_history.append(session)
        dashboard_state.inference_session_id = session.id
        
        return session
        
    except Exception as e:
        print(f"Error during inference: {e}")
        traceback.print_exc()
        
        # Even on error, track the final component activations
        end_time = time.time()
        for component_name in mind.components:
            if component_name not in component_activations:
                component_activations[component_name] = ComponentActivation(
                    name=component_name,
                    activation_levels=[mind.components[component_name].activation_level],
                    timestamps=[end_time]
                )
        
        # Create error session record
        error_session = InferenceSession(
            id=str(uuid.uuid4()),
            user_input=input_text,
            lmm_response=f"Error during inference: {str(e)}",
            timestamp=time.time(),
            component_activations=component_activations,
            processing_time_ms=(end_time - start_time) * 1000
        )
        
        # Save in history
        dashboard_state.inference_history.append(error_session)
        dashboard_state.inference_session_id = error_session.id
        
        return error_session

# Create navigation bar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.Row(
                [
                    dbc.Col(
                        html.Img(
                            src="/assets/logo.png",
                            height="40px",
                        ),
                        width="auto",
                    ),
                    dbc.Col(
                        html.H2("Neural Child Dashboard", className="ms-2"), 
                        width="auto",
                    ),
                ],
                align="center",
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.ButtonGroup(
                            [
                                dbc.Button(
                                    "Training Observatory", 
                                    id="btn-mode-training",
                                    color="primary" if dashboard_state.mode == "training" else "secondary",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    "Model Management", 
                                    id="btn-mode-management",
                                    color="primary" if dashboard_state.mode == "management" else "secondary",
                                    className="me-1",
                                ),
                                dbc.Button(
                                    "Inference", 
                                    id="btn-mode-inference",
                                    color="primary" if dashboard_state.mode == "inference" else "secondary",
                                    disabled=dashboard_state.active_checkpoint is None or 
                                            not (dashboard_state.active_checkpoint.inference_capable 
                                                 if dashboard_state.active_checkpoint else False),
                                ),
                            ]
                        ),
                        width="auto",
                    ),
                ],
                align="center",
                className="ms-auto",
            ),
        ],
        fluid=True,
    ),
    color="dark",
    dark=True,
)

# Footer with status information
footer = dbc.Container(
    dbc.Row(
        [
            dbc.Col(
                [
                    html.Div(id="status-info", className="text-muted"),
                ],
                width=6,
            ),
            dbc.Col(
                [
                    html.Div(id="training-status", className="text-right text-muted"),
                ],
                width=6,
                className="text-end",
            ),
        ],
        className="mt-4 mb-4",
    ),
    fluid=True,
)

# Training Observatory Tab
training_observatory_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Training Controls"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Button(
                                                            "Start Training", 
                                                            id="btn-start-training",
                                                            color="success",
                                                            className="me-2",
                                                            size="md",
                                                            disabled=dashboard_state.training_status == "running",
                                                            style={"min-width": "140px"}
                                                        ),
                                                        dbc.Button(
                                                            "Stop Training", 
                                                            id="btn-stop-training",
                                                            color="danger",
                                                            size="md",
                                                            disabled=dashboard_state.training_status != "running",
                                                            style={"min-width": "140px"}
                                                        ),
                                                        dbc.Button(
                                                            "Save Checkpoint", 
                                                            id="btn-save-checkpoint",
                                                            color="info",
                                                            className="ms-2",
                                                            size="md",
                                                            style={"min-width": "140px"}
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Development Speed"),
                                                        dcc.Slider(
                                                            id="slider-speed",
                                                            min=0.1,
                                                            max=100,
                                                            step=0.1,
                                                            value=dashboard_state.speed_multiplier,
                                                            marks={
                                                                0.1: '0.1x',
                                                                1: '1x',
                                                                10: '10x',
                                                                100: '100x'
                                                            },
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                            className="mb-3",
                                        ),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        dbc.Label("Target Age (months)"),
                                                        dbc.Input(
                                                            id="input-target-age",
                                                            type="number",
                                                            min=0,
                                                            max=360,
                                                            step=1,
                                                            value=dashboard_state.target_age_months,
                                                            placeholder="No limit",
                                                        ),
                                                    ],
                                                    width=12,
                                                ),
                                            ],
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Child Status"),
                                dbc.CardBody(id="child-status-content")
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Recent Interactions"),
                                dbc.CardBody(id="recent-interactions-content")
                            ],
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Developmental Metrics"),
                                dbc.CardBody(
                                    [
                                        dcc.Tabs(
                                            [
                                                dcc.Tab(
                                                    label="Language Development",
                                                    children=[
                                                        dcc.Graph(id="graph-language-development"),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Cognitive Development",
                                                    children=[
                                                        dcc.Graph(id="graph-cognitive-development"),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Emotional Development",
                                                    children=[
                                                        dcc.Graph(id="graph-emotional-development"),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Social Development",
                                                    children=[
                                                        dcc.Graph(id="graph-social-development"),
                                                    ],
                                                ),
                                                dcc.Tab(
                                                    label="Overall Progress",
                                                    children=[
                                                        dcc.Graph(id="graph-overall-development"),
                                                    ],
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Neural Component Activations"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="graph-component-activations"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=8,
                ),
            ],
        ),
    ],
    fluid=True,
)

# Model Management Tab
model_management_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Available Checkpoints"),
                                dbc.CardBody(id="checkpoints-list-content")
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Create New Checkpoint"),
                                dbc.CardBody(
                                    [
                                        dbc.Input(
                                            id="input-checkpoint-name",
                                            placeholder="Checkpoint name (optional)",
                                            type="text",
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Create Checkpoint", 
                                            id="btn-create-checkpoint",
                                            color="primary",
                                            size="md",
                                            style={"min-width": "140px"}
                                        ),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Checkpoint Details"),
                                dbc.CardBody(id="checkpoint-details-content")
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Developmental Comparison"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="graph-checkpoint-comparison"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=8,
                ),
            ],
        ),
    ],
    fluid=True,
)

# Inference Tab
inference_tab = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Current Model Status"),
                                dbc.CardBody(id="inference-model-status")
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Inference History"),
                                dbc.CardBody(id="inference-history-content")
                            ],
                        ),
                    ],
                    width=4,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Inference Interface"),
                                dbc.CardBody(
                                    [
                                        dbc.Textarea(
                                            id="inference-input",
                                            placeholder="Enter text to interact with the LMM...",
                                            style={"height": "150px"},
                                            className="mb-3",
                                        ),
                                        dbc.Button(
                                            "Submit", 
                                            id="btn-submit-inference",
                                            color="primary",
                                            disabled=dashboard_state.active_checkpoint is None or 
                                                    not (dashboard_state.active_checkpoint.inference_capable 
                                                         if dashboard_state.active_checkpoint else False),
                                        ),
                                        html.Div(id="inference-response", className="mt-4"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                        dbc.Card(
                            [
                                dbc.CardHeader("Component Activation"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="graph-inference-activation"),
                                    ]
                                ),
                            ],
                        ),
                    ],
                    width=8,
                ),
            ],
        ),
    ],
    fluid=True,
)

# Toast container for notifications
toast_container = html.Div(
    id="toast-container",
    style={"position": "fixed", "top": 10, "right": 10, "zIndex": 9999}
)

# Error message container
error_message_container = html.Div(
    id="error-message-container",
    style={"position": "fixed", "bottom": 10, "right": 10, "zIndex": 9998}
)

# Create the app layout with tabs
app.layout = html.Div(
    [
        navbar,
        dbc.Container(
            [
                dcc.Store(id="store-mode", data=dashboard_state.mode),
                dcc.Store(id="store-training-status", data=dashboard_state.training_status),
                dcc.Store(id="store-active-checkpoint", data=dashboard_state.active_checkpoint.id if dashboard_state.active_checkpoint else None),
                dcc.Store(id="store-inference-session", data=dashboard_state.inference_session_id),
                html.Div(id="content"),
                dcc.Interval(
                    id="interval-update",
                    interval=dashboard_state.update_interval_ms,
                    n_intervals=0,
                ),
                toast_container,
                error_message_container,
            ],
            fluid=True,
            className="mt-4",
        ),
        footer,
    ]
)

# Callback to switch between modes
@callback(
    Output("store-mode", "data"),
    [
        Input("btn-mode-training", "n_clicks"),
        Input("btn-mode-management", "n_clicks"),
        Input("btn-mode-inference", "n_clicks"),
    ],
    [State("store-mode", "data")],
    prevent_initial_call=True,
)
def switch_mode(btn_training, btn_management, btn_inference, current_mode):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_mode
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-mode-training":
        dashboard_state.mode = "training"
    elif button_id == "btn-mode-management":
        dashboard_state.mode = "management"
    elif button_id == "btn-mode-inference":
        # Make sure we have an inference-capable model
        if dashboard_state.active_checkpoint and dashboard_state.active_checkpoint.inference_capable:
            dashboard_state.mode = "inference"
        else:
            # Show error or notification here
            return current_mode
    
    return dashboard_state.mode

# Callback to update content based on mode
@callback(
    Output("content", "children"),
    [Input("store-mode", "data")],
)
def update_content(mode):
    if mode == "training":
        return training_observatory_tab
    elif mode == "management":
        return model_management_tab
    elif mode == "inference":
        return inference_tab
    
    # Default to training tab
    return training_observatory_tab

# Callback to handle training control buttons
@callback(
    Output("store-training-status", "data"),
    [
        Input("btn-start-training", "n_clicks"),
        Input("btn-stop-training", "n_clicks"),
    ],
    [State("store-training-status", "data")],
    prevent_initial_call=True,
)
def control_training(btn_start, btn_stop, current_status):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        return current_status
    
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if button_id == "btn-start-training" and current_status != "running":
        # Initialize Mind if needed
        global mind
        if mind is None:
            initialize_mind()
        
        # Start training
        start_training()
        return "running"
    
    elif button_id == "btn-stop-training" and current_status == "running":
        # Stop training
        stop_training()
        return "stopped"
    
    return current_status

# Callback to update training controls appearance
@callback(
    [
        Output("btn-start-training", "disabled"),
        Output("btn-stop-training", "disabled"),
    ],
    [Input("store-training-status", "data")],
)
def update_training_controls(status):
    return status == "running", status != "running"

# Callback to handle save checkpoint button
@callback(
    Output("toast-container", "children"),
    [Input("btn-save-checkpoint", "n_clicks")],
    prevent_initial_call=True,
)
def handle_save_checkpoint(n_clicks):
    if n_clicks:
        # Create a checkpoint
        checkpoint = create_checkpoint()
        if checkpoint:
            # Show success toast
            return dbc.Toast(
                f"Checkpoint created: {checkpoint.id}",
                id="toast-save-success",
                header="Success",
                icon="success",
                dismissable=True,
                duration=4000,
                is_open=True,
                style={"background-color": "var(--success-color)", "color": "#000"}
            )
        else:
            # Show error toast
            return dbc.Toast(
                "Failed to create checkpoint. Check logs for details.",
                id="toast-save-error",
                header="Error",
                icon="danger",
                dismissable=True,
                duration=4000,
                is_open=True,
                style={"background-color": "var(--danger-color)"}
            )
    
    # If we get here, return empty (no toast)
    return []

# Callback to create checkpoint
@callback(
    Output("checkpoints-list-content", "children"),
    [Input("btn-create-checkpoint", "n_clicks"),
     Input("interval-update", "n_intervals")],
    [State("input-checkpoint-name", "value")],
    prevent_initial_call=True,
)
def handle_create_checkpoint(btn_clicks, n_intervals, name):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "btn-create-checkpoint" and btn_clicks:
        # Create a checkpoint
        checkpoint = create_checkpoint(name)
    
    # Either way, update the checkpoints list
    return create_checkpoints_list()

def create_checkpoints_list():
    """Create the list of checkpoints for the UI."""
    checkpoints = dashboard_state.available_checkpoints
    
    if not checkpoints:
        return html.Div("No checkpoints available yet.")
    
    # Sort by timestamp (newest first)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x.timestamp, reverse=True)
    
    # Create list items
    items = []
    for cp in sorted_checkpoints:
        # Format the timestamp
        timestamp_str = datetime.fromtimestamp(cp.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        
        # Create list item
        item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.H5(cp.id, className="mb-1"),
                        html.P(
                            [
                                f"Age: {cp.age_months:.1f} months",
                                html.Br(),
                                f"Stage: {cp.developmental_stage}",
                                html.Br(),
                                f"Vocabulary: {cp.vocabulary_size} words",
                                html.Br(),
                                f"Created: {timestamp_str}",
                            ],
                            className="mb-1",
                        ),
                        dbc.Badge(
                            "Inference Ready" if cp.inference_capable else "Training Only",
                            color="success" if cp.inference_capable else "warning",
                            className="me-1",
                        ),
                        dbc.Button(
                            "Load", 
                            id={"type": "btn-load-checkpoint", "index": cp.id},
                            color="primary",
                            size="sm",
                            className="mt-2",
                        ),
                    ]
                ),
            ],
            className="d-flex justify-content-between align-items-center",
        )
        items.append(item)
    
    return dbc.ListGroup(items)

# Callback to load checkpoint
@callback(
    Output("store-active-checkpoint", "data"),
    [Input({"type": "btn-load-checkpoint", "index": dash.dependencies.ALL}, "n_clicks")],
    prevent_initial_call=True,
)
def load_selected_checkpoint(btn_clicks):
    ctx = dash.callback_context
    
    if not ctx.triggered:
        raise PreventUpdate
    
    # Get the button that was clicked
    button_id = ctx.triggered[0]["prop_id"].split(".")[0]
    checkpoint_id = json.loads(button_id)["index"]
    
    # Load the checkpoint
    success = load_checkpoint(checkpoint_id)
    
    if success:
        return checkpoint_id
    else:
        raise PreventUpdate

# Callback to update checkpoint details
@callback(
    Output("checkpoint-details-content", "children"),
    [Input("store-active-checkpoint", "data")],
)
def update_checkpoint_details(checkpoint_id):
    if not checkpoint_id:
        return html.Div("No checkpoint selected.")
    
    # Find the checkpoint
    checkpoint = next((cp for cp in dashboard_state.available_checkpoints if cp.id == checkpoint_id), None)
    
    if checkpoint is None:
        return html.Div("Checkpoint not found.")
    
    # Format timestamp
    timestamp_str = datetime.fromtimestamp(checkpoint.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    return html.Div(
        [
            html.H4(checkpoint.id),
            html.P(f"Created: {timestamp_str}"),
            html.Hr(),
            html.H5("Developmental Status"),
            html.P(f"Age: {checkpoint.age_months:.1f} months"),
            html.P(f"Stage: {checkpoint.developmental_stage}"),
            html.P(f"Vocabulary Size: {checkpoint.vocabulary_size} words"),
            html.Hr(),
            html.H5("Capabilities"),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.language_development * 100, 
                        color="info",
                        bar=True,
                        label=f"Language: {checkpoint.language_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.cognitive_development * 100, 
                        color="success",
                        bar=True,
                        label=f"Cognitive: {checkpoint.cognitive_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.emotional_development * 100, 
                        color="danger",
                        bar=True,
                        label=f"Emotional: {checkpoint.emotional_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.social_development * 100, 
                        color="warning",
                        bar=True,
                        label=f"Social: {checkpoint.social_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            html.Hr(),
            html.Div(
                [
                    html.H5("Inference Status"),
                    dbc.Badge(
                        "Ready for Inference" if checkpoint.inference_capable else "Not Ready for Inference",
                        color="success" if checkpoint.inference_capable else "danger",
                        className="me-1",
                    ),
                    html.P(
                        "This model has reached sufficient developmental milestones to engage in direct interaction."
                        if checkpoint.inference_capable else
                        "This model has not yet reached the developmental milestones required for direct interaction. Continue training to develop language and cognitive capabilities.",
                        className="mt-2",
                    ),
                ]
            ),
        ]
    )

# Callback to handle inference submission
@callback(
    [Output("inference-response", "children"),
     Output("store-inference-session", "data")],
    [Input("btn-submit-inference", "n_clicks")],
    [State("inference-input", "value")],
    prevent_initial_call=True,
)
def handle_inference(n_clicks, input_text):
    if not n_clicks or not input_text:
        raise PreventUpdate
    
    # Check if we have an inference-capable model
    if dashboard_state.active_checkpoint is None or not dashboard_state.active_checkpoint.inference_capable:
        return html.Div(
            "Current model is not mature enough for inference. Please load a model that has reached sufficient developmental milestones.",
            className="text-danger",
        ), None
    
    try:
        # Perform inference
        inference_session = inference_with_mind(input_text)
        
        return html.Div(
            [
                html.H5("Response:"),
                html.Div(
                    inference_session.lmm_response,
                    className="p-3 bg-light rounded",
                ),
                html.Div(
                    f"Processing time: {inference_session.processing_time_ms:.2f} ms",
                    className="text-muted mt-2",
                ),
            ]
        ), inference_session.id
    except Exception as e:
        return html.Div(
            [
                html.H5("Error:"),
                html.Div(
                    str(e),
                    className="p-3 bg-light rounded text-danger",
                ),
            ]
        ), None

# Callback to update child status
@callback(
    Output("child-status-content", "children"),
    [Input("interval-update", "n_intervals")],
)
def update_child_status(n_intervals):
    global mind
    
    if mind is None:
        return html.Div("Mind not initialized. Please start training.")
    
    # Get current status
    status = mind.get_status()
    
    # Extract key info
    age_months = status.get("age_months", 0)
    dev_stage = status.get("developmental_stage", "Unknown")
    vocab_size = status.get("vocabulary_size", 0)
    dominant_emotion = status.get("dominant_emotion", ("neutral", 0))
    
    # Create status display
    return html.Div(
        [
            html.H4(f"Age: {age_months:.1f} months"),
            html.H5(f"Stage: {dev_stage}"),
            html.Hr(),
            html.Div(
                [
                    html.H5("Developmental Metrics"),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div("Language", className="text-center"),
                                    dbc.Progress(
                                        value=status.get("development_progress", {}).get("language", 0) * 100,
                                        color="info",
                                        className="mb-2",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Div("Cognitive", className="text-center"),
                                    dbc.Progress(
                                        value=status.get("development_progress", {}).get("cognitive", 0) * 100,
                                        color="success",
                                        className="mb-2",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Div("Emotional", className="text-center"),
                                    dbc.Progress(
                                        value=status.get("development_progress", {}).get("emotional", 0) * 100,
                                        color="danger",
                                        className="mb-2",
                                    ),
                                ],
                                className="col-6",
                            ),
                            html.Div(
                                [
                                    html.Div("Social", className="text-center"),
                                    dbc.Progress(
                                        value=status.get("development_progress", {}).get("social", 0) * 100,
                                        color="warning",
                                        className="mb-2",
                                    ),
                                ],
                                className="col-6",
                            ),
                        ],
                        className="row",
                    ),
                ],
                className="mb-3",
            ),
            html.Hr(),
            html.Div(
                [
                    html.H5("Capabilities"),
                    html.P(f"Vocabulary Size: {vocab_size} words"),
                    html.P(f"Dominant Emotion: {dominant_emotion[0]} ({dominant_emotion[1]:.2f})"),
                    html.P(f"Memory Items: {status.get('memory_counts', {}).get('episodic', 0)} episodic, {status.get('memory_counts', {}).get('semantic', 0)} semantic"),
                ]
            ),
        ]
    )

# Callback to update recent interactions
@callback(
    Output("recent-interactions-content", "children"),
    [Input("interval-update", "n_intervals")],
)
def update_recent_interactions(n_intervals):
    # Get recent interactions
    recent = dashboard_state.interaction_history.get_recent_interactions(5)
    
    if not recent:
        return html.Div("No interactions recorded yet.")
    
    # Create list of interactions
    items = []
    for interaction in reversed(recent):  # Show newest first
        # Extract relevant info
        child_utterance = interaction.get("child_state", {}).get("verbal_response", "")
        mother_utterance = interaction.get("mother_state", {}).get("verbal_response", "")
        
        # Format timestamp
        timestamp = interaction.get("timestamp", 0)
        timestamp_str = datetime.fromtimestamp(timestamp).strftime("%H:%M:%S")
        
        # Create list item
        item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.Small(timestamp_str, className="text-muted"),
                        html.P(
                            [
                                html.Strong("Child: "),
                                html.Span(child_utterance or "[non-verbal]"),
                            ],
                            className="mb-1",
                        ),
                        html.P(
                            [
                                html.Strong("Mother: "),
                                html.Span(mother_utterance),
                            ],
                            className="mb-0",
                        ),
                    ]
                ),
            ]
        )
        items.append(item)
    
    return dbc.ListGroup(items)

# Callback to update inference model status
@callback(
    Output("inference-model-status", "children"),
    [Input("store-active-checkpoint", "data"),
     Input("interval-update", "n_intervals")],
)
def update_inference_model_status(checkpoint_id, n_intervals):
    if not checkpoint_id:
        return html.Div("No model loaded. Please load a model from the Model Management tab.")
    
    # Find the checkpoint
    checkpoint = next((cp for cp in dashboard_state.available_checkpoints if cp.id == checkpoint_id), None)
    
    if checkpoint is None:
        return html.Div("Model information not found.")
    
    # Check if model is inference capable
    if not checkpoint.inference_capable:
        return html.Div(
            [
                html.H4("Model Not Ready"),
                html.P(
                    "This model has not yet reached the developmental milestones required for inference. "
                    "Please load a model that has sufficient language and cognitive capabilities.",
                    className="text-danger",
                ),
                html.Hr(),
                html.H5("Current Capabilities"),
                html.P(f"Age: {checkpoint.age_months:.1f} months"),
                html.P(f"Stage: {checkpoint.developmental_stage}"),
                html.P(f"Vocabulary: {checkpoint.vocabulary_size} words"),
                html.P(f"Language Development: {checkpoint.language_development:.2f}"),
                html.P(f"Cognitive Development: {checkpoint.cognitive_development:.2f}"),
                html.Hr(),
                html.H5("Required for Inference"),
                html.P("Age: at least 36 months"),
                html.P("Language Development: at least 0.7"),
                html.P("Cognitive Development: at least 0.6"),
                html.P("Vocabulary: at least 1000 words"),
            ]
        )
    
    # Model is ready for inference
    return html.Div(
        [
            html.H4("Model Ready for Inference"),
            html.P(
                "This model has reached sufficient developmental milestones for direct interaction.",
                className="text-success",
            ),
            html.Hr(),
            html.H5("Model Capabilities"),
            html.P(f"Age: {checkpoint.age_months:.1f} months"),
            html.P(f"Stage: {checkpoint.developmental_stage}"),
            html.P(f"Vocabulary: {checkpoint.vocabulary_size} words"),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.language_development * 100, 
                        color="info",
                        bar=True,
                        label=f"Language: {checkpoint.language_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            dbc.Progress(
                [
                    dbc.Progress(
                        value=checkpoint.cognitive_development * 100, 
                        color="success",
                        bar=True,
                        label=f"Cognitive: {checkpoint.cognitive_development:.2f}",
                    )
                ],
                className="mb-3",
            ),
            html.Hr(),
            html.H5("Interaction Guidelines"),
            html.P(
                "The LMM's responses are generated entirely from its learned neural patterns, "
                "without using any external LLMs. Responses reflect its current developmental "
                "capabilities and may be limited compared to traditional LLMs."
            ),
        ]
    )

# Callback to update inference history
@callback(
    Output("inference-history-content", "children"),
    [Input("store-inference-session", "data"),
     Input("interval-update", "n_intervals")],
)
def update_inference_history(session_id, n_intervals):
    history = dashboard_state.inference_history
    
    if not history:
        return html.Div("No inference interactions recorded yet.")
    
    # Create list of interactions
    items = []
    for session in reversed(history):  # Show newest first
        # Format timestamp
        timestamp_str = datetime.fromtimestamp(session.timestamp).strftime("%H:%M:%S")
        
        # Create list item
        item = dbc.ListGroupItem(
            [
                html.Div(
                    [
                        html.Small(timestamp_str, className="text-muted"),
                        html.P(
                            [
                                html.Strong("You: "),
                                html.Span(session.user_input),
                            ],
                            className="mb-1",
                        ),
                        html.P(
                            [
                                html.Strong("LMM: "),
                                html.Span(session.lmm_response),
                            ],
                            className="mb-0",
                        ),
                    ]
                ),
            ]
        )
        items.append(item)
    
    return dbc.ListGroup(items)

# Callback to update language development graph
@callback(
    Output("graph-language-development", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_language_graph(n_intervals):
    df = dashboard_state.developmental_data.to_dataframe()
    
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No developmental data available yet. Start training to see metrics.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["receptive_language"],
            mode='lines',
            name='Receptive Language',
            line=dict(color='royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["expressive_language"],
            mode='lines',
            name='Expressive Language',
            line=dict(color='firebrick')
        )
    )
    
    # Add vocabulary size on secondary y-axis
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["vocabulary_size"],
            mode='lines',
            name='Vocabulary Size',
            line=dict(color='green', dash='dash'),
            yaxis='y2'
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Language Development Over Time",
        xaxis_title="Age (months)",
        yaxis_title="Development Level",
        yaxis2=dict(
            title="Vocabulary Size",
            overlaying="y",
            side="right"
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

# Callback to update cognitive development graph
@callback(
    Output("graph-cognitive-development", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_cognitive_graph(n_intervals):
    df = dashboard_state.developmental_data.to_dataframe()
    
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No developmental data available yet. Start training to see metrics.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["attention"],
            mode='lines',
            name='Attention',
            line=dict(color='royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["memory"],
            mode='lines',
            name='Memory',
            line=dict(color='firebrick')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["problem_solving"],
            mode='lines',
            name='Problem Solving',
            line=dict(color='green')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["abstract_thinking"],
            mode='lines',
            name='Abstract Thinking',
            line=dict(color='purple')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Cognitive Development Over Time",
        xaxis_title="Age (months)",
        yaxis_title="Development Level",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

# Callback to update emotional development graph
@callback(
    Output("graph-emotional-development", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_emotional_graph(n_intervals):
    df = dashboard_state.developmental_data.to_dataframe()
    
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No developmental data available yet. Start training to see metrics.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["basic_emotions"],
            mode='lines',
            name='Basic Emotions',
            line=dict(color='royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["emotional_regulation"],
            mode='lines',
            name='Emotional Regulation',
            line=dict(color='firebrick')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["emotional_complexity"],
            mode='lines',
            name='Emotional Complexity',
            line=dict(color='green')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Emotional Development Over Time",
        xaxis_title="Age (months)",
        yaxis_title="Development Level",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

# Callback to update social development graph
@callback(
    Output("graph-social-development", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_social_graph(n_intervals):
    df = dashboard_state.developmental_data.to_dataframe()
    
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No developmental data available yet. Start training to see metrics.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["attachment"],
            mode='lines',
            name='Attachment',
            line=dict(color='royalblue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["social_awareness"],
            mode='lines',
            name='Social Awareness',
            line=dict(color='firebrick')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["empathy"],
            mode='lines',
            name='Empathy',
            line=dict(color='green')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["theory_of_mind"],
            mode='lines',
            name='Theory of Mind',
            line=dict(color='purple')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Social Development Over Time",
        xaxis_title="Age (months)",
        yaxis_title="Development Level",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

# Callback to update overall development graph
@callback(
    Output("graph-overall-development", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_overall_graph(n_intervals):
    df = dashboard_state.developmental_data.to_dataframe()
    
    if df.empty:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No developmental data available yet. Start training to see metrics.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Calculate overall development metrics
    df["language_overall"] = (df["receptive_language"] + df["expressive_language"]) / 2
    df["cognitive_overall"] = (df["attention"] + df["memory"] + df["problem_solving"] + df["abstract_thinking"]) / 4
    df["emotional_overall"] = (df["basic_emotions"] + df["emotional_regulation"] + df["emotional_complexity"]) / 3
    df["social_overall"] = (df["attachment"] + df["social_awareness"] + df["empathy"] + df["theory_of_mind"]) / 4
    
    # Create figure
    fig = go.Figure()
    
    # Add traces
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["language_overall"],
            mode='lines',
            name='Language',
            line=dict(color='blue')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["cognitive_overall"],
            mode='lines',
            name='Cognitive',
            line=dict(color='green')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["emotional_overall"],
            mode='lines',
            name='Emotional',
            line=dict(color='red')
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df["age_months"],
            y=df["social_overall"],
            mode='lines',
            name='Social',
            line=dict(color='orange')
        )
    )
    
    # Update layout
    fig.update_layout(
        title="Overall Development Progress",
        xaxis_title="Age (months)",
        yaxis_title="Development Level",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark"
    )
    
    return fig

# Callback to update component activations graph
@callback(
    Output("graph-component-activations", "figure"),
    [Input("interval-update", "n_intervals")],
)
def update_component_activations(n_intervals):
    global mind
    
    if mind is None:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Mind not initialized. Please start training.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Get current activations
    activations = mind.mind_state.component_activations
    
    # Create figure
    fig = go.Figure(data=[
        go.Bar(
            x=list(activations.keys()),
            y=list(activations.values()),
            marker_color=['blue', 'red', 'green', 'purple', 'orange']
        )
    ])
    
    # Update layout
    fig.update_layout(
        title="Neural Component Activations",
        xaxis_title="Component",
        yaxis_title="Activation Level",
        yaxis=dict(range=[0, 1]),
        template="plotly_dark"
    )
    
    return fig

# Callback to update checkpoint comparison graph
@callback(
    Output("graph-checkpoint-comparison", "figure"),
    [Input("store-active-checkpoint", "data")],
)
def update_checkpoint_comparison(active_checkpoint_id):
    # Get all checkpoints
    checkpoints = dashboard_state.available_checkpoints
    
    if not checkpoints:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="No checkpoints available for comparison.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Prepare data for radar chart
    categories = ['Language', 'Cognitive', 'Emotional', 'Social', 'Memory']
    
    # Sort by age (oldest to youngest)
    sorted_checkpoints = sorted(checkpoints, key=lambda x: x.age_months)
    
    # Take at most 5 checkpoints to avoid cluttering
    if len(sorted_checkpoints) > 5:
        # Take 5 checkpoints evenly distributed
        indices = np.linspace(0, len(sorted_checkpoints) - 1, 5, dtype=int)
        display_checkpoints = [sorted_checkpoints[i] for i in indices]
    else:
        display_checkpoints = sorted_checkpoints
    
    # Create figure
    fig = go.Figure()
    
    # Add a trace for each checkpoint
    for cp in display_checkpoints:
        values = [
            cp.language_development,
            cp.cognitive_development,
            cp.emotional_development,
            cp.social_development,
            cp.memory_development
        ]
        
        # Close the polygon
        categories_closed = categories + [categories[0]]
        values_closed = values + [values[0]]
        
        # Highlight active checkpoint
        if cp.id == active_checkpoint_id:
            line_width = 3
            opacity = 0.8
        else:
            line_width = 1
            opacity = 0.6
        
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=categories_closed,
            name=f"{cp.developmental_stage} ({cp.age_months:.1f} mo)",
            line=dict(width=line_width),
            opacity=opacity
        ))
    
    # Update layout
    fig.update_layout(
        title="Developmental Comparison Across Checkpoints",
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True,
        template="plotly_dark"
    )
    
    return fig

# Callback to update inference activation graph
@callback(
    Output("graph-inference-activation", "figure"),
    [Input("store-inference-session", "data")],
)
def update_inference_activation(session_id):
    if not session_id:
        # Return empty figure with message
        fig = go.Figure()
        fig.add_annotation(
            text="Submit an inference to see component activation patterns.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Find the inference session
    session = next((s for s in dashboard_state.inference_history if s.id == session_id), None)
    
    if not session:
        # Session not found
        fig = go.Figure()
        fig.add_annotation(
            text="Inference session data not found.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Create figure for activation heatmap
    component_names = list(session.component_activations.keys())
    
    if not component_names:
        # No component data
        fig = go.Figure()
        fig.add_annotation(
            text="No component activation data available for this inference session.",
            showarrow=False,
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
        )
        return fig
    
    # Find component with the most timesteps
    max_timesteps = max(len(session.component_activations[c].activation_levels) for c in component_names)
    
    # Prepare data for heatmap
    activation_matrix = np.zeros((len(component_names), max_timesteps))
    
    # Fill matrix with activation values
    for i, component in enumerate(component_names):
        levels = session.component_activations[component].activation_levels
        # Pad if necessary
        padded_levels = levels + [levels[-1]] * (max_timesteps - len(levels)) if levels else [0] * max_timesteps
        activation_matrix[i, :] = padded_levels
    
    # Create timestep labels
    timesteps = [f"t{i+1}" for i in range(max_timesteps)]
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=activation_matrix,
        x=timesteps,
        y=component_names,
        colorscale='Viridis',
        zmin=0,
        zmax=1
    ))
    
    # Update layout
    fig.update_layout(
        title="Component Activation During Inference",
        xaxis_title="Processing Time Steps",
        yaxis_title="Neural Component",
        template="plotly_dark"
    )
    
    return fig

# Callback for status info in footer
@callback(
    Output("status-info", "children"),
    [Input("interval-update", "n_intervals")],
)
def update_status_info(n_intervals):
    global mind
    
    if mind is None:
        return "Status: Mind not initialized"
    
    # Get basic status info
    return f"Status: Age {mind.mind_state.age_months:.1f} months | Stage: {mind.mind_state.developmental_stage}"

# Callback for training status in footer
@callback(
    Output("training-status", "children"),
    [Input("store-training-status", "data")],
)
def update_training_status_footer(status):
    if status == "running":
        return "Training: Running"
    elif status == "paused":
        return "Training: Paused"
    elif status == "stopped":
        return "Training: Stopped"
    else:
        return "Training: Idle"

# Callback to update speed multiplier
@callback(
    Output("slider-speed", "value"),
    [Input("slider-speed", "value")],
    prevent_initial_call=True,
)
def update_speed_multiplier(value):
    if value is None:
        raise PreventUpdate
    
    # Update dashboard state
    dashboard_state.speed_multiplier = value
    
    # Update mind configuration if initialized
    global mind
    if mind is not None:
        mind.config.development_speed = value
    
    return value

# Callback to update target age
@callback(
    Output("input-target-age", "value"),
    [Input("input-target-age", "value")],
    prevent_initial_call=True,
)
def update_target_age(value):
    # Update dashboard state
    dashboard_state.target_age_months = value
    
    return value

# Initialize app data directories and Mind
def initialize_app_data():
    """Initialize data directories and Mind instance."""
    # Create data directories
    dashboard_state.data_dir.mkdir(parents=True, exist_ok=True)
    dashboard_state.models_dir.mkdir(parents=True, exist_ok=True)
    dashboard_state.logs_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Mind
    initialize_mind()

# Run the app
if __name__ == "__main__":
    # Initialize the app before running
    initialize_app_data()
    app.run_server(debug=True)

# Callback to display training errors
@callback(
    Output("error-message-container", "children"),
    [Input("interval-update", "n_intervals")],
)
def update_error_messages(n_intervals):
    global training_error
    
    if training_error:
        error_msg = training_error
        
        # Determine severity level based on error message content
        if "WARNING" in error_msg:
            severity = "warning"
            header = "Warning"
            icon = "warning"
            bg_color = "var(--bs-warning)"
            duration = 10000  # 10 seconds for warnings
        else:
            severity = "danger"
            header = "Error"
            icon = "danger" 
            bg_color = "var(--bs-danger)"
            duration = 15000  # 15 seconds for errors
        
        # Only reset transient errors (keep persistent warnings/errors)
        persistent_errors = ["CPU", "cuda", "device", "WARNING"]
        if not any(term in error_msg for term in persistent_errors):
            training_error = None
            
        return dbc.Toast(
            error_msg,
            id=f"toast-{severity}",
            header=header,
            icon=icon,
            dismissable=True,
            duration=duration,
            is_open=True,
            style={"background-color": bg_color, "color": "white", "font-weight": "bold"}
        )
    
    return []

# Add a helper function for safely resetting model dimensions
def reset_model_dimensions():
    """Helper function to reset the model with correct tensor dimensions.
    
    This function attempts to fix matrix dimension mismatch errors
    by completely reinitializing the model architecture.
    """
    global mind, dashboard_state, training_error
    
    try:
        print("Resetting model dimensions...")
        
        # Get current device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Setup device
        torch.set_default_device(device)
        torch.set_default_dtype(torch.float32)
        
        # Save current state age and development
        if mind is not None:
            current_age = mind.mind_state.age_months
            current_stage = mind.mind_state.developmental_stage
        else:
            current_age = 0.0
            current_stage = "newborn"
        
        # Create a fresh configuration
        config = dashboard_state.neural_child_config.copy()
        config.initial_age_months = current_age
        
        # Completely reinitialize from scratch (don't load existing)
        # This rebuilds all neural networks with proper dimensions
        success = initialize_mind(config=config, load_existing=False)
        
        if success:
            # Force the developmental stage to match previous state
            mind.mind_state.developmental_stage = current_stage
            
            # Ensure device consistency
            ensure_device_consistency()
            
            print(f"Model dimensions reset successful (age: {current_age:.1f} months, stage: {current_stage})")
            return True
        else:
            print("Failed to reset model dimensions")
            if training_error is None:
                training_error = "Failed to reset model dimensions"
            return False
    
    except Exception as e:
        print(f"Error resetting model dimensions: {e}")
        if training_error is None:
            training_error = f"Error resetting model dimensions: {str(e)}"
        traceback.print_exc()
        return False