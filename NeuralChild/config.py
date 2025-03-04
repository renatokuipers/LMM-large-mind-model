# config.py
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal
from enum import Enum
from pydantic import BaseModel, Field, field_validator, model_validator

# Import NetworkType for network configurations
from networks.network_types import NetworkType

class MemoryConfig(BaseModel):
    """Configuration for memory systems"""
    working_memory_capacity: int = Field(5, ge=1, description="How many items can be held in working memory")
    long_term_decay_rate: float = Field(0.01, ge=0.0, le=1.0, description="Rate of forgetting in long-term memory")
    episodic_memory_capacity: int = Field(1000, ge=10, description="Maximum number of episodic memories")
    associative_strength_threshold: float = Field(0.3, ge=0.0, le=1.0, description="Minimum strength for associative links")
    consolidation_rate: float = Field(0.2, ge=0.0, le=1.0, description="Rate at which memories are consolidated")
    memory_embedding_dim: int = Field(128, ge=16, description="Dimension for memory embeddings")

class DevelopmentConfig(BaseModel):
    """Configuration for developmental tracking"""
    milestones_path: Path = Field(Path("./data/milestones"), description="Path to developmental milestones data")
    metrics_log_interval: int = Field(10, ge=1, description="How often to log developmental metrics (interactions)")
    stage_transition_thresholds: Dict[str, float] = Field(
        default_factory=lambda: {
            "infancy_to_early_childhood": 30.0,  # days
            "early_to_middle_childhood": 180.0,  # days
            "middle_childhood_to_adolescence": 365.0,  # days
        },
        description="Age thresholds for developmental stage transitions"
    )

class SystemConfig(BaseModel):
    """System-wide configuration"""
    data_dir: Path = Field(Path("./data"), description="Root directory for all data")
    logs_dir: Path = Field(Path("./logs"), description="Directory for log files")
    save_state_interval: int = Field(100, ge=1, description="Save state every N interactions")
    debug_mode: bool = Field(False, description="Enable debug mode with extra logging")
    random_seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    llm_base_url: str = Field("http://192.168.2.12:1234", description="Base URL for LLM API")
    llm_model: str = Field("qwen2.5-7b-instruct", description="Default LLM model to use")
    
    @model_validator(mode='after')
    def create_directories(self) -> 'SystemConfig':
        """Ensure required directories exist"""
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.logs_dir.mkdir(exist_ok=True, parents=True)
        (self.data_dir / "neural_child").mkdir(exist_ok=True)
        (self.data_dir / "memory").mkdir(exist_ok=True)
        (self.data_dir / "vocabulary").mkdir(exist_ok=True)
        (self.data_dir / "training").mkdir(exist_ok=True)
        return self

class TrainingConfig(BaseModel):
    """Configuration for training the neural child"""
    interaction_batch_size: int = Field(10, ge=1, description="Number of interactions per training batch")
    learning_rate: float = Field(0.001, gt=0.0, description="Learning rate for neural components")
    max_training_sessions: int = Field(1000, ge=1, description="Maximum number of training sessions")
    validation_interval: int = Field(10, ge=1, description="Validate every N training sessions")
    checkpoint_dir: Path = Field(Path("./checkpoints"), description="Directory for model checkpoints")
    export_format: Literal["pth", "onnx"] = Field("pth", description="Format for exported models")
    use_cuda: bool = Field(True, description="Use CUDA for accelerated training if available")
    
    @model_validator(mode='after')
    def create_checkpoint_dir(self) -> 'TrainingConfig':
        """Ensure checkpoint directory exists"""
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        return self

class GlobalConfig(BaseModel):
    """Global configuration for the NeuralChild system"""
    system: SystemConfig = Field(default_factory=SystemConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    development: DevelopmentConfig = Field(default_factory=DevelopmentConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    
    # Network emphasis - how strongly each network contributes
    network_emphasis: Dict[NetworkType, float] = Field(default_factory=dict)
    
    @field_validator('network_emphasis')
    @classmethod
    def validate_network_emphasis(cls, v):
        """Ensure all network types have emphasis values and they sum to 1.0"""
        # Set default values for any missing network types
        for network_type in NetworkType:
            if network_type not in v:
                v[network_type] = 0.1
        
        # Normalize to ensure sum is 1.0
        total = sum(v.values())
        if total > 0:
            for network_type in v:
                v[network_type] /= total
        
        return v
    
    @classmethod
    def load_from_file(cls, path: Path) -> 'GlobalConfig':
        """Load configuration from a file"""
        import json
        if path.exists():
            with open(path, 'r') as f:
                data = json.load(f)
                # Convert string paths to Path objects
                if "system" in data and "data_dir" in data["system"]:
                    data["system"]["data_dir"] = Path(data["system"]["data_dir"])
                if "system" in data and "logs_dir" in data["system"]:
                    data["system"]["logs_dir"] = Path(data["system"]["logs_dir"])
                if "development" in data and "milestones_path" in data["development"]:
                    data["development"]["milestones_path"] = Path(data["development"]["milestones_path"])
                if "training" in data and "checkpoint_dir" in data["training"]:
                    data["training"]["checkpoint_dir"] = Path(data["training"]["checkpoint_dir"])
                return cls(**data)
        return cls()
    
    def save_to_file(self, path: Path) -> None:
        """Save configuration to a file"""
        import json
        # Convert Path objects to strings for JSON serialization
        data = self.model_dump()
        if "system" in data and "data_dir" in data["system"]:
            data["system"]["data_dir"] = str(data["system"]["data_dir"])
        if "system" in data and "logs_dir" in data["system"]:
            data["system"]["logs_dir"] = str(data["system"]["logs_dir"])
        if "development" in data and "milestones_path" in data["development"]:
            data["development"]["milestones_path"] = str(data["development"]["milestones_path"])
        if "training" in data and "checkpoint_dir" in data["training"]:
            data["training"]["checkpoint_dir"] = str(data["training"]["checkpoint_dir"])
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

# Create a default global configuration instance
CONFIG = GlobalConfig()

# Function to get the global configuration
def get_config() -> GlobalConfig:
    """Get the global configuration instance"""
    return CONFIG

# Function to initialize configuration from a file
def init_config(config_path: Optional[Path] = None) -> GlobalConfig:
    """Initialize configuration from a file if provided"""
    global CONFIG
    if config_path:
        CONFIG = GlobalConfig.load_from_file(config_path)
    return CONFIG