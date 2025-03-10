# Large Mind Model (LMM) Project

The Large Mind Model (LMM) is a cognitive architecture that models developmental psychology principles to create an artificial mind that learns genuinely from zero, developing capabilities over time through experience.

## Project Overview

The LMM project is built around these core principles:

- **Developmental Approach**: The system progresses through distinct developmental stages similar to human cognitive development
- **Modular Architecture**: Specialized cognitive modules handle different aspects of cognition
- **Event-Based Communication**: Modules communicate through a central event bus
- **Pure Learning**: Capabilities emerge from learning rather than being pre-programmed
- **Neural Substrate**: All cognitive functions are built on a simulated neural substrate

## Architecture

The system is organized into the following components:

### Core System
- **Mind**: Central coordinator that manages modules and overall system state
- **EventBus**: Communication system enabling modules to exchange messages
- **StateManager**: Tracks and persists the system's state
- **Message System**: Defines structured messages for inter-module communication

### Neural Substrate
- **Neurons**: Simulated neural building blocks with activation functions
- **Synapses**: Connections between neurons 
- **Neural Clusters**: Functional groupings of neurons
- **Hebbian Learning**: Implementation of "neurons that fire together, wire together"

### Cognitive Modules
- **Perception**: Processes sensory input
- **Attention**: Controls focus and salience detection
- **Memory**: Stores and retrieves experiences (working, episodic, semantic, associative)
- **Language**: Handles language acquisition and processing
- **Emotion**: Manages emotional states
- **Consciousness**: Integrates information across modules
- **Executive**: Handles planning and decision-making
- **Social**: Implements theory of mind and relationship modeling
- **Motivation**: Handles drives, needs, and rewards
- **Temporal**: Processes sequences, predictions, and causality
- **Creativity**: Enables imagination and novel idea generation
- **Self-Regulation**: Controls emotional regulation and impulse control
- **Learning**: Implements learning mechanisms
- **Identity**: Manages self-concept and personality
- **Belief**: Handles belief formation and updating

### Interfaces
- **Mother Interface**: Provides nurturing and teaching through an LLM
- **Researcher Interface**: Allows monitoring and analyzing development

### Storage and Persistence
- **Vector Storage**: Manages embedding-based memory
- **State Persistence**: Saves and loads system states
- **Experience Logger**: Records experiences for analysis

## Development Stages

The LMM progresses through these developmental stages:

1. **Prenatal** (0.0-0.1 age units): Basic neural formation and simple patterns
2. **Infant** (0.1-1.0 age units): Object permanence, early language, basic emotions
3. **Child** (1.0-3.0 age units): Expanding vocabulary, episodic memory, social awareness
4. **Adolescent** (3.0-6.0 age units): Abstract thinking, identity formation, complex reasoning
5. **Adult** (6.0+ age units): Integrated thinking, self-directed learning, philosophical reasoning

## Implementation

The system is implemented in Python with these key technologies:

- **PyTorch**: For neural network operations
- **FAISS**: For efficient vector storage and retrieval
- **Pydantic**: For structured data validation
- **CUDA Acceleration**: GPU acceleration for neural processing

## Getting Started

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure CUDA 12.1 is installed if using GPU acceleration
4. Configure the system by editing config.yml

### Running the System

To start the system:

```python
from lmm_project.core import get_mind

# Initialize and start the mind
mind = get_mind(config_path="config.yml")
mind.start()

# To stop the system
mind.stop()
```

## Module Extension

The system is designed to be extensible. To add a new module:

1. Create a class that inherits from BaseModule
2. Implement the required methods (process_input, update_development, get_state, save_state)
3. Register the module with the mind:

```python
from lmm_project.core import ModuleType, get_mind
from my_module import MyModule

mind = get_mind()
my_module = MyModule()
mind.register_module("my_module_id", my_module, ModuleType.PERCEPTION)
```

## Configuration

The system configuration is stored in config.yml. Key configuration options include:

- Development rate
- Checkpoint intervals
- LLM API endpoints
- Module-specific settings

## Project Status

This project is currently in active development. The core architecture is established, but many cognitive modules are still in early stages of implementation.

## License

[MIT License](LICENSE)
