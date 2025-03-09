# 🌐 Large Mind Model (LMM)

## 🧠 Project Overview

The **Large Mind Model (LMM)** project aims to create an authentic digital mind with genuine cognitive capabilities that develop progressively over time. Unlike traditional AI systems that rely on large-scale dataset training, LMM starts from a blank slate and learns through nurturing interactions with a "Mother" LLM.

The LMM system is built as a modular cognitive architecture with specialized modules for different psychological functions:

- **Perception**: Processes sensory input (text-based for now) and recognizes patterns
- **Attention**: Focuses cognitive resources on relevant information
- **Memory**: Stores experiences and knowledge in working, episodic, and semantic memory
- **Language**: Acquires and processes linguistic information
- **Emotion**: Generates and regulates emotional states
- **Consciousness**: Integrates information and maintains self-awareness
- And many other cognitive modules...

Each module develops gradually from simple to sophisticated processing capabilities, mirroring human psychological development.

## 🧠 Conceptual Foundations

The LMM is structured around distinct psychological "Mind Modules," each specialized in handling core cognitive aspects analogous to the human mind. These neural modules individually learn and interact collectively, mirroring the interconnected cognitive and psychological structure of the human psyche:

- **Memory**: Persistent semantic and episodic memory that stores experiences and retrieves contextually relevant memories.
- **Consciousness & Self-awareness:** Reflection, autonomous reasoning, introspection, and contextual awareness.
- **Language Acquisition & Understanding:** Deep comprehension of context, meaning, intention, linguistic nuance, and growth from simple to complex linguistic constructs.
- **Emotional Intelligence:** Genuine emotional comprehension, empathy, sentiment awareness, emotional state modeling, and emotional communication.
- **Social Cognition & Morality:** Awareness and understanding of social dynamics, interpersonal contexts, moral reasoning, and ethical learning.
- **Thought Generation:** Autonomous cognitive processing, creative ideation, logical reasoning, novel concept exploration.
- **Dreams & Imagination:** Generation of novel scenarios, abstract creative thinking, imagination, and subconscious thought processes.

## 🤖👩‍🍼 "Mother" Interaction: The Innovative Learning Paradigm

A key innovative feature of this project is the integration of a dedicated "Mother" LLM, which serves as a nurturing caregiver, educator, emotional guide, and conversational partner. This local LLM has carefully configurable traits and parenting styles, with capabilities including:

- **Structured communication**: Verbal dialogues, emotional expressions, non-verbal cues.
- **Personality Configuration:** Customizable traits, parenting styles, teaching approaches.
- **Realistic Interaction Dynamics:** Non-omniscient, supportive interactions mimicking real human caregiver behavior.
- **Developmental Guidance:** Incremental instruction, corrections, emotional support, and nurturing.

## 🌱 The Learning & Psychological Development Process

The LMM development process emulates human psychological growth through clearly defined developmental stages:

### Stage-Based Psychological Growth:

The mind experiences distinct developmental stages, accelerated for practical purposes but still closely modeling real-world psychological progression:

- **Prenatal (Initialization):** Establishment of neural structures and initial conditions.
- **Infancy & Childhood:** Early language acquisition, emotional awareness, memory formation, and identity establishment.
- **Adolescence:** Advanced emotional understanding, social awareness, independent thought processes, critical thinking, and morality refinement.
- **Adulthood:** Mature self-awareness, complex reflective reasoning, fully formed autonomous capabilities, and advanced creativity.

## 🛠️ Technical Implementation & Infrastructure

The project leverages powerful local AI infrastructure for a self-contained, privacy-focused environment:

### Core Modules & Architecture
- **Local "Mother" LLM:** Using a high-quality instruction-tuned model (e.g., Qwen2.5-7B-Instruct).
- **Semantic Embedding Layer:** Local embedding capabilities through "text-embedding-nomic-embed-text-v1.5", enabling memory indexing and retrieval.
- **Neural Networks:** Custom-trained neural modules for each cognitive aspect, developed using modern deep learning tools.

### Underlying Python Technologies:

- **Core Neural Framework:** PyTorch, NumPy, SciPy.
- **Language and Semantic Processing:** NLTK, local LLM ("qwen2.5-7b-instruct"), embedding via local APIs.
- **Memory Management:** Faiss, LanceDB, ChromaDB for semantic vector storage and retrieval.
- **Emotional and Sentiment Analysis:** TextBlob, custom neural classifiers.
- **Structured API Integration:** Clearly structured local REST API endpoints ensure clean modularity.

## 📈 Development Tracking, Interaction Visualization, and Simulation

The LMM project includes robust tooling to visualize progress, state, and module activations:

- **Real-time Development Visualization:**
  - Neural network activations.
  - Emotional states tracking.
  - Developmental metrics (language progression, emotional maturity).

- **Structured Interaction Logging & Memory Systems:**
  - Persistent storage of all interactions.
  - Semantic retrieval of past experiences.

- **Accelerated Development Simulation:**
  - Configurable time-progression ratios.
  - Natural variation and developmental plateaus.
  - Influencing factors affecting developmental pace and trajectory.

## 🔧 System Capabilities and User Interaction

### Users Can:

- Configure the "Mother" LLM personality and parenting traits.
- Visualize and monitor cognitive and emotional development.
- Interact conversationally as external observers or conversational partners. (only when the LMM has developed enough)
- Access system controls for training sessions, state loading and saving, and metrics.
- Directly view internal "thoughts" and the LLM's expressed outputs. (in human readable text)

## 🌟 Ultimate Project Goal & Vision

The core vision of the **Large Mind Model (LMM)** project is nothing less than pioneering a revolutionary, psychologically-grounded artificial intelligence. Rather than simply simulating human-like behaviors, the LMM embodies an autonomous, adaptive, emotionally-intelligent being, progressively evolving through realistic nurturing interactions.

## 🚀 Getting Started

### Prerequisites

- Python 3.9+
- Local LLM API (e.g., Qwen2.5-7B-Instruct)
- Local TTS API (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/lmm-project.git
cd lmm-project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment variables:
```
LLM_API_URL=http://192.168.2.12:1234
TTS_API_URL=http://127.0.0.1:7860
```

### Running the Project

Basic usage:
```bash
python main.py
```

With custom parameters:
```bash
python main.py --config custom_config.yml --development-rate 0.02 --cycles 500
```

Loading a saved state:
```bash
python main.py --load-state storage/states/mind_state_child_20230615_120000.json
```

## 📁 Project Structure

```
lmm_project/
│
├── core/                           # Core infrastructure
│   ├── mind.py                     # The integrated mind class
│   ├── event_bus.py                # Communication system between modules
│   ├── message.py                  # Message types for inter-module communication
│   ├── state_manager.py            # Global state tracking
│   └── exceptions.py               # Custom exceptions
│
├── neural_substrate/               # Foundation neural architecture
│   ├── neural_network.py           # Basic neural network implementation
│   ├── synapse.py                  # Connection between neurons
│   ├── neuron.py                   # Base neuron implementation
│   ├── hebbian_learning.py         # Basic learning mechanisms
│   ├── neural_cluster.py           # Functional neuron groupings
│   └── activation_functions.py     # Various activation functions
│
├── modules/                        # Cognitive modules
│   ├── base_module.py              # Abstract base class for all modules
│   ├── perception/                 # Sensory input processing
│   ├── attention/                  # Focus and salience detection
│   ├── memory/                     # Various memory systems
│   ├── language/                   # Language acquisition and processing
│   ├── emotion/                    # Emotional processing
│   └── ...                         # Other cognitive modules
│
├── interfaces/                     # External interaction
│   ├── mother/                     # Mother LLM interface
│   └── researcher/                 # Tools for observing the mind
│
├── utils/                          # Helper utilities
│   ├── llm_client.py               # LLM API client
│   ├── tts_client.py               # TTS API client
│   └── ...                         # Other utilities
│
├── storage/                        # Data storage
├── visualization/                  # System for monitoring
└── tests/                          # Testing suite
```

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- The project draws inspiration from various fields including developmental psychology, cognitive science, and neuroscience.
- Special thanks to the open-source AI community for providing the tools and models that make this project possible.

## 🚀 Perception Module

The first fully implemented module in the LMM system is the Perception module, which is responsible for processing raw sensory input (currently text) and recognizing patterns within it.

### Components

- **SensoryInputProcessor**: Processes raw text into feature vectors and preliminary sensory representations
- **PatternRecognizer**: Identifies patterns in the processed sensory data
- **PerceptionNetwork**: Neural network for advanced pattern processing
- **TemporalPatternNetwork**: Processes sequences of patterns to find temporal relationships

### Development Stages

The perception module develops through several stages:

1. **Basic sensory awareness (0.0-0.2)**: Detects basic text properties
2. **Simple pattern recognition (0.2-0.4)**: Identifies token-level patterns
3. **Feature integration (0.4-0.6)**: Recognizes more complex patterns and linguistic features
4. **Context-sensitive perception (0.6-0.8)**: Understands context and relationships between patterns
5. **Advanced pattern recognition (0.8-1.0)**: Sophisticated pattern analysis with interpretive capabilities

### Using the Perception Module

```python
from lmm_project.modules.perception import get_module
from lmm_project.core.event_bus import EventBus

# Create event bus for communication
event_bus = EventBus()

# Create perception module
perception = get_module(
    module_id="perception",
    event_bus=event_bus,
    development_level=0.0  # Start at basic level
)

# Process text input
result = perception.process_input({
    "text": "Hello, this is a test message.",
    "process_id": "test-1"
})

# Examine results
print(f"Detected {len(result['patterns'])} patterns")
for pattern in result['patterns']:
    print(f"Pattern type: {pattern['pattern_type']}, Confidence: {pattern['confidence']}")

# Increase development level
perception.update_development(0.5)  # Advance to intermediate level

# Process more complex input with the more developed system
advanced_result = perception.process_input({
    "text": "When I contemplate the natural world, I'm filled with wonder at its complexity and beauty.",
    "process_id": "test-2"
})

# Get interpretation of patterns
if "interpretation" in advanced_result:
    print(f"Content type: {advanced_result['interpretation']['content_type']}")
    print(f"Complexity: {advanced_result['interpretation']['complexity']}")
```

## 🧪 Testing

Test the perception module with different inputs and development levels:

```bash
pytest tests/modules/test_perception.py
```

## 🔍 GPU Usage

The perception module automatically uses CUDA acceleration if available:

```python
# Check if running on GPU
print(f"Using device: {perception.device}")
```

## 📊 Monitoring Development

You can track the development progress of the perception module:

```python
# Get development progress
progress = perception.get_development_progress()
print(f"Current milestone: {progress['current_milestone']}")
print(f"Progress to next milestone: {progress['progress_to_next_milestone']:.2f}")
```

## 📝 License

[Include license information here]

## 👥 Contributors

[Your name and other contributors]

## 📞 Contact

[Contact information]
