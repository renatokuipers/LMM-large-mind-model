# 🧠 NeuralChild: A Psychological Mind Simulation Framework

NeuralChild creates a simulated mind that develops through interactions rather than traditional large-scale dataset training. The system models psychological development by implementing the psychological functions that emerge from biological brain structures.

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## Core Concept

The key innovation of NeuralChild is that this mind learns to communicate, feel, and develop a sense of self through nurturing interactions with a "Mother" LLM. The system isn't static; it literally "grows up" from infancy to maturity through these interactions.

Unlike traditional LLMs, NeuralChild represents a "Large Mind Model" (LMM) with the following characteristics:
- Independent thought and decision-making capabilities
- Adaptable norms and values through interactions
- Persistent memory and life experience accumulation
- Development of unique preferences and "personality"
- Self-awareness similar to human consciousness

## Installation

### Prerequisites

- Python 3.8 or higher
- A local LLM with structured output capabilities

### Installing from source

```bash
# Clone the repository
git clone https://github.com/yourusername/neuralchild.git
cd neuralchild

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"
```

## Usage

### Running the Dashboard

The simplest way to interact with NeuralChild is through the provided dashboard:

```bash
# Start the dashboard with default settings
neuralchild

# Specify a custom configuration file
neuralchild --config my_config.json

# Load a previously saved state
neuralchild --load-state saved_state.json

# Run on a different port
neuralchild --port 8051
```

### Headless Mode

For server environments or batch processing, you can run NeuralChild without the dashboard:

```bash
# Run simulation for 1000 interactions
neuralchild --no-dashboard --simulate 1000

# Save state after simulation
neuralchild --no-dashboard --simulate 1000 --save-after-simulate final_state.json
```

### Programmatic Usage

```python
from NeuralChild import NeuralChild, DevelopmentalStage

# Create a neural child instance
child = NeuralChild()

# Run a single interaction
result = child.interact_with_mother()

# Get developmental metrics
metrics = child.get_developmental_metrics()
print(f"Current developmental stage: {metrics['developmental_stage']}")
print(f"Vocabulary size: {metrics['vocabulary_size']}")

# Save and load state
child.save_state("my_child.json")
child.load_state("my_child.json")

# Run simulation for multiple interactions
simulation_result = child.simulate_development(
    num_interactions=100,
    callback=lambda info: print(f"Progress: {info['interaction']}/{info['total']}")
)
```

## Architectural Components

### The Mother Component

The Mother component functions as a nurturing caregiver that:
- Communicates through a structured output format
- Responds realistically (not omnisciently) to the child's state
- Has configurable personality traits and parenting styles
- Provides verbal responses, emotional states, non-verbal cues, and teaching elements

### The Neural Child's Mind

The child's mind consists of:
- A network of interconnected neural components representing psychological functions
- Specialized components for emotions, language, memory, and consciousness
- Dynamic activation patterns based on interactions
- Developmental metrics tracking growth over time
- Stage-based progression from prenatal to adulthood

### The Dashboard Interface

A comprehensive visualization system featuring:
- Real-time display of the child's mental state and development
- Interactive chat interface (available after sufficient development)
- System controls for training, saving/loading states
- Metrics tracking (vocabulary size, emotional development, training time)
- Neural network activation visualization

## Learning and Development Process

NeuralChild follows a psychological development framework with:

1. **Developmental Stages**:
   - Prenatal: Neural architecture formation
   - Infancy: Babbling, basic recognition
   - Early childhood: Rapid vocabulary acquisition
   - Middle childhood: Grammar emergence
   - Adolescence: Abstract thinking development
   - Various adult stages: Refinement and specialization

2. **Language Acquisition**:
   - Word-emotion associations
   - Context-based understanding
   - Progression from single words to complex sentences
   - Both explicit teaching and passive absorption

3. **Emotional Development**:
   - Emotional contagion from the Mother
   - Association of emotions with experiences
   - Increasing emotional stability over time
   - Development of complex emotional responses

## Configuration

NeuralChild is highly configurable through JSON configuration files:

```json
{
  "mother": {
    "personality_traits": {
      "openness": 0.7,
      "conscientiousness": 0.8,
      "extraversion": 0.6,
      "agreeableness": 0.9,
      "neuroticism": 0.2
    },
    "parenting_style": "authoritative",
    "teaching_style": "socratic"
  },
  "development": {
    "time_acceleration": 1000,
    "learning_rate_multiplier": 1.0,
    "vocabulary_acquisition_rate": 0.8,
    "emotional_development_rate": 0.7
  }
}
```

## Technical Considerations

- Processing is done on the CPU rather than GPU (psychological networks are smaller)
- Development is simulated at an accelerated rate
- Language development uses simplified models compared to biological reality
- The modular architecture allows for extensions with additional components

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

NeuralChild is inspired by various theories of developmental psychology, cognitive science, and artificial intelligence research.