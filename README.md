# Neural Child Project

A computational simulation of child cognitive and emotional development using neural components.

## Overview

The Neural Child project is a sophisticated psychological simulation that models a child's development from infancy through early childhood. It incorporates various aspects of child development:

- **Emotional Development**: Simulates how a child's emotional responses develop over time
- **Language Acquisition**: Models the process of learning language from caregiver interactions
- **Memory Formation**: Simulates episodic, semantic, and procedural memory development
- **Cognitive Development**: Models the development of attention, reasoning, and problem-solving abilities
- **Social Development**: Simulates how the child learns to interact with caregivers

The system does not rely on external LLMs for generating responses, but instead uses neural components to simulate the internal psychological processes of a developing child.

## Project Structure

```
neural_child/
├── mind/               # Core mind architecture
│   ├── base.py         # Base classes for neural components
│   └── mind.py         # Central Mind class integrating all components
├── emotion/            # Emotional development components
├── language/           # Language acquisition components
├── memory/             # Memory formation and retrieval
├── cognition/          # Cognitive processing components
├── development/        # Developmental stage management
└── social/             # Social interaction components

mother/                 # Mother interaction components
├── mother.py           # Mother agent implementation

dashboard/              # Visualization and interaction UI
├── app.py              # Dash application
└── assets/             # Dashboard assets

utils/                  # Utility functions and configurations
```

## Key Features

- **Developmental Stages**: The child progresses through realistic developmental stages
- **Emotional Simulation**: Models basic emotions and their development over time
- **Natural Language Processing**: Simulates language acquisition from simple sounds to complex sentences
- **Memory Systems**: Models different types of memory formation and recall
- **Interactive Dashboard**: Provides visualization and interaction capabilities
- **Mother Agent**: Simulates a caregiver that interacts with and nurtures the Neural Child

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages listed in `requirements.txt`

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/neural-child.git
   cd neural-child
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Running the Dashboard

1. Start the Dash application:
   ```
   python -m dashboard.app
   ```

2. Open your browser and navigate to `http://127.0.0.1:8050/`

## Using the Dashboard

The dashboard provides a comprehensive interface for interacting with and monitoring the Neural Child:

### Child Status Panel

- **Age Display**: Shows the current age of the child
- **Development Speed**: Adjust how quickly the child develops
- **Advance Time**: Move forward in time to see development progress
- **Physiological Needs**: Monitor and address the child's basic needs

### Development Metrics

- **Language Development**: Track vocabulary size, grammar complexity, etc.
- **Emotional Development**: Monitor emotional regulation, complexity, etc.
- **Cognitive Development**: Track attention span, problem-solving abilities, etc.
- **Memory Development**: Monitor memory capacity and retention

### Interaction Panel

- **Mother's Utterances**: Enter text to communicate with the child
- **Emotional State**: Set the mother's emotional state during interaction
- **Teaching Elements**: Introduce new concepts to the child
- **Conversation History**: View the history of interactions

## Development Timeline and Milestones

The Neural Child progresses through several developmental stages:

1. **Prenatal** (before birth): Basic neural architecture formation
2. **Neonatal** (0-1 month): Reflexive responses, primitive emotions
3. **Early Infancy** (1-6 months): Basic emotions, pattern recognition
4. **Late Infancy** (6-12 months): Attachment, primitive communication
5. **Toddlerhood** (1-3 years): Language explosion, emotional complexity
6. **Early Childhood** (3-6 years): Advanced language, social understanding

Each stage includes specific milestones in language, emotional, cognitive, and social domains.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 