# Psychological Mind

A sophisticated neural network system that simulates psychological processes of the human mind. This project implements various neural network architectures to model different aspects of cognition, emotion, and consciousness.

## System Architecture

The system consists of multiple specialized neural networks, each representing different psychological components:

### Core Components
- Archetypes (GRU)
- Instincts (LSTM)
- Repressed Memories (Autoencoder)
- Unconsciousness (Deep Belief Network)
- ID (Self-Organizing Map)
- Drives (Feed-Forward Neural Network)
- And many more...

Each component communicates in a web-like structure, simulating the interconnected nature of human cognition.

## Development Setup

### Prerequisites
- Python 3.9.13
- CUDA 12.1
- Local LLM setup for "mother" training

### Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd psychological_mind
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -e .
```

## Project Structure

```
psychological_mind/
├── src/
│   ├── networks/           # Neural network implementations
│   │   ├── archetypes.py
│   │   ├── instincts.py
│   │   └── ...
│   ├── training/          # Training utilities
│   │   ├── mother_llm.py
│   │   └── trainer.py
│   ├── utils/            # Helper utilities
│   │   ├── logging.py
│   │   └── visualization.py
│   └── core/             # Core system components
│       ├── brain.py
│       └── communication.py
├── tests/                # Test suite
├── configs/              # Configuration files
├── notebooks/           # Jupyter notebooks for experiments
└── logs/                # Training logs
```

## License
MIT License
