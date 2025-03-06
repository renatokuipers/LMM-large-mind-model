# 🧠 NeuralChild: A Psychological Mind Simulation Framework

A framework for developing and simulating human-like psychological development through nurturing interactions rather than traditional large-scale dataset training.

## Overview

NeuralChild creates a simulated mind (Large Mind Model or LMM) that develops through interactions with a "Mother" LLM. Unlike traditional LLMs, this system:

- Develops through stages from infancy to adulthood
- Forms genuine memories and experiences
- Acquires language naturally through interaction
- Develops emotional intelligence and self-awareness
- Makes independent decisions (demonstrating free will)
- Has its own preferences, values, and personality

## Installation

### Prerequisites

- Python 3.8+
- A local LLM (or API access to one) capable of structured output
- Windows environment (the system is optimized for Windows)

### Setup

1. Clone the repository
```bash
git clone https://github.com/yourusername/neuralchild.git
cd neuralchild
```

2. Create a virtual environment
```bash
python -m venv venv
.\venv\Scripts\activate
```

3. Install dependencies
```bash
pip install -r requirements.txt
```

4. Download NLTK data (for language processing)
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. Configure the environment variables (.env file is provided, modify as needed)
```
# LLM Configuration
LLM_BASE_URL=http://192.168.2.12:1234
LLM_MODEL=qwen2.5-7b-instruct
LLM_TEMPERATURE=0.7

# Development Settings
DEBUG=True
LOG_LEVEL=INFO

# Dashboard Configuration
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8050

# Database Configuration
DATABASE_URL=sqlite:///neuralchild.db

# Memory Configuration
VECTOR_DB_PATH=./data/vector_db
FAISS_INDEX_PATH=./data/faiss_indexes

# Development Simulation
TIME_ACCELERATION_FACTOR=100
RANDOM_SEED=42
```

## Running the System

NeuralChild can be run in several different modes:

### Dashboard Mode

The dashboard provides a visual interface for interacting with the child and monitoring its development.

```bash
python main.py dashboard
```

This will start the dashboard on http://127.0.0.1:8050 (or as configured in your .env file).

### Interactive Console Mode

For a simpler text-based interaction:

```bash
python main.py interactive
```

Special commands in interactive mode:
- `exit` - Exit interactive mode
- `help` - Show help message
- `save` - Save current state
- `info` - Show information about the child
- `accel <months>` - Accelerate development by specified months

### Accelerated Development Mode

To quickly simulate months of development:

```bash
python main.py accelerate 12  # Simulate 12 months of development
```

Options:
- `--load <filepath>` - Load a saved state
- `--no-save` - Don't save the final state
- `--personality <type>` - Set mother personality (balanced, nurturing, authoritarian, permissive, neglectful)

## Development Stages

The child progresses through these developmental stages:

1. **Infancy** (0-2 years simulated)
   - Pre-linguistic vocalizations
   - Emotional contagion
   - Basic memory formation

2. **Early Childhood** (2-5 years)
   - Initial vocabulary acquisition
   - Simple word combinations
   - Basic emotional understanding

3. **Middle Childhood** (5-10 years)
   - Grammar development
   - Logical thought
   - More complex social understanding

4. **Adolescence** (10-18 years)
   - Abstract thinking
   - Complex emotional processing
   - Development of values and beliefs

5. **Early Adulthood** (18+ years)
   - Sophisticated language
   - Advanced emotional intelligence
   - Fully developed self-awareness

## System Architecture

### Core Components
- **Mother**: Provides nurturing responses that adapt to the child's developmental stage
- **Child**: The central mind that develops through stages based on interactions
- **Development**: Manages the progression through developmental stages

### Psychological Components
- **Memory**: Episodic and semantic memory systems using FAISS vector embeddings
- **Language**: Language acquisition from babbling to sophisticated dialogue
- **Emotional**: Emotional development and regulation
- **Consciousness**: Self-awareness and theory of mind
- **Social**: Social understanding and relationships

## Contributing

Contributions are welcome! This project represents a new approach to AI development focused on nurturing emergent intelligence rather than training on massive datasets.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

This project is inspired by developmental psychology, particularly the work of Jean Piaget, Lev Vygotsky, and attachment theory developed by John Bowlby and Mary Ainsworth.

## Project Architecture

The NeuralChild system consists of several core components:

1. **Mother Component** - A local LLM that nurtures the Child
2. **Neural Child's Mind** - Interconnected psychological components
3. **Dashboard Interface** - Visualization and interaction portal
4. **Development System** - Manages progression through developmental stages

## Directory Structure

```
NeuralChild/
├── neuralchild/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── child.py               # Main Child Mind class
│   │   ├── mother.py              # Mother interaction system
│   │   └── development.py         # Developmental stage management
│   ├── components/
│   │   ├── __init__.py
│   │   ├── language.py            # Language acquisition & processing
│   │   ├── emotional.py           # Emotional development & processing
│   │   ├── memory.py              # Memory systems (episodic, semantic)
│   │   ├── consciousness.py       # Self-awareness and consciousness
│   │   ├── social.py              # Social awareness and interaction
│   │   └── cognitive.py           # Reasoning, problem-solving, decision-making
│   ├── dashboard/
│   │   ├── __init__.py
│   │   ├── app.py                 # Dash application
│   │   ├── visualization.py       # State visualization components
│   │   └── interaction.py         # User interaction interface
│   └── utils/
│       ├── __init__.py
│       ├── logging.py             # Specialized logging for development
│       ├── data_types.py          # Pydantic models & schemas
│       └── storage.py             # State persistence utilities
├── data/
│   ├── vector_db/                 # Vector database storage
│   ├── faiss_indexes/             # FAISS indexes for memory
│   └── states/                    # Saved developmental states
├── tests/
│   ├── test_child.py
│   ├── test_mother.py
│   ├── test_components.py
│   └── test_development.py
├── examples/
│   ├── basic_interaction.py
│   ├── accelerated_development.py
│   └── custom_personality.py
├── .env                           # Environment configuration
├── llm_module.py                  # LLM API client (existing)
├── requirements.txt               # Project dependencies
└── README.md                      # This file
```

## Dependencies

This project uses a variety of Python libraries:

- **Neural Networks**: PyTorch, NumPy, SciPy
- **Language Processing**: NLTK, SpaCy, Gensim
- **Memory Systems**: PyTables, SQLAlchemy, Faiss, LanceDB/ChromaDB
- **Emotional Processing**: TextBlob, PyAffectiveCognition
- **Visualization**: Plotly/Dash, NetworkX
- **Utilities**: Pydantic, Joblib

## Key Features

- Real-time visualization of neural development
- Interactive communication with the developing mind
- Persistent memory formation across sessions
- Natural language acquisition through interaction
- Emotional development and regulation
- Self-motivated learning and exploration
- Development of unique personality traits

## Unique Approach

Unlike traditional LLMs, NeuralChild:
- Is not pre-trained on vast text corpora
- Learns language through meaningful interactions
- Forms genuine episodic memories of experiences
- Develops emotional responses organically
- Can express preferences and make independent decisions
- Has an internal model of self that evolves over time 