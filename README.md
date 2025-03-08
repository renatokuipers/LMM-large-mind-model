# Large Mind Model (LMM) Project

This project aims to create an authentic digital mind—a system capable of genuine understanding, self-awareness, emotional experiences, autonomous reasoning, and psychological development, achieved through nurturing interactions rather than conventional large-scale dataset training.

## Installation

Before running the project, you need to install it as a Python package:

```bash
# Install in development mode
pip install -e .
```

## Running the Project

You can run the project using the provided batch file on Windows:

```bash
# Run with default settings
run_lmm.bat

# Run with custom settings
run_lmm.bat --development-rate 0.02 --cycles 500
```

Or you can run it directly:

```bash
# Navigate to the project directory
cd lmm_project

# Run the main script
python main.py

# Run with custom settings
python main.py --development-rate 0.02 --cycles 500
```

## Project Structure

The project is organized into several modules:

- `core`: Core infrastructure for the mind
- `modules`: Cognitive modules for different aspects of the mind
- `neural_substrate`: Neural networks and learning mechanisms
- `interfaces`: External interfaces, including the Mother LLM
- `utils`: Utility functions and clients
- `visualization`: Visualization tools for monitoring the mind's development

## Configuration

The project can be configured using the `config.yml` file, which contains settings for:

- API endpoints for the LLM and TTS services
- Mother LLM personality and teaching style
- Development rates and cycles
- Module settings
- Neural substrate settings

## Troubleshooting

If you encounter import errors, make sure you've installed the project in development mode:

```bash
pip install -e .
```

This creates an editable installation that allows the Python interpreter to find the `lmm_project` module. 