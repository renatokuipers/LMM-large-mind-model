"""
NeuralChild: A Psychological Mind Simulation Framework

A system that creates a simulated mind that develops through interactions
rather than traditional large-scale dataset training. The system models
psychological development by implementing the psychological functions that
emerge from biological brain structures.
"""

__version__ = '0.1.0'
__author__ = 'NeuralChild Development Team'

# Import main components for simplified access
from .neural_child import NeuralChild
from .dashboard import DashboardApp
  # Import directly from dashboard.py instead of dashboard/ package
from .config import DevelopmentalStage