"""
Researcher Interface Module

This module provides tools for researchers to observe, measure, and analyze
the development of the LMM system. It includes components for:

- Tracking developmental progress across cognitive modules
- Collecting performance metrics and developmental indicators
- Observing internal states and activation patterns
- Analyzing trends and developmental trajectories

These tools enable scientific study of the developing mind model.
"""

from lmm_project.interfaces.researcher.models import (
    ResearchMetrics,
    DevelopmentalStage,
    MetricCategory,
    CognitiveModuleState,
    DevelopmentalMilestone,
    DevelopmentalEvent,
    LearningAnalysis,
    NeuralActivitySnapshot,
    DevelopmentalTrajectory,
    VisualizationRequest,
    SystemStateSnapshot
)
from lmm_project.interfaces.researcher.development_tracker import DevelopmentTracker
from lmm_project.interfaces.researcher.metrics_collector import MetricsCollector
from lmm_project.interfaces.researcher.state_observer import StateObserver

__all__ = [
    # Main classes
    'DevelopmentTracker',
    'MetricsCollector',
    'StateObserver',
    
    # Models
    'ResearchMetrics',
    'DevelopmentalStage',
    'MetricCategory',
    'CognitiveModuleState',
    'DevelopmentalMilestone',
    'DevelopmentalEvent',
    'LearningAnalysis',
    'NeuralActivitySnapshot',
    'DevelopmentalTrajectory',
    'VisualizationRequest',
    'SystemStateSnapshot'
] 
