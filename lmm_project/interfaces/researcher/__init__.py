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

from lmm_project.interfaces.researcher.models import ResearchMetrics
from lmm_project.interfaces.researcher.development_tracker import DevelopmentTracker
from lmm_project.interfaces.researcher.metrics_collector import MetricsCollector
from lmm_project.interfaces.researcher.state_observer import StateObserver

__all__ = [
    'ResearchMetrics',
    'DevelopmentTracker',
    'MetricsCollector',
    'StateObserver',
] 
