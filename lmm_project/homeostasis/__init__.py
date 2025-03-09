"""
Homeostasis Module for the Large Mind Model (LMM)

This module implements regulatory systems that maintain internal balance and stability
in the developing cognitive system, similar to how biological organisms maintain homeostasis.

The homeostasis systems regulate:
1. Energy levels - Overall system energy and resource management
2. Arousal - Activation/stimulation level affecting learning and attention
3. Cognitive Load - Processing resource management and working memory
4. Social Needs - Regulation of social interaction requirements
5. Coherence - Internal consistency between beliefs and experiences

Homeostatic regulation is critical for healthy cognitive development, as it ensures
the system operates within optimal parameters while adapting to developmental stages.
"""

from .models import (
    HomeostaticSystem, 
    HomeostaticNeedType, 
    HomeostaticResponse, 
    NeedState
)
from .energy_regulation import EnergyRegulator
from .arousal_control import ArousalController
from .cognitive_load_balancer import CognitiveLoadBalancer
from .social_need_manager import SocialNeedManager
from .coherence import CoherenceManager

__all__ = [
    # Core models
    'HomeostaticSystem',
    'HomeostaticNeedType',
    'HomeostaticResponse',
    'NeedState',
    
    # Regulatory systems
    'EnergyRegulator',
    'ArousalController',
    'CognitiveLoadBalancer',
    'SocialNeedManager',
    'CoherenceManager',
]

# Mapping of need types to their primary regulators
NEED_REGULATORS = {
    HomeostaticNeedType.ENERGY: EnergyRegulator,
    HomeostaticNeedType.AROUSAL: ArousalController,
    HomeostaticNeedType.COGNITIVE_LOAD: CognitiveLoadBalancer,
    HomeostaticNeedType.SOCIAL: SocialNeedManager,
    HomeostaticNeedType.COHERENCE: CoherenceManager
} 
