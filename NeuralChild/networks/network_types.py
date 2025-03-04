# networks/network_types.py
from enum import Enum, auto

class NetworkType(str, Enum):
    """Types of neural networks in the child's mind"""
    ARCHETYPES = "archetypes"
    INSTINCTS = "instincts"
    UNCONSCIOUSNESS = "unconsciousness"
    DRIVES = "drives"
    EMOTIONS = "emotions"
    MOODS = "moods"
    ATTENTION = "attention"
    PERCEPTION = "perception"
    CONSCIOUSNESS = "consciousness"
    THOUGHTS = "thoughts"

class ConnectionType(str, Enum):
    """Types of connections between neural networks"""
    EXCITATORY = "excitatory"  # Increases activation of target network
    INHIBITORY = "inhibitory"  # Decreases activation of target network
    MODULATORY = "modulatory"  # Changes parameters of target network
    FEEDBACK = "feedback"      # Provides feedback to source network
    ASSOCIATIVE = "associative"  # Creates associations between networks