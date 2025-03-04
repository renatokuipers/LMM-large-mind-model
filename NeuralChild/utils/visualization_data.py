# visualization_data.py - Prepare data for dashboard display
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
import logging
from pathlib import Path
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger("VisualizationData")

class NetworkVisualizationData(BaseModel):
    """Data for visualizing a neural network"""
    name: str
    network_type: str
    activation: float = Field(0.0, ge=0.0, le=1.0)
    confidence: float = Field(0.0, ge=0.0, le=1.0)
    training_progress: float = Field(0.0, ge=0.0, le=1.0)
    error_rate: float = Field(0.0, ge=0.0, le=1.0)
    last_active: str  # ISO format datetime
    connections: Dict[str, Dict[str, Union[str, float]]] = Field(default_factory=dict)
    
    @field_validator('activation', 'confidence', 'training_progress', 'error_rate', mode='before')
    @classmethod
    def validate_float_range(cls, v):
        """Ensure float values are in range 0-1"""
        if isinstance(v, (int, float)):
            return max(0.0, min(1.0, float(v)))
        return v

class EmotionalStateData(BaseModel):
    """Data for visualizing emotional state"""
    joy: float = Field(0.0, ge=0.0, le=1.0)
    sadness: float = Field(0.0, ge=0.0, le=1.0)
    anger: float = Field(0.0, ge=0.0, le=1.0)
    fear: float = Field(0.0, ge=0.0, le=1.0)
    surprise: float = Field(0.0, ge=0.0, le=1.0)
    disgust: float = Field(0.0, ge=0.0, le=1.0)
    trust: float = Field(0.0, ge=0.0, le=1.0)
    anticipation: float = Field(0.0, ge=0.0, le=1.0)
    
    def dominant_emotion(self) -> Tuple[str, float]:
        """Return the dominant emotion and its intensity"""
        emotions = {
            "joy": self.joy,
            "sadness": self.sadness,
            "anger": self.anger,
            "fear": self.fear,
            "surprise": self.surprise,
            "disgust": self.disgust,
            "trust": self.trust,
            "anticipation": self.anticipation
        }
        dominant = max(emotions, key=emotions.get)
        return dominant, emotions[dominant]

class VocabularyData(BaseModel):
    """Data for visualizing vocabulary development"""
    total_words: int = Field(0, ge=0)
    active_vocabulary: int = Field(0, ge=0)
    passive_vocabulary: int = Field(0, ge=0)
    recent_words: List[str] = Field(default_factory=list)
    by_category: Dict[str, int] = Field(default_factory=dict)
    average_understanding: float = Field(0.0, ge=0.0, le=1.0)
    average_production: float = Field(0.0, ge=0.0, le=1.0)
    learning_acceleration: float = Field(0.0)

class DashboardData(BaseModel):
    """Main data container for dashboard visualization"""
    timestamp: str
    system_status: Dict[str, Any] = Field(default_factory=dict)
    networks: Dict[str, NetworkVisualizationData] = Field(default_factory=dict)
    emotional_state: EmotionalStateData = Field(default_factory=EmotionalStateData)
    vocabulary: VocabularyData = Field(default_factory=VocabularyData)
    recent_interactions: List[Dict[str, Any]] = Field(default_factory=list)
    development_metrics: Dict[str, Any] = Field(default_factory=dict)

def prepare_network_visualization_data(network_states: Dict[str, Dict[str, Any]]) -> Dict[str, NetworkVisualizationData]:
    """Prepare neural network data for visualization
    
    Args:
        network_states: Raw network state data
        
    Returns:
        Processed network visualization data
    """
    visualization_data = {}
    
    for network_type, state in network_states.items():
        # Initialize with defaults
        network_data = {
            "name": state.get("name", network_type),
            "network_type": network_type,
            "activation": state.get("activation", 0.0),
            "confidence": state.get("confidence", 0.0),
            "training_progress": state.get("training_progress", 0.0),
            "error_rate": state.get("error_rate", 0.0),
            "last_active": state.get("last_active", datetime.now().isoformat()),
            "connections": state.get("connections", {})
        }
        
        # Validate and convert to model
        visualization_data[network_type] = NetworkVisualizationData(**network_data)
    
    return visualization_data

def prepare_emotional_state_data(emotional_state: Dict[str, float]) -> EmotionalStateData:
    """Prepare emotional state data for visualization
    
    Args:
        emotional_state: Raw emotional state data
        
    Returns:
        Processed emotional state data
    """
    # Create default empty state, then update with values
    state_data = EmotionalStateData()
    
    for emotion, intensity in emotional_state.items():
        if hasattr(state_data, emotion):
            setattr(state_data, emotion, float(intensity))
    
    return state_data

def prepare_vocabulary_data(vocabulary_stats: Dict[str, Any]) -> VocabularyData:
    """Prepare vocabulary data for visualization
    
    Args:
        vocabulary_stats: Raw vocabulary statistics
        
    Returns:
        Processed vocabulary data
    """
    # Extract relevant fields with defaults
    vocab_data = {
        "total_words": vocabulary_stats.get("total_words", 0),
        "active_vocabulary": vocabulary_stats.get("active_vocabulary", 0),
        "passive_vocabulary": vocabulary_stats.get("passive_vocabulary", 0),
        "recent_words": vocabulary_stats.get("recent_words", [])[:10],  # Limit to 10 
        "by_category": vocabulary_stats.get("by_category", {}),
        "average_understanding": vocabulary_stats.get("average_understanding", 0.0),
        "average_production": vocabulary_stats.get("average_production", 0.0),
        "learning_acceleration": vocabulary_stats.get("learning_acceleration", 0.0)
    }
    
    return VocabularyData(**vocab_data)

def prepare_dashboard_data(
    system_status: Dict[str, Any],
    network_states: Dict[str, Dict[str, Any]],
    emotional_state: Dict[str, float],
    vocabulary_stats: Dict[str, Any],
    recent_interactions: List[Dict[str, Any]],
    development_metrics: Dict[str, Any]
) -> DashboardData:
    """Prepare complete dashboard data
    
    Args:
        system_status: System status data
        network_states: Network state data
        emotional_state: Emotional state data
        vocabulary_stats: Vocabulary statistics
        recent_interactions: Recent interaction history
        development_metrics: Development metrics
        
    Returns:
        Complete dashboard data
    """
    networks = prepare_network_visualization_data(network_states)
    emotional = prepare_emotional_state_data(emotional_state)
    vocabulary = prepare_vocabulary_data(vocabulary_stats)
    
    # Create dashboard data
    dashboard_data = DashboardData(
        timestamp=datetime.now().isoformat(),
        system_status=system_status,
        networks=networks,
        emotional_state=emotional,
        vocabulary=vocabulary,
        recent_interactions=recent_interactions[-10:],  # Limit to 10 most recent
        development_metrics=development_metrics
    )
    
    return dashboard_data

def generate_network_graph_data(network_states: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Generate data for network graph visualization
    
    Args:
        network_states: Network state data
        
    Returns:
        Network graph data (nodes and edges)
    """
    nodes = []
    edges = []
    
    # Process each network as a node
    for network_type, state in network_states.items():
        # Create node
        node = {
            "id": network_type,
            "label": state.get("name", network_type),
            "size": 30 + (state.get("activation", 0.0) * 20),  # Size based on activation
            "color": _get_node_color(state.get("activation", 0.0)),
            "activation": state.get("activation", 0.0),
            "confidence": state.get("confidence", 0.0),
            "training_progress": state.get("training_progress", 0.0)
        }
        nodes.append(node)
        
        # Process connections as edges
        for target, connection in state.get("connections", {}).items():
            # Create edge
            edge = {
                "source": network_type,
                "target": target,
                "id": f"{network_type}_to_{target}",
                "label": connection.get("type", "connection"),
                "size": max(1, connection.get("strength", 0.5) * 5),  # Edge size based on strength
                "color": _get_edge_color(connection.get("type", "connection")),
                "strength": connection.get("strength", 0.5),
                "type": connection.get("type", "connection")
            }
            edges.append(edge)
    
    return {
        "nodes": nodes,
        "edges": edges
    }

def _get_node_color(activation: float) -> str:
    """Get node color based on activation level
    
    Args:
        activation: Activation level (0-1)
        
    Returns:
        Hex color string
    """
    # Blue gradient based on activation level
    if activation < 0.25:
        return "#4dabf5"  # Light blue
    elif activation < 0.5:
        return "#2196f3"  # Medium blue
    elif activation < 0.75:
        return "#1976d2"  # Darker blue
    else:
        return "#0d47a1"  # Very dark blue

def _get_edge_color(connection_type: str) -> str:
    """Get edge color based on connection type
    
    Args:
        connection_type: Connection type
        
    Returns:
        Hex color string
    """
    # Colors for different connection types
    colors = {
        "excitatory": "#4caf50",  # Green
        "inhibitory": "#f44336",  # Red
        "modulatory": "#ff9800",  # Orange
        "feedback": "#9c27b0",    # Purple
        "associative": "#2196f3"  # Blue
    }
    
    return colors.get(connection_type.lower(), "#757575")  # Default to gray

def generate_emotion_chart_data(emotional_state: EmotionalStateData) -> Dict[str, Any]:
    """Generate data for emotional state radar chart
    
    Args:
        emotional_state: Emotional state data
        
    Returns:
        Chart data
    """
    # Extract emotions and values
    emotions = [
        "joy", "anticipation", "trust", "surprise", 
        "fear", "sadness", "disgust", "anger"
    ]
    
    values = [
        emotional_state.joy, 
        emotional_state.anticipation,
        emotional_state.trust,
        emotional_state.surprise,
        emotional_state.fear,
        emotional_state.sadness,
        emotional_state.disgust,
        emotional_state.anger
    ]
    
    # Close the radar plot by repeating first value
    emotions.append(emotions[0])
    values.append(values[0])
    
    return {
        "emotions": emotions,
        "values": values,
        "dominant": emotional_state.dominant_emotion()[0],
        "dominant_intensity": emotional_state.dominant_emotion()[1]
    }

def generate_vocabulary_chart_data(vocabulary: VocabularyData) -> Dict[str, Any]:
    """Generate data for vocabulary charts
    
    Args:
        vocabulary: Vocabulary data
        
    Returns:
        Chart data
    """
    # Vocabulary distribution
    distribution = {
        "active": vocabulary.active_vocabulary,
        "passive": vocabulary.passive_vocabulary,
        "total": vocabulary.total_words
    }
    
    # Category breakdown
    categories = vocabulary.by_category
    
    # Skills levels
    skills = {
        "understanding": vocabulary.average_understanding,
        "production": vocabulary.average_production
    }
    
    return {
        "distribution": distribution,
        "categories": categories,
        "skills": skills,
        "recent_words": vocabulary.recent_words,
        "learning_acceleration": vocabulary.learning_acceleration
    }

def save_visualization_data(data: DashboardData, output_dir: Path) -> str:
    """Save visualization data to file
    
    Args:
        data: Dashboard data
        output_dir: Directory to save to
        
    Returns:
        Path to saved file
    """
    # Create directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"dashboard_data_{timestamp}.json"
    filepath = output_dir / filename
    
    # Save data
    with open(filepath, 'w') as f:
        json.dump(data.model_dump(), f, indent=2)
    
    logger.info(f"Saved dashboard data to {filepath}")
    return str(filepath)

def load_visualization_data(filepath: Path) -> Optional[DashboardData]:
    """Load visualization data from file
    
    Args:
        filepath: Path to file
        
    Returns:
        Dashboard data or None if loading fails
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return DashboardData(**data)
    
    except Exception as e:
        logger.error(f"Error loading visualization data: {str(e)}")
        return None

def generate_development_timeline(milestones: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Generate development timeline from milestones
    
    Args:
        milestones: List of milestone events
        
    Returns:
        Timeline data for visualization
    """
    # Sort milestones by timestamp
    sorted_milestones = sorted(milestones, key=lambda x: x.get("timestamp", ""))
    
    # Convert to timeline format
    timeline_data = []
    
    for i, milestone in enumerate(sorted_milestones):
        # Calculate position on timeline (0-100)
        if len(sorted_milestones) > 1:
            position = (i / (len(sorted_milestones) - 1)) * 100
        else:
            position = 50
        
        # Format milestone for timeline
        timeline_item = {
            "id": i,
            "timestamp": milestone.get("timestamp", ""),
            "title": milestone.get("milestone", "Unknown milestone"),
            "category": milestone.get("category", "general"),
            "importance": milestone.get("importance", 1),
            "position": position,
            "details": milestone.get("details", {})
        }
        
        timeline_data.append(timeline_item)
    
    return timeline_data

def aggregate_metrics_over_time(metrics_file: Path) -> Dict[str, List[Any]]:
    """Aggregate metrics over time for trend visualization
    
    Args:
        metrics_file: Path to metrics file (JSONL format)
        
    Returns:
        Aggregated metrics
    """
    if not metrics_file.exists():
        logger.error(f"Metrics file not found: {metrics_file}")
        return {}
    
    # Initialize data structure
    aggregated = {
        "timestamps": [],
        "metrics": {}
    }
    
    try:
        # Read JSONL file line by line
        with open(metrics_file, 'r') as f:
            for line in f:
                record = json.loads(line.strip())
                
                # Extract timestamp
                timestamp = record.get("timestamp", "")
                aggregated["timestamps"].append(timestamp)
                
                # Extract metrics
                metrics = record.get("metrics", {})
                for key, value in metrics.items():
                    if key not in aggregated["metrics"]:
                        aggregated["metrics"][key] = []
                    
                    # Store numeric values only
                    if isinstance(value, (int, float)):
                        aggregated["metrics"][key].append(value)
                    else:
                        # For non-numeric values, store None to maintain index alignment
                        aggregated["metrics"][key].append(None)
    
    except Exception as e:
        logger.error(f"Error aggregating metrics: {str(e)}")
    
    return aggregated