# app.py
import os
import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple, Union, Literal
from pathlib import Path
import threading
import queue
import uuid
import plotly.graph_objs as go
import numpy as np
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from pydantic import BaseModel, Field

# Import NeuralChild components
from neural_child import NeuralChild
from llm_module import LLMClient, Message
from mother import MotherConfig, Mother, ChildState, InteractionHistory
from memory.memory_manager import MemoryType, MemoryManager
from networks.network_types import NetworkType
from config import get_config, init_config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("neural_child_dashboard.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("NeuralChildDashboard")

# Initialize configuration
config = init_config(Path("./config.json") if os.path.exists("./config.json") else None)

# Initialize LLM client
llm_client = LLMClient(base_url=config.system.llm_base_url)

# Constants
REFRESH_INTERVAL = 2  # seconds
CHAT_MILESTONE_THRESHOLD = 0.75  # The milestone completion threshold to enable chat
AUTO_SAVE_INTERVAL = 30 * 60  # 30 minutes in seconds

# Pydantic models for dashboard data
class DashboardState(BaseModel):
    training_active: bool = False
    paused: bool = False
    current_milestone_pct: float = 0.0
    child_age_days: float = 0.0
    chat_enabled: bool = False
    mother_state: Dict[str, Any] = Field(default_factory=dict)
    neural_networks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    memory_stats: Dict[str, Any] = Field(default_factory=dict)
    language_stats: Dict[str, Any] = Field(default_factory=dict)
    consciousness_status: Dict[str, Any] = Field(default_factory=dict)
    identity_status: Dict[str, Any] = Field(default_factory=dict)
    latest_thoughts: List[str] = Field(default_factory=list)
    latest_emotions: Dict[str, float] = Field(default_factory=dict)
    latest_perception: List[str] = Field(default_factory=list)
    latest_interaction: Dict[str, str] = Field(default_factory=dict)
    milestone_progress: Dict[str, float] = Field(default_factory=dict)
    
    @property
    def overall_milestone_progress(self) -> float:
        """Calculate overall milestone progress"""
        if not self.milestone_progress:
            return 0.0
        return sum(self.milestone_progress.values()) / len(self.milestone_progress)

class ChatMessage(BaseModel):
    role: Literal["user", "assistant", "system", "mother"] 
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)

# Global state
dashboard_state = DashboardState()
neural_child: Optional[NeuralChild] = None
mother: Optional[Mother] = None
interaction_queue = queue.Queue()
chat_history: List[ChatMessage] = []
last_save_time = time.time()

# Initialize Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True
)
server = app.server
app.title = "NeuralChild Mind Simulation"

# Helper functions
def format_time(seconds: float) -> str:
    """Format seconds into a readable time string"""
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

def create_network_activation_figure(networks: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Create a radar chart showing network activations"""
    network_names = list(networks.keys())
    activations = [networks[net].get("activation", 0) for net in network_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=activations,
        theta=network_names,
        fill='toself',
        name='Activation Level',
        line_color='rgba(57, 255, 20, 0.8)',
        fillcolor='rgba(57, 255, 20, 0.2)'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=300
    )
    
    return fig

def create_emotion_chart(emotions: Dict[str, float]) -> go.Figure:
    """Create a bar chart showing emotions"""
    if not emotions:
        emotions = {"joy": 0, "sadness": 0, "anger": 0, "fear": 0, "surprise": 0, "trust": 0}
        
    # Sort emotions by intensity
    sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
    emotion_names = [e[0] for e in sorted_emotions]
    emotion_values = [e[1] for e in sorted_emotions]
    
    # Choose colors based on emotion
    color_map = {
        "joy": "rgba(255, 215, 0, 0.8)",      # Gold
        "trust": "rgba(0, 191, 255, 0.8)",    # Deep Sky Blue
        "fear": "rgba(138, 43, 226, 0.8)",    # Blue Violet
        "surprise": "rgba(255, 140, 0, 0.8)", # Dark Orange
        "sadness": "rgba(30, 144, 255, 0.8)", # Dodger Blue
        "disgust": "rgba(50, 205, 50, 0.8)",  # Lime Green
        "anger": "rgba(220, 20, 60, 0.8)",    # Crimson
        "anticipation": "rgba(240, 230, 140, 0.8)" # Khaki
    }
    
    colors = [color_map.get(e, "rgba(200, 200, 200, 0.8)") for e in emotion_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=emotion_names,
        y=emotion_values,
        marker_color=colors
    ))
    
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=250,
        yaxis=dict(
            range=[0, 1],
            title=None
        ),
        xaxis=dict(
            title=None
        )
    )
    
    return fig

def create_milestone_progress_chart(milestones: Dict[str, float]) -> go.Figure:
    """Create a horizontal bar chart showing milestone progress"""
    if not milestones:
        milestones = {"No milestones tracked yet": 0}
        
    milestone_names = list(milestones.keys())
    milestone_values = list(milestones.values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=milestone_names,
        x=milestone_values,
        orientation='h',
        marker=dict(
            color='rgba(57, 255, 20, 0.8)',
            line=dict(color='rgba(57, 255, 20, 1.0)', width=1)
        )
    ))
    
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=300,
        xaxis=dict(
            range=[0, 1],
            title="Progress",
            tickformat=".0%"
        ),
        yaxis=dict(
            title=None
        )
    )
    
    return fig

def create_vocab_chart(vocab_history: List[Tuple[float, int]]) -> go.Figure:
    """Create a line chart showing vocabulary growth over time"""
    if not vocab_history or len(vocab_history) < 2:
        # Create placeholder data if not enough real data
        ages = [0, 1, 2]
        vocab_sizes = [0, 10, 20]
    else:
        ages = [point[0] for point in vocab_history]
        vocab_sizes = [point[1] for point in vocab_history]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=ages,
        y=vocab_sizes,
        mode='lines+markers',
        name='Vocabulary Size',
        line=dict(color='rgba(57, 255, 20, 0.8)', width=2),
        marker=dict(size=8, color='rgba(57, 255, 20, 1.0)')
    ))
    
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=250,
        xaxis=dict(
            title="Age (days)"
        ),
        yaxis=dict(
            title="Words"
        )
    )
    
    return fig

def create_network_training_chart(networks: Dict[str, Dict[str, Any]]) -> go.Figure:
    """Create a bar chart showing network training progress"""
    network_names = list(networks.keys())
    training_progress = [networks[net].get("training_progress", 0) for net in network_names]
    confidence = [networks[net].get("confidence", 0) for net in network_names]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=network_names,
        y=training_progress,
        name='Training Progress',
        marker_color='rgba(57, 255, 20, 0.8)'
    ))
    
    fig.add_trace(go.Bar(
        x=network_names,
        y=confidence,
        name='Confidence',
        marker_color='rgba(255, 215, 0, 0.8)'
    ))
    
    fig.update_layout(
        barmode='group',
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=300,
        yaxis=dict(
            range=[0, 1],
            title=None
        ),
        xaxis=dict(
            title=None,
            tickangle=45
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def generate_network_cards(networks: Dict[str, Dict[str, Any]]) -> List[dbc.Card]:
    """Generate cards for each neural network"""
    cards = []
    
    for network_name, network_data in networks.items():
        card = dbc.Card([
            dbc.CardHeader(html.H5(network_name.capitalize())),
            dbc.CardBody([
                html.Div([
                    html.P(f"Activation: {network_data.get('activation', 0):.2f}", className="network-stat"),
                    html.P(f"Confidence: {network_data.get('confidence', 0):.2f}", className="network-stat"),
                    html.P(f"Training: {network_data.get('training_progress', 0):.2f}", className="network-stat"),
                    html.Hr(),
                    html.Div(
                        [html.P(f"{k}: {v}") for k, v in network_data.items() 
                         if k not in ['activation', 'confidence', 'training_progress', 'name', 'network_type']],
                        className="network-details"
                    )
                ])
            ])
        ], className="network-card mb-3")
        
        cards.append(card)
    
    return cards

def memory_pie_chart(memory_stats: Dict[str, Any]) -> go.Figure:
    """Create a pie chart showing memory type distribution"""
    if not memory_stats or "items_by_type" not in memory_stats:
        memory_types = ["WORKING", "EPISODIC", "LONG_TERM", "ASSOCIATIVE"]
        memory_counts = [0, 0, 0, 0]
    else:
        memory_types = list(memory_stats["items_by_type"].keys())
        memory_counts = list(memory_stats["items_by_type"].values())
    
    fig = go.Figure()
    
    fig.add_trace(go.Pie(
        labels=memory_types,
        values=memory_counts,
        hole=.3,
        marker=dict(
            colors=[
                'rgba(57, 255, 20, 0.8)',   # Green
                'rgba(0, 191, 255, 0.8)',   # Blue
                'rgba(255, 215, 0, 0.8)',   # Gold
                'rgba(220, 20, 60, 0.8)'    # Red
            ]
        )
    ))
    
    fig.update_layout(
        margin=dict(l=30, r=30, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0.1)',
        font=dict(color='rgba(200, 200, 200, 0.8)'),
        height=300
    )
    
    return fig

# Training process management
def process_interaction(interaction_type: str, message: Optional[str] = None) -> None:
    """Process an interaction with the neural child"""
    global neural_child, mother, dashboard_state
    
    if not neural_child or not mother:
        logger.error("Neural child or mother not initialized")
        return
    
    try:
        # Prepare child state for mother
        child_state = ChildState(
            message=message or "",
            apparent_emotion=neural_child.get_apparent_emotion(),
            vocabulary_size=neural_child.get_vocabulary_size(),
            age_days=neural_child.age_days,
            recent_concepts_learned=neural_child.get_recent_concepts_learned(),
            attention_span=neural_child.get_attention_span()
        )
        
        # Get mother's response
        mother_response = mother.respond_to_child(child_state)
        
        # Process mother's response in the neural child
        neural_child.process_mother_response(mother_response)
        
        # Update dashboard state
        update_dashboard_state()
        
        # Record interaction
        dashboard_state.latest_interaction = {
            "child_message": message or "",
            "mother_response": mother_response.verbal.text,
            "mother_emotion": mother_response.emotional.primary_emotion,
            "timestamp": datetime.now().isoformat()
        }
        
        # Add to chat history if we're enabling chat view of mother interactions
        chat_history.append(ChatMessage(
            role="assistant",
            content=message or neural_child.generate_utterance()
        ))
        
        chat_history.append(ChatMessage(
            role="mother",
            content=mother_response.verbal.text
        ))
        
    except Exception as e:
        logger.error(f"Error processing interaction: {str(e)}")

def training_loop() -> None:
    """Main training loop for the neural child"""
    global neural_child, mother, dashboard_state, last_save_time
    
    try:
        while dashboard_state.training_active:
            if dashboard_state.paused:
                time.sleep(1)
                continue
            
            try:
                # Process any queued interactions first
                if not interaction_queue.empty():
                    interaction = interaction_queue.get_nowait()
                    process_interaction(interaction["type"], interaction.get("message"))
                else:
                    # Generate a child utterance
                    utterance = neural_child.generate_utterance()
                    process_interaction("auto", utterance)
                
                # Check if milestones reached for chat enabling
                if not dashboard_state.chat_enabled:
                    if dashboard_state.overall_milestone_progress >= CHAT_MILESTONE_THRESHOLD:
                        dashboard_state.chat_enabled = True
                        logger.info(f"Chat enabled! Milestone progress: {dashboard_state.overall_milestone_progress:.2f}")
                
                # Auto-save periodically
                current_time = time.time()
                if current_time - last_save_time > AUTO_SAVE_INTERVAL:
                    save_state()
                    last_save_time = current_time
                
            except Exception as e:
                logger.error(f"Error in training iteration: {str(e)}")
            
            # Add a delay to prevent excessive CPU usage
            time.sleep(0.5)  
            
    except Exception as e:
        logger.error(f"Error in training loop: {str(e)}")
        dashboard_state.training_active = False

def update_dashboard_state() -> None:
    """Update the dashboard state with current neural child state"""
    global neural_child, dashboard_state
    
    if not neural_child:
        return
    
    try:
        # Get network states
        networks = neural_child.get_network_states()
        dashboard_state.neural_networks = networks
        
        # Get memory stats
        memory_stats = neural_child.get_memory_stats()
        dashboard_state.memory_stats = memory_stats
        
        # Get language stats
        language_stats = neural_child.get_language_stats()
        dashboard_state.language_stats = language_stats
        
        # Get consciousness state
        consciousness = networks.get("consciousness", {})
        dashboard_state.consciousness_status = consciousness
        
        # Get identity info (from long-term memory)
        if "memory_stats" in memory_stats and "identity_state" in memory_stats:
            dashboard_state.identity_status = memory_stats["identity_state"]
        
        # Get latest thoughts
        thoughts_network = networks.get("thoughts", {})
        dashboard_state.latest_thoughts = thoughts_network.get("thoughts", [])
        
        # Get latest emotions
        emotions_network = networks.get("emotions", {})
        dashboard_state.latest_emotions = emotions_network.get("emotional_state", {})
        
        # Get latest perceptions
        perception_network = networks.get("perception", {})
        dashboard_state.latest_perception = perception_network.get("percepts", [])
        
        # Get milestone progress
        milestones = neural_child.get_milestone_progress()
        dashboard_state.milestone_progress = milestones
        
        # Age and other basic info
        dashboard_state.child_age_days = neural_child.age_days
        dashboard_state.current_milestone_pct = dashboard_state.overall_milestone_progress
        
    except Exception as e:
        logger.error(f"Error updating dashboard state: {str(e)}")

def init_neural_child() -> None:
    """Initialize the neural child and mother"""
    global neural_child, mother, dashboard_state
    
    try:
        # Initialize neural child
        neural_child = NeuralChild()
        
        # Initialize mother
        mother = Mother(llm_client=llm_client)
        
        # Update dashboard state
        update_dashboard_state()
        
        logger.info("Neural child and mother initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing neural child: {str(e)}")

def start_training() -> None:
    """Start the training process"""
    global dashboard_state
    
    if dashboard_state.training_active:
        return
    
    dashboard_state.training_active = True
    dashboard_state.paused = False
    
    # Start training in a separate thread
    training_thread = threading.Thread(target=training_loop, daemon=True)
    training_thread.start()
    
    logger.info("Training started")

def pause_training() -> None:
    """Pause the training process"""
    global dashboard_state
    
    if not dashboard_state.training_active:
        return
    
    dashboard_state.paused = True
    logger.info("Training paused")

def resume_training() -> None:
    """Resume the training process"""
    global dashboard_state
    
    if not dashboard_state.training_active:
        start_training()
        return
    
    dashboard_state.paused = False
    logger.info("Training resumed")

def stop_training() -> None:
    """Stop the training process"""
    global dashboard_state
    
    dashboard_state.training_active = False
    dashboard_state.paused = False
    logger.info("Training stopped")

def save_state() -> None:
    """Save the current state of the neural child"""
    global neural_child
    
    if not neural_child:
        return
    
    try:
        # Create a timestamp for the save file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"./saved_states/neural_child_{timestamp}")
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save neural child state
        neural_child.save_state(save_path)
        
        logger.info(f"State saved to {save_path}")
        
    except Exception as e:
        logger.error(f"Error saving state: {str(e)}")

def load_state(state_path: str) -> None:
    """Load a saved state"""
    global neural_child, mother, dashboard_state
    
    try:
        # Ensure path exists
        if not os.path.exists(state_path):
            logger.error(f"State file not found: {state_path}")
            return
        
        # Initialize a new neural child
        neural_child = NeuralChild()
        
        # Load the saved state
        neural_child.load_state(Path(state_path))
        
        # Reinitialize the mother
        mother = Mother(llm_client=llm_client)
        
        # Update dashboard state
        update_dashboard_state()
        
        # Check if milestones reached for chat enabling
        if dashboard_state.overall_milestone_progress >= CHAT_MILESTONE_THRESHOLD:
            dashboard_state.chat_enabled = True
        
        logger.info(f"State loaded from {state_path}")
        
    except Exception as e:
        logger.error(f"Error loading state: {str(e)}")

# Define the layout
app.layout = dbc.Container([
    dcc.Store(id="chat-store", data={"history": []}),
    dcc.Interval(id="refresh-interval", interval=REFRESH_INTERVAL * 1000, n_intervals=0),
    
    # Header
    dbc.Row([
        dbc.Col([
            html.H1("NeuralChild Mind Simulation", className="header-title"),
            html.P("A psychological mind simulation framework", className="header-subtitle")
        ], width=8),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.H5("Simulation Status", className="status-title"),
                        html.P(id="simulation-status", className="status-text")
                    ]),
                    html.Div([
                        html.H5("Child Age", className="status-title"),
                        html.P(id="child-age", className="status-text")
                    ]),
                    html.Div([
                        html.H5("Milestone Progress", className="status-title"),
                        dbc.Progress(id="milestone-progress-bar", animated=True, 
                                     style={"height": "15px"}, className="mb-2")
                    ])
                ])
            ], className="status-card")
        ], width=4)
    ], className="header-row"),
    
    # Control Panel
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Control Panel"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Initialize", id="initialize-btn", color="primary", className="control-btn"),
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Start Training", id="start-btn", color="success", className="control-btn"),
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Pause", id="pause-btn", color="warning", className="control-btn"),
                        ], width=3),
                        dbc.Col([
                            dbc.Button("Stop", id="stop-btn", color="danger", className="control-btn"),
                        ], width=3)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Button("Save State", id="save-btn", color="info", className="control-btn"),
                        ], width=6),
                        dbc.Col([
                            dbc.Button(
                                "Load State", 
                                id="load-btn", 
                                color="info", 
                                className="control-btn",
                                **{
                                    'data-bs-toggle': 'modal',
                                    'data-bs-target': '#load-modal'
                                }
                            ),
                        ], width=6)
                    ])
                ])
            ], className="control-card")
        ], width=12)
    ], className="mb-3"),
    
    # Main Content Tabs
    dbc.Row([
        dbc.Col([
            dbc.Tabs([
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Network Activation"),
                                dbc.CardBody([
                                    dcc.Graph(id="network-activation-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Emotional State"),
                                dbc.CardBody([
                                    dcc.Graph(id="emotion-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Network Training Progress"),
                                dbc.CardBody([
                                    dcc.Graph(id="network-training-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Memory Distribution"),
                                dbc.CardBody([
                                    dcc.Graph(id="memory-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6)
                    ], className="mb-3"),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Milestone Progress"),
                                dbc.CardBody([
                                    dcc.Graph(id="milestone-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Vocabulary Growth"),
                                dbc.CardBody([
                                    dcc.Graph(id="vocab-chart")
                                ])
                            ], className="dashboard-card")
                        ], width=6)
                    ], className="mb-3")
                ], label="Dashboard"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            html.Div(id="network-cards-container", className="network-cards-grid")
                        ], width=12)
                    ])
                ], label="Neural Networks"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Memory Stats"),
                                dbc.CardBody(id="memory-stats-container")
                            ], className="dashboard-card mb-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Recent Memories"),
                                dbc.CardBody(id="recent-memories-container")
                            ], className="dashboard-card mb-3")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Long-term Memory"),
                                dbc.CardBody(id="longterm-memory-container")
                            ], className="dashboard-card mb-3")
                        ], width=12)
                    ])
                ], label="Memory Systems"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Language Stats"),
                                dbc.CardBody(id="language-stats-container")
                            ], className="dashboard-card mb-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Recent Words Learned"),
                                dbc.CardBody(id="recent-words-container")
                            ], className="dashboard-card mb-3")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Grammar Development"),
                                dbc.CardBody(id="grammar-container")
                            ], className="dashboard-card mb-3")
                        ], width=12)
                    ])
                ], label="Language Development"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Identity Formation"),
                                dbc.CardBody(id="identity-container")
                            ], className="dashboard-card mb-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Belief Systems"),
                                dbc.CardBody(id="beliefs-container")
                            ], className="dashboard-card mb-3")
                        ], width=6)
                    ]),
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Latest Thoughts"),
                                dbc.CardBody(id="latest-thoughts-container")
                            ], className="dashboard-card mb-3")
                        ], width=6),
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Latest Perceptions"),
                                dbc.CardBody(id="latest-perceptions-container")
                            ], className="dashboard-card mb-3")
                        ], width=6)
                    ])
                ], label="Consciousness"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Mother-Child Interaction"),
                                dbc.CardBody([
                                    html.Div(id="interaction-history-container", className="interaction-history"),
                                    html.Hr(),
                                    dbc.InputGroup([
                                        dbc.Input(id="custom-interaction-input", placeholder="Simulate a specific utterance from the child..."),
                                        dbc.Button("Send", id="custom-interaction-btn", color="primary")
                                    ])
                                ])
                            ], className="dashboard-card")
                        ], width=12)
                    ])
                ], label="Parent-Child Interaction"),
                dbc.Tab([
                    dbc.Row([
                        dbc.Col([
                            dbc.Card([
                                dbc.CardHeader("Chat with Neural Child"),
                                dbc.CardBody([
                                    html.Div(id="chat-container", className="chat-container"),
                                    html.Hr(),
                                    dbc.InputGroup([
                                        dbc.Input(id="chat-input", placeholder="Type your message...", disabled=True),
                                        dbc.Button("Send", id="chat-btn", color="primary", disabled=True)
                                    ])
                                ])
                            ], className="dashboard-card")
                        ], width=12)
                    ])
                ], label="Chat", id="chat-tab")
            ], className="tabs-container")
        ], width=12)
    ])
], fluid=True, className="main-container")

# Load state modal
app.layout = html.Div([
    app.layout,
    dbc.Modal([
        dbc.ModalHeader("Load Saved State"),
        dbc.ModalBody([
            html.Div(id="saved-states-list"),
            dbc.Input(id="state-path-input", placeholder="Path to saved state...", className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Cancel", id="cancel-load-btn", className="me-2", color="secondary", 
                       data_bs_dismiss="modal"),
            dbc.Button("Load", id="confirm-load-btn", color="primary", data_bs_dismiss="modal")
        ])
    ], id="load-modal")
])

# Define callbacks
@app.callback(
    [
        Output("simulation-status", "children"),
        Output("child-age", "children"),
        Output("milestone-progress-bar", "value"),
        Output("milestone-progress-bar", "label"),
        Output("milestone-progress-bar", "color")
    ],
    [Input("refresh-interval", "n_intervals")]
)
def update_status(n_intervals):
    """Update the status indicators"""
    status_text = "Not Started"
    
    if dashboard_state.training_active:
        if dashboard_state.paused:
            status_text = "Paused"
        else:
            status_text = "Training Active"
    
    age_text = f"{dashboard_state.child_age_days:.2f} days"
    
    milestone_pct = dashboard_state.overall_milestone_progress * 100
    milestone_label = f"{milestone_pct:.1f}%"
    
    # Determine color based on percentage
    if milestone_pct < 30:
        color = "danger"
    elif milestone_pct < 70:
        color = "warning"
    else:
        color = "success"
    
    return status_text, age_text, milestone_pct, milestone_label, color

@app.callback(
    Output("network-activation-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_network_activation_chart(n_intervals):
    """Update the network activation chart"""
    return create_network_activation_figure(dashboard_state.neural_networks)

@app.callback(
    Output("emotion-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_emotion_chart(n_intervals):
    """Update the emotion chart"""
    return create_emotion_chart(dashboard_state.latest_emotions)

@app.callback(
    Output("network-training-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_network_training_chart(n_intervals):
    """Update the network training chart"""
    return create_network_training_chart(dashboard_state.neural_networks)

@app.callback(
    Output("memory-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_memory_chart(n_intervals):
    """Update the memory chart"""
    return memory_pie_chart(dashboard_state.memory_stats)

@app.callback(
    Output("milestone-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_milestone_chart(n_intervals):
    """Update the milestone chart"""
    return create_milestone_progress_chart(dashboard_state.milestone_progress)

@app.callback(
    Output("vocab-chart", "figure"),
    [Input("refresh-interval", "n_intervals")]
)
def update_vocab_chart(n_intervals):
    """Update the vocabulary chart"""
    if "vocabulary_history" in dashboard_state.language_stats:
        return create_vocab_chart(dashboard_state.language_stats["vocabulary_history"])
    return create_vocab_chart([])

@app.callback(
    Output("network-cards-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_network_cards(n_intervals):
    """Update the network cards"""
    return generate_network_cards(dashboard_state.neural_networks)

@app.callback(
    Output("memory-stats-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_memory_stats(n_intervals):
    """Update memory stats display"""
    stats = dashboard_state.memory_stats
    
    if not stats:
        return html.P("No memory stats available yet.")
    
    # Format memory stats
    return html.Div([
        html.H5(f"Total Memories: {stats.get('total_items', 0)}"),
        html.P(f"Average Salience: {stats.get('avg_salience', 0):.2f}"),
        html.H6("Memory Types:"),
        html.Ul([
            html.Li(f"{mem_type}: {count}") 
            for mem_type, count in stats.get('items_by_type', {}).items()
        ]),
        html.H6("Association Strength:"),
        html.P(f"Total Associations: {stats.get('total_associations', 0)}")
    ])

@app.callback(
    Output("recent-memories-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_recent_memories(n_intervals):
    """Update recent memories display"""
    if not neural_child:
        return html.P("Neural child not initialized yet.")
    
    try:
        recent_memories = neural_child.get_recent_memories(5)
        
        if not recent_memories:
            return html.P("No recent memories available yet.")
        
        memory_items = []
        for memory in recent_memories:
            memory_card = dbc.Card([
                dbc.CardBody([
                    html.P(f"Type: {memory.get('type', 'Unknown')}"),
                    html.P(f"Content: {str(memory.get('content', 'No content'))[:100]}..."),
                    html.P(f"Salience: {memory.get('salience', 0):.2f}")
                ])
            ], className="memory-card mb-2")
            
            memory_items.append(memory_card)
        
        return html.Div(memory_items)
        
    except Exception as e:
        logger.error(f"Error displaying recent memories: {str(e)}")
        return html.P(f"Error retrieving memories: {str(e)}")

@app.callback(
    Output("longterm-memory-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_longterm_memory(n_intervals):
    """Update long-term memory display"""
    if not neural_child:
        return html.P("Neural child not initialized yet.")
    
    try:
        longterm_stats = neural_child.get_longterm_memory_stats()
        
        if not longterm_stats:
            return html.P("No long-term memory data available yet.")
        
        return html.Div([
            html.H5("Long-term Memory Overview"),
            html.Div([
                html.Div([
                    html.H6("Domain Stats"),
                    html.Ul([
                        html.Li(f"{domain}: {count}") 
                        for domain, count in longterm_stats.get('domain_counts', {}).items()
                    ])
                ], className="col-md-6"),
                html.Div([
                    html.H6("Identity Structure"),
                    html.P(f"Beliefs: {longterm_stats.get('belief_count', 0)}"),
                    html.P(f"Values: {longterm_stats.get('value_count', 0)}"),
                    html.P(f"Relationships: {longterm_stats.get('relationship_count', 0)}"),
                    html.P(f"Self-attributes: {longterm_stats.get('attribute_count', 0)}")
                ], className="col-md-6")
            ], className="row")
        ])
        
    except Exception as e:
        logger.error(f"Error displaying long-term memory: {str(e)}")
        return html.P(f"Error retrieving long-term memory: {str(e)}")

@app.callback(
    Output("language-stats-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_language_stats(n_intervals):
    """Update language stats display"""
    stats = dashboard_state.language_stats
    
    if not stats:
        return html.P("No language stats available yet.")
    
    return html.Div([
        html.H5(f"Vocabulary Size: {stats.get('vocabulary_size', 0)}"),
        html.P(f"Active Words: {stats.get('active_vocabulary', 0)}"),
        html.P(f"Passive Words: {stats.get('passive_vocabulary', 0)}"),
        html.H6("Linguistic Development:"),
        html.P(f"Comprehension: {stats.get('comprehension', 0):.2f}"),
        html.P(f"Expression: {stats.get('expression', 0):.2f}"),
        html.P(f"Grammar Complexity: {stats.get('grammar_complexity', 0):.2f}"),
        html.P(f"Current Stage: {stats.get('language_stage', 'Unknown')}")
    ])

@app.callback(
    Output("recent-words-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_recent_words(n_intervals):
    """Update recent words display"""
    stats = dashboard_state.language_stats
    
    if not stats or "recent_words" not in stats:
        return html.P("No recent words available yet.")
    
    words = stats.get("recent_words", [])
    
    word_items = []
    for word in words:
        word_card = dbc.Card([
            dbc.CardBody([
                html.H5(word, className="word-title"),
                html.P(f"Understanding: {stats.get('word_stats', {}).get(word, {}).get('understanding', 0):.2f}"),
                html.P(f"Production: {stats.get('word_stats', {}).get(word, {}).get('production', 0):.2f}")
            ])
        ], className="word-card mb-2")
        
        word_items.append(word_card)
    
    return html.Div(word_items) if word_items else html.P("No recent words available yet.")

@app.callback(
    Output("grammar-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_grammar(n_intervals):
    """Update grammar development display"""
    stats = dashboard_state.language_stats
    
    if not stats or "grammar_rules" not in stats:
        return html.P("No grammar development data available yet.")
    
    grammar_rules = stats.get("grammar_rules", {})
    
    # Sort rules by mastery level
    sorted_rules = sorted(grammar_rules.items(), key=lambda x: x[1]["mastery"], reverse=True)
    
    rule_items = []
    for rule_name, rule_data in sorted_rules:
        mastery = rule_data.get("mastery", 0)
        color = "success" if mastery > 0.7 else "warning" if mastery > 0.3 else "danger"
        
        rule_card = dbc.Card([
            dbc.CardBody([
                html.H5(rule_name.replace("_", " ").title(), className="rule-title"),
                html.P(rule_data.get("description", "No description")),
                dbc.Progress(value=mastery*100, color=color, className="mb-2"),
                html.P(f"Examples: {', '.join(rule_data.get('examples', ['No examples'])[:3])}")
            ])
        ], className="rule-card mb-2")
        
        rule_items.append(rule_card)
    
    return html.Div(rule_items) if rule_items else html.P("No grammar rules available yet.")

@app.callback(
    Output("identity-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_identity(n_intervals):
    """Update identity display"""
    identity = dashboard_state.identity_status
    
    if not identity:
        return html.P("No identity data available yet.")
    
    # Format self-concept
    self_concept = identity.get("self_concept", {})
    
    return html.Div([
        html.H5("Self-Concept"),
        html.Div([
            html.H6("Attributes:"),
            html.Ul([
                html.Li(f"{attr}: {certainty:.2f}") 
                for attr, certainty in self_concept.get("attributes", {}).items()
            ]) if self_concept.get("attributes") else html.P("No attributes yet.")
        ]),
        html.Div([
            html.H6("Preferences:"),
            html.Div([
                html.Div([
                    html.H6(category.title()),
                    html.Ul([
                        html.Li(f"{item}: {strength:.2f}") 
                        for item, strength in prefs.items()
                    ])
                ]) for category, prefs in self_concept.get("preferences", {}).items()
            ]) if self_concept.get("preferences") else html.P("No preferences yet.")
        ]),
        html.Div([
            html.H6("Core Values:"),
            html.Ul([
                html.Li(f"{value}: {importance:.2f}") 
                for value, importance in self_concept.get("values", {}).items()
            ]) if self_concept.get("values") else html.P("No values developed yet.")
        ])
    ])

@app.callback(
    Output("beliefs-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_beliefs(n_intervals):
    """Update beliefs display"""
    identity = dashboard_state.identity_status
    
    if not identity:
        return html.P("No belief data available yet.")
    
    belief_systems = identity.get("belief_systems", {})
    
    if not belief_systems:
        return html.P("No belief systems developed yet.")
    
    belief_items = []
    for name, system_data in belief_systems.items():
        belief_card = dbc.Card([
            dbc.CardHeader(name.title()),
            dbc.CardBody([
                html.P(f"Values: {system_data.get('value_count', 0)}"),
                html.P(f"Beliefs: {system_data.get('belief_count', 0)}"),
                html.P(f"Consistency: {system_data.get('consistency', 0):.2f}"),
                html.P(f"Age: {system_data.get('age_days', 0)} days")
            ])
        ], className="belief-card mb-2")
        
        belief_items.append(belief_card)
    
    return html.Div(belief_items)

@app.callback(
    Output("latest-thoughts-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_latest_thoughts(n_intervals):
    """Update latest thoughts display"""
    thoughts = dashboard_state.latest_thoughts
    
    if not thoughts:
        return html.P("No thoughts available yet.")
    
    thought_items = []
    for thought in thoughts:
        thought_items.append(html.Div(thought, className="thought-bubble"))
    
    return html.Div(thought_items)

@app.callback(
    Output("latest-perceptions-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_latest_perceptions(n_intervals):
    """Update latest perceptions display"""
    perceptions = dashboard_state.latest_perception
    
    if not perceptions:
        return html.P("No perceptions available yet.")
    
    perception_items = []
    for perception in perceptions:
        perception_items.append(html.Div(perception, className="perception-item"))
    
    return html.Div(perception_items)

@app.callback(
    Output("interaction-history-container", "children"),
    [Input("refresh-interval", "n_intervals")]
)
def update_interaction_history(n_intervals):
    """Update interaction history display"""
    if not neural_child:
        return html.P("Neural child not initialized yet.")
    
    try:
        # Get recent interactions
        if hasattr(neural_child, "get_recent_interactions"):
            interactions = neural_child.get_recent_interactions(10)
        else:
            # Fallback to global state
            interactions = [dashboard_state.latest_interaction] if dashboard_state.latest_interaction else []
        
        if not interactions:
            return html.P("No interactions available yet.")
        
        interaction_items = []
        for interaction in interactions:
            if isinstance(interaction, dict):
                child_message = interaction.get("child_message", "")
                mother_response = interaction.get("mother_response", "")
                mother_emotion = interaction.get("mother_emotion", "neutral")
                
                if child_message or mother_response:
                    interaction_card = dbc.Card([
                        dbc.CardBody([
                            html.Div([
                                html.P(f"Child: {child_message}", className="child-message")
                            ]),
                            html.Div([
                                html.P(f"Mother ({mother_emotion}): {mother_response}", className="mother-message")
                            ])
                        ])
                    ], className="interaction-card mb-2")
                    
                    interaction_items.append(interaction_card)
        
        return html.Div(interaction_items)
        
    except Exception as e:
        logger.error(f"Error displaying interactions: {str(e)}")
        return html.P(f"Error retrieving interactions: {str(e)}")

@app.callback(
    Output("chat-container", "children"),
    [Input("refresh-interval", "n_intervals"),
     Input("chat-store", "data")]
)
def update_chat_display(n_intervals, chat_data):
    """Update chat display"""
    global chat_history
    
    if not dashboard_state.chat_enabled:
        return html.Div([
            html.P("Chat is not enabled yet. The neural child needs to reach sufficient development first.", 
                   className="chat-disabled-message"),
            html.P(f"Current milestone progress: {dashboard_state.overall_milestone_progress:.2%}", 
                   className="chat-progress-message"),
            html.P(f"Required milestone progress: {CHAT_MILESTONE_THRESHOLD:.2%}", 
                   className="chat-threshold-message")
        ])
    
    if not chat_history:
        # Add a welcome message
        chat_history.append(ChatMessage(
            role="system",
            content="Welcome to the Neural Child chat! You can now interact directly with the developed mind."
        ))
    
    chat_messages = []
    for msg in chat_history:
        if msg.role == "user":
            chat_messages.append(html.Div(msg.content, className="user-message"))
        elif msg.role == "assistant":
            chat_messages.append(html.Div(msg.content, className="assistant-message"))
        elif msg.role == "mother":
            chat_messages.append(html.Div([
                html.Span("Mother: ", className="mother-tag"),
                msg.content
            ], className="mother-message"))
        elif msg.role == "system":
            chat_messages.append(html.Div(msg.content, className="system-message"))
    
    return html.Div(chat_messages, id="chat-messages")

@app.callback(
    [Output("chat-input", "disabled"),
     Output("chat-btn", "disabled")],
    [Input("refresh-interval", "n_intervals")]
)
def update_chat_controls(n_intervals):
    """Enable or disable chat controls based on development progress"""
    chat_disabled = not dashboard_state.chat_enabled
    return chat_disabled, chat_disabled

@app.callback(
    Output("chat-store", "data"),
    [Input("chat-btn", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-store", "data")]
)
def handle_chat_message(n_clicks, message, chat_data):
    """Handle user chat messages"""
    global neural_child, chat_history
    
    if not n_clicks or not message or not neural_child or not dashboard_state.chat_enabled:
        raise PreventUpdate
    
    try:
        # Add user message to chat history
        chat_history.append(ChatMessage(
            role="user",
            content=message
        ))
        
        # Process the message with the neural child
        response = neural_child.process_chat_message(message)
        
        # Add neural child's response to chat history
        chat_history.append(ChatMessage(
            role="assistant",
            content=response
        ))
        
        # Update chat data store to trigger refresh
        return {"history": [msg.model_dump() for msg in chat_history], "updated": datetime.now().isoformat()}
        
    except Exception as e:
        logger.error(f"Error processing chat message: {str(e)}")
        # Add error message to chat history
        chat_history.append(ChatMessage(
            role="system",
            content=f"Error: {str(e)}"
        ))
        return {"history": [msg.model_dump() for msg in chat_history], "error": str(e)}

@app.callback(
    Output("custom-interaction-input", "value"),
    [Input("custom-interaction-btn", "n_clicks")],
    [State("custom-interaction-input", "value")]
)
def handle_custom_interaction(n_clicks, message):
    """Handle custom interaction from the UI"""
    if not n_clicks or not message or not neural_child:
        raise PreventUpdate
    
    try:
        # Add to interaction queue
        interaction_queue.put({
            "type": "custom",
            "message": message
        })
        
        return ""  # Clear the input
        
    except Exception as e:
        logger.error(f"Error processing custom interaction: {str(e)}")
        return no_update

@app.callback(
    Output("saved-states-list", "children"),
    [Input("load-btn", "n_clicks")]
)
def update_saved_states_list(n_clicks):
    """Update the list of saved states"""
    if not n_clicks:
        raise PreventUpdate
    
    saved_states_dir = Path("./saved_states")
    if not saved_states_dir.exists():
        return html.P("No saved states found.")
    
    try:
        saved_states = list(saved_states_dir.glob("neural_child_*"))
        
        if not saved_states:
            return html.P("No saved states found.")
        
        # Sort by modification time (newest first)
        saved_states.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        state_items = []
        for state_path in saved_states:
            # Extract timestamp from filename
            try:
                timestamp_str = state_path.name.replace("neural_child_", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                formatted_time = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            except:
                formatted_time = "Unknown date"
            
            state_items.append(
                dbc.Button(
                    f"{formatted_time}",
                    id={"type": "state-item", "index": str(state_path)},
                    color="link",
                    className="saved-state-item"
                )
            )
        
        return html.Div(state_items)
        
    except Exception as e:
        logger.error(f"Error getting saved states: {str(e)}")
        return html.P(f"Error: {str(e)}")

@app.callback(
    Output("state-path-input", "value"),
    [Input({"type": "state-item", "index": dash.dependencies.ALL}, "n_clicks")],
    [State({"type": "state-item", "index": dash.dependencies.ALL}, "id")]
)
def select_state_item(n_clicks_list, ids):
    """Select a saved state item"""
    if not n_clicks_list or not any(n_clicks_list):
        raise PreventUpdate
    
    # Find which button was clicked
    for i, n_clicks in enumerate(n_clicks_list):
        if n_clicks:
            # Extract path from id
            state_path = ids[i]["index"]
            return state_path
    
    return no_update

@app.callback(
    Output("initialize-btn", "disabled"),
    [Input("initialize-btn", "n_clicks")]
)
def initialize_system(n_clicks):
    """Initialize the system"""
    if not n_clicks:
        return False
    
    try:
        init_neural_child()
        return False  # Keep button enabled
    except Exception as e:
        logger.error(f"Error initializing system: {str(e)}")
        return False

@app.callback(
    [Output("start-btn", "disabled"),
     Output("pause-btn", "disabled"),
     Output("stop-btn", "disabled")],
    [Input("start-btn", "n_clicks"),
     Input("pause-btn", "n_clicks"),
     Input("stop-btn", "n_clicks"),
     Input("refresh-interval", "n_intervals")]
)
def handle_training_controls(start_clicks, pause_clicks, stop_clicks, n_intervals):
    """Handle training control buttons"""
    ctx = callback_context
    
    if not ctx.triggered:
        # Default state
        start_disabled = neural_child is None
        pause_disabled = not dashboard_state.training_active or dashboard_state.paused
        stop_disabled = not dashboard_state.training_active
        return start_disabled, pause_disabled, stop_disabled
    
    trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    if trigger_id == "start-btn" and start_clicks:
        start_training()
    elif trigger_id == "pause-btn" and pause_clicks:
        pause_training()
    elif trigger_id == "stop-btn" and stop_clicks:
        stop_training()
    
    # Update button states based on current state
    start_disabled = dashboard_state.training_active and not dashboard_state.paused
    pause_disabled = not dashboard_state.training_active or dashboard_state.paused
    stop_disabled = not dashboard_state.training_active
    
    return start_disabled, pause_disabled, stop_disabled

@app.callback(
    Output("save-btn", "disabled"),
    [Input("save-btn", "n_clicks")]
)
def handle_save(n_clicks):
    """Handle save button"""
    if not n_clicks:
        return neural_child is None
    
    save_state()
    return neural_child is None

@app.callback(
    Output("confirm-load-btn", "disabled"),
    [Input("confirm-load-btn", "n_clicks")],
    [State("state-path-input", "value")]
)
def handle_load(n_clicks, state_path):
    """Handle load button"""
    if not n_clicks or not state_path:
        return False
    
    # Stop training if active
    if dashboard_state.training_active:
        stop_training()
    
    load_state(state_path)
    return False

# CSS for the dashboard
app.layout = html.Div([
    app.layout,
    html.Link(rel="stylesheet", href="/assets/dashboard.css")
])

# Create assets folder and CSS file if they don't exist
assets_dir = Path("./assets")
assets_dir.mkdir(exist_ok=True)

css_path = assets_dir / "dashboard.css"
if not css_path.exists():
    with open(css_path, "w") as f:
        f.write("""
/* Main container */
.main-container {
    background-color: #111111;
    color: #e0e0e0;
    min-height: 100vh;
    padding: 20px;
}

/* Header */
.header-title {
    color: #39ff14;
    font-weight: bold;
    margin-bottom: 0;
}

.header-subtitle {
    color: #cccccc;
    font-style: italic;
}

.header-row {
    margin-bottom: 20px;
}

/* Status card */
.status-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
    height: 100%;
}

.status-title {
    color: #39ff14;
    font-size: 1rem;
    margin-bottom: 5px;
}

.status-text {
    color: #e0e0e0;
    font-size: 1.25rem;
    font-weight: bold;
    margin-bottom: 10px;
}

/* Control card */
.control-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

.control-btn {
    width: 100%;
}

/* Dashboard cards */
.dashboard-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
    height: 100%;
}

/* Network cards */
.network-cards-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
    gap: 20px;
    padding: 10px;
}

.network-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

.network-stat {
    font-weight: bold;
    margin-bottom: 5px;
}

.network-details {
    font-size: 0.9rem;
    max-height: 200px;
    overflow-y: auto;
}

/* Memory card */
.memory-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

/* Word card */
.word-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

.word-title {
    color: #39ff14;
}

/* Rule card */
.rule-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

.rule-title {
    color: #39ff14;
}

/* Belief card */
.belief-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

/* Thought bubble */
.thought-bubble {
    background-color: #2a2a2a;
    border-radius: 15px;
    padding: 10px;
    margin-bottom: 10px;
    position: relative;
}

.thought-bubble:after {
    content: '';
    position: absolute;
    bottom: -10px;
    left: 20px;
    border-width: 10px 10px 0;
    border-style: solid;
    border-color: #2a2a2a transparent;
}

/* Perception item */
.perception-item {
    background-color: #2a2a2a;
    border-radius: 5px;
    padding: 10px;
    margin-bottom: 5px;
}

/* Interaction card */
.interaction-card {
    background-color: #1a1a1a;
    border: 1px solid #2c2c2c;
    border-radius: 5px;
}

.interaction-history {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 20px;
}

.child-message {
    color: #39ff14;
    margin-bottom: 5px;
}

.mother-message {
    color: #e0e0e0;
    margin-bottom: 5px;
}

/* Chat container */
.chat-container {
    height: 400px;
    overflow-y: auto;
    margin-bottom: 20px;
    padding: 10px;
}

.user-message {
    background-color: #1e3b1e;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    max-width: 80%;
    margin-left: auto;
    color: #ffffff;
}

.assistant-message {
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    max-width: 80%;
    color: #39ff14;
}

.mother-message {
    background-color: #2a2a2a;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    max-width: 80%;
    color: #e0e0e0;
}

.mother-tag {
    color: #ffd700;
    font-weight: bold;
}

.system-message {
    background-color: #333333;
    border-radius: 10px;
    padding: 10px;
    margin-bottom: 10px;
    max-width: 90%;
    margin-left: auto;
    margin-right: auto;
    color: #cccccc;
    font-style: italic;
    text-align: center;
}

.chat-disabled-message {
    color: #cccccc;
    text-align: center;
    margin-top: 50px;
    font-size: 1.2rem;
}

.chat-progress-message {
    color: #39ff14;
    text-align: center;
    font-size: 1.1rem;
}

.chat-threshold-message {
    color: #ffd700;
    text-align: center;
    font-size: 1.1rem;
}

/* Tabs container */
.tabs-container {
    margin-top: 10px;
}

/* Modal */
.saved-state-item {
    display: block;
    text-align: left;
    margin-bottom: 5px;
}
""")

# Main entry point
if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)