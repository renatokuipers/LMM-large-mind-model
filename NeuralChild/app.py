import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from typing import Dict, List, Optional, Union, Literal
from pydantic import BaseModel, Field, field_validator
import json
import os

# ==== PYDANTIC MODELS ====

class NeuralMetrics(BaseModel):
    """Metrics for each neural network component"""
    network_name: str
    confidence: float = Field(ge=0.0, le=1.0)
    activation_level: float = Field(ge=0.0, le=1.0)
    training_progress: float = Field(ge=0.0, le=1.0)
    last_active: datetime
    error_rate: float = Field(ge=0.0)
    connection_strength: Dict[str, float] = Field(default_factory=dict)
    
    @field_validator('connection_strength')
    @classmethod
    def validate_connections(cls, v):
        for key, value in v.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"Connection strength for {key} must be between 0 and 1")
        return v

class EmotionalState(BaseModel):
    """Current emotional state of the neural child"""
    joy: float = Field(ge=0.0, le=1.0, default=0.0)
    sadness: float = Field(ge=0.0, le=1.0, default=0.0)
    anger: float = Field(ge=0.0, le=1.0, default=0.0)
    fear: float = Field(ge=0.0, le=1.0, default=0.0)
    surprise: float = Field(ge=0.0, le=1.0, default=0.0)
    disgust: float = Field(ge=0.0, le=1.0, default=0.0)
    trust: float = Field(ge=0.0, le=1.0, default=0.0)
    anticipation: float = Field(ge=0.0, le=1.0, default=0.0)
    
    def dominant_emotion(self) -> str:
        emotions = {
            "Joy": self.joy,
            "Sadness": self.sadness,
            "Anger": self.anger,
            "Fear": self.fear,
            "Surprise": self.surprise,
            "Disgust": self.disgust,
            "Trust": self.trust,
            "Anticipation": self.anticipation
        }
        return max(emotions, key=emotions.get)

class TrainingSession(BaseModel):
    """Information about a training session"""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    duration_minutes: Optional[float] = None
    interactions_count: int = 0
    emotional_growth: Dict[str, float] = Field(default_factory=dict)
    vocabulary_size: int = 0
    
    @field_validator('duration_minutes', mode='before')
    @classmethod
    def calculate_duration(cls, v, info):
        values = info.data
        if values.get('end_time') and values.get('start_time'):
            return (values['end_time'] - values['start_time']).total_seconds() / 60
        return v

class ChatMessage(BaseModel):
    """Model for chat messages"""
    sender: Literal["user", "mother", "neural_child"]
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    emotional_context: Optional[EmotionalState] = None
    active_networks: List[str] = Field(default_factory=list)

class SystemStatus(BaseModel):
    """Overall system status"""
    is_training: bool = False
    total_training_time: float = 0  # in hours
    total_interactions: int = 0
    child_age: float = 0  # in simulated days
    chat_ready: bool = False
    vocabulary_size: int = 0
    active_networks: List[str] = Field(default_factory=list)
    system_load: float = 0.0
    last_updated: datetime = Field(default_factory=datetime.now)

# ==== DEMO DATA GENERATION ====

def generate_demo_data():
    """Generate demo data for visualizations"""
    # System status
    system_status = SystemStatus(
        is_training=True,
        total_training_time=72.5,
        total_interactions=4328,
        child_age=15.2,
        chat_ready=False,
        vocabulary_size=1250,
        active_networks=["Perception", "Attention", "Emotions", "Consciousness"],
        system_load=0.67
    )
    
    # Neural network components with metrics
    neural_components = [
        NeuralMetrics(
            network_name="Archetypes",
            confidence=0.45,
            activation_level=0.38,
            training_progress=0.52,
            last_active=datetime.now() - timedelta(minutes=35),
            error_rate=0.08,
            connection_strength={"Unconsciousness": 0.75, "Emotions": 0.62}
        ),
        NeuralMetrics(
            network_name="Instincts",
            confidence=0.72,
            activation_level=0.65,
            training_progress=0.78,
            last_active=datetime.now() - timedelta(minutes=12),
            error_rate=0.05,
            connection_strength={"Drives": 0.88, "Emotions": 0.79}
        ),
        NeuralMetrics(
            network_name="Emotions",
            confidence=0.68,
            activation_level=0.81,
            training_progress=0.65,
            last_active=datetime.now() - timedelta(minutes=5),
            error_rate=0.04,
            connection_strength={"Moods": 0.92, "Consciousness": 0.85}
        ),
        NeuralMetrics(
            network_name="Consciousness",
            confidence=0.58,
            activation_level=0.73,
            training_progress=0.62,
            last_active=datetime.now() - timedelta(minutes=2),
            error_rate=0.06,
            connection_strength={"Thoughts": 0.81, "Attention": 0.77}
        )
    ]
    
    # Emotional state
    emotion_state = EmotionalState(
        joy=0.65,
        sadness=0.15,
        anger=0.05,
        fear=0.08,
        surprise=0.42,
        disgust=0.03,
        trust=0.72,
        anticipation=0.58
    )
    
    # Chat history
    chat_history = [
        ChatMessage(
            sender="mother",
            content="Hello, little one. How are you feeling today?",
            timestamp=datetime.now() - timedelta(minutes=15),
            active_networks=["Perception", "Attention"]
        ),
        ChatMessage(
            sender="neural_child",
            content="H-hello. I... feel... good?",
            timestamp=datetime.now() - timedelta(minutes=14),
            emotional_context=EmotionalState(joy=0.6, surprise=0.4, trust=0.7),
            active_networks=["Emotions", "Consciousness", "Perception"]
        ),
        ChatMessage(
            sender="mother",
            content="That's wonderful! Would you like to learn something new today?",
            timestamp=datetime.now() - timedelta(minutes=13),
            active_networks=["Perception", "Attention"]
        ),
        ChatMessage(
            sender="neural_child",
            content="Yes! Learn... new things. I want.",
            timestamp=datetime.now() - timedelta(minutes=12),
            emotional_context=EmotionalState(joy=0.8, anticipation=0.75, trust=0.85),
            active_networks=["Emotions", "Consciousness", "Attention"]
        )
    ]
    
    # Training session history
    training_sessions = [
        TrainingSession(
            session_id="sess_001",
            start_time=datetime.now() - timedelta(days=5, hours=8),
            end_time=datetime.now() - timedelta(days=5, hours=4),
            interactions_count=485,
            emotional_growth={"joy": 0.15, "trust": 0.22},
            vocabulary_size=320
        ),
        TrainingSession(
            session_id="sess_002",
            start_time=datetime.now() - timedelta(days=4, hours=9),
            end_time=datetime.now() - timedelta(days=4, hours=3),
            interactions_count=612,
            emotional_growth={"joy": 0.18, "trust": 0.25, "surprise": 0.12},
            vocabulary_size=520
        ),
        TrainingSession(
            session_id="sess_003",
            start_time=datetime.now() - timedelta(days=2, hours=10),
            end_time=datetime.now() - timedelta(days=2, hours=4),
            interactions_count=758,
            emotional_growth={"joy": 0.22, "trust": 0.28, "surprise": 0.15},
            vocabulary_size=850
        ),
        TrainingSession(
            session_id="sess_004",
            start_time=datetime.now() - timedelta(days=1, hours=7),
            end_time=datetime.now() - timedelta(days=1, hours=1),
            interactions_count=823,
            emotional_growth={"joy": 0.25, "trust": 0.32, "surprise": 0.18, "sadness": 0.08},
            vocabulary_size=1100
        ),
        TrainingSession(
            session_id="sess_005",
            start_time=datetime.now() - timedelta(hours=8),
            end_time=None,
            interactions_count=1650,
            emotional_growth={"joy": 0.32, "trust": 0.38, "surprise": 0.22, "sadness": 0.12},
            vocabulary_size=1250
        )
    ]
    
    return {
        "system_status": system_status,
        "neural_components": neural_components,
        "emotion_state": emotion_state,
        "chat_history": chat_history,
        "training_sessions": training_sessions
    }

# ==== DASH APP SETUP ====

# Initialize the Dash app with dark theme
app = dash.Dash(
    __name__, 
    external_stylesheets=[dbc.themes.DARKLY],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}]
)

app.title = "NeuralChild Mind Dashboard"

# Custom CSS for rounded corners and modern look
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            :root {
                --primary-bg: #121212;
                --secondary-bg: #1e1e1e;
                --accent-color: #00b4d8;
                --text-color: #ffffff;
                --card-bg: #2a2a2a;
                --success-color: #4ade80;
                --warning-color: #fbbf24;
                --danger-color: #f87171;
                --info-color: #60a5fa;
            }
            
            body {
                background-color: var(--primary-bg);
                color: var(--text-color);
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }
            
            .dash-card {
                background-color: var(--card-bg);
                border-radius: 16px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                padding: 16px;
                margin-bottom: 16px;
                transition: transform 0.2s;
            }
            
            .dash-card:hover {
                transform: translateY(-5px);
            }
            
            .nav-link {
                border-radius: 8px;
                margin: 5px 0;
            }
            
            .nav-link.active {
                background-color: var(--accent-color) !important;
            }
            
            .btn {
                border-radius: 8px;
            }
            
            .form-control {
                border-radius: 8px;
                background-color: var(--secondary-bg);
                border: 1px solid #444;
                color: var(--text-color);
            }
            
            .gauge-container {
                display: flex;
                align-items: center;
                justify-content: center;
            }
            
            .chat-container {
                height: 400px;
                overflow-y: auto;
                background-color: var(--secondary-bg);
                border-radius: 16px;
                padding: 16px;
            }
            
            .user-message {
                background-color: var(--accent-color);
                padding: 10px 15px;
                border-radius: 18px 18px 0 18px;
                margin: 10px 0;
                max-width: 80%;
                align-self: flex-end;
                margin-left: auto;
            }
            
            .ai-message {
                background-color: #444;
                padding: 10px 15px;
                border-radius: 18px 18px 18px 0;
                margin: 10px 0;
                max-width: 80%;
            }
            
            .mother-message {
                background-color: #7b2cbf;
                padding: 10px 15px;
                border-radius: 18px 18px 18px 0;
                margin: 10px 0;
                max-width: 80%;
            }
            
            .status-badge {
                border-radius: 12px;
                padding: 5px 10px;
                font-size: 0.8rem;
                font-weight: bold;
            }
            
            .network-card {
                background: linear-gradient(145deg, #2a2a2a, #333333);
                border-radius: 16px;
                padding: 16px;
                margin-bottom: 16px;
                border-left: 4px solid var(--accent-color);
            }
            
            .progress {
                height: 10px;
                border-radius: 5px;
            }
            
            .tab-content {
                padding: 20px 0;
            }
            
            .nav-tabs .nav-link {
                border-radius: 8px 8px 0 0;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# ==== DASHBOARD COMPONENTS ====

# Generate demo data
demo_data = generate_demo_data()

# System status indicators
system_status_card = dbc.Card([
    dbc.CardHeader([
        html.H4("System Status", className="d-flex align-items-center"),
        dbc.Badge(
            "Training" if demo_data["system_status"].is_training else "Idle",
            color="success" if demo_data["system_status"].is_training else "warning",
            className="ml-auto status-badge"
        )
    ], className="d-flex justify-content-between align-items-center"),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.H6("Neural Child Age"),
                    html.H3(f"{demo_data['system_status'].child_age:.1f} days", className="text-info")
                ], className="text-center mb-3"),
                html.Div([
                    html.H6("Vocabulary Size"),
                    html.H3(f"{demo_data['system_status'].vocabulary_size} words", className="text-success")
                ], className="text-center")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H6("Total Training Time"),
                    html.H3(f"{demo_data['system_status'].total_training_time:.1f} hours", className="text-warning")
                ], className="text-center mb-3"),
                html.Div([
                    html.H6("Total Interactions"),
                    html.H3(f"{demo_data['system_status'].total_interactions}", className="text-primary")
                ], className="text-center")
            ], width=4),
            dbc.Col([
                html.Div([
                    html.H6("Chat Ready"),
                    html.H3(
                        "Yes" if demo_data["system_status"].chat_ready else "No", 
                        className=f"text-{'success' if demo_data['system_status'].chat_ready else 'danger'}"
                    )
                ], className="text-center mb-3"),
                html.Div([
                    html.H6("System Load"),
                    dbc.Progress(
                        value=int(demo_data["system_status"].system_load * 100),
                        color="success" if demo_data["system_status"].system_load < 0.7 else "warning"
                    )
                ], className="text-center")
            ], width=4),
        ])
    ])
], className="mb-4 dash-card")

# Emotion state radar chart
def create_emotion_radar():
    emotions = demo_data["emotion_state"].model_dump()
    categories = list(emotions.keys())
    values = list(emotions.values())
    values.append(values[0])  # Close the radar plot
    categories.append(categories[0])  # Close the radar plot
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values,
        theta=categories,
        fill='toself',
        fillcolor='rgba(0, 180, 216, 0.3)',
        line=dict(color='#00b4d8', width=3)
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False,
        margin=dict(l=40, r=40, t=30, b=30),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    
    return fig

emotional_state_card = dbc.Card([
    dbc.CardHeader([
        html.H4("Emotional State"),
        dbc.Badge(
            demo_data["emotion_state"].dominant_emotion(),
            color="info",
            className="ml-auto status-badge"
        )
    ], className="d-flex justify-content-between align-items-center"),
    dbc.CardBody([
        dcc.Graph(
            figure=create_emotion_radar(),
            config={'displayModeBar': False},
            style={"height": "300px"}
        )
    ])
], className="mb-4 dash-card")

# Neural network components
neural_components_list = []
for component in demo_data["neural_components"]:
    neural_components_list.append(
        dbc.Card([
            dbc.CardHeader([
                html.H5(component.network_name),
                dbc.Badge(
                    f"{int(component.activation_level * 100)}% Active",
                    color="success" if component.activation_level > 0.5 else "warning",
                    className="ml-auto status-badge"
                )
            ], className="d-flex justify-content-between align-items-center"),
            dbc.CardBody([
                html.Div([
                    html.Span("Training Progress:", className="mr-2"),
                    dbc.Progress(
                        value=int(component.training_progress * 100),
                        color="info",
                        className="mb-3"
                    )
                ]),
                html.Div([
                    html.Span("Confidence:", className="mr-2"),
                    dbc.Progress(
                        value=int(component.confidence * 100),
                        color="success",
                        className="mb-3"
                    )
                ]),
                html.Div([
                    html.Span("Error Rate:", className="mr-2"),
                    dbc.Progress(
                        value=int(min(component.error_rate * 100, 100)),
                        color="danger",
                        className="mb-3"
                    )
                ]),
                html.Hr(),
                html.H6("Connected Networks", className="mb-3"),
                html.Div([
                    dbc.Badge(
                        f"{network}: {strength:.2f}",
                        color="primary" if strength > 0.7 else "secondary",
                        className="mr-2 mb-2"
                    ) for network, strength in component.connection_strength.items()
                ])
            ])
        ], className="mb-3 network-card")
    )

neural_components_card = dbc.Card([
    dbc.CardHeader(html.H4("Neural Network Components")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col(neural_components_list)
        ])
    ])
], className="mb-4 dash-card")

# Training history chart
def create_training_history_chart():
    sessions = demo_data["training_sessions"]
    dates = [session.start_time.strftime('%Y-%m-%d') for session in sessions]
    interactions = [session.interactions_count for session in sessions]
    vocabulary = [session.vocabulary_size for session in sessions]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=dates,
        y=interactions,
        name='Interactions',
        marker_color='#00b4d8'
    ))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=vocabulary,
        name='Vocabulary Size',
        marker_color='#4ade80',
        mode='lines+markers'
    ))
    
    fig.update_layout(
        title='Training Progress',
        xaxis_title='Date',
        yaxis_title='Count',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        template="plotly_dark",
        margin=dict(l=40, r=40, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )
    
    return fig

training_history_card = dbc.Card([
    dbc.CardHeader(html.H4("Training History")),
    dbc.CardBody([
        dcc.Graph(
            figure=create_training_history_chart(),
            config={'displayModeBar': False}
        )
    ])
], className="mb-4 dash-card")

# Chat interface
chat_interface = dbc.Card([
    dbc.CardHeader([
        html.H4("Interaction Console"),
        dbc.Badge(
            "Ready" if demo_data["system_status"].chat_ready else "Not Ready",
            color="success" if demo_data["system_status"].chat_ready else "danger",
            className="ml-auto status-badge"
        )
    ], className="d-flex justify-content-between align-items-center"),
    dbc.CardBody([
        html.Div([
            # Chat messages container
            html.Div(id="chat-messages-container", className="d-flex flex-column", children=[
                html.Div([
                    html.Strong(f"{msg.sender.capitalize()}: ", className="mr-2"),
                    html.Span(msg.content)
                ], className=f"{'user-message' if msg.sender == 'user' else ('mother-message' if msg.sender == 'mother' else 'ai-message')}")
                for msg in demo_data["chat_history"]
            ], style={"height": "350px", "overflowY": "auto"}),
            
            # Input area with updated InputGroup syntax
            html.Div([
                dbc.InputGroup([
                    dbc.Input(
                        id="chat-input",
                        placeholder="Type your message here..." if demo_data["system_status"].chat_ready else "Chat will be available after sufficient training",
                        disabled=not demo_data["system_status"].chat_ready
                    ),
                    dbc.Button(
                        "Send", 
                        id="send-button", 
                        color="primary",
                        disabled=not demo_data["system_status"].chat_ready
                    )
                ], className="mt-3")
            ])
        ])
    ])
], className="mb-4 dash-card")

# Network visualization
def create_network_graph():
    import math
    # Build node list from neural_components and add any missing targets from connection_strength
    nodes = [component.network_name for component in demo_data["neural_components"]]
    for component in demo_data["neural_components"]:
        for target, strength in component.connection_strength.items():
            if target not in nodes:
                nodes.append(target)
    
    # Compute positions using a circular layout
    n = len(nodes)
    positions = {}
    radius = 3
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n
        positions[node] = (radius * math.cos(angle), radius * math.sin(angle))
    
    # Create node trace with computed positions
    node_x = [positions[node][0] for node in nodes]
    node_y = [positions[node][1] for node in nodes]
    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode='markers+text',
        marker=dict(
            size=20,
            color='#00b4d8',
            line=dict(width=2, color='white')
        ),
        text=nodes,
        textposition="top center"
    )
    
    # Create edges using the computed positions
    edge_traces = []
    for component in demo_data["neural_components"]:
        for target, weight in component.connection_strength.items():
            if target not in positions:
                continue  # safety check, though all targets are in positions now
            x0, y0 = positions[component.network_name]
            x1, y1 = positions[target]
            edge_trace = go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=weight * 5, color='rgba(255, 255, 255, 0.3)'),
                hoverinfo='none'
            )
            edge_traces.append(edge_trace)
    
    # Create the figure with edges and node trace
    fig = go.Figure(data=edge_traces + [node_trace])
    fig.update_layout(
        title='Neural Network Connections (Simplified Visualization)',
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20, l=5, r=5, t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        template="plotly_dark",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    return fig

network_visualization_card = dbc.Card([
    dbc.CardHeader(html.H4("Network Visualization")),
    dbc.CardBody([
        dcc.Graph(
            figure=create_network_graph(),
            config={'displayModeBar': False}
        )
    ])
], className="mb-4 dash-card")

# Control panel
control_panel = dbc.Card([
    dbc.CardHeader(html.H4("Control Panel")),
    dbc.CardBody([
        dbc.Row([
            dbc.Col([
                html.H5("Training Controls", className="mb-3"),
                dbc.Button("Start Training", color="success", className="mr-2 mb-2"),
                dbc.Button("Pause Training", color="warning", className="mr-2 mb-2"),
                dbc.Button("Reset NeuralChild", color="danger", className="mb-2"),
                html.Hr(),
                html.H5("Monitoring Settings", className="mb-3"),
                html.Div([
                    dbc.Label("Update Interval"),
                    dbc.Select(
                        options=[
                            {"label": "Real-time", "value": "1"},
                            {"label": "Every 5 seconds", "value": "5"},
                            {"label": "Every 10 seconds", "value": "10"},
                            {"label": "Every 30 seconds", "value": "30"},
                            {"label": "Manual refresh", "value": "0"}
                        ],
                        value="5"
                    )
                ], className="mb-3"),
                html.Div([
                    dbc.Label("Display Components"),
                    dbc.Checklist(
                        options=[
                            {"label": "System Status", "value": "status"},
                            {"label": "Neural Networks", "value": "networks"},
                            {"label": "Emotional State", "value": "emotions"},
                            {"label": "Training History", "value": "history"},
                            {"label": "Network Visualization", "value": "visualization"}
                        ],
                        value=["status", "networks", "emotions", "history", "visualization"],
                        id="display-components"
                    )
                ], className="mb-3")
            ], width=6),
            dbc.Col([
                html.H5("System Configuration", className="mb-3"),
                html.Div([
                    dbc.Label("Chat Readiness Threshold"),
                    dbc.Input(type="number", value=10, min=1, max=100, step=1)
                ], className="mb-3"),
                html.Div([
                    dbc.Label("Mother LLM Model"),
                    dbc.Select(
                        options=[
                            {"label": "qwen2.5-7b-instruct", "value": "qwen2.5-7b-instruct"},
                            {"label": "qwen2.5-72b-instruct", "value": "qwen2.5-72b-instruct"},
                            {"label": "llama3-8b", "value": "llama3-8b"},
                            {"label": "llama3-70b", "value": "llama3-70b"},
                            {"label": "gemma-7b", "value": "gemma-7b"}
                        ],
                        value="qwen2.5-7b-instruct"
                    )
                ], className="mb-3"),
                html.Div([
                    dbc.Label("Learning Speed Multiplier"),
                    dbc.Input(type="number", value=1.0, min=0.1, max=10, step=0.1)
                ], className="mb-3"),
                html.Hr(),
                html.H5("Export/Import", className="mb-3"),
                dbc.Button("Export NeuralChild State", color="info", className="mr-2 mb-2"),
                dbc.Button("Import NeuralChild State", color="info", className="mb-2"),
            ], width=6)
        ])
    ])
], className="mb-4 dash-card")

# ==== LAYOUT STRUCTURE ====

# Sidebar
sidebar = html.Div([
    html.Div([
        html.H2("NeuralChild", className="display-4 text-center mb-4"),
        html.Hr(),
        dbc.Nav([
            dbc.NavLink("Dashboard", href="#", active=True, className="mb-1"),
            dbc.NavLink("Neural Networks", href="#", className="mb-1"),
            dbc.NavLink("Training History", href="#", className="mb-1"),
            dbc.NavLink("Chat Interface", href="#", className="mb-1"),
            dbc.NavLink("Controls", href="#", className="mb-1"),
            dbc.NavLink("Settings", href="#", className="mb-1"),
            dbc.NavLink("Help", href="#", className="mb-1")
        ], vertical=True, pills=True)
    ], className="p-3")
], className="dash-card", style={"height": "100vh", "position": "fixed", "width": "16rem"})

# Main content
content = html.Div([
    dbc.Tabs([
        dbc.Tab([
            dbc.Row([
                dbc.Col(system_status_card, width=8),
                dbc.Col(emotional_state_card, width=4)
            ]),
            dbc.Row([
                dbc.Col(training_history_card, width=12)
            ]),
            dbc.Row([
                dbc.Col(network_visualization_card, width=12)
            ])
        ], label="Overview", tab_id="tab-overview"),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col(neural_components_card, width=12)
            ])
        ], label="Neural Networks", tab_id="tab-networks"),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col(training_history_card, width=12)
            ])
        ], label="Training History", tab_id="tab-history"),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col(chat_interface, width=12)
            ])
        ], label="Chat Interface", tab_id="tab-chat"),
        
        dbc.Tab([
            dbc.Row([
                dbc.Col(control_panel, width=12)
            ])
        ], label="Controls", tab_id="tab-controls")
    ], id="tabs", active_tab="tab-overview")
], style={"margin-left": "18rem", "margin-right": "2rem", "padding-top": "2rem"})

# Overall app layout
app.layout = html.Div([
    sidebar,
    content
])

# ==== CALLBACKS ====

@app.callback(
    Output("chat-messages-container", "children"),
    [Input("send-button", "n_clicks")],
    [State("chat-input", "value"),
     State("chat-messages-container", "children")]
)
def update_chat(n_clicks, input_value, current_messages):
    if n_clicks is None or not input_value:
        return current_messages
    
    # Add user message
    user_message = html.Div([
        html.Strong("User: ", className="mr-2"),
        html.Span(input_value)
    ], className="user-message")
    
    # In a real implementation, you would:
    # 1. Send the message to the NeuralChild
    # 2. Get the response
    # 3. Update the state
    
    # This is just a placeholder response
    neural_child_message = html.Div([
        html.Strong("NeuralChild: ", className="mr-2"),
        html.Span("I... understand your words. Feel... curious about that.")
    ], className="ai-message")
    
    return current_messages + [user_message, neural_child_message]

# ==== RUN SERVER ====

if __name__ == "__main__":
    app.run_server(debug=True)